"""Run OFF and the locked learned candidate once on all49 confirmation tasks."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

from gemma_particle.confirmation import (load_confirmation_access,claim_confirmation_pair,
    confirmation_pair_identity,validate_confirmation_server,confirmation_asset_names)
from gemma_particle.confirmation_environments import load_confirmation_environments
from gemma_particle.confirmation_results import summarize_confirmation_rows,compare_confirmation_pair
from gemma_particle.conversion_handoff import read
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import candidate_preflight
from gemma_particle.dev_evidence import audit_verified_task,journal_digest,validate_matching_contract
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import sha256_file,wheel_identity
from gemma_particle.lora_serving import BASE_ALIAS


def write(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['access','environments','executor','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--executor-commit',required=True)
    p.add_argument('--selection-plan',type=Path,default=Path('research/dev-selection-v1.json'))
    p.add_argument('--port',type=int,default=8034)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.out.exists():raise FileExistsError('paired confirmation is immutable; preserve and reconcile partial attempts')
    scope=None if args.plan_only else load_confirmation_access(args.access)
    environments=None if args.plan_only else load_confirmation_environments(args.environments,scope)
    selection=load_selection_plan(args.selection_plan)
    if scope is not None and scope.identity['selection_plan_sha256']!=sha256_file(args.selection_plan):
        raise ValueError('original locked confirmation selection plan required')
    if not 1024<=args.port<=65535:raise ValueError('explicit unprivileged loopback port required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()!=args.executor_commit:
        raise ValueError('exact frozen protected generation/verifier executor required')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=args.executor,check=True)
    root=Path(__file__).resolve().parents[1]
    source_files=subprocess.check_output(['git','ls-files','scripts','src','configs','submission','sources.lock.json'],cwd=args.executor,text=True).splitlines()
    sources={str(args.executor/name):sha256_file(args.executor/name) for name in source_files}
    controller_sources={str(root/name):sha256_file(root/name) for name in ['scripts/run_confirmation_pair.py',
        'src/gemma_particle/confirmation_results.py','src/gemma_particle/confirmation.py','src/gemma_particle/confirmation_environments.py']}
    declaration={'schema':'gemma4-confirmation-pair-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'access':str(args.access),'environments':str(args.environments),'executor_commit':args.executor_commit,
        'selection_plan_sha256':sha256_file(args.selection_plan),'device':'cuda:0','memory_fraction':.6,
        'startup_headroom_GiB':8,'port':args.port,'tasks_per_condition':49,'conditions':2,
        'order':'all49 OFF then all49 selected, original task order','sources':sources,'controller_sources':controller_sources,
        'scope':'one locked confirmation pair and its offline audit; no new model selection or official submission'}
    if args.plan_only:
        args.out.mkdir(parents=True);write(args.out/'plan.json',declaration)
        write(args.out/'progress.json',{'state':'prepared','holdout_task_contents_parsed':False,'model_inference':False})
        print(json.dumps({'state':'prepared','tasks':98,'holdout_task_contents_parsed':False}));return
    with socket.socket() as check:check.bind(('127.0.0.1',args.port))
    claim=claim_confirmation_pair(scope,args.out)
    identity=confirmation_pair_identity(scope,claim);packet=scope.receipt['selected_packet']
    environments['qualification_sha256']=sha256_file(args.environments);plan=environments['plan']
    declaration.update(confirmation=identity,pair_claim=str(claim),environment_qualification_sha256=environments['qualification_sha256'])
    write(args.out/'plan.json',declaration);logs=args.out/'logs';logs.mkdir()
    status={'state':'waiting_shared_gpu0_headroom','finished_conditions':[],'model_inference':False}
    def publish(state,**values):
        status.update(state=state,as_of=datetime.now(timezone.utc).isoformat(),**values)
        write(args.out/'progress.json',status);print(json.dumps(status),flush=True)
    def unchanged(task=None):
        if server is not None and server.poll() is not None:
            raise RuntimeError(f'owned confirmation server exited{server.returncode}; preserve actual attempts')
        if (any(sha256_file(path)!=value for path,value in {**sources,**controller_sources}.items())
                or sha256_file(args.access)!=identity['access_receipt_sha256']
                or sha256_file(args.environments)!=environments['qualification_sha256']):
            raise ValueError('fixed confirmation source/access/calibration changed')
        if task is None:load_confirmation_environments(args.environments,scope)
        else:
            paths=[*confirmation_asset_names(scope,[task]),'sandbox/setup.py']
            if (any(sha256_file(Path(plan['inputs'])/name)!=plan['input_files'][name] for name in paths)
                    or wheel_identity(plan['wheel_profile'])!=plan['wheels']):
                raise ValueError('current task inputs changed after full-panel calibration')
    publish(status['state']);server=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(args.executor/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    endpoint=f'http://127.0.0.1:{args.port}'
    try:
        while True:
            raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
            free,total=(int(v.strip())*1024**2 for v in raw.strip().split(','))
            if free>=total*.6+8*1024**3:break
            time.sleep(10)
        unchanged();publish('starting_fixed_candidate_server')
        serving=args.out/'server';server_path=serving/'server-identity.json'
        with (logs/'server.log').open('x') as stream:
            server=subprocess.Popen([str(args.executor/'.vllm-venv/bin/python'),'scripts/serve_vllm_lora.py',
                '--adapter',packet['adapter'],'--conversion-report',packet['conversion_report'],'--device','cuda:0',
                '--memory-fraction','.60','--port',str(args.port),'--out',str(serving)],cwd=args.executor,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while not server_path.exists():
            if server.poll() is not None:raise RuntimeError(f'actual server exited{server.returncode}; preserve startup log')
            time.sleep(1)
        server_identity=read(server_path);validate_confirmation_server(scope,server_identity,learned=True)
        with httpx.Client(timeout=600,headers={'X-Gemma-Identity':digest(server_identity)}) as client:
            while True:
                if server.poll() is not None:raise RuntimeError(f'actual server exited{server.returncode}; preserve startup log')
                try:
                    if client.get(endpoint+'/health',timeout=5).status_code==200:break
                except (httpx.ConnectError,httpx.TimeoutException):pass
                time.sleep(5)
            def call(c,base,path,payload=None):
                with (args.out/'preflight-http.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase':'attempt','path':path,'request':payload})+'\n')
                response=c.get(base+path) if payload is None else c.post(base+path,json=payload)
                value=response.json()
                with (args.out/'preflight-http.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase':'response','path':path,'status_code':response.status_code,'response':value})+'\n')
                response.raise_for_status();return value
            write(args.out/'preflight.json',candidate_preflight(client,endpoint,server_identity,call))
        condition_paths=[]
        for condition in scope.receipt['permitted_conditions']:
            unchanged();finished=[];outcomes=[];canonical=None;destination=args.out/condition;destination.mkdir()
            for task in scope.rows:
                task_id=task['instance_id'];unchanged(task_id)
                cohort=destination/(task_id+'-agent');verification=destination/(task_id+'-verification')
                command=[str(args.executor/'.harness-venv/bin/python'),'scripts/run_trajectories.py',
                    '--tasks',scope.receipt['holdout_manifest'],'--task-id',task_id,'--arm','off','--model',BASE_ALIAS,
                    '--confirmation-access',str(args.access),'--confirmation-pair',str(claim),
                    '--endpoint',endpoint+'/v1','--server-identity',str(server_path),'--inputs',plan['inputs'],
                    '--wheels',plan['wheel_profile'],'--image',plan['image_id'],'--out',str(cohort)]
                for option in ['prompt_profile','instruction_protocol','budget_profile','history_profile']:
                    command.extend(['--'+option.replace('_','-'),scope.plan['matching'][option]])
                command.extend(['--capture-protocol','context-budget-off-v1'])
                if condition!='off':command.extend(['--adapter',packet['adapter']])
                publish('running_confirmation',current_condition=condition,current_task=task_id,model_inference=True)
                with (logs/(condition+'-'+task_id+'-agent.log')).open('x') as stream:
                    child=subprocess.run(command,cwd=args.executor,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if child.returncode:raise RuntimeError(f'actual confirmation agent process exited{child.returncode}; preserve reserved attempt')
                unchanged(task_id)
                with (logs/(condition+'-'+task_id+'-verify.log')).open('x') as stream:
                    child=subprocess.run([str(args.executor/'.harness-venv/bin/python'),'scripts/verify_trajectories.py',
                        '--cohort',str(cohort),'--tasks',plan['raw_tasks'],'--inputs',plan['inputs'],'--wheels',plan['wheel_profile'],
                        '--image',plan['image_id'],'--confirmation-access',str(args.access),'--confirmation-pair',str(claim),
                        '--endpoint',endpoint+'/v1','--out',str(verification)],cwd=args.executor,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if child.returncode:raise RuntimeError(f'actual confirmation verifier exited{child.returncode}; preserve captured attempt')
                row,contract=audit_verified_task(cohort,verification,task_id,expected_arm='off',expected_alias='particle_lora' if condition!='off' else BASE_ALIAS)
                validate_matching_contract(contract,scope.plan);fingerprint=digest(contract)
                if canonical is not None and canonical!=fingerprint:raise ValueError('unmatched within-condition confirmation contract')
                canonical=fingerprint;outcomes.append(row)
                finished.append({'instance_id':task_id,'cohort':str(cohort),'verification':str(verification),
                    'outcomes_sha256':sha256_file(verification/'outcomes.jsonl'),'model_journal_sha256':journal_digest(cohort,task_id)})
                write(destination/'progress.json',{'state':'running','condition_id':condition,'finished':finished,'planned':49})
            result={'schema':'gemma4-confirmation-condition-v1','status':'complete','condition_id':condition,
                'confirmation':identity,'environment_qualification_sha256':environments['qualification_sha256'],
                'sources':sources,'server_identity':str(server_path),'server_identity_sha256':digest(server_identity),
                'finished':finished,'canonical_contract_sha256':canonical,
                'summary':summarize_confirmation_rows(scope.rows,outcomes,qualified_ids=environments['qualified_task_ids'])}
            path=destination/'result.json';write(path,result);condition_paths.append(path)
            publish('condition_complete',finished_conditions=[str(p) for p in condition_paths])
        unchanged();comparison=compare_confirmation_pair(condition_paths,scope,claim,environments)
        write(args.out/'comparison.json',comparison)
        publish('complete_confirmation',comparison_sha256=sha256_file(args.out/'comparison.json'),
            off_resolved=comparison['off']['resolved'],selected_resolved=comparison['selected']['resolved'])
    except Exception as exc:
        publish('error',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT);server.wait()


if __name__=='__main__':main()
