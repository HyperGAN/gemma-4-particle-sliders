"""Collect qualified TRAIN tasks in fixed order until a genuine target is ready.

Each task has a fresh OFF control and the declared conditional teacher policy.
All outcomes remain recorded. This adaptive TRAIN collection is not a benchmark.
Child execution uses an explicitly frozen checkout, independent of this script.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

from gemma_particle.budgets import OFFICIAL_BUDGET,budget_values
from gemma_particle.capture import CONTEXT_CAPTURE_PROTOCOL
from gemma_particle.curation import instruction_set
from gemma_particle.data import read_jsonl,write_jsonl
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.harness_identity import sha256_file,wheel_identity
from gemma_particle.history import NO_COMPACTION,OFFICIAL_COMPACTION,HISTORY_PROFILES,history_contract

EXECUTION_SOURCES=['src/gemma_particle/curation.py','scripts/build_submission.py',
    'scripts/run_trajectories.py','scripts/collect_arms.py','src/gemma_particle/budgets.py',
    'src/gemma_particle/capture.py','src/gemma_particle/context_budget.py','scripts/verify_trajectories.py']


def select_tasks(visible,qualification,excluded):
    validate_agent_tasks(visible)
    if any(r['split']!='train' for r in visible):raise ValueError('TRAIN pool cannot contain DEV or holdout')
    controls=qualification['controls']
    if (qualification['schema']!='gemma4-train-environment-qualification-v1'
            or len(controls)!=len(visible) or {r['instance_id'] for r in controls}!={r['instance_id'] for r in visible}
            or len({r['instance_id'] for r in controls})!=len(controls)):
        raise ValueError('qualification must retain the whole fixed TRAIN cohort')
    if sum(r['status']=='passed' for r in controls)!=qualification['qualified']:
        raise ValueError('qualification count differs from its recorded controls')
    if not set(excluded)<={r['instance_id'] for r in visible}:raise ValueError('unknown excluded TRAIN task')
    by_id={r['instance_id']:r for r in controls}
    selected=[by_id[r['instance_id']] for r in visible if by_id[r['instance_id']]['status']=='passed' and r['instance_id'] not in excluded]
    if not selected:raise ValueError('no qualified TRAIN tasks remain')
    return selected


def write_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def stop_owned_server(pid,identity_path):
    proc=Path(f'/proc/{pid}')
    if not proc.exists():return 'process already exited'
    command=[os.fsdecode(s) for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
    if not any(s=='scripts/serve_vllm.py' or s.endswith('/scripts/serve_vllm.py') for s in command):
        raise ValueError('refusing to stop a process other than the owned serving launcher')
    if '--out' not in command:raise ValueError('owned server output argument required')
    output=Path(command[command.index('--out')+1])
    if not output.is_absolute():output=(proc/'cwd').resolve()/output
    if output.resolve()!=identity_path.parent.resolve():raise ValueError('owned serving directory differs')
    os.kill(pid,signal.SIGINT)
    return 'SIGINT sent to matching owned server; release requires subsequent observation'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--executor',type=Path,required=True)
    p.add_argument('--executor-commit',required=True)
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--qualification',type=Path,default=Path('reports/train-environment-qualification-v1.json'))
    p.add_argument('--visible-tasks',type=Path,default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--image',default='gemma-particle-sandbox:fastapi-deps-v2')
    p.add_argument('--exclude-task',action='append',default=[])
    p.add_argument('--stop-server-pid',type=int)
    p.add_argument('--history-profile',choices=HISTORY_PROFILES,default=OFFICIAL_COMPACTION)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    args.root=args.root.resolve();args.executor=args.executor.resolve();args.server_identity=args.server_identity.resolve()
    if args.root.exists():raise FileExistsError('TRAIN pool is immutable; incomplete runs require explicit reconciliation')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()
    if commit!=args.executor_commit:raise ValueError('executor checkout moved')
    source_files=list(EXECUTION_SOURCES)
    if args.history_profile!=NO_COMPACTION:
        source_files.extend(['src/gemma_particle/history.py','src/gemma_particle/adk_capture.py'])
    sources={s:sha256_file(args.executor/s) for s in source_files}
    server=json.loads(args.server_identity.read_text());qualification=json.loads(args.qualification.read_text())
    if sha256_file(args.raw_tasks)!=qualification['source_sha256']:raise ValueError('TRAIN source differs')
    selected=select_tasks(read_jsonl(args.visible_tasks),qualification,args.exclude_task)
    if server.get('kind')!='gemma4' or not server.get('pretrained_inference'):
        raise ValueError('pretrained Gemma server required')
    config=json.loads((args.executor/'configs/31b-ct-pilot.json').read_text())
    if (server.get('strength')!=0 or server.get('native_checkpoint') is not None
            or server['model_identity']['repo']!=config['model']
            or server['model_identity']['revision']!=config['model_revision']):
        raise ValueError('frozen pinned collection model required')
    plan={'schema':'gemma4-train-pool-plan-v1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'task_ids':[r['instance_id'] for r in selected],'excluded_tasks':args.exclude_task,
        'qualification_sha256':sha256_file(args.qualification),'visible_tasks_sha256':sha256_file(args.visible_tasks),
        'raw_tasks_sha256':qualification['source_sha256'],'server_identity':server,
        'executor':str(args.executor),'executor_commit':commit,'execution_sources':sources,
        'pool_source_sha256':sha256_file(Path(__file__)),'teacher_policy':'failed-off-only',
        'instruction_protocol':'tools-v2','capture_protocol':CONTEXT_CAPTURE_PROTOCOL,
        'history_profile':args.history_profile,'history':history_contract(args.history_profile),
        'budget_profile':OFFICIAL_BUDGET,'budget':budget_values(OFFICIAL_BUDGET),
        'stop_rule':'first completed task bringing admitted contexts to at least two; preserve every earlier outcome',
        'scope':'adaptive TRAIN data collection only; no DEV/holdout or benchmark score'}
    if args.plan_only:print(json.dumps(plan,indent=2));return 0
    args.root.mkdir(parents=True);(args.root/'logs').mkdir();write_json(args.root/'plan.json',plan)
    env=dict(os.environ,PYTHONPATH=str(args.executor/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    python=args.executor/'.harness-venv/bin/python';finished=[];admitted=[]
    def execute(command,log,*,cpu=False):
        child_env=dict(env)
        if cpu:child_env['CUDA_VISIBLE_DEVICES']=''
        interpreter=args.executor/'.train-venv/bin/python' if cpu else python
        with log.open('w') as stream:
            r=subprocess.run([str(interpreter),*command],cwd=args.executor,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'pipeline exit{r.returncode}; see{log}')
    for record in selected:
        if {s:sha256_file(args.executor/s) for s in sources}!=sources:raise ValueError('executor sources changed')
        task_id=record['instance_id'];control_path=Path(record['receipt']).resolve();wheels=Path(record['wheels']).resolve()
        actual_image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image],text=True).strip()
        if actual_image!=qualification['image_id']:raise ValueError('qualified sandbox image tag changed')
        if sha256_file(control_path)!=record['receipt_sha256']:raise ValueError('qualified control changed')
        control=json.loads(control_path.read_text())
        if (control['status']!='passed' or control['image_id']!=qualification['image_id']
                or control['tasks_sha256']!=qualification['source_sha256']
                or sha256_file(args.inputs/'snapshots'/f'{task_id}.tgz')!=control['snapshot_sha256']
                or sha256_file(wheels/'receipt.json')!=control['wheel_selection_sha256']
                or wheel_identity(wheels)!=json.loads((wheels/'receipt.json').read_text())['wheels']):
            raise ValueError('qualified task environment changed')
        task_root=args.root/task_id;task_root.mkdir()
        task_plan={'schema':'gemma4-curation-plan-v1','task_ids':[task_id],
            'arms':['off','reproduce','localize','verify'],'server_identity':server,
            'target_instructions':instruction_set('tools-v2'),'execution_sources':sources,
            'source_git':commit,'source_checkout':str(args.executor),
            'environment_control_sha256':record['receipt_sha256'],'pool_plan_sha256':sha256_file(args.root/'plan.json'),
            **{k:plan[k] for k in ('teacher_policy','instruction_protocol','capture_protocol','history_profile','history','budget_profile','budget','scope')}}
        write_json(task_root/'plan.json',task_plan)
        execute(['scripts/collect_arms.py','--root',str(task_root),'--visible-tasks',str(args.visible_tasks.resolve()),
            '--raw-tasks',str(args.raw_tasks.resolve()),'--inputs',str(args.inputs.resolve()),'--wheels',str(wheels),
            '--image',args.image,'--server-identity',str(args.server_identity),'--endpoint',args.endpoint],
            args.root/'logs'/f'{task_id}-collection.log')
        result=json.loads((task_root/'progress.json').read_text())
        if result['state']!='complete':raise ValueError('nonterminal task result')
        contexts=read_jsonl(task_root/'contexts.jsonl')
        if result['admitted_contexts']!=len(contexts) or any(r['instance_id']!=task_id or r['split']!='train' for r in contexts):
            raise ValueError('task contexts differ from the terminal selection')
        admitted.extend(contexts)
        finished.append({'instance_id':task_id,'result':result,'contexts':len(contexts),
            'progress_sha256':sha256_file(task_root/'progress.json'),'contexts_sha256':sha256_file(task_root/'contexts.jsonl')})
        write_json(args.root/'progress.json',{'state':'collecting','finished':finished,'planned':len(selected),'admitted_contexts':len(admitted)})
        print(json.dumps(finished[-1]),flush=True)
        if len(admitted)>=2:
            context_path=args.root/'contexts.jsonl';write_jsonl(context_path,admitted)
            transport=args.root/'prompt-transport.json'
            execute(['scripts/tokenize_contexts.py','--contexts',str(context_path),
                '--server-identity',str(args.server_identity),'--endpoint',args.endpoint,'--out',str(transport)],
                args.root/'logs'/f'{task_id}-tokenize.log',cpu=True)
            status={'state':'target_ready','finished':finished,'planned':len(selected),
                'contexts':str(context_path),'admitted_contexts':len(admitted),'prompt_transport':str(transport),
                'prompt_transport_sha256':sha256_file(transport),'scope':plan['scope']}
            if args.stop_server_pid:status['server_stop']=stop_owned_server(args.stop_server_pid,args.server_identity)
            write_json(args.root/'progress.json',status);print(json.dumps(status),flush=True);return 0
    if admitted:write_jsonl(args.root/'contexts.jsonl',admitted)
    status={'state':'complete_without_sufficient_targets','finished':finished,'planned':len(selected),
        'admitted_contexts':len(admitted),'scope':plan['scope']}
    if args.stop_server_pid:status['server_stop']=stop_owned_server(args.stop_server_pid,args.server_identity)
    write_json(args.root/'progress.json',status);print(json.dumps(status),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
