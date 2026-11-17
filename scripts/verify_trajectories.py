"""Verify captured patches in fresh containers, with gold data isolated here.

CPU protocol fixtures may exercise this handoff but cannot become training data.
The model-running process never imports or reads reference patches.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from adk_submission import ModelRegistry
from swegemma.budget import EvaluationBudget,HarnessLimits
from swegemma.config import EvalConfig
from swegemma.models import load_tasks
from swegemma.sandbox import ContainerConfig
from swegemma.harness.verification import verify_task

from gemma_particle.capture import load_requests
from gemma_particle.data import VISIBLE_FIELDS,REPO_SPLITS,digest,write_jsonl
from gemma_particle.harness_identity import package_identity,wheel_identity,sha256_file
from gemma_particle.sandbox import isolate_sandbox_cache,SourceBoundManager


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--tasks',type=Path,default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/control-inputs/root/.cache/kagglehub/competitions/gemma-4-developer-agent'))
    p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('verification cohort is immutable')
    manifest=json.loads((args.cohort/'cohort.json').read_text())
    contract=manifest['contract']
    if digest(contract)!=manifest['evaluation_contract_sha256']:
        raise ValueError('evaluation contract changed')
    environment=manifest['environment']
    if (package_identity()!=environment['packages'] or wheel_identity(args.wheels)!=environment['wheels']
            or sha256_file(args.inputs/'sandbox/setup.py')!=environment['sandbox_setup_sha256']):
        raise ValueError('verification must use the captured harness and sandbox inputs')
    if [r['instance_id'] for r in manifest['runs']]!=manifest['task_ids']:
        raise ValueError('every selected task needs a terminal trajectory')
    raw={t.instance_id:t for t in load_tasks(args.tasks)}
    args.out.mkdir(parents=True,mode=0o700)
    os.environ['KAGGLE_SANDBOX_DIR']=str((args.inputs/'sandbox').resolve())
    outcomes=[]
    for captured in manifest['runs']:
        if (captured['evaluation_contract_sha256']!=manifest['evaluation_contract_sha256']
                or captured['generator_kind']!=manifest['generator_kind'] or captured['arm']!=manifest['arm']
                or captured['model_identity']!=manifest['server_identity'].get('model_identity')):
            raise ValueError('terminal result differs from the captured evaluation contract')
        task_id=captured['instance_id'];task=raw[task_id]
        if REPO_SPLITS[task.repo] not in ('train','dev'):
            raise ValueError('this curation verifier excludes the untouched holdout')
        visible={key:getattr(task,key) for key in VISIBLE_FIELDS}
        visible['split']=REPO_SPLITS[task.repo]
        folder=args.cohort/task_id
        if (json.loads((folder/'visible-task.json').read_text())!=visible
                or captured['visible_task_sha256']!=digest(visible)
                or json.loads((folder/'trajectory.json').read_text())!=captured):
            raise ValueError('captured task or terminal result differs from the official task')
        patch=(folder/'agent.patch').read_text()
        snapshot=args.inputs/'snapshots'/f'{task_id}.tgz'
        if sha256_file(snapshot)!=captured['snapshot_sha256'] or sha256_file(folder/'agent.patch')!=captured['patch_sha256']:
            raise ValueError('snapshot or agent patch changed')
        requests=load_requests(folder/'requests') if captured['capture_complete'] else []
        if digest(requests)!=captured['requests_sha256']:
            raise ValueError('request capture changed')
        if any(r['instance_id']!=task_id or r['arm']!=captured['arm'] for r in requests):
            raise ValueError('captured request belongs to a different task or arm')
        first_path=folder/'requests/0000-request.json'
        first_request=None
        if first_path.exists():
            first=json.loads(first_path.read_text())
            if (first['instance_id']!=task_id or first['arm']!=captured['arm'] or first['decision']!=0
                    or first['request_sha256']!=digest(first['request'])):
                raise ValueError('initial model request identity changed')
            first_request=first['request']
        destination=args.out/task_id;destination.mkdir(mode=0o700)
        isolate_sandbox_cache(destination/'tmp')
        package={'psf/requests':'requests','fastapi/fastapi':'fastapi'}[task.repo]
        manager=SourceBoundManager(ContainerConfig(image=args.image,network_mode='none',mem_limit='4g',cpu_quota=200000),package=package)
        if manager.client.images.get(args.image).id!=manifest['image_id']:
            raise ValueError('sandbox image differs from the agent run')
        config=EvalConfig(tasks_path=args.tasks,snapshots_dir=args.inputs/'snapshots',results_dir=destination,
            submission_dir=args.cohort/'artifact/agent',models=ModelRegistry(),wheels_dir=args.wheels,
            budget=EvaluationBudget(**contract['budget']),harness=HarnessLimits(**contract['harness']))
        start=time.perf_counter()
        result=await verify_task(manager,config,task,snapshot,agent_patch=patch,start_time=start)
        (destination/'test-output.txt').write_text(result.test_output or '')
        receipt={'stage':'fresh-container-tests','instance_id':task_id,'patch_sha256':captured['patch_sha256'],
            'requests_sha256':digest(requests),'resolved':result.resolved,'test_exit_code':result.test_exit_code,
            'first_request_sha256':digest(first_request),
            'error':result.error,'image_id':manifest['image_id'],'origin_checks':manager.origin_checks,
            'evaluation_contract_sha256':manifest['evaluation_contract_sha256'],
            'tasks_sha256':sha256_file(args.tasks),'test_output_sha256':sha256_file(destination/'test-output.txt')}
        receipt['verification_script_sha256']=sha256_file(Path(__file__))
        if result.resolved and (not manager.origin_checks or not all(r['workspace'] for r in manager.origin_checks)):
            raise ValueError('passing score must test the edited workspace')
        timeout=(str(captured.get('agent_error','')).startswith('Agent exceeded session timeout')
            and captured['elapsed_seconds']>=contract['budget']['time_minutes']*60)
        outcome={**captured,'schema':'gemma4-verified-trajectory-v3','resolved':result.resolved,
            'requests':requests,'first_request':first_request,'task_budget_exhausted':timeout,'verification':receipt,
            'model_calls_attempted':len(list((folder/'requests').glob('*-request.json'))),
            'model_calls_completed':len(list((folder/'requests').glob('*-response.json')))}
        outcomes.append(outcome)
        (destination/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        write_jsonl(args.out/'outcomes.jsonl',outcomes)
        print(json.dumps({'instance_id':task_id,'generator_kind':manifest['generator_kind'],
                          'resolved':result.resolved,'test_exit_code':result.test_exit_code}),flush=True)


if __name__=='__main__':asyncio.run(main())
