"""Qualify every original TRAIN environment on two CPU workers.

Reuse explicit, matching prior controls. Retain every new failure; reference
patches are read only by the separate official-verifier child processes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gemma_particle.data import read_jsonl
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.harness_identity import package_identity,sha256_file,wheel_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visible-tasks',type=Path,default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--wheel-sources',type=Path,default=Path('data/task-wheel-sources-with-annotated-doc-v1'))
    p.add_argument('--existing-wheels',type=Path,default=Path('data/task-wheels'))
    p.add_argument('--reuse-control',action='append',type=Path,default=[])
    p.add_argument('--reuse-source-commit',action='append',default=[],help='explicit audited historical verifier source accepted for reuse; recorded separately')
    p.add_argument('--image',default='gemma-particle-sandbox:fastapi-deps-v2')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('TRAIN environment cohort is immutable')
    visible=read_jsonl(args.visible_tasks);validate_agent_tasks(visible)
    if any(r['split']!='train' or r['repo']!='fastapi/fastapi' for r in visible):
        raise ValueError('original FastAPI TRAIN manifest required')
    ids=[r['instance_id'] for r in visible]
    image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image],text=True).strip()
    source_sha=sha256_file(args.raw_tasks)
    script_sha=sha256_file('scripts/verify_controls.py')
    historical_sources={}
    for revision in args.reuse_source_commit:
        commit=subprocess.check_output(['git','rev-parse','--verify',revision+'^{commit}'],text=True).strip()
        source=subprocess.check_output(['git','show',commit+':scripts/verify_controls.py'])
        historical_sources[commit]=hashlib.sha256(source).hexdigest()
    profiles={sha256_file(p):p.parent for p in args.existing_wheels.glob('*/receipt.json')}
    reused={}
    for path in args.reuse_control:
        receipt=json.loads(path.read_text());task_id=receipt['task']
        if (task_id not in ids or task_id in reused or receipt['split']!='train'
                or receipt['status']!='passed' or receipt['image_id']!=image_id
                or receipt['tasks_sha256']!=source_sha
                or receipt['execution_script_sha256'] not in {script_sha,*historical_sources.values()}
                or receipt['snapshot_sha256']!=sha256_file(args.inputs/'snapshots'/f'{task_id}.tgz')
                or receipt['arms']['empty']['resolved'] or not receipt['arms']['reference']['resolved']
                or receipt['arms']['empty']['test_exit_code'] in (0,None)
                or receipt['arms']['reference']['test_exit_code']!=0
                or receipt['arms']['empty']['error'] is not None or receipt['arms']['reference']['error'] is not None
                or len(receipt['origin_checks'])!=2 or receipt['gpu_used'] is not False
                or not all(o['workspace'] and o['origin'].startswith('/workspace/') for o in receipt['origin_checks'])):
            raise ValueError('reused TRAIN control differs from the declared environment')
        profile=profiles[receipt['wheel_selection_sha256']]
        if wheel_identity(profile)!=json.loads((profile/'receipt.json').read_text())['wheels']:
            raise ValueError('reused TRAIN wheel bytes changed')
        reused[task_id]={'instance_id':task_id,'status':'passed','reused':True,
            'receipt':str(path),'receipt_sha256':sha256_file(path),'wheels':str(profile),
            'execution_script_sha256':receipt['execution_script_sha256']}
    args.out.mkdir(parents=True)
    (args.out/'logs').mkdir();(args.out/'wheel-profiles').mkdir()
    plan={'schema':'gemma4-train-environment-plan-v1','task_ids':ids,'workers':2,
        'visible_tasks_sha256':sha256_file(args.visible_tasks),'raw_tasks_sha256':source_sha,
        'image_id':image_id,'reused_controls':list(reused.values()),
        'accepted_historical_verifier_sources':historical_sources,
        'wheel_sources':wheel_identity(args.wheel_sources),'harness':package_identity(),
        'execution_sources':{s:sha256_file(s) for s in ['scripts/verify_controls.py',
            'scripts/prepare_task_wheels.py','scripts/verify_train_environments.py','src/gemma_particle/sandbox.py']},
        'scope':'CPU environment qualification only; no model inference, training targets or benchmark score'}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def run(task_id):
        profile=args.out/'wheel-profiles'/task_id
        folder=args.out/task_id
        commands=[['scripts/prepare_task_wheels.py','--snapshot',str(args.inputs/'snapshots'/f'{task_id}.tgz'),
            '--wheels',str(args.wheel_sources),'--out',str(profile)],
            ['scripts/verify_controls.py','--tasks',str(args.raw_tasks),'--inputs',str(args.inputs),
            '--task-id',task_id,'--split','train','--wheels',str(profile),'--image',args.image,'--out',str(folder)]]
        returncode=0
        with (args.out/'logs'/f'{task_id}.log').open('w') as stream:
            for command in commands:
                returncode=subprocess.run([sys.executable,*command],stdout=stream,stderr=subprocess.STDOUT).returncode
                if returncode:break
        receipt=folder/'result.json'
        status=json.loads(receipt.read_text())['status'] if receipt.exists() else 'blocked before verifier receipt'
        return {'instance_id':task_id,'status':status,'returncode':returncode,'reused':False,
            'receipt':str(receipt) if receipt.exists() else None,
            'receipt_sha256':sha256_file(receipt) if receipt.exists() else None,'wheels':str(profile)}
    results=list(reused.values())
    def record():
        status={'planned':len(ids),'finished':results,'qualified':sum(r['status']=='passed' for r in results),
            'scope':plan['scope']}
        (args.out/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    record()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,i) for i in ids if i not in reused]
        for future in as_completed(futures):
            results.append(future.result());record();print(json.dumps(results[-1]),flush=True)
    summary={'planned':len(ids),'results':results,'qualified':sum(r['status']=='passed' for r in results),
        'status':'passed' if all(r['status']=='passed' and r.get('returncode',0)==0 for r in results) else 'failures retained',
        'scope':plan['scope']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if summary['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
