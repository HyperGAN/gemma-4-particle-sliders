"""Calibrate all13 original Requests DEV tasks with explicit immutable inputs.

Every task remains in the report, including network/TLS failures. Only the
separate official-verifier child reads reference-bearing data. No agent runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from gemma_particle.data import read_jsonl
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.harness_identity import package_identity,sha256_file,wheel_identity


def write(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visible-tasks',type=Path,default=Path('data/splits-v1/dev.jsonl'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--image',required=True,help='exact qualified immutable image ID')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('original DEV calibration is immutable')
    visible=read_jsonl(args.visible_tasks);validate_agent_tasks(visible)
    if len(visible)!=13 or any(r['split']!='dev' or r['repo']!='psf/requests' for r in visible):
        raise ValueError('the complete original13-task DEV manifest is required')
    image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image],text=True).strip()
    if args.image!=image_id:raise ValueError('use the immutable image ID rather than a mutable tag')
    snapshots={r['instance_id']:sha256_file(args.inputs/'snapshots'/(r['instance_id']+'.tgz')) for r in visible}
    wheels=wheel_identity(args.wheels)
    if not wheels:raise ValueError('explicit original wheel bytes required')
    args.out.mkdir(parents=True);(args.out/'logs').mkdir();profile=args.out/'wheels';profile.mkdir()
    for name in wheels:
        try:os.link((args.wheels/name).resolve(),profile/name)
        except OSError:shutil.copyfile(args.wheels/name,profile/name)
    write(profile/'receipt.json',{'schema':'gemma4-original-dev-wheel-profile-v1','wheels':wheels,
        'policy':'exact unmodified original wheel pool; workspace imports remain checked'})
    if wheel_identity(profile)!=wheels:raise ValueError('copied original wheel bytes changed')
    plan={'schema':'gemma4-original-dev-environment-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'task_ids':[r['instance_id'] for r in visible],'visible_tasks_sha256':sha256_file(args.visible_tasks),
        'raw_tasks_sha256':sha256_file(args.raw_tasks),'snapshots':snapshots,'image_id':image_id,
        'wheel_profile_sha256':sha256_file(profile/'receipt.json'),'wheels':wheels,
        'sandbox_setup_sha256':sha256_file(args.inputs/'sandbox/setup.py'),'harness':package_identity(),
        'sources':{name:sha256_file(name) for name in ['scripts/verify_controls.py',
            'scripts/verify_original_dev_environments.py','src/gemma_particle/sandbox.py']},
        'workers':2,'gpu_used':False,'model_inference':False,
        'scope':'all original13 DEV environment controls; no agent accuracy, target or holdout evaluation'}
    write(args.out/'plan.json',plan)
    def run(task_id):
        destination=args.out/task_id
        command=[sys.executable,'scripts/verify_controls.py','--tasks',str(args.raw_tasks),
            '--inputs',str(args.inputs),'--task-id',task_id,'--split','dev','--image',image_id,
            '--wheels',str(profile),'--out',str(destination)]
        with (args.out/'logs'/(task_id+'.log')).open('x') as stream:
            r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        path=destination/'result.json'
        result=json.loads(path.read_text()) if path.exists() else None
        if result and (result['image_id']!=image_id or result['tasks_sha256']!=plan['raw_tasks_sha256']
                or result['snapshot_sha256']!=snapshots[task_id]
                or result['wheel_selection_sha256']!=plan['wheel_profile_sha256']):
            raise ValueError('DEV control differs from its declared inputs')
        return {'instance_id':task_id,'returncode':r.returncode,
            'status':result['status'] if result else 'failed before verifier receipt',
            'receipt':str(path) if result else None,'receipt_sha256':sha256_file(path) if result else None,
            'arms':result['arms'] if result else None,'origin_checks':result.get('origin_checks') if result else None}
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,row['instance_id']) for row in visible]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write(args.out/'progress.json',{'planned':13,'finished':results,
                'qualified':sum(r['status']=='passed' for r in results)})
            print(json.dumps({'task':result['instance_id'],'status':result['status']}),flush=True)
    if wheel_identity(profile)!=wheels:raise ValueError('DEV wheel profile changed during calibration')
    result={'schema':'gemma4-original-dev-environment-qualification-v1','state':'complete',
        'as_of':datetime.now(timezone.utc).isoformat(),'planned':13,'qualified':sum(r['status']=='passed' for r in results),
        'controls':sorted(results,key=lambda r:r['instance_id']),'plan_sha256':sha256_file(args.out/'plan.json'),
        'image_id':image_id,'wheels':str(profile),'wheel_profile_sha256':plan['wheel_profile_sha256'],
        'visible_tasks_sha256':plan['visible_tasks_sha256'],'source_sha256':plan['raw_tasks_sha256'],
        'gpu_used':False,'model_inference':False,'scope':plan['scope']}
    write(args.out/'qualification.json',result)
    return 0 if result['qualified']==13 else 2


if __name__=='__main__':raise SystemExit(main())
