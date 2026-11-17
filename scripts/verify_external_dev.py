"""Independently calibrate every fixed supplementary DEV task on two CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/external-dev-v1/manifest.json'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/freshswe/raw/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/freshswe/selected'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('external control cohort is immutable')
    manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True)
    wheels=args.out/'empty-wheels';wheels.mkdir()
    (wheels/'receipt.json').write_text(json.dumps({'schema':'external-dev-image-wheelhouse-v1',
        'injected_wheels':[], 'reason':'exact per-task locks and wheelhouse are baked into the provider image',
        'inputs_receipt_sha256':hashlib.sha256((args.inputs/'receipt.json').read_bytes()).hexdigest()},indent=2)+'\n')
    (args.out/'plan.json').write_text(json.dumps({'task_ids':manifest['task_ids'],'workers':2,
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'image':'gemma-particle-sandbox:freshswe-v1','scope':'environment qualification only; no model score'},indent=2)+'\n')
    def run(task_id):
        command=[sys.executable,'scripts/verify_controls.py','--tasks',str(args.raw_tasks),
            '--inputs',str(args.inputs),'--task-id',task_id,'--split','dev','--wheels',str(wheels),
            '--image','gemma-particle-sandbox:freshswe-v1','--external-dev-manifest',str(args.manifest),
            '--out',str(args.out/task_id)]
        with (args.out/(task_id+'.log')).open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        receipt=args.out/task_id/'result.json'
        return {'instance_id':task_id,'returncode':r.returncode,
            'status':json.loads(receipt.read_text())['status'] if receipt.exists() else 'failed before receipt',
            'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None}
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,i) for i in manifest['task_ids']]
        for f in as_completed(futures):
            results.append(f.result())
            (args.out/'progress.json').write_text(json.dumps({'planned':len(futures),'finished':results},indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'planned':len(manifest['task_ids']),'results':results,
        'status':'passed' if all(r['status']=='passed' for r in results) else 'incomplete qualification; retain every failure'},indent=2)+'\n')


if __name__=='__main__':main()
