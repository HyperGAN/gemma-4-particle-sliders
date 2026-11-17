"""Independently calibrate every fixed supplementary DEV task on two CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from gemma_particle.dev import load_external_dev,qualified_dev_identity


def write_qualification(args,selection):
    """Summarize terminal controls without re-running them or reading gold."""
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=json.loads((args.out/'plan.json').read_text())
    summary=json.loads((args.out/'summary.json').read_text())
    if (plan['manifest_sha256']!=selection.manifest_sha256
            or plan['task_ids']!=selection.manifest['task_ids']
            or summary['planned']!=len(selection.rows)
            or len(summary['results'])!=len(selection.rows)
            or {r['instance_id'] for r in summary['results']}!=set(plan['task_ids'])):
        raise ValueError('terminal controls differ from the frozen DEV plan')
    controls=[]
    for entry in summary['results']:
        source=args.out/entry['instance_id']/'result.json'
        if not source.exists():
            controls.append({'task':entry['instance_id'],'status':entry['status']});continue
        if sha(source)!=entry['receipt_sha256']:raise ValueError('individual control receipt changed')
        record=json.loads(source.read_text())
        if (record['task']!=entry['instance_id'] or record['split']!='dev' or record['gpu_used'] is not False
                or record['tasks_sha256']!=selection.manifest['source_sha256']
                or record['external_dev_manifest_sha256']!=selection.manifest_sha256):
            raise ValueError('individual control source/selection differs')
        controls.append({**record,'receipt_sha256':entry['receipt_sha256']})
    images={r.get('image_id') for r in controls}
    passed=(summary['status']=='passed' and len(images)==1 and None not in images
        and all(e['returncode']==0 for e in summary['results']))
    receipt={'schema':'gemma4-supplementary-dev-qualification-v1','as_of':datetime.now(timezone.utc).isoformat(),
        'status':'passed' if passed else 'failed','planned':len(selection.rows),
        'qualified':sum(c['status']=='passed' for c in controls),'repositories':len(selection.manifest['repositories']),
        'selection_manifest_sha256':selection.manifest_sha256,'source_sha256':selection.manifest['source_sha256'],
        'visible_sha256':selection.manifest['visible_sha256'],'input_receipt_sha256':sha(args.inputs/'receipt.json'),
        'image_id':next(iter(images)) if len(images)==1 else None,'raw_summary_sha256':sha(args.out/'summary.json'),
        'no_model_inference':True,'forbidden_training':True,'controls':controls,
        'scope':'CPU environment controls only; original Requests DEV retained and Rich/HTTPX holdout untouched.'}
    target=args.out/'qualification.json'
    if target.exists():raise FileExistsError('qualification receipt is immutable')
    if passed:
        # Use the same strict checks required at a later agent/verifier boundary.
        temporary=args.out/'qualification.pending.json';temporary.write_text(json.dumps(receipt,indent=2)+'\n')
        qualified_dev_identity(selection,temporary,inputs=args.inputs,image_id=receipt['image_id'],wheels=args.out/'empty-wheels')
        temporary.rename(target)
    else:target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'qualification':str(target),'status':receipt['status']}),flush=True)
    return 0 if passed else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/external-dev-v1/manifest.json'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/freshswe/raw/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/freshswe/selected'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summarize-existing',action='store_true',help='summarize a terminal immutable control cohort without restarting any controls')
    args=p.parse_args()
    selection=load_external_dev(args.manifest,source_sha256=hashlib.sha256(args.raw_tasks.read_bytes()).hexdigest())
    if args.summarize_existing:return write_qualification(args,selection)
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
    return write_qualification(args,selection)


if __name__=='__main__':raise SystemExit(main())
