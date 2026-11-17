"""Extract only fixed DEV inputs and offline dependencies from the public bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from gemma_particle.data import read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('data/freshswe/bundle/freshswe-py.zip'))
    p.add_argument('--tasks',type=Path,default=Path('data/external-dev-v1/dev.jsonl'))
    p.add_argument('--out',type=Path,default=Path('data/freshswe/selected'))
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('external input extraction is immutable')
    rows=read_jsonl(args.tasks)
    if any(r['split']!='dev' for r in rows):raise ValueError('fixed DEV manifest required')
    needed={'docker/Dockerfile','sandbox/setup.py'}
    for row in rows:
        needed.update([f"snapshots/{row['instance_id']}.tgz",f"locks/{row['base_commit']}.txt"])
        prefix=row['repo'].split('/')[-1]+'_'+row['base_commit']
        needed.update([f'graphs/{prefix}.json',f'embeddings/{prefix}.npz'])
    records=[]
    with zipfile.ZipFile(args.bundle) as archive:
        names=set(archive.namelist())
        if needed-names:raise ValueError('fixed DEV inputs missing from bundle: '+repr(sorted(needed-names)))
        needed.update(n for n in names if n.startswith('wheelhouse/') and not n.endswith('/'))
        args.out.mkdir(parents=True)
        for name in sorted(needed):
            target=args.out/name
            if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(name) as source,target.open('wb') as sink:shutil.copyfileobj(source,sink)
            with target.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
            records.append({'path':name,'size':target.stat().st_size,'sha256':h})
    with args.bundle.open('rb') as stream:bundle_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt={'schema':'gemma4-external-dev-inputs-v1','dataset':'moshegolderson/freshswe-py','bundle_sha256':bundle_hash,
        'visible_tasks_sha256':hashlib.sha256(args.tasks.read_bytes()).hexdigest(),'files':records,
        'excluded':'baseline patches/results, other snapshots, all training use',
        'scope':'input integrity only; reference/negative controls pending'}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['size'] for r in records),'status':'extracted'}))


if __name__=='__main__':main()
