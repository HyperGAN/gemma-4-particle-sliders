"""Download selected official inputs using Kaggle's own private authentication.

Run in an isolated environment with kaggle installed. The CLI reads the token
from its standard location; this script never copies it into run artifacts.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path

from kaggle import api

COMPETITION='gemma-4-developer-agent'
WHEELHOUSE='metric/gemma-4-developer-agent-wheelhouse'


def file_hash(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--visible-tasks',type=Path)
    p.add_argument('--task-id',action='append',default=[])
    p.add_argument('--wheelhouse',action='store_true')
    p.add_argument('--max-download-gib',type=float,default=1)
    args=p.parse_args()
    names=['HARNESS_README.md','tasks.jsonl','sandbox/setup.py','docker/Dockerfile.sandbox','docker/imp.py','docker/telnetlib.py']
    if args.task_id:
        if not args.visible_tasks:raise ValueError('sanitized task manifest required')
        tasks=[json.loads(line) for line in args.visible_tasks.read_text().splitlines() if line.strip()]
        if any('patch' in r or 'test_patch' in r or r['split'] not in ('train','dev') for r in tasks):
            raise ValueError('preparation accepts sanitized TRAIN/DEV tasks only')
        lookup={r['instance_id']:r for r in tasks}
        for task_id in args.task_id:
            task=lookup[task_id];repo=task['repo'].split('/')[-1]
            names.extend([f'snapshots/{task_id}.tgz',f'graphs/{repo}_{task["base_commit"]}.json',
                          f'embeddings/{repo}_{task["base_commit"]}.npz'])
    listing=[];token=None
    for _ in range(100):
        page=api.competition_list_files(COMPETITION,page_token=token,page_size=200)
        listing.extend({'name':f.name,'bytes':f.total_bytes} for f in page.files)
        token=page.next_page_token
        if not token:break
    else:raise ValueError('unexpected file pagination')
    sizes={r['name']:r['bytes'] for r in listing}
    names=sorted(set(names))
    if sum(sizes[n] for n in names)>args.max_download_gib*1024**3:
        raise ValueError('selected official inputs exceed the declared transfer budget')
    args.out.mkdir(parents=True,exist_ok=True)
    receipt_path=args.out/'download-receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'schema':'gemma4-official-download-v1','competition':COMPETITION,'wheelhouse':WHEELHOUSE,'files':{}}
    for source,name in [('competition',n) for n in names]+([('wheelhouse',n) for n in (
            'adk_eval_core-0.1.0-py3-none-any.whl','adk_submission-0.2.13-py3-none-any.whl',
            'swegemma-0.2.11-py3-none-any.whl')] if args.wheelhouse else []):
        relative=name if source=='competition' else 'wheelhouse/'+name
        target=args.out/relative
        previous=receipt['files'].get(relative)
        if previous:
            if not target.is_file() or file_hash(target)!=previous['sha256']:
                raise ValueError('a previously downloaded official input changed')
            continue
        destination=args.out if source=='competition' else args.out/'wheelhouse'
        destination.mkdir(parents=True,exist_ok=True)
        if source=='competition':
            api.competition_download_file(COMPETITION,name,path=str(destination),quiet=True,unzip=True)
        else:
            api.dataset_download_file(WHEELHOUSE,name,path=str(destination),quiet=True)
        if not target.is_file():raise FileNotFoundError(target)
        receipt['files'][relative]={'source':source,'bytes':target.stat().st_size,'sha256':file_hash(target)}
        receipt['fetched_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'downloaded':relative,'bytes':target.stat().st_size}),flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        board=api.competition_leaderboard_view(COMPETITION,page_size=20)
    snapshot={'fetched_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source':f'https://www.kaggle.com/competitions/{COMPETITION}/leaderboard',
        'entries':[{'team':r.team_name,'score':r.score,'submission_date':str(r.submission_date)} for r in board]}
    (args.out/'leaderboard.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    print(json.dumps({'verified_files':len(receipt['files']),'leading_public_score':snapshot['entries'][0]['score']}))


if __name__=='__main__':main()
