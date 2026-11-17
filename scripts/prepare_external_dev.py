"""Freeze a supplementary repo-disjoint issue-only DEV cohort before inference.

This does not replace the original Requests DEV or the untouched Rich/HTTPX
holdout. Provider validation is a claim; our negative/reference controls must
qualify each environment independently before this cohort can guide selection.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from gemma_particle.data import VISIBLE_FIELDS,REPO_SPLITS,digest,read_jsonl,write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/freshswe/raw/tasks.jsonl'))
    p.add_argument('--out',type=Path,default=Path('data/external-dev-v1'))
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('supplementary DEV split is immutable')
    grouped=defaultdict(list)
    for row in read_jsonl(args.source):
        if row['statement_source']!='issue':continue
        if row['repo'] in REPO_SPLITS or row['repo'].split('/')[0] in ['fastapi','Textualize']:continue
        grouped[row['repo']].append(row)
    # Fixed per-repository cap and SHA order; no model or reference outcomes enter.
    selected=[]
    for repo,rows in sorted(grouped.items()):
        selected.extend(sorted(rows,key=lambda r:hashlib.sha256(('freshswe-dev-v1:'+r['instance_id']).encode()).hexdigest())[:2])
    visible=[{**{k:r[k] for k in VISIBLE_FIELDS},'split':'dev'} for r in selected]
    manifest={'schema':'gemma4-supplementary-dev-v1','dataset':'moshegolderson/freshswe-py',
        'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'selection':'statement_source=issue; exclude original split repos and fastapi/Textualize organizations; at most two per repo by SHA256(freshswe-dev-v1:instance_id)',
        'count':len(visible),'visible_sha256':digest(visible),'task_ids':[r['instance_id'] for r in visible],
        'repositories':sorted(grouped),'status':'unqualified; independent negative/reference controls pending',
        'original_requests_dev':'preserved; blocked full-test controls remain blocked',
        'original_rich_httpx_holdout':'untouched; excluded from this cohort',
        'forbidden_training':True}
    args.out.mkdir(parents=True)
    write_jsonl(args.out/'dev.jsonl',visible)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'tasks':len(visible),'repositories':len(grouped),'status':manifest['status']}))


if __name__=='__main__':main()
