"""Freeze initial TRAIN OFF prompts and exact tokens for conversion diagnostics."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from gemma_particle.conversion_probe import initial_off_request,validate_probe
from gemma_particle.data import digest,read_jsonl
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.prompting import template_options
from gemma_particle.runtime import file_hash,model_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outcomes',type=Path,nargs='+',required=True)
    p.add_argument('--train-tasks',type=Path,default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--config',type=Path,default=Path('configs/31b-ct-pilot.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('conversion probe is immutable')
    tasks=read_jsonl(args.train_tasks);validate_agent_tasks(tasks)
    if any(t['split']!='train' for t in tasks):raise ValueError('TRAIN manifest required')
    train_ids={t['instance_id'] for t in tasks}
    server=json.loads(args.server_identity.read_text());identity=digest(server)
    response=model_identity(json.loads(args.config.read_text()))
    sources=[]
    for path in args.outcomes:
        outcomes=read_jsonl(path)
        if len(outcomes)!=1:raise ValueError('explicit single-OFF outcome files required')
        outcome=outcomes[0];request=initial_off_request(outcome,train_ids)
        if outcome['model_identity']!=server['model_identity']:raise ValueError('OFF server/model differs')
        sources.append((path,outcome['instance_id'],request))
    url=urlsplit(args.endpoint)
    if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1','::1') or url.username or url.password:
        raise ValueError('local bound tokenization server required')
    endpoint=args.endpoint.rstrip('/').removesuffix('/v1')
    entries=[]
    with httpx.Client(base_url=endpoint,headers={'X-Gemma-Identity':identity},timeout=60) as client:
        def check():
            r=client.get('/health');r.raise_for_status()
            if r.json().get('identity_sha256')!=identity:raise ValueError('live server identity differs')
        check()
        for path,task,request in sources:
            body={'model':server['served_model'],'messages':request['messages'],'tools':request['tools'],
                  'chat_template_kwargs':template_options(request['generation'])}
            r=client.post('/tokenize',json=body);r.raise_for_status();ids=r.json()['tokens']
            entries.append({'instance_id':task,'request':request,'request_sha256':digest(request),
                'outcomes_sha256':file_hash(path),'neutral_ids':ids,'neutral_ids_sha256':digest(ids),
                'tokenize_request_sha256':digest(body)})
        check()
    packet={'schema':'gemma4-conversion-probe-v1','purpose':'conversion diagnostic only; forbidden as a training target',
        'selection':'explicit completed OFF tasks, fixed before any pretrained optimizer update; initial system/user prompt only',
        'train_manifest_sha256':file_hash(args.train_tasks),'server_identity':server,'server_identity_sha256':identity,
        'response_model_identity':response,'entries':entries,
        'sources':{str(f):file_hash(f) for f in [Path(__file__),Path('src/gemma_particle/conversion_probe.py')]}}
    validate_probe(packet,response_identity=response,train_ids=train_ids,fit_ids=set())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps({'tasks':[e['instance_id'] for e in entries],'tokens':[len(e['neutral_ids']) for e in entries],
                      'sha256':file_hash(args.out),'pretrained_inference':False}),flush=True)


if __name__=='__main__':main()
