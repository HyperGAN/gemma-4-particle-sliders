"""Capture exact official-vLLM tokens for admitted TRAIN context pairs.

The separate native cache will compute its own positive/neutral hidden states
for these exact tokens. Backend activations are never transferred or equated.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from gemma_particle.data import digest,read_jsonl
from gemma_particle.prompting import template_options
from gemma_particle.runtime import file_hash,model_identity
from gemma_particle.transport import validate_transport


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=Path,required=True)
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--config',type=Path,default=Path('configs/31b-ct-pilot.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('prompt transport is immutable')
    u=urlsplit(args.endpoint)
    if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost','::1') or u.username or u.password:
        raise ValueError('local official server required')
    endpoint=args.endpoint.rstrip('/').removesuffix('/v1')
    server=json.loads(args.server_identity.read_text());identity=digest(server)
    contexts=read_jsonl(args.contexts)
    if not contexts or any(r['split']!='train' or r.get('schema')!='gemma4-paired-context-v2' for r in contexts):
        raise ValueError('nonempty admitted TRAIN context pairs required')
    with httpx.Client(base_url=endpoint,headers={'X-Gemma-Identity':identity},timeout=60) as client:
        def check():
            r=client.get('/health');r.raise_for_status()
            if r.json().get('identity_sha256')!=identity:raise ValueError('live server identity differs')
        check();entries=[]
        for row in contexts:
            entry={k:row[k] for k in ('instance_id','decision')};entry['context_sha256']=digest(row)
            for side in ('neutral','positive'):
                body={'model':server['served_model'],'messages':row[side+'_messages'],'tools':row['tools'],
                      'chat_template_kwargs':template_options(row['generation'])}
                r=client.post('/tokenize',json=body);r.raise_for_status();ids=r.json()['tokens']
                entry[side+'_ids']=ids;entry[side+'_ids_sha256']=digest(ids)
                entry[side+'_tokenize_request_sha256']=digest(body)
            entries.append(entry)
            print(json.dumps({'tokenized':len(entries),'total':len(contexts)}),flush=True)
        check()
    transport={'schema':'gemma4-prompt-transport-v1','contexts_sha256':file_hash(args.contexts),
        'contexts_digest':digest(contexts),'generator_model_identity':server['model_identity'],
        'response_model_identity':model_identity(json.loads(args.config.read_text())),
        'server_identity':server,'server_identity_sha256':identity,'entries':entries,
        'sources':{str(f):file_hash(f) for f in [Path(__file__),Path('src/gemma_particle/transport.py')]},
        'scope':'exact input tokens only; native response targets are independently computed; no cross-backend activation or production parity claim'}
    validate_transport(transport,contexts,contexts_sha256=file_hash(args.contexts),response_identity=transport['response_model_identity'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(transport,indent=2)+'\n')


if __name__=='__main__':main()
