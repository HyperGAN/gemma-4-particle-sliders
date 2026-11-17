"""Observe real local vLLM alias loading, zero parity and nonzero effects.

Uses a fixed synthetic prompt, exact shared token IDs and one generated token.
This cannot supply TRAIN targets or establish learned coding quality.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from gemma_particle.data import digest
from gemma_particle.lora_serving import BASE_ALIAS,BASE_REPO,BASE_REVISION,compare_alias_effects


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',required=True,help='Local server root, without /v1')
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();endpoint=urlsplit(args.endpoint)
    if endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','localhost','::1') or endpoint.username or endpoint.password:
        raise ValueError('explicit local HTTP endpoint required')
    identity=json.loads(args.server_identity.read_text())
    if (identity.get('kind')!='gemma4_lora_transport_fixture' or not identity.get('qualification_only')
            or identity.get('model_identity',{}).get('repo')!=BASE_REPO
            or identity['model_identity'].get('revision')!=BASE_REVISION):
        raise ValueError('exact synthetic transport server identity required')
    args.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    receipt={'schema':'gemma4-vllm-lora-effect-control-v1','status':'started','created_at':datetime.now(timezone.utc).isoformat(),
        'server_identity_sha256':digest(identity),'server_identity_file_sha256':hashlib.sha256(args.server_identity.read_bytes()).hexdigest(),
        'pretrained_inference':True,'synthetic_adapters':True,'training_updates':0,'requests_completed':0,
        'requests_attempted':0,'http_responses_received':0,
        'sources':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [__file__,'src/gemma_particle/lora_serving.py']},
        'scope':'local official-vLLM alias/input/effect controls; no trained adapter, native approximation or coding-quality claim'}
    def save():(args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save();start=time.perf_counter();root=args.endpoint.rstrip('/')
    headers={'X-Gemma-Identity':digest(identity)}
    try:
        with httpx.Client(timeout=600,headers=headers) as client:
            def call(path,payload=None):
                index=receipt['requests_attempted'];receipt['requests_attempted']+=1
                with (args.out/'http-requests.jsonl').open('a') as stream:
                    stream.write(json.dumps({'index':index,'path':path,'request':payload})+'\n')
                receipt['phase']=path;save()
                response=client.get(root+path) if payload is None else client.post(root+path,json=payload)
                value=response.json();receipt['http_responses_received']+=1
                with (args.out/'http-calls.jsonl').open('a') as stream:
                    stream.write(json.dumps({'index':index,'path':path,'request':payload,'status_code':response.status_code,'response':value})+'\n')
                response.raise_for_status()
                receipt['requests_completed']+=1;save();return value
            if call('/health').get('identity_sha256')!=digest(identity):raise ValueError('live identity differs')
            available={row['id'] for row in call('/v1/models')['data']}
            names=[BASE_ALIAS,'particle_zero','particle_probe']
            if not set(names)<=available:raise ValueError('declared LoRA aliases not advertised by the actual server')
            prompt='Repository repair protocol control.\nReturn the word ready.\nAnswer:'
            encoded={name:call('/tokenize',{'model':name,'prompt':prompt})['tokens'] for name in names}
            if not encoded[BASE_ALIAS] or any(encoded[name]!=encoded[BASE_ALIAS] for name in names):
                raise ValueError('alias tokenization differs for the fixed prompt')
            receipt['prompt_token_ids']=encoded[BASE_ALIAS]
            responses={name:call('/v1/completions',{'model':name,'prompt':encoded[BASE_ALIAS],
                'temperature':0,'seed':0,'max_tokens':1,'logprobs':5,'return_tokens_as_token_ids':True,'ignore_eos':True}) for name in names}
            measured=compare_alias_effects(responses,tolerance=1e-5)
            receipt.update(measured,status='passed' if measured['passed'] else 'failed',
                model_aliases_advertised=True,prompt_token_ids_equal=True)
    except Exception as exc:
        receipt.update(status='error',error_type=type(exc).__name__,error=str(exc))
    receipt['seconds']=time.perf_counter()-start;save();print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
