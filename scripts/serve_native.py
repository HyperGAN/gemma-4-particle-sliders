"""Serve pinned BF16 native Gemma locally on an explicitly shared device.

This is a native HF research condition. It does not establish production W4A16
vLLM parity. The eligible Kaggle model name is only a routing alias here; actual
weights, precision, runtime and strength are recorded in server-identity.json.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading

from fastapi import FastAPI,HTTPException,Request
import torch
from transformers import AutoTokenizer
import uvicorn

from gemma_particle.generation import NativeGenerator
from gemma_particle.official_thinking import state_machine,DEFAULT_WHEEL,WHEEL_SHA256,SOURCE_SHA256
from gemma_particle.runtime import load_native,model_identity,restore_native,file_hash
from gemma_particle.data import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/pilot.json'))
    p.add_argument('--device',required=True)
    p.add_argument('--memory-fraction',type=float,default=.35)
    p.add_argument('--served-model',default='gemma-4-e2b-it-qat-w4a16-ct')
    p.add_argument('--native',type=Path)
    p.add_argument('--strength',type=float,default=0)
    p.add_argument('--port',type=int,default=8017)
    p.add_argument('--max-request-seconds',type=float,default=120)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('server run is immutable')
    if args.strength and not args.native:raise ValueError('nonzero serving strength requires trained native weights')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    config=json.loads(args.config.read_text())
    state_machine(DEFAULT_WHEEL)  # Validate official policy before allocating GPU memory.
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['model_revision'],local_files_only=True)
    args.out.mkdir(parents=True,mode=0o700)
    native=load_native(config,device=args.device,memory_fraction=args.memory_fraction,checkpointing=False)
    trained=restore_native(native,args.native) if args.native else None
    generator=NativeGenerator(native,tokenizer,served_model=args.served_model,strength=args.strength,max_seconds=args.max_request_seconds)
    source_paths=[Path(__file__),Path('src/gemma_particle/generation.py'),Path('src/gemma_particle/official_thinking.py'),
        Path('src/gemma_particle/prompting.py'),Path('src/gemma_particle/model.py'),Path('src/gemma_particle/runtime.py'),Path('sources.lock.json')]
    if config.get('frozen_quantization'):source_paths.append(Path('src/gemma_particle/ct_linear.py'))
    sources={str(p):file_hash(p) for p in source_paths}
    identity={'kind':'gemma4','served_model':args.served_model,'model_identity':model_identity(config),
        'pretrained_inference':True,'runtime':'native transformers; separate from production vLLM',
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','particle-sliders-core','particlegan','fastapi','uvicorn','pydantic']},
        'device':str(device),'memory_fraction':args.memory_fraction,'strength':args.strength,'native_checkpoint':trained,
        'max_context':32768,'max_request_seconds':args.max_request_seconds,'sources':sources,
        'thinking_policy':{'official_vllm_wheel_sha256':WHEEL_SHA256,'unchanged_state_methods_sha256':SOURCE_SHA256,
                           'host':'one HF sequence; native runtime has not established vLLM output parity'}}
    (args.out/'server-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({'status':'loaded','device':str(device),'memory_allocated':torch.cuda.memory_allocated(device) if device.type=='cuda' else None}),flush=True)
    app=FastAPI();lock=threading.Lock()
    def complete(body):
        with lock:
            response,audit=generator.complete(body)
            with (args.out/'generation.jsonl').open('a') as stream:stream.write(json.dumps(audit,ensure_ascii=False)+'\n')
            return response
    @app.get('/health')
    def health():return {'status':'ok','model_identity':identity['model_identity'],'identity_sha256':digest(identity)}
    @app.get('/v1/models')
    def models():return {'object':'list','data':[{'id':args.served_model,'object':'model','owned_by':'local'}]}
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        if request.headers.get('X-Gemma-Identity')!=digest(identity):
            raise HTTPException(status_code=412,detail='request must name this exact server identity')
        body=await request.json()
        try:return await asyncio.to_thread(complete,body)
        except torch.OutOfMemoryError as exc:
            # Under a fixed cap the same prompt is permanently unservable.
            # Expose it as a terminal request error rather than repeated 5xx.
            with (args.out/'generation-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'kind':'out_of_memory','message':str(exc)})+'\n')
            torch.cuda.empty_cache()
            raise HTTPException(status_code=422,detail='native request exceeded the declared GPU memory cap') from exc
        except (ValueError,KeyError) as exc:raise HTTPException(status_code=400,detail=str(exc)) from exc
    try:uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')
    finally:native.close()


if __name__=='__main__':main()
