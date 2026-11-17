"""Separate LoRA-serving cohort; never mutate the live frozen collection server."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from serve_vllm import sha,verify_wheel
from gemma_particle.lora_serving import BASE_ALIAS,prepare_lora,lora_launch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/31b-ct-pilot.json'))
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--adapter',type=Path)
    group.add_argument('--synthetic-fixtures',type=Path)
    group.add_argument('--base-only',action='store_true',help='unadapted base; same LoRA-enabled engine settings, no adapter weights')
    p.add_argument('--conversion-report',type=Path)
    p.add_argument('--device',required=True,choices=['cuda:0','cuda:1'])
    p.add_argument('--memory-fraction',type=float,default=.60)
    p.add_argument('--port',type=int,default=8027)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true',help='CPU preparation only; no nvidia-smi, model or CUDA access')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('LoRA serving run/plan is immutable')
    config=json.loads(args.config.read_text())
    spec=prepare_lora(config,adapter=args.adapter,fixtures=args.synthetic_fixtures,conversion_report=args.conversion_report,base_only=args.base_only)
    launch=lora_launch(config,spec,port=args.port,memory_fraction=args.memory_fraction)
    sources={str(path):sha(path) for path in [Path(__file__),Path('scripts/serve_vllm.py'),args.config,
        Path('src/gemma_particle/lora_serving.py'),Path('src/gemma_particle/adapters.py'),Path('src/gemma_particle/vllm_identity.py')]}
    if args.synthetic_fixtures:sources[str(args.synthetic_fixtures)]=sha(args.synthetic_fixtures)
    elif args.adapter:sources[spec['provenance']['conversion_report']]=sha(spec['provenance']['conversion_report'])
    if args.plan_only:
        args.out.mkdir(parents=True,mode=0o700)
        plan={'schema':'gemma4-lora-serving-plan-v1','status':'prepared','gpu_used':False,
            'environment_verified':False,'inference_executed':False,'spec':spec,'launch':launch,
            'device':args.device,'memory_fraction':args.memory_fraction,'startup_headroom_GiB':8,
            'sources':sources,'scope':'CPU launch preparation; no vLLM adapter loading, effect or quality evidence'}
        (args.out/'launch-plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan));return
    physical=args.device.split(':')[1]
    memory=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    free,total=(int(x.strip())*1024**2 for x in memory.strip().split(','))
    if total*args.memory_fraction>free-8*1024**3:raise ValueError('serving target leaves less than eight GiB observed headroom')
    os.environ.update(CUDA_VISIBLE_DEVICES=physical,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                      VLLM_ALLOW_RUNTIME_LORA_UPDATING='0')
    wheels=Path('data/kaggle-official/wheelhouse')
    checked={name:verify_wheel(wheels/file,expected,name) for name,file,expected in [
        ('vllm','vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl','73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310'),
        ('compressed-tensors','compressed_tensors-0.15.0.1-py3-none-any.whl','e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9'),
        ('transformers','transformers-5.13.1-py3-none-any.whl','53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd')]}
    identity={'kind':spec['kind'],'served_model':BASE_ALIAS,'pretrained_inference':True,
        'model_identity':{'repo':config['model'],'revision':config['model_revision'],'dtype':'bfloat16',
                          'weight_format':'ct-w4g32','attn_implementation':'vllm:auto'},
        'device':args.device,'memory_fraction':args.memory_fraction,'observed_free_bytes':free,'total_device_bytes':total,
        'memory_budget_kind':'vLLM engine target; not a CUDA allocator cap','strength':0,'native_checkpoint':None,
        'max_context':32768,'tensor_parallel_size':1,'max_num_seqs':1,'adapters':spec['adapters'],
        'adapter_provenance':spec['provenance'],'qualification_only':spec['qualification_only'],
        'base_only':spec['base_only'],
        'adapter_dtype':'vLLM auto; BF16 base dtype','runtime_adapter_updates':False,
        'official_sources':checked,'sources':sources,'launch':launch,
        'packages':{n:importlib.metadata.version(n) for n in ['torch','vllm','transformers','compressed-tensors']},
        'scope':'local base/LoRA alias serving; adapter effect, native conversion fidelity and task quality require separate evidence'}
    args.out.mkdir(parents=True,mode=0o700);target=(args.out/'server-identity.json').resolve()
    target.write_text(json.dumps(identity,indent=2)+'\n');target.chmod(0o600)
    os.environ['GEMMA_SERVER_IDENTITY']=str(target)
    print(json.dumps({'status':'launching','identity':str(target),'qualification_only':spec['qualification_only']}),flush=True)
    sys.argv=launch
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__=='__main__':main()
