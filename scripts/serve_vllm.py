"""Run the official vLLM wheel on one shared GPU with an immutable identity.

Serving uses official Gemma4 parsers and thinking policy. Tensor parallelism,
hardware and memory budget differ from Kaggle's four L4 GPUs and remain explicit.
No native-HF numerical or adapter-transport parity is implied by serving alone.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_wheel(path,expected,distribution):
    if sha(path)!=expected:raise ValueError(f'{distribution} wheel differs')
    installed=importlib.metadata.distribution(distribution)
    with zipfile.ZipFile(path) as archive:
        members=[n for n in archive.namelist() if n.endswith('.py')]
        if any(installed.locate_file(n).read_bytes()!=archive.read(n) for n in members):
            raise ValueError(f'{distribution} installed Python source differs')
    return {'version':installed.version,'wheel_sha256':expected,'verified_python_files':len(members)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/31b-ct-pilot.json'))
    p.add_argument('--device',required=True,choices=['cuda:0','cuda:1'])
    p.add_argument('--memory-fraction',type=float,default=.60)
    p.add_argument('--port',type=int,default=8025)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('server run is immutable')
    if not 0<args.memory_fraction<=.65:raise ValueError('shared serving fraction must be positive and at most .65')
    config=json.loads(args.config.read_text())
    if config['model']!='google/gemma-4-31B-it-qat-w4a16-ct' or config['model_revision']!='52f3f65bc7a02d555763bc923bd1d9094898219d':
        raise ValueError('only the pinned official 31B CT serving cohort is qualified here')
    physical=args.device.split(':')[1]
    memory=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    free,total=(int(s.strip())*1024**2 for s in memory.strip().split(','))
    if total*args.memory_fraction>free-8*1024**3:raise ValueError('requested serving budget leaves less than 8 GiB of observed headroom')
    os.environ['CUDA_VISIBLE_DEVICES']=physical
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    wheels=Path('data/kaggle-official/wheelhouse')
    checked={name:verify_wheel(wheels/filename,expected,name) for name,filename,expected in [
        ('vllm','vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl','73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310'),
        ('compressed-tensors','compressed_tensors-0.15.0.1-py3-none-any.whl','e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9'),
        ('transformers','transformers-5.13.1-py3-none-any.whl','53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd')]}
    model='gemma-4-31b-it-qat-w4a16-ct'
    launch=['vllm','serve',config['model'],'--revision',config['model_revision'],
        '--tokenizer-revision',config['model_revision'],'--served-model-name',model,
        '--host','127.0.0.1','--port',str(args.port),'--dtype','bfloat16',
        '--tensor-parallel-size','1','--gpu-memory-utilization',str(args.memory_fraction),
        '--max-model-len','32768','--max-num-seqs','1','--max-num-batched-tokens','2048',
        '--enforce-eager','--language-model-only','--no-scheduler-reserve-full-isl',
        '--enable-auto-tool-choice','--tool-call-parser','gemma4','--reasoning-parser','gemma4',
        '--reasoning-config',json.dumps({'reasoning_start_str':'<|channel>','reasoning_end_str':'<channel|>'}),
        '--middleware','gemma_particle.vllm_identity.guard']
    identity={'kind':'gemma4','served_model':model,'pretrained_inference':True,
        'runtime':'official vLLM 0.19.1; local single-GPU research cohort',
        'model_identity':{'repo':config['model'],'revision':config['model_revision'],'dtype':'bfloat16',
                          'weight_format':'ct-w4g32','attn_implementation':'vllm:auto'},
        'device':args.device,'memory_fraction':args.memory_fraction,'memory_budget_kind':'vLLM engine target; not a CUDA allocator cap',
        'observed_free_bytes':free,'total_device_bytes':total,'strength':0,'native_checkpoint':None,
        'max_context':32768,'tensor_parallel_size':1,'max_num_seqs':1,
        'packages':{n:importlib.metadata.version(n) for n in ['torch','vllm','transformers','compressed-tensors','fastapi','pydantic']},
        'official_sources':checked,'launch':launch,
        'sources':{str(f):sha(f) for f in [Path(__file__),args.config,Path('src/gemma_particle/vllm_identity.py'),Path('src/gemma_particle/data.py')]},
        'scope':'frozen-base generation; native HF targets and production four-L4 parity are separate qualifications'}
    args.out.mkdir(parents=True,mode=0o700)
    target=(args.out/'server-identity.json').resolve()
    target.write_text(json.dumps(identity,indent=2)+'\n');target.chmod(0o600)
    os.environ['GEMMA_SERVER_IDENTITY']=str(target)
    print(json.dumps({'status':'launching','identity':str(target),'device':args.device,'memory_fraction':args.memory_fraction}),flush=True)
    sys.argv=launch
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__=='__main__':main()
