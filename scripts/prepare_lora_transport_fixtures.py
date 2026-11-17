"""Create fixed zero/nonzero valid-shape LoRAs using cached header shapes only.

No base-weight values, task/reference data, model forward or GPU is accessed.
These adapters test transport and cannot establish learned behavior or accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from huggingface_hub import try_to_load_from_cache
from safetensors.torch import save_file
import torch

from gemma_particle.adapters import adapter_identity
from gemma_particle.export import peft_target
from gemma_particle.lora_serving import BASE_REPO,BASE_REVISION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError('fixture packet is immutable')
    index=try_to_load_from_cache(BASE_REPO,'model.safetensors.index.json',revision=BASE_REVISION)
    names=sorted(set(json.loads(Path(index).read_text())['weight_map'].values())) if isinstance(index,str) else ['model.safetensors']
    shapes={};headers={}
    for name in names:
        path=try_to_load_from_cache(BASE_REPO,name,revision=BASE_REVISION)
        if not isinstance(path,str):raise FileNotFoundError('pinned cached model header required; no download attempted')
        with open(path,'rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            if not 0<size<=16*1024**2:raise ValueError('bounded safetensors header required')
            raw=stream.read(size);header=json.loads(raw)
        headers[name]=hashlib.sha256(raw).hexdigest()
        for key,row in header.items():
            if '.self_attn.' not in key or not key.endswith('.weight_packed'):continue
            module=key.removesuffix('.weight_packed')
            if module.rsplit('.',1)[-1] not in ('q_proj','k_proj','v_proj','o_proj'):continue
            if row['dtype']!='I32' or len(row['shape'])!=2 or module in shapes:
                raise ValueError('unique official INT32 packed attention matrices required')
            shapes[module]=(row['shape'][0],row['shape'][1]*8)
    if not shapes:raise ValueError('no native attention shapes found')
    generator=torch.Generator(device='cpu').manual_seed(0);rank=8
    zero={};probe={}
    for name,(out_width,in_width) in sorted(shapes.items()):
        target='base_model.model.'+peft_target(name)
        down=torch.randn(rank,in_width,generator=generator)/math.sqrt(in_width)
        up=torch.randn(out_width,rank,generator=generator)*.05
        zero[target+'.lora_A.weight']=down;probe[target+'.lora_A.weight']=down
        zero[target+'.lora_B.weight']=torch.zeros_like(up);probe[target+'.lora_B.weight']=up
    args.out.mkdir(parents=True,mode=0o700);adapters={}
    config={'peft_type':'LORA','task_type':'CAUSAL_LM','r':rank,'lora_alpha':rank,'lora_dropout':0.,
        'bias':'none','inference_mode':True,'fan_in_fan_out':False,'use_rslora':False,'use_dora':False,
        'base_model_name_or_path':BASE_REPO,'target_modules':['k_proj','o_proj','q_proj','v_proj']}
    for alias,weights,folder in [('particle_zero',zero,'zero'),('particle_probe',probe,'probe')]:
        directory=args.out/folder;directory.mkdir()
        save_file(weights,directory/'adapter_model.safetensors',metadata={'format':'pt','scope':'synthetic fixed LoRA transport control; no training or quality claim'})
        (directory/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
        adapters[alias]={'directory':folder,'identity':adapter_identity(directory)}
    receipt={'schema':'gemma4-lora-transport-fixtures-v1','kind':'synthetic_lora_transport',
        'model_identity':{'repo':BASE_REPO,'revision':BASE_REVISION},'seed':0,'rank':rank,
        'attention_matrices':len(shapes),'header_sha256':headers,'adapters':adapters,
        'nonzero_up_std':.05,'base_weight_values_read':False,'gpu_used':False,'model_inference':False,'training_updates':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'zero/nonzero alias, loading and effect controls only; forbidden as targets or coding-quality evidence'}
    (args.out/'fixtures.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
