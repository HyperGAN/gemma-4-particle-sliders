"""Bounded pretrained native forward/backward qualification, without updates.

The scalar probe tests an input/output derivative, not a training objective.
No targets, optimizer, adapter checkpoint or coding score are produced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import torch
from transformers import AutoTokenizer

from gemma_particle.runtime import load_native,model_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--memory-fraction',type=float,required=True)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--batch-size',type=int,choices=[1,2],default=1)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('resource profile is immutable')
    if not 8<=args.max_tokens<=32768:raise ValueError('bounded 8..32768 token resource probe required')
    torch.set_num_threads(4)
    config=json.loads(args.config.read_text())
    identity=model_identity(config)
    tokenizer=AutoTokenizer.from_pretrained(identity['repo'],revision=identity['revision'],local_files_only=True)
    seed_ids=tokenizer.encode('Inspect the repository, reproduce a failure, implement a focused repair, and verify it. ',
        add_special_tokens=False)
    ids=([tokenizer.bos_token_id]+seed_ids*((args.max_tokens+len(seed_ids)-1)//len(seed_ids)))[:args.max_tokens]
    args.out.mkdir(parents=True,mode=0o700)
    receipt={'schema':'gemma4-native-resource-qualification-v1','model_identity':identity,'device':args.device,
        'memory_fraction':args.memory_fraction,'context_tokens':len(ids),'batch_size':args.batch_size,'training_updates':0,
        'context_kind':'synthetic resource/derivative probe; forbidden as a training target',
        'scope':'memory, finite forward, exact initial OFF and live adapter gradients; not learned quality',
        'allocator_environment':{key:os.environ.get(key) for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
        'status':'started','phase':'loading','on_forwards_completed':0}
    sources=[Path(__file__),args.config,Path('src/gemma_particle/model.py'),Path('src/gemma_particle/runtime.py')]
    if config.get('checkpoint_profile')=='nested_positional_attention_mlp':
        sources.append(Path('src/gemma_particle/checkpoint_inputs.py'))
    if config.get('activation_storage')=='cpu':
        sources.append(Path('src/gemma_particle/activation_storage.py'))
    if config.get('frozen_quantization'):
        from gemma_particle.ct_linear import verify_ct_sources
        receipt['ct_sources']=verify_ct_sources()
        sources.append(Path('src/gemma_particle/ct_linear.py'))
    receipt['execution_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    def save():(args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save();start=time.perf_counter();native=None;device=torch.device(args.device)
    try:
        native=load_native(config,device=args.device,memory_fraction=args.memory_fraction,checkpointing=True)
        receipt['native_metadata']=native.metadata()
        if device.type=='cuda':torch.cuda.synchronize(device)
        receipt['load_seconds']=time.perf_counter()-start
        receipt['phase']='off_forward';save()
        before={k:v.detach().cpu().clone() for k,v in native.network.state_dict().items()}
        start=time.perf_counter()
        with torch.no_grad():off=native.hidden(ids,strength=0).detach().clone()
        if device.type=='cuda':torch.cuda.synchronize(device)
        receipt['off_seconds']=time.perf_counter()-start
        receipt['phase']='on_forward_0';save()
        storage=[]
        start=time.perf_counter()
        outputs=[]
        for _ in range(args.batch_size):
            outputs.append(native.hidden(ids,strength=1))
            storage.append(native.last_activation_storage)
            receipt['on_forwards_completed']=len(outputs)
            receipt['activation_storage_forwards']=storage
            receipt['phase']=f'on_forward_{len(outputs)}' if len(outputs)<args.batch_size else 'forward_comparison'
            save()
        response=torch.cat(outputs,dim=0)
        receipt['activation_storage_forwards']=storage
        receipt['initial_on_equals_off']=torch.equal(response.detach(),off.expand_as(response))
        receipt['phase']='backward';save()
        direction=torch.linspace(.1,1,native.width,device=response.device)
        gradients=torch.autograd.grad((response*direction).sum(),tuple(native.network.parameters()),allow_unused=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        receipt.update(backward_seconds=time.perf_counter()-start,response_shape=list(response.shape),
            gradient_tensors=len(gradients),connected_gradient_tensors=sum(g is not None for g in gradients),
            nonzero_gradient_tensors=sum(g is not None and bool(torch.count_nonzero(g)) for g in gradients),
            finite_gradients=all(g is None or bool(torch.isfinite(g).all()) for g in gradients),
            adapter_unchanged=all(torch.equal(v,native.network.state_dict()[k].detach().cpu()) for k,v in before.items()),
            frozen_base=all(not p.requires_grad for p in native.model.parameters()),
            host_profile=native.slider.host_profile)
        receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
        receipt['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        passed=(receipt['initial_on_equals_off'] and receipt['finite_gradients'] and receipt['nonzero_gradient_tensors']>0
            and receipt['connected_gradient_tensors']==receipt['gradient_tensors'] and receipt['adapter_unchanged'] and receipt['frozen_base'])
        receipt['status']='passed' if passed else 'failed'
        receipt['phase']='completed'
    except Exception as exc:
        receipt.update(status='error',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        receipt['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if device.type=='cuda' and torch.cuda.is_initialized():
            receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
            receipt['allocator_backend']=torch.cuda.memory.get_allocator_backend()
        save()
        if native is not None:native.close()
    print(json.dumps(receipt,indent=2),flush=True)
    return 0 if receipt['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
