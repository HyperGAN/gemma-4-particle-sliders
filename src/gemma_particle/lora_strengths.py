"""Derive LoRA doses on CPU from one measured unit-strength conversion."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file,save_file
import torch

from .adapters import adapter_identity


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def derive_peft_strength(source,destination,*,strength,conversion_report=None):
    source,destination=Path(source),Path(destination)
    report=Path(conversion_report) if conversion_report else source.with_name(source.name+'-conversion.json')
    output_report=destination.with_name(destination.name+'-conversion.json')
    if destination.exists() or output_report.exists():raise FileExistsError('derived adapter/report is immutable')
    if type(strength) not in (int,float) or not math.isfinite(strength) or strength not in (.5,1.5):
        raise ValueError('declared derived strengths are .5 and1.5; use the measured source for strength1')
    identity=adapter_identity(source);receipt=json.loads(report.read_text());report_sha=sha(report)
    if (receipt.get('schema')!='gemma4-routed-to-peft-v1' or receipt.get('exact_merge') is not False
            or receipt.get('strength')!=1 or receipt.get('strength_derivation') is not None
            or receipt.get('adapter_sha256')!=identity['files']['adapter_model.safetensors']):
        raise ValueError('original measured unit-strength conversion and unchanged adapter required')
    weights=load_file(source/'adapter_model.safetensors',device='cpu');rank=identity['rank']
    modules={k.removesuffix('.lora_A.weight') for k in weights if k.endswith('.lora_A.weight')}
    if not modules or set(weights)!={m+suffix for m in modules for suffix in ('.lora_A.weight','.lora_B.weight')}:
        raise ValueError('complete paired ordinary LoRA matrices required')
    for module in modules:
        a,b=[weights[module+suffix] for suffix in ('.lora_A.weight','.lora_B.weight')]
        if (a.ndim!=2 or b.ndim!=2 or a.shape[0]!=rank or b.shape[1]!=rank
                or a.dtype!=torch.float32 or b.dtype!=torch.float32
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError('finite FP32 rank-matching measured LoRA matrices required')
        scaled=(b*strength).contiguous()
        if not torch.isfinite(scaled).all():raise ValueError('derived LoRA dose is nonfinite')
        weights[module+'.lora_B.weight']=scaled
    if adapter_identity(source)!=identity or sha(report)!=report_sha:raise ValueError('source changed during CPU derivation')
    with safe_open(source/'adapter_model.safetensors',framework='pt',device='cpu') as stream:
        metadata=stream.metadata()
    destination.mkdir(parents=True)
    (destination/'adapter_config.json').write_bytes((source/'adapter_config.json').read_bytes())
    save_file(weights,destination/'adapter_model.safetensors',metadata=metadata)
    derived=deepcopy(receipt);derived.update(strength=strength,adapter_sha256=sha(destination/'adapter_model.safetensors'))
    derived['strength_derivation']={'method':'FP32 B times strength; A and config unchanged',
        'source_strength':1,'source_adapter':str(source.resolve()),'source_adapter_identity':identity,
        'source_conversion_report':str(report.resolve()),'source_conversion_report_sha256':report_sha,
        'layer_metrics_scope':'original unscaled native-delta fit/probe metrics; no new fitting or quality measurement'}
    output_report.write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
    return derived
