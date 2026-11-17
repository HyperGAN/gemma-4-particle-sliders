"""Pinned Gemma loading and safe, tensor-only cache/artifact serialization."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import Native, ContextRow, Prepared


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def model_identity(config):
    revision = config.get("model_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model revision must be a pinned commit")
    attention=config.get('attn_implementation','eager')
    if attention not in ('eager','sdpa'):
        raise ValueError('qualified attention implementation must be eager or sdpa')
    identity={"repo": config["model"], "revision": revision, "dtype": "bfloat16", 'attn_implementation':attention}
    if config.get('frozen_quantization'):
        if config['frozen_quantization']!='ct-w4g32':raise ValueError('unqualified frozen quantization format')
        identity['weight_format']='ct-w4g32'
    return identity


def load_native(config, *, device, memory_fraction=.25, checkpointing=True):
    from .source_identity import require_core_source
    require_core_source()
    activation_storage=config.get('activation_storage','device') if checkpointing else 'device'
    if activation_storage not in ('device','cpu'):
        raise ValueError('unqualified activation storage profile')
    from transformers import AutoModelForImageTextToText,CompressedTensorsConfig
    identity = model_identity(config)
    device = torch.device(device)
    quantized=config.get('frozen_quantization')=='ct-w4g32'
    quant_options={}
    if quantized:
        from .ct_linear import verify_ct_sources,replace_compressed_linears
        verify_ct_sources()
        quant_options['quantization_config']=CompressedTensorsConfig(run_compressed=True)
    if device.type == 'cuda':
        limit=.75 if quantized else .5
        if not 0 < memory_fraction <= limit:
            raise ValueError(f"shared GPU memory fraction must be positive and at most {limit}")
        torch.cuda.set_device(device)
        free,total=torch.cuda.mem_get_info(device)
        if total*memory_fraction>free-8*1024**3:
            raise ValueError('requested memory cap would leave less than 8 GiB of observed GPU headroom')
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    elif device.type != 'cpu':
        raise ValueError("CPU or explicitly shared CUDA device required")
    model = AutoModelForImageTextToText.from_pretrained(identity['repo'], revision=identity['revision'],
        dtype=torch.bfloat16, attn_implementation=identity['attn_implementation'], device_map=str(device),**quant_options)
    if quantized:
        replace_compressed_linears(model)
    return Native(model, identity=identity, checkpointing=checkpointing, activation_storage=activation_storage,
                  rank=config['rank'], alpha=config['alpha'],
                  host_profile='frozen-ct-w4g32' if quantized else 'float-linear',
                  particles=config['particles'], particle_width=config['particle_width'], seed=config['seed'])


def save_cache(path, rows, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError("cache is immutable")
    if len(rows) < 2:
        raise ValueError("at least two native response rows required")
    path.mkdir(parents=True)
    save_file({"target": torch.cat([p.target for p in rows]).detach().float().cpu().contiguous(),
               "neutral": torch.cat([p.neutral for p in rows]).detach().float().cpu().contiguous()}, path/'responses.safetensors')
    index = {"schema": "gemma4-matched-response-cache-v1", "metadata": metadata,
             "responses_sha256": file_hash(path/'responses.safetensors'),
             "rows": [dict(neutral_ids=list(p.row.neutral_ids), positive_ids=list(p.row.positive_ids),
                           instance_id=p.row.instance_id, decision=p.row.decision) for p in rows]}
    (path/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    return index


def load_cache(path, *, identity, train_ids, device='cpu'):
    path = Path(path)
    index = json.loads((path/'index.json').read_text())
    if index['schema'] != 'gemma4-matched-response-cache-v1' or index['metadata']['model_identity'] != identity:
        raise ValueError("cache schema or pinned model mismatch")
    if file_hash(path/'responses.safetensors') != index['responses_sha256']:
        raise ValueError("cache tensors changed")
    tensors = load_file(path/'responses.safetensors', device=str(device))
    if (set(tensors) != {'target', 'neutral'} or tensors['target'].shape != tensors['neutral'].shape
            or tensors['target'].ndim != 2 or len(tensors['target']) != len(index['rows'])
            or any(not torch.isfinite(t).all() for t in tensors.values())):
        raise ValueError("cache tensor schema/values differ")
    rows = []
    for i, record in enumerate(index['rows']):
        if record['instance_id'] not in train_ids:
            raise ValueError("non-TRAIN cache row")
        row = ContextRow(tuple(record['neutral_ids']), tuple(record['positive_ids']), record['instance_id'], record['decision'])
        rows.append(Prepared(row, tensors['target'][i:i+1], tensors['neutral'][i:i+1]))
    return rows, index


def save_native(path, native, *, game_signature, completed_steps):
    path = Path(path)
    if path.exists():
        raise FileExistsError("native inference artifact is immutable")
    path.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in native.slider.state_dict().items()}, path/'native.safetensors')
    metadata = {"schema": "gemma4-native-slider-v1", "model_identity": native.identity,
                "rank": native.slider.rank, "alpha": native.slider.alpha,
                "bank_shape": list(native.particles.shape), "completed_steps": completed_steps,
                "game_signature": game_signature, "weights": "raw", "ema_decay": None,
                "sha256": file_hash(path/'native.safetensors')}
    (path/'native.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def restore_native(native, path):
    path = Path(path)
    metadata = json.loads((path/'native.json').read_text())
    if (metadata['schema'] != 'gemma4-native-slider-v1' or metadata['model_identity'] != native.identity
            or metadata['rank'] != native.slider.rank or metadata['alpha'] != native.slider.alpha
            or metadata['bank_shape'] != list(native.particles.shape)
            or metadata['weights'] != 'raw' or metadata['ema_decay'] is not None
            or metadata['sha256'] != file_hash(path/'native.safetensors')):
        raise ValueError('native inference artifact identity differs')
    native.slider.load_state_dict(load_file(path/'native.safetensors', device=str(native.particles.device)), strict=True)
    return metadata
