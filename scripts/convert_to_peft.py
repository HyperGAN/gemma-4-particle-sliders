"""Fit a PEFT approximation of a native checkpoint on TRAIN activations."""
import argparse
import hashlib
import json
from pathlib import Path
from gemma_particle.data import read_jsonl
from gemma_particle.runtime import load_native, load_cache, model_identity, restore_native, file_hash
from gemma_particle.export import collect_activations, export_peft

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
p.add_argument('--native', type=Path, required=True)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--train-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--memory-fraction', type=float, default=.25)
p.add_argument('--tokens-per-layer', type=int, default=256)
p.add_argument('--strength', type=float, default=1)
args = p.parse_args()
config = json.loads(args.config.read_text())
tasks = read_jsonl(args.train_tasks)
if any(row['split'] != 'train' for row in tasks):
    raise ValueError('TRAIN manifest required')
rows, _ = load_cache(args.cache, identity=model_identity(config), train_ids={r['instance_id'] for r in tasks})
ids = sorted({r.row.instance_id for r in rows}, key=lambda v: hashlib.sha256(v.encode()).hexdigest())
if len(ids) < 2:
    raise ValueError('at least two TRAIN tasks required for disjoint fit and probe')
fit_ids, probe_ids = ids[:max(1,len(ids)*3//4)], ids[max(1,len(ids)*3//4):]
native = load_native(config, device=args.device, memory_fraction=args.memory_fraction, checkpointing=False)
try:
    restore_native(native, args.native)
    fit = collect_activations(native, [r.row for r in rows if r.row.instance_id in fit_ids], tokens_per_layer=args.tokens_per_layer)
    probe = collect_activations(native, [r.row for r in rows if r.row.instance_id in probe_ids], tokens_per_layer=args.tokens_per_layer)
    receipt = export_peft(native, fit, probe, args.out, base_model_name=config['model'], strength=args.strength,
        provenance={'fit_task_ids':fit_ids, 'probe_task_ids':probe_ids, 'task_split':'TRAIN only',
                    'native_metadata_sha256':file_hash(args.native/'native.json'), 'cache_index_sha256':file_hash(args.cache/'index.json')})
    print(json.dumps(receipt, indent=2))
finally:
    native.close()
