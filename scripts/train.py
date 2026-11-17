"""Run a bounded chunk of the shared PG0.10 conditional game; exact resume."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from gemma_particle.data import read_jsonl
from gemma_particle.model import make_game, Prepared
from gemma_particle.runtime import load_native, load_cache, model_identity, save_native, file_hash

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--train-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
p.add_argument('--preset', choices=['rhbi_optimized','empf'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--memory-fraction', type=float, default=.25)
p.add_argument('--chunk-updates', type=int, default=64)
p.add_argument('--resume', type=Path)
p.add_argument('--gpu-native-polar', action='store_true', help='explicit RHBI CUDA execution derivative; recorded in signature')
args = p.parse_args()
if not 1 <= args.chunk_updates <= 1024:
    raise ValueError('bounded positive chunk size required')
config = json.loads(args.config.read_text())
tasks = read_jsonl(args.train_tasks)
if any(r['split'] != 'train' for r in tasks):
    raise ValueError('TRAIN manifest required')
if args.out.exists() and not args.resume:
    raise FileExistsError('existing run requires its exact checkpoint')
rows, index = load_cache(args.cache, identity=model_identity(config), train_ids={r['instance_id'] for r in tasks})
native = load_native(config, device=args.device, memory_fraction=args.memory_fraction)
try:
    rows = [Prepared(row.row, row.target.to(native.particles.device), row.neutral.to(native.particles.device)) for row in rows]
    game = make_game(native, rows, preset=args.preset, steps=config['updates'], batch=config['batch_size'],
                     optimizer_dense_profile='gpu_native_polar' if args.gpu_native_polar else 'recipe')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        pin = json.loads(args.resume.with_suffix('.json').read_text())
        if file_hash(args.resume) != pin['sha256']:
            raise ValueError('checkpoint hash mismatch')
        game.load_state_dict(torch.load(args.resume, map_location=args.device, weights_only=True))
    target_step = min(game.completed_steps + args.chunk_updates, config['updates'])
    while game.completed_steps < target_step:
        start = time.monotonic()
        metrics = game.step()
        metrics['wall_seconds'] = time.monotonic() - start
        with (args.out/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics), flush=True)
        if game.completed_steps % 64 == 0 or game.completed_steps == target_step:
            step = game.completed_steps
            checkpoint = args.out/f'FULL-{step:04d}.pt'
            if checkpoint.exists():
                raise FileExistsError('checkpoint already exists')
            torch.save(game.state_dict(), checkpoint)
            checkpoint.with_suffix('.json').write_text(json.dumps({'sha256':file_hash(checkpoint), 'steps':step},indent=2)+'\n')
            save_native(args.out/f'RAW-{step:04d}', native, game_signature=game.signature, completed_steps=step)
finally:
    native.close()
