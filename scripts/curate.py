"""Select matched contexts from complete verified TRAIN trajectory runs."""
import argparse
from pathlib import Path
from gemma_particle.curation import select_contexts
from gemma_particle.data import read_jsonl, write_jsonl

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
p.add_argument('--outcomes', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
if args.out.exists():
    raise FileExistsError('curated contexts are immutable')
pairs = select_contexts(read_jsonl(args.tasks), read_jsonl(args.outcomes))
args.out.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(args.out, pairs)
print(f'Selected {len(pairs)} decision contexts from {len({p["instance_id"] for p in pairs})} TRAIN tasks')
