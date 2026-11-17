import argparse
import json
from pathlib import Path
from gemma_particle.data import prepare_tasks

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, default=Path("data/splits-v1"))
args = p.parse_args()
manifest = prepare_tasks(args.source, args.out)
print(json.dumps({k: v["count"] for k, v in manifest["splits"].items()}, indent=2))
