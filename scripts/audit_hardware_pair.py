"""Reaudit a complete recorded numerical control offline; never activate it."""
import argparse
import json
from pathlib import Path

from gemma_particle.hardware_evidence import audit_hardware_pair

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--qualification', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
if args.out.exists(): raise FileExistsError('offline audit output is immutable')
result = audit_hardware_pair(args.qualification)
args.out.mkdir(parents=True)
(args.out / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': 'passed', 'pairs': len(result['pairs']), 'resource_supplement_activated': False}))
