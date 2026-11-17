"""Bind the prospective DEV resource declaration to actual numerical evidence.

This performs no server launch, task inference, driver query or HTTP request.
"""
import argparse
import json
from pathlib import Path

from gemma_particle.resource_supplement import prepare_resource_supplement

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--declaration', type=Path, default=Path('research/dev-resource-declaration-v1.json'))
p.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
p.add_argument('--qualification', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
if args.out.exists(): raise FileExistsError('qualified resource record is immutable')
result = prepare_resource_supplement(args.declaration, args.selection_plan, args.qualification)
args.out.mkdir(parents=True)
(args.out / 'resource.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': result['status'], 'serving_launched': False, 'model_task_inference': False}))
