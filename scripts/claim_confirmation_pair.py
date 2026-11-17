"""Claim the one fixed post-selection pair; never run a model or task here."""
import argparse
import json
from pathlib import Path

from gemma_particle.confirmation import load_confirmation_access, claim_confirmation_pair

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--access', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
scope = load_confirmation_access(args.access)
claim = claim_confirmation_pair(scope, args.out)
print(json.dumps({'status': 'claimed_one_pair', 'claim': str(claim), 'tasks_per_condition': 49,
    'permitted_conditions': scope.receipt['permitted_conditions'], 'model_inference': False}))
