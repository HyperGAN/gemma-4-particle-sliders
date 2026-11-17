"""Reaudit a complete locked confirmation pair offline; never choose a model."""
import argparse
import json
from pathlib import Path

from gemma_particle.confirmation import load_confirmation_access
from gemma_particle.confirmation_environments import load_confirmation_environments
from gemma_particle.confirmation_results import compare_confirmation_pair
from gemma_particle.harness_identity import sha256_file

p=argparse.ArgumentParser(description=__doc__)
for name in ['access','environments','pair','out']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--condition-result',type=Path,action='append',required=True)
args=p.parse_args()
if args.out.exists():raise FileExistsError('confirmation comparison is immutable')
scope=load_confirmation_access(args.access)
environments=load_confirmation_environments(args.environments,scope)
environments['qualification_sha256']=sha256_file(args.environments)
result=compare_confirmation_pair(args.condition_result,scope,args.pair,environments)
args.out.mkdir(parents=True)
(args.out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'complete','off_resolved':result['off']['resolved'],
    'selected_resolved':result['selected']['resolved'],'new_candidate_selected':False,'official_score':None}))
