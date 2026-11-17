"""CPU source-upgrade probe; random Gemma fixture, never training targets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_gemma_binding import fixture,assert_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=['rhbi_optimized','empf'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--foreign-checkpoint',type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('source probe is immutable')
    args.out.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(0)
    _,native,rows,game=fixture(args.preset)
    try:
        values={}
        def snapshot(prefix):
            for name,tensor in native.network.state_dict().items():values[prefix+'/adapter/'+name]=tensor.detach().cpu().contiguous().clone()
            for name,tensor in game.critic.state_dict().items():values[prefix+'/critic/'+name]=tensor.detach().cpu().contiguous().clone()
            for i,row in enumerate(rows):
                values[prefix+f'/target/{i}']=row.target.detach().cpu().contiguous().clone()
                values[prefix+f'/neutral/{i}']=row.neutral.detach().cpu().contiguous().clone()
        snapshot('initial');metrics=game.step();snapshot('after-one-update')
        packet=game.state_dict();torch.save(packet,args.out/'full.pt')
        rejected=False
        if args.foreign_checkpoint:
            foreign=torch.load(args.foreign_checkpoint,weights_only=True)
            before=deepcopy(game.state_dict())
            try:game.load_state_dict(foreign)
            except ValueError:rejected=True
            if not rejected:raise AssertionError('foreign source checkpoint was admitted')
            assert_equal(before,game.state_dict())
        save_file(values,args.out/'numeric-state.safetensors')
        receipt={'preset':args.preset,'pg_version':game.signature['pg_version'],
            'pg_source_commit':game.signature['pg_source_commit'],'pg_source_profile':game.signature['pg_source_profile'],
            'numeric_metrics':{k:v for k,v in metrics.items() if type(v) in (float,int,bool)},
            'foreign_checkpoint_rejected_without_mutation':rejected if args.foreign_checkpoint else None,
            'scope':'random CPU fixture, one optimizer update; no pretrained data, quality or learned target'}
        (args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    finally:native.close()


if __name__=='__main__':main()
