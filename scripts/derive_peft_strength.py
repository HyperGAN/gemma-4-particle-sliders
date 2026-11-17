"""Scale a measured unit-strength PEFT conversion on CPU without model loading."""
import argparse
import json
from pathlib import Path

from gemma_particle.lora_strengths import derive_peft_strength

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--conversion-report',type=Path)
p.add_argument('--strength',type=float,choices=[.5,1.5],required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
receipt=derive_peft_strength(args.source,args.out,strength=args.strength,conversion_report=args.conversion_report)
print(json.dumps({'status':'derived','strength':receipt['strength'],'adapter_sha256':receipt['adapter_sha256'],
    'gpu_used':False,'new_fitting':False,'scope':'CPU adapter-dose derivation; learned fidelity/quality require actual evaluation'}))
