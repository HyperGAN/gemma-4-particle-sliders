"""Run CPU engineering gates and retain a compact, source-bound receipt."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK=json.loads((ROOT/'sources.lock.json').read_text())
from gemma_particle.source_identity import require_core_source
source_cohort=require_core_source()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
logs = ROOT/'runs/preflight'
logs.mkdir(parents=True, exist_ok=True)
commands = [
    [sys.executable, '-m', 'pytest', '-q', 'tests', str(Path('.deps')/LOCK['particle_sliders'].get('directory','particle-sliders')/'packages/particle-sliders-core/tests/test_pg0100_integration.py')],
    [str(ROOT/'.harness-venv/bin/python'), 'scripts/validate_submission.py', 'dist/baseline-31b/agent'],
]
checks = []
for i, command in enumerate(commands):
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (logs/f'check-{i}.log').write_text(result.stdout)
    print(result.stdout, end='')
    checks.append({'command':command, 'exit_code':result.returncode,
                   'output_sha256':hashlib.sha256(result.stdout.encode()).hexdigest()})
    if result.returncode:
        raise SystemExit(result.returncode)
sources = {}
for folder in ('src','tests','scripts','submission','patches','configs'):
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'schema':'gemma4-cpu-preflight-v1', 'status':'passed', 'gpu_used':False,
           'source_cohort':source_cohort,'sources_lock_sha256':hashlib.sha256((ROOT/'sources.lock.json').read_bytes()).hexdigest(),
           'checks':checks, 'sources':sources,
           'packages':{p:importlib.metadata.version(p) for p in ['torch','particlegan','particle-sliders-core','transformers','peft','numpy']
                       + (['compressed-tensors'] if importlib.util.find_spec('compressed_tensors') else [])},
           'harness_receipt_sha256':hashlib.sha256((ROOT/'reports/harness-environment.json').read_bytes()).hexdigest(),
           'scope':'CPU random Gemma 4 fixture, software checks and competition compiler only; no pretrained task accuracy or SOTA claim'}
(ROOT/'reports/cpu-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
