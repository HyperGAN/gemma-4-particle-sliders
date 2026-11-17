"""Check imported package ownership after the exact official sandbox setup."""
import argparse
import json
import os
from pathlib import Path
import shlex

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.harness.container_setup import (setup_container_wheels, extract_snapshot,
    setup_synthetic_hardware, setup_git_exclude, install_editable_package,
    install_test_dependencies, setup_workspace_test_config, setup_baseline_commit)
from swegemma.sandbox import ContainerConfig, ContainerManager
from gemma_particle.sandbox import isolate_sandbox_cache

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', type=Path, default=Path('data/control-inputs/root/.cache/kagglehub/competitions/gemma-4-developer-agent'))
p.add_argument('--task-id', default='requests_6757')
p.add_argument('--repo', default='psf/requests')
p.add_argument('--package', default='requests')
p.add_argument('--image', default='gemma-particle-sandbox:requests-deps-v1')
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
if args.out.exists():
    raise FileExistsError('inspection receipt is immutable')
args.out.parent.mkdir(parents=True,exist_ok=True)
isolate_sandbox_cache(args.out.with_suffix('.tmp'))
os.environ['KAGGLE_SANDBOX_DIR'] = str((args.inputs/'sandbox').resolve())
manager = ContainerManager(ContainerConfig(image=args.image, network_mode='none'))
config = EvalConfig(tasks_path=Path('data/splits-v1/dev.jsonl'), snapshots_dir=args.inputs/'snapshots',
    results_dir=args.out.parent, submission_dir=Path('submission'), models=ModelRegistry(), wheels_dir=args.inputs/'wheels')
cid = manager.start()
try:
    setup_container_wheels(manager,cid,config)
    extract_snapshot(manager,cid,args.inputs/'snapshots'/f'{args.task_id}.tgz')
    setup_synthetic_hardware(manager,cid)
    setup_git_exclude(manager,cid)
    install_editable_package(manager,cid)
    install_test_dependencies(manager,cid,args.repo,config=config)
    setup_workspace_test_config(manager,cid,args.repo)
    setup_baseline_commit(manager,cid,'ownership_inspection')
    script = ('import importlib,json,sys; m=importlib.import_module('+repr(args.package)+'); '
              'print(json.dumps(dict(file=m.__file__, version=getattr(m,"__version__",None), path=sys.path)))')
    result = manager.exec(cid,'cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c '+shlex.quote(script))
    receipt = {'task':args.task_id, 'image_id':manager.client.images.get(args.image).id,
               'exit_code':result.exit_code, 'stdout':result.stdout, 'stderr':result.stderr,
               'scope':'official setup with no test or solution patch; package import ownership check'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
finally:
    manager.stop(cid)
