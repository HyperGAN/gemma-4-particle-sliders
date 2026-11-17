"""CPU-only scorer calibration using genuine public task reference patches.

Reference patches are isolated in this verifier. They never supply an agent
answer, training target, conversion activation, or benchmark agent score.
Each arm calls the unchanged official verify_task in its own fresh container.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.models import load_tasks
from swegemma.harness.verification import verify_task
from swegemma.sandbox import ContainerConfig, ContainerManager
from gemma_particle.sandbox import isolate_sandbox_cache,SourceBoundManager
from gemma_particle.data import VISIBLE_FIELDS,digest,read_jsonl


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('data/public-mirror/root/.cache/kagglehub/competitions/gemma-4-developer-agent/tasks.jsonl'))
    parser.add_argument('--inputs', type=Path, default=Path('data/control-inputs/root/.cache/kagglehub/competitions/gemma-4-developer-agent'))
    parser.add_argument('--task-id', default='requests_6757')
    parser.add_argument('--image', default='gemma-particle-sandbox:mirror-v30')
    parser.add_argument('--split', choices=['train','dev'], default='dev')
    parser.add_argument('--wheels',type=Path)
    parser.add_argument('--external-dev-manifest',type=Path)
    parser.add_argument('--out', type=Path, default=Path('runs/verifier-controls-v1'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('control run is immutable')
    task = next(row for row in load_tasks(args.tasks) if row.instance_id == args.task_id)
    allowed_repos = {'train':{'fastapi/fastapi'}, 'dev':{'psf/requests'}}
    external=None
    if args.external_dev_manifest:
        external=json.loads(args.external_dev_manifest.read_text())
        visible=read_jsonl(args.external_dev_manifest.parent/'dev.jsonl')
        if (args.split!='dev' or external.get('schema')!='gemma4-supplementary-dev-v1'
                or external['source_sha256']!=hashlib.sha256(args.tasks.read_bytes()).hexdigest()
                or external['visible_sha256']!=digest(visible)
                or task.instance_id not in external['task_ids']):
            raise ValueError('supplementary DEV source/selection differs')
        expected=next(r for r in visible if r['instance_id']==task.instance_id)
        if {k:getattr(task,k) for k in VISIBLE_FIELDS}!={k:expected[k] for k in VISIBLE_FIELDS}:
            raise ValueError('reference task differs from the frozen visible DEV task')
        allowed_repos['dev']=set(external['repositories'])
    if task.repo not in allowed_repos[args.split] or not task.patch.strip() or not task.test_patch.strip():
        raise ValueError('declared TRAIN/DEV calibration task with nonempty references required')
    snapshot = args.inputs/'snapshots'/f'{task.instance_id}.tgz'
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    os.environ['KAGGLE_SANDBOX_DIR'] = str((args.inputs/'sandbox').resolve())
    args.out.mkdir(parents=True)
    isolate_sandbox_cache(args.out/'tmp')
    packages={'psf/requests':'requests','fastapi/fastapi':'fastapi','pypa/packaging':'packaging',
        'psf/black':'black','scrapy/w3lib':'w3lib','pydantic/pydantic-settings':'pydantic_settings',
        'tox-dev/platformdirs':'platformdirs','PyCQA/isort':'isort','strawberry-graphql/strawberry':'strawberry',
        'litestar-org/litestar':'litestar','pallets/click':'click','simonw/sqlite-utils':'sqlite_utils',
        'encode/uvicorn':'uvicorn','pallets/werkzeug':'werkzeug','more-itertools/more-itertools':'more_itertools',
        'pytest-dev/pluggy':'pluggy','scrapy/parsel':'parsel','urllib3/urllib3':'urllib3','jd/tenacity':'tenacity',
        'python-humanize/humanize':'humanize','python-attrs/attrs':'attrs','Python-Markdown/markdown':'markdown',
        'marshmallow-code/marshmallow':'marshmallow','pallets/flask':'flask',
        'executablebooks/markdown-it-py':'markdown_it','Delgan/loguru':'loguru'}
    package=packages[task.repo]
    manager = SourceBoundManager(ContainerConfig(image=args.image, network_mode='none', mem_limit='4g', cpu_quota=200000),package=package)
    config = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.inputs/'snapshots',
                        results_dir=args.out, submission_dir=Path('submission'), models=ModelRegistry(),
                        wheels_dir=args.wheels or args.inputs/'wheels')
    receipt = {'schema':'gemma4-real-task-verifier-controls-v1', 'task':task.instance_id,
               'kind':'verifier calibration; reference oracle is not an agent', 'gpu_used':False,
               'image_id':manager.client.images.get(args.image).id,
               'tasks_sha256':hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
               'split':args.split,
               'wheel_selection_sha256':hashlib.sha256((args.wheels/'receipt.json').read_bytes()).hexdigest() if args.wheels else None,
               'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               'execution_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms':{}}
    if external:
        receipt['external_dev_manifest_sha256']=hashlib.sha256(args.external_dev_manifest.read_bytes()).hexdigest()
        receipt['external_inputs_sha256']=hashlib.sha256((args.inputs/'receipt.json').read_bytes()).hexdigest()
    for arm, patch in [('empty', ''), ('reference', task.patch)]:
        start = time.perf_counter()
        result = await verify_task(manager, config, task, snapshot, agent_patch=patch, start_time=start)
        record = {'resolved':result.resolved, 'test_exit_code':result.test_exit_code,
                  'error':result.error, 'wall_seconds':time.perf_counter()-start}
        receipt['arms'][arm] = record
        (args.out/f'{arm}-test-output.txt').write_text(result.test_output or '')
        print(json.dumps({'arm':arm, **record}), flush=True)
    empty, reference = receipt['arms']['empty'], receipt['arms']['reference']
    receipt['origin_checks']=manager.origin_checks
    cache=args.out/'tmp/swegemma_sp_cache_v8/sp_base.tar'
    receipt['injected_wheel_tar_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None
    if not reference['resolved']:
        receipt['status'] = 'blocked: reference control did not pass'
    elif empty['resolved'] or empty['test_exit_code'] == 0:
        receipt['status'] = 'failed: negative control did not distinguish a fix'
    else:
        receipt['status'] = 'passed'
    (args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2), flush=True)
    return 0 if receipt['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
