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


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('data/public-mirror/root/.cache/kagglehub/competitions/gemma-4-developer-agent/tasks.jsonl'))
    parser.add_argument('--inputs', type=Path, default=Path('data/control-inputs/root/.cache/kagglehub/competitions/gemma-4-developer-agent'))
    parser.add_argument('--task-id', default='requests_6757')
    parser.add_argument('--image', default='gemma-particle-sandbox:mirror-v30')
    parser.add_argument('--out', type=Path, default=Path('runs/verifier-controls-v1'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('control run is immutable')
    task = next(row for row in load_tasks(args.tasks) if row.instance_id == args.task_id)
    if task.repo != 'psf/requests' or not task.patch.strip() or not task.test_patch.strip():
        raise ValueError('declared DEV calibration task with nonempty references required')
    snapshot = args.inputs/'snapshots'/f'{task.instance_id}.tgz'
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    os.environ['KAGGLE_SANDBOX_DIR'] = str((args.inputs/'sandbox').resolve())
    args.out.mkdir(parents=True)
    manager = ContainerManager(ContainerConfig(image=args.image, network_mode='none', mem_limit='4g', cpu_quota=200000))
    config = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.inputs/'snapshots',
                        results_dir=args.out, submission_dir=Path('submission'), models=ModelRegistry(),
                        wheels_dir=args.inputs/'wheels')
    receipt = {'schema':'gemma4-real-task-verifier-controls-v1', 'task':task.instance_id,
               'kind':'verifier calibration; reference oracle is not an agent', 'gpu_used':False,
               'image_id':manager.client.images.get(args.image).id,
               'tasks_sha256':hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
               'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               'execution_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms':{}}
    for arm, patch in [('empty', ''), ('reference', task.patch)]:
        start = time.perf_counter()
        result = await verify_task(manager, config, task, snapshot, agent_patch=patch, start_time=start)
        record = {'resolved':result.resolved, 'test_exit_code':result.test_exit_code,
                  'error':result.error, 'wall_seconds':time.perf_counter()-start}
        receipt['arms'][arm] = record
        (args.out/f'{arm}-test-output.txt').write_text(result.test_output or '')
        print(json.dumps({'arm':arm, **record}), flush=True)
    empty, reference = receipt['arms']['empty'], receipt['arms']['reference']
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
