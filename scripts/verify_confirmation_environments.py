"""Calibrate all49 released holdout environments on two shared CPU workers.

Reference data is read only by the separately gated official verifier children.
Plan-only checks source/image/wheel metadata without loading confirmation tasks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from gemma_particle.confirmation import load_confirmation_access
from gemma_particle.confirmation_environments import (confirmation_input_hashes,
    control_qualified, audit_confirmation_control)
from gemma_particle.conversion_handoff import read
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import package_identity, sha256_file, wheel_identity


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['access', 'executor', 'out']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--executor-commit', required=True)
    p.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    p.add_argument('--raw-tasks', type=Path, default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs', type=Path, default=Path('data/kaggle-official'))
    p.add_argument('--wheels', type=Path, default=Path('runs/original-dev-environments-v1/wheels'))
    p.add_argument('--image', help='explicit immutable image ID; defaults to the original secondary DEV image')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    if args.out.exists(): raise FileExistsError('confirmation environment evidence is immutable')
    scope = None if args.plan_only else load_confirmation_access(args.access)
    selection = load_selection_plan(args.selection_plan); selection_sha = sha256_file(args.selection_plan)
    if scope is not None and scope.identity['selection_plan_sha256'] != selection_sha:
        raise ValueError('original access selection plan required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.executor, text=True).strip() != args.executor_commit:
        raise ValueError('exact frozen confirmation verifier executor required')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=args.executor, check=True)
    source_sha = read(selection['secondary']['qualification'])['source_sha256']
    if sha256_file(args.raw_tasks) != source_sha: raise ValueError('original official reference dataset required')
    image = args.image or selection['secondary']['image_id']
    actual = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    if actual != image: raise ValueError('immutable image ID rather than mutable tag required')
    wheels = wheel_identity(args.wheels)
    if not wheels: raise ValueError('explicit unchanged wheel bytes required')
    root = Path(__file__).resolve().parents[1]
    execution_sources = {str(args.executor / name): sha256_file(args.executor / name) for name in [
        'scripts/verify_controls.py', 'src/gemma_particle/sandbox.py', 'src/gemma_particle/confirmation.py',
        'src/gemma_particle/dev.py', 'src/gemma_particle/harness_identity.py']}
    controller_sources = {str(root / name): sha256_file(root / name) for name in [
        'scripts/verify_confirmation_environments.py', 'src/gemma_particle/confirmation_environments.py',
        'src/gemma_particle/confirmation.py']}
    declaration = {'schema': 'gemma4-confirmation-environment-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'selection_plan_sha256': selection_sha, 'access_receipt': str(args.access), 'executor_commit': args.executor_commit,
        'raw_tasks': str(args.raw_tasks), 'raw_tasks_sha256': source_sha, 'inputs': str(args.inputs),
        'image_id': image, 'source_wheel_directory': str(args.wheels), 'wheels': wheels,
        'download_receipt_sha256':sha256_file(args.inputs/'download-receipt.json'),
        'sandbox_setup_sha256': sha256_file(args.inputs / 'sandbox/setup.py'),
        'harness': package_identity(), 'execution_sources': execution_sources, 'controller_sources': controller_sources,
        'workers': 2, 'gpu_used': False, 'model_inference': False, 'planned': 49,
        'scope': 'all original49 negative/reference environment controls; failures retained, never agent accuracy'}
    if scope is not None:
        declaration.update(access_identity=scope.identity, task_ids=[r['instance_id'] for r in scope.rows],
            input_files=confirmation_input_hashes(scope, args.inputs))
    args.out.mkdir(parents=True)
    if args.plan_only:
        write(args.out / 'plan.json', declaration)
        write(args.out / 'progress.json', {'state': 'prepared', 'planned': 49, 'holdout_task_contents_parsed': False,
            'model_inference': False, 'gpu_used': False})
        print(json.dumps({'state': 'prepared', 'planned': 49, 'holdout_task_contents_parsed': False})); return 0
    logs = args.out / 'logs'; logs.mkdir(); profile = args.out / 'wheels'; profile.mkdir()
    for name in wheels:
        try: os.link((args.wheels / name).resolve(), profile / name)
        except OSError: shutil.copyfile(args.wheels / name, profile / name)
    write(profile / 'receipt.json', {'schema': 'gemma4-confirmation-wheel-profile-v1', 'wheels': wheels,
        'policy': 'one explicit unchanged wheel profile for the complete panel; workspace origins checked'})
    declaration.update(wheel_profile=str(profile), wheel_profile_sha256=sha256_file(profile / 'receipt.json'))
    write(args.out / 'plan.json', declaration); finished = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.executor / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def unchanged():
        if (any(sha256_file(path) != value for path, value in {**execution_sources, **controller_sources}.items())
                or wheel_identity(profile) != wheels or confirmation_input_hashes(scope, args.inputs) != declaration['input_files']
                or sha256_file(args.raw_tasks) != source_sha):
            raise ValueError('confirmation execution source or declared input changed')
    def run(task_id):
        log = logs / (task_id + '.log'); destination = args.out / task_id
        command = [str(args.executor / '.harness-venv/bin/python'), 'scripts/verify_controls.py', '--split', 'holdout',
            '--confirmation-access', str(args.access), '--tasks', str(args.raw_tasks), '--inputs', str(args.inputs),
            '--task-id', task_id, '--image', image, '--wheels', str(profile), '--out', str(destination)]
        with log.open('x') as stream:
            child = subprocess.run(command, cwd=args.executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
        path = destination / 'result.json'; receipt = read(path) if path.exists() else None
        record = {'instance_id': task_id, 'returncode': child.returncode, 'log': str(log), 'log_sha256': sha256_file(log),
            'status': receipt['status'] if receipt else 'failed before verifier receipt',
            'receipt': str(path) if receipt else None, 'receipt_sha256': sha256_file(path) if receipt else None,
            'qualified': bool(control_qualified(receipt, child.returncode))}
        audit_confirmation_control(record, declaration, scope); return record
    try:
        write(args.out / 'progress.json', {'state': 'calibrating', 'planned': 49, 'finished': finished, 'qualified': 0})
        unchanged()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, task_id) for task_id in declaration['task_ids']]
            for future in as_completed(futures):
                record = future.result(); finished.append(record)
                write(args.out / 'progress.json', {'state': 'calibrating', 'planned': 49, 'finished': finished,
                    'qualified': sum(r['qualified'] for r in finished)})
                print(json.dumps({'instance_id': record['instance_id'], 'qualified': record['qualified'], 'status': record['status']}), flush=True)
        unchanged(); by_id = {r['instance_id']: r for r in finished}
        result = {'schema': 'gemma4-confirmation-environments-v1', 'state': 'complete',
            'access_identity': scope.identity, 'planned': 49, 'qualified': sum(r['qualified'] for r in finished),
            'controls': [by_id[task_id] for task_id in declaration['task_ids']], 'image_id': image, 'wheels': str(profile),
            'plan': str(args.out / 'plan.json'), 'plan_sha256': sha256_file(args.out / 'plan.json'),
            'gpu_used': False, 'model_inference': False, 'scope': declaration['scope']}
        write(args.out / 'qualification.json', result)
        write(args.out / 'progress.json', {'state': 'complete', 'planned': 49, 'finished': result['controls'], 'qualified': result['qualified']})
        return 0 if result['qualified'] == 49 else 2
    except Exception as exc:
        write(args.out / 'progress.json', {'state': 'error', 'planned': 49, 'finished': finished,
            'error_type': type(exc).__name__, 'error': str(exc)}); raise


if __name__ == '__main__': raise SystemExit(main())
