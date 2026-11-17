"""Reaudit the actual DEV choice, then release the original visible holdout.

Plan-only hashes the locked manifest without parsing its task contents. Neither
mode runs a model, verifier container, confirmation task, or Kaggle submission.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from gemma_particle.confirmation import validate_dev_choice, build_confirmation_access
from gemma_particle.conversion_handoff import read
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dev-grid', 'dev-executor', 'packets', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dev-executor-commit', required=True)
    parser.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits-v1/holdout.jsonl'))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.out.exists():
        raise FileExistsError('confirmation access evidence is immutable')
    plan = load_selection_plan(args.selection_plan)
    plan_sha = sha256_file(args.selection_plan)
    grid_plan = read(args.dev_grid / 'plan.json')
    packet_plan = read(args.packets / 'plan.json')
    if (grid_plan['executor_commit'] != args.dev_executor_commit
            or Path(grid_plan['executor']).resolve() != args.dev_executor
            or grid_plan['selection_plan_sha256'] != plan_sha or packet_plan['selection_plan_sha256'] != plan_sha
            or sha256_file(args.manifest) != plan['confirmation']['manifest_sha256']
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.dev_executor, text=True).strip() != args.dev_executor_commit):
        raise ValueError('original declared DEV executor, packet producer and locked manifest required')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=args.dev_executor, check=True)
    args.out.mkdir(parents=True)
    pins = {str(p): sha256_file(p) for p in [args.selection_plan, args.dev_grid / 'plan.json',
        args.packets / 'plan.json', args.dev_executor / 'scripts/compare_dev_conditions.py']}
    root = Path(__file__).resolve().parents[1]
    declaration = {'schema': 'gemma4-confirmation-access-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'dev_grid': str(args.dev_grid), 'dev_executor_commit': args.dev_executor_commit,
        'packet_producer': str(args.packets), 'holdout_manifest_sha256': sha256_file(args.manifest),
        'selection_plan_sha256': plan_sha, 'declaration_pins': pins,
        'sources': {name: sha256_file(root / name) for name in ['scripts/unlock_confirmation.py', 'src/gemma_particle/confirmation.py']},
        'gpu_used': False, 'model_inference': False, 'holdout_task_contents_parsed': False,
        'scope': 'explicit access preparation; no confirmation evaluation, quality or official score'}
    write(args.out / 'plan.json', declaration)
    if args.plan_only:
        write(args.out / 'result.json', {'status': 'prepared', 'holdout_task_contents_parsed': False,
            'gpu_used': False, 'model_inference': False})
        print(json.dumps({'status': 'prepared', 'holdout_task_contents_parsed': False})); return
    try:
        choice, original = validate_dev_choice(args.dev_grid, plan, plan_sha)
        choice_sha = sha256_file(original)
        command = [str(args.dev_executor / '.harness-venv/bin/python'), 'scripts/compare_dev_conditions.py',
            '--selection-plan', str(args.selection_plan), '--out', str(args.out / 'reaudited-dev-choice')]
        for record in choice['condition_results'].values():
            command.extend(['--condition-result', record['path']])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.dev_executor / 'src'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (args.out / 'dev-reaudit.log').open('x') as stream:
            result = subprocess.run(command, cwd=args.dev_executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('actual complete DEV evidence failed the independent offline reaudit')
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=args.dev_executor, check=True)
        if sha256_file(original) != choice_sha or any(sha256_file(p) != value for p, value in pins.items()):
            raise ValueError('locked choice or declaration changed during offline reaudit')
        access = build_confirmation_access(args.dev_grid, plan, plan_sha,
            args.out / 'reaudited-dev-choice/selection.json', args.manifest, args.packets)
        access.update(selection_plan=str(args.selection_plan), access_plan=str((args.out / 'plan.json').resolve()),
            access_plan_sha256=sha256_file(args.out / 'plan.json'),
            offline_dev_auditor_executor_commit=args.dev_executor_commit,
            offline_dev_audit_log_sha256=sha256_file(args.out / 'dev-reaudit.log'),
            offline_dev_auditor_returncode=result.returncode)
        write(args.out / 'access.json', access)
        write(args.out / 'result.json', {'status': 'access_released', 'access_sha256': sha256_file(args.out / 'access.json'),
            'holdout_task_contents_parsed': True, 'model_inference': False, 'confirmation_results': None})
        print(json.dumps({'status': 'access_released', 'selected': choice['selected'], 'confirmation_tasks': 49}))
    except Exception as exc:
        write(args.out / 'result.json', {'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
            'access_released': False, 'model_inference': False})
        raise


if __name__ == '__main__':
    main()
