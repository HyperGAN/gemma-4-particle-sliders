"""Independently reaudit a resource-qualified DEV choice before releasing holdout.

Generation stays on its original frozen executor. The separately frozen offline
auditor must preserve the original audit/selection helpers. No model is run.
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
from gemma_particle.resource_dispatch import task_audit_bridge
from gemma_particle.resource_supplement import bind_resource_choice


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dev-grid', 'dev-executor', 'resource-auditor', 'resource-supplement', 'packets', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dev-executor-commit', required=True)
    p.add_argument('--resource-auditor-commit', required=True)
    p.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    p.add_argument('--manifest', type=Path, default=Path('data/splits-v1/holdout.jsonl'))
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    if args.out.exists(): raise FileExistsError('resource confirmation access evidence is immutable')
    plan = load_selection_plan(args.selection_plan); plan_sha = sha256_file(args.selection_plan)
    grid_plan = read(args.dev_grid / 'plan.json'); packet_plan = read(args.packets / 'plan.json')
    if (grid_plan['executor_commit'] != args.dev_executor_commit
            or Path(grid_plan['executor']).resolve() != args.dev_executor
            or grid_plan['selection_plan_sha256'] != plan_sha or packet_plan['selection_plan_sha256'] != plan_sha
            or sha256_file(args.manifest) != plan['confirmation']['manifest_sha256']):
        raise ValueError('original generation executor,packets,selection and locked manifest required')
    task_executor = Path(grid_plan.get('task_evidence_executor', args.dev_executor)).resolve()
    task_commit = grid_plan.get('task_evidence_executor_commit', args.dev_executor_commit)
    source_sets = {}
    for executor, commit in [(args.dev_executor, args.dev_executor_commit), (task_executor, task_commit),
            (args.resource_auditor, args.resource_auditor_commit)]:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=executor, text=True).strip() != commit:
            raise ValueError('exact clean frozen generation and offline audit sources required')
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=executor, check=True)
        names = subprocess.check_output(['git', 'ls-files', 'scripts', 'src', 'configs', 'submission', 'sources.lock.json'],
            cwd=executor, text=True).splitlines()
        source_sets[executor] = {str(executor / name): sha256_file(executor / name) for name in names}
    bridge = task_audit_bridge(args.dev_executor, task_executor, source_sets[args.dev_executor], source_sets[task_executor])
    if 'task_audit_bridge' in grid_plan:
        if (grid_plan['task_audit_bridge'] != bridge
                or Path(grid_plan['offline_auditor']).resolve() != args.resource_auditor
                or grid_plan['offline_auditor_commit'] != args.resource_auditor_commit
                or any(grid_plan['declaration_pins'].get(path) != value
                    for scope in source_sets.values() for path, value in scope.items())):
            raise ValueError('original frozen DEV task and offline audit bindings required')
    elif task_executor != args.dev_executor or task_commit != args.dev_executor_commit:
        raise ValueError('separate task evidence executor requires a declared audit bridge')
    for name in ['src/gemma_particle/dev_evidence.py', 'src/gemma_particle/dev_selection.py']:
        if sha256_file(task_executor / name) != sha256_file(args.resource_auditor / name):
            raise ValueError('offline auditor must match the declared DEV task audit and selection helpers')
    pins = {str(path): sha256_file(path) for path in [args.selection_plan, args.dev_grid / 'plan.json',
        args.packets / 'plan.json']}
    for scope in source_sets.values(): pins.update(scope)
    if args.resource_supplement.exists(): pins[str(args.resource_supplement)] = sha256_file(args.resource_supplement)
    root = Path(__file__).resolve().parents[1]
    sources = {str(root / name): sha256_file(root / name) for name in ['scripts/unlock_resource_confirmation.py',
        'src/gemma_particle/confirmation.py', 'src/gemma_particle/resource_supplement.py',
        'src/gemma_particle/resource_dispatch.py']}
    args.out.mkdir(parents=True)
    declaration = {'schema': 'gemma4-confirmation-access-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'dev_grid': str(args.dev_grid), 'packet_producer': str(args.packets),
        'dev_executor_commit': args.resource_auditor_commit,
        'offline_auditor': str(args.resource_auditor),
        'generation_executor': str(args.dev_executor), 'generation_executor_commit': args.dev_executor_commit,
        'task_evidence_executor': str(task_executor), 'task_evidence_executor_commit': task_commit,
        'task_audit_bridge': bridge,
        'resource_supplement': str(args.resource_supplement), 'selection_plan_sha256': plan_sha,
        'holdout_manifest_sha256': sha256_file(args.manifest), 'declaration_pins': pins, 'sources': sources,
        'gpu_used': False, 'model_inference': False, 'holdout_task_contents_parsed': False,
        'executor_field_semantics': 'dev_executor_commit names the actual offline auditor;generation executor recorded separately',
        'scope': 'independent qualified-resource DEV reaudit before confirmation access;no model or task evaluation'}
    write(args.out / 'plan.json', declaration)
    if args.plan_only:
        write(args.out / 'result.json', {'status': 'prepared', 'holdout_task_contents_parsed': False,
            'resource_policy_qualified': False, 'model_inference': False})
        print(json.dumps({'status': 'prepared', 'protected_tasks_parsed': False, 'resource_policy_qualified': False})); return
    try:
        choice, original = validate_dev_choice(args.dev_grid, plan, plan_sha)
        choice_sha = sha256_file(original)
        policy = bind_resource_choice(choice, grid_plan, args.resource_supplement, plan_sha)
        if str(args.resource_supplement) not in pins:
            raise ValueError('resource policy must exist in the original access declaration')
        command = [str(args.resource_auditor / '.harness-venv/bin/python'), 'scripts/compare_dev_conditions.py',
            '--selection-plan', str(args.selection_plan), '--resource-supplement', str(args.resource_supplement),
            '--out', str(args.out / 'reaudited-dev-choice')]
        for record in choice['condition_results'].values(): command.extend(['--condition-result', record['path']])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.resource_auditor / 'src'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (args.out / 'dev-reaudit.log').open('x') as stream:
            result = subprocess.run(command, cwd=args.resource_auditor, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('actual qualified-resource DEV evidence failed independent offline reaudit')
        if (sha256_file(original) != choice_sha
                or any(sha256_file(path) != value for path, value in {**pins, **sources}.items())):
            raise ValueError('original choice,resource,generation or audit source changed during reaudit')
        access = build_confirmation_access(args.dev_grid, plan, plan_sha,
            args.out / 'reaudited-dev-choice/selection.json', args.manifest, args.packets)
        access.update(selection_plan=str(args.selection_plan), access_plan=str(args.out / 'plan.json'),
            access_plan_sha256=sha256_file(args.out / 'plan.json'),
            offline_dev_auditor_executor_commit=args.resource_auditor_commit,
            offline_dev_audit_log_sha256=sha256_file(args.out / 'dev-reaudit.log'), offline_dev_auditor_returncode=result.returncode)
        write(args.out / 'access.json', access)
        write(args.out / 'result.json', {'status': 'access_released', 'access_sha256': sha256_file(args.out / 'access.json'),
            'resource_supplement': policy, 'holdout_task_contents_parsed': True, 'model_inference': False})
        print(json.dumps({'status': 'access_released', 'selected': choice['selected'], 'confirmation_tasks': 49}))
    except Exception as exc:
        write(args.out / 'result.json', {'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
            'access_released': False, 'model_inference': False})
        raise


if __name__ == '__main__': main()
