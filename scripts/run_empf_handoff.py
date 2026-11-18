"""Wait for the EMPF DEV choice, confirm it once, and prepare the local ZIP."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gemma_particle.conversion_handoff import read
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import live, process_binding
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import sha256_file


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fixed_grid_sha(plan):
    # DEV publishes its measured qualification into this one field.
    return digest({key: value for key, value in plan.items() if key != 'resource_supplement'})


def wait_for_dev(grid, binding):
    while True:
        active = binding is not None and live(binding)
        status = read(grid / 'progress.json')
        if status['state'] == 'error':
            raise RuntimeError(f'DEV stopped with an error. See {grid / "progress.json"}')
        if not active:
            if status['state'] != 'complete_dev_choice':
                raise RuntimeError(f'DEV exited before completing the choice. See {grid / "progress.json"}')
            choice_path = grid / 'choice/selection.json'
            choice = read(choice_path)
            if (sha256_file(choice_path) != status['selection_sha256']
                    or choice['selected'] != status['selected']):
                raise ValueError('Completed DEV choice differs from its progress receipt')
            return choice['selected']
        time.sleep(15)


def require_checkout(root, commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if actual != commit:
        raise ValueError('Expected the frozen confirmation checkout')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['grid', 'packets', 'selection-plan', 'manifest', 'wheels', 'kaggle-python', 'out']:
        # The Kaggle interpreter must retain its virtual environment path.
        parser.add_argument('--' + name, type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument('--grid-pid', type=int, required=True)
    for name in ['executor-commit', 'owner', 'notebook-slug']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require_checkout(root, args.executor_commit)
    grid_plan_path = args.grid / 'plan.json'
    packet_plan_path = args.packets / 'plan.json'
    grid = read(grid_plan_path)
    plan = load_selection_plan(args.selection_plan)
    plan_sha = sha256_file(args.selection_plan)
    packet_sha = sha256_file(packet_plan_path)
    learned = [c for c in plan['conditions'] if c['role'] == 'learned']
    if (len(learned) != 9 or {c['preset'] for c in learned} != {'empf'}
            or grid['condition_ids'] != [c['id'] for c in learned]
            or grid['selection_plan_sha256'] != plan_sha
            or read(packet_plan_path)['selection_plan_sha256'] != plan_sha
            or args.packets.resolve() not in {Path(p['directory']).resolve() for p in grid['producers']}):
        raise ValueError('Expected the declared EMPF grid and its conversion producer')
    try:
        binding = process_binding(args.grid_pid, 'run_resource_dev_grid.py', args.grid.resolve())
    except (FileNotFoundError, ProcessLookupError, ValueError):
        if read(args.grid / 'progress.json')['state'] != 'complete_dev_choice':
            raise
        binding = None
    grid_sha = fixed_grid_sha(grid)
    args.out.mkdir(parents=True)
    write(args.out / 'plan.json', {
        'schema': 'gemma4-empf-handoff-plan-v1', 'grid': str(args.grid), 'grid_producer': binding,
        'fixed_grid_sha256': grid_sha, 'packets': str(args.packets), 'packet_plan_sha256': packet_sha,
        'selection_plan': str(args.selection_plan), 'selection_plan_sha256': plan_sha,
        'manifest': str(args.manifest), 'executor_commit': args.executor_commit, 'executor': str(root),
        'kaggle_python': str(args.kaggle_python), 'wheels': str(args.wheels), 'owner': args.owner,
        'notebook_slug': args.notebook_slug, 'confirmation_tasks_per_condition': plan['confirmation']['count'],
        'account_writes': False,
    })

    def publish(state, **details):
        update = {'state': state, 'as_of': datetime.now(timezone.utc).isoformat(), **details}
        write(args.out / 'progress.json', update)
        print(json.dumps(update), flush=True)

    if args.plan_only:
        publish('prepared', protected_inputs_opened=False, network_requests=False, model_inference=False)
        return

    env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    stage = 'waiting_for_completed_dev'
    publish(stage, protected_inputs_opened=False, network_requests=False, model_inference=False)

    def run(state, name, *options, python=sys.executable):
        nonlocal stage
        stage = state
        require_checkout(root, args.executor_commit)
        log = args.out / (name + '.log')
        publish(stage, selected=selected, log=str(log))
        command = [str(python), str(root / 'scripts' / (name + '.py')), *map(str, options)]
        with log.open('x') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        selected = wait_for_dev(args.grid, binding)
        measured = read(grid_plan_path)
        if (fixed_grid_sha(measured) != grid_sha or sha256_file(packet_plan_path) != packet_sha
                or sha256_file(args.selection_plan) != plan_sha):
            raise ValueError('Declared DEV grid, conversion plan or selection policy changed')
        if selected is None:
            publish('no_eligible_dev_choice', protected_inputs_opened=False, network_requests=False, model_inference=False)
            return
        resource = measured['resource_supplement']
        supplement = Path(resource['path'])
        if sha256_file(supplement) != resource['sha256']:
            raise ValueError('Measured DEV resource qualification changed')

        access_dir = args.out / 'access'
        access = access_dir / 'access.json'
        official = args.out / 'official-inputs'
        environments_dir = args.out / 'environments'
        environments = environments_dir / 'qualification.json'
        pair = args.out / 'pair'
        handoff = args.out / 'handoff'
        scope = ('--selection-plan', args.selection_plan)
        executor = ('--executor', root, '--executor-commit', args.executor_commit)
        run('reauditing_dev', 'unlock_resource_confirmation', '--dev-grid', args.grid,
            '--dev-executor', grid['executor'], '--dev-executor-commit', grid['executor_commit'],
            '--resource-auditor', grid['offline_auditor'], '--resource-auditor-commit', grid['offline_auditor_commit'],
            '--resource-supplement', supplement, '--packets', args.packets, *scope,
            '--manifest', args.manifest, '--out', access_dir)
        run('downloading_confirmation_inputs', 'fetch_official_inputs', '--confirmation-access', access,
            '--all-confirmation', '--max-download-gib', shutil.disk_usage(args.out).free / 1024**3,
            '--out', official, python=args.kaggle_python)
        run('qualifying_confirmation_environments', 'verify_confirmation_environments', '--access', access,
            *executor, *scope, '--raw-tasks', official / 'tasks.jsonl', '--inputs', official,
            '--wheels', args.wheels, '--out', environments_dir)
        run('confirming_fixed_pair', 'run_confirmation_pair', '--access', access, '--environments', environments,
            *executor, *scope, '--out', pair)
        run('preparing_local_zip', 'prepare_selected_candidate', '--access', access, '--environments', environments,
            '--pair', read(pair / 'plan.json')['pair_claim'], '--comparison', pair / 'comparison.json',
            '--owner', args.owner, '--notebook-slug', args.notebook_slug, '--out', handoff)
        readiness_path = handoff / 'readiness.json'
        readiness = read(readiness_path)
        archive = handoff / 'submission.zip'
        artifact_sha = sha256_file(archive)
        if (readiness['status'] != 'prepared' or readiness['handoff_zip'] != str(archive)
                or readiness['artifact_sha256'] != artifact_sha or readiness['account_requests'] is not False):
            raise ValueError('Local ZIP differs from the evaluated artifact receipt')
        publish('handoff_ready', selected=selected, submission_zip=str(archive), artifact_sha256=artifact_sha,
            readiness=str(readiness_path), readiness_sha256=sha256_file(readiness_path), account_writes=False)
    except Exception as exc:
        publish('error', stage=stage, error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
