"""Observe a genuinely published learned submission and fresh public records.

No publishing, submission, model execution or selection occurs here. Pending
and unknown statuses retain the exact submission ID; watch has no time limit.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from requests.exceptions import ConnectionError, Timeout

from gemma_particle.harness_identity import sha256_file
from gemma_particle.kaggle_publication import safe_error
from gemma_particle.official_results import audit_publication, observe_official_record
from gemma_particle.selected_publication import audit_selected_publication


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['publication', 'readiness', 'access', 'environments', 'pair', 'comparison', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root: raise ValueError('run inside the frozen monitor checkout')
    if args.out.exists(): raise FileExistsError('observations are immutable;use a fresh output for another read')
    names = ['scripts/monitor_selected_submission.py', 'scripts/publish_selected_candidate.py',
        'scripts/prepare_kaggle_candidate.py', 'src/gemma_particle/official_results.py',
        'src/gemma_particle/kaggle_publication.py', 'src/gemma_particle/selected_publication.py',
        'research/kaggle-publication-runtime-v1.json']
    sources = {str(root / name): sha256_file(root / name) for name in names}
    plan = {'schema': 'gemma4-official-learned-observation-plan-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
        'selection_or_mutation': False, 'model_inference': False, 'goal_marked_complete': False,
        'scope': 'exact learned submission and fresh public score observations;full goal completion remains a separate audit'}
    if args.plan_only:
        args.out.mkdir(parents=True); write(args.out / 'plan.json', plan)
        write(args.out / 'progress.json', {'state': 'prepared', 'protected_inputs_opened': False, 'account_requests': 0})
        print(json.dumps({'state': 'prepared', 'protected_inputs_opened': False, 'account_requests': 0})); return
    preparer = load_script(root / 'scripts/prepare_kaggle_candidate.py', 'fixed_monitor_payload')
    publisher = load_script(root / 'scripts/publish_selected_candidate.py', 'fixed_monitor_publisher')
    candidate = audit_selected_publication(args.readiness, args.access, args.environments, args.pair,
        args.comparison, root, preparer.notebook_source)
    binding = audit_publication(args.publication, candidate, sha256_file(args.readiness))
    runtime, sdk_sources = publisher.validate_runtime(root / 'research/kaggle-publication-runtime-v1.json')
    sources.update(sdk_sources)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    def unchanged():
        pins = {**sources, **candidate['experiment_pins'], **binding['publication_files']}
        if any(sha256_file(path) != value for path, value in pins.items()):
            raise ValueError('original selected experiment,publication or observation source changed')
    unchanged()
    from kaggle import api
    args.out.mkdir(parents=True, mode=0o700)
    plan.update(submission_id=binding['submission_id'], artifact_sha256=binding['artifact_sha256'],
        runtime_versions=runtime['versions'], publication=str(args.publication), publication_files=binding['publication_files'])
    write(args.out / 'plan.json', plan)
    cycle = 0
    while True:
        unchanged()
        try:
            result = observe_official_record(api, binding, args.out / 'api-observations.jsonl')
        except Exception as exc:
            result = {'state': 'observation_incomplete', 'submission_id': binding['submission_id'],
                'verified_strict_public_record': False, 'goal_marked_complete': False, **safe_error(exc)}
            code = result['http_status']
            transient = (isinstance(exc, (ConnectionError, Timeout, TimeoutError))
                or code in [408, 429] or isinstance(code, int) and 500 <= code <= 599)
            write(args.out / f'observation-{cycle:06d}.json', result); write(args.out / 'progress.json', result)
            if not args.watch or not transient: raise
            time.sleep(60); cycle += 1; continue
        unchanged()
        write(args.out / f'observation-{cycle:06d}.json', result)
        write(args.out / 'progress.json', result)
        print(json.dumps({'state': result['state'], 'submission_id': binding['submission_id'],
            'verified_strict_public_record': result['verified_strict_public_record']}), flush=True)
        if not args.watch or result['state'] != 'waiting_official_evaluation': return
        time.sleep(60); cycle += 1


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'state': 'error', **safe_error(exc)})); raise SystemExit(1)
