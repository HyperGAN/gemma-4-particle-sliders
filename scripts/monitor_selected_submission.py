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

from gemma_particle.conversion_handoff import read
from gemma_particle.harness_identity import sha256_file
from gemma_particle.kaggle_publication import safe_error
from gemma_particle.official_results import audit_publication, observe_official_record
from gemma_particle.selected_publication import audit_selected_publication
from gemma_particle.user_submission import (SubmissionNotVisible, bind_user_submission,
    capture_user_reference, load_user_reference)


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def retryable_observation(exc):
    status = safe_error(exc)['http_status']
    return (isinstance(exc, (ConnectionError, Timeout, TimeoutError, SubmissionNotVisible))
        or status in [408, 429] or isinstance(status, int) and 500 <= status <= 599)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['readiness', 'access', 'environments', 'pair', 'comparison', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--publication', type=Path)
    modes.add_argument('--capture-user-reference', action='store_true')
    modes.add_argument('--submission-id', type=int)
    p.add_argument('--prior-reference', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    if (args.submission_id is not None) != (args.prior_reference is not None) or (
            args.submission_id is not None and args.submission_id <= 0):
        p.error('--submission-id requires a positive ID and --prior-reference')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root: raise ValueError('run inside the frozen monitor checkout')
    if args.out.exists(): raise FileExistsError('observations are immutable;use a fresh output for another read')
    names = ['scripts/monitor_selected_submission.py', 'scripts/publish_selected_candidate.py',
        'scripts/prepare_kaggle_candidate.py', 'src/gemma_particle/official_results.py',
        'src/gemma_particle/kaggle_publication.py', 'src/gemma_particle/selected_publication.py',
        'research/kaggle-publication-runtime-v1.json', 'src/gemma_particle/user_submission.py']
    sources = {str(root / name): sha256_file(root / name) for name in names}
    plan = {'schema': 'gemma4-official-learned-observation-plan-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
        'origin': 'publication' if args.publication else 'user_submission',
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
    readiness_sha = sha256_file(args.readiness)
    binding = None
    reference = None
    manual_pins = {}
    if args.publication:
        binding = audit_publication(args.publication, candidate, readiness_sha)
    else:
        archive = Path(read(args.readiness)['handoff_zip'])
        if archive.is_symlink() or sha256_file(archive) != candidate['artifact_sha256']:
            raise ValueError('Expected the exact local handoff ZIP')
        manual_pins[str(archive)] = candidate['artifact_sha256']
        if args.prior_reference:
            manual_pins[str(args.prior_reference)] = sha256_file(args.prior_reference)
            reference = load_user_reference(args.prior_reference, candidate, readiness_sha)
    runtime, sdk_sources = publisher.validate_runtime(root / 'research/kaggle-publication-runtime-v1.json')
    sources.update(sdk_sources)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    def unchanged():
        publication_files = binding['publication_files'] if binding else {}
        pins = {**sources, **candidate['experiment_pins'], **manual_pins, **publication_files}
        if any(sha256_file(path) != value for path, value in pins.items()):
            raise ValueError('original selected experiment,publication or observation source changed')
    unchanged()
    from kaggle import api
    args.out.mkdir(parents=True, mode=0o700)
    plan.update(runtime_versions=runtime['versions'])
    if args.publication:
        plan.update(submission_id=binding['submission_id'], artifact_sha256=binding['artifact_sha256'],
            publication=str(args.publication), publication_files=binding['publication_files'])
    else:
        plan.update(submission_id=args.submission_id, artifact_sha256=candidate['artifact_sha256'],
            readiness_sha256=readiness_sha, handoff_pins=manual_pins)
    write(args.out / 'plan.json', plan)
    if args.capture_user_reference:
        path = args.out / 'user-reference.json'
        while True:
            unchanged()
            try:
                reference = capture_user_reference(api, candidate, readiness_sha, path)
                break
            except Exception as exc:
                write(args.out / 'progress.json', {'state': 'user_reference_observation_incomplete',
                    'account_mutations': 0, **safe_error(exc)})
                if not args.watch or not retryable_observation(exc):
                    raise
                time.sleep(60)
        unchanged()
        result = {'state': 'user_reference_ready', 'reference': str(path),
            'reference_sha256': sha256_file(path), 'best_score': reference['leaderboard']['best_score'],
            'artifact_sha256': candidate['artifact_sha256'], 'account_mutations': 0}
        write(args.out / 'progress.json', result)
        print(json.dumps(result), flush=True)
        return
    cycle = 0
    while True:
        unchanged()
        try:
            if binding is None:
                binding = bind_user_submission(api, args.submission_id, candidate, args.prior_reference,
                    reference, args.out / f'submission-binding-{cycle:06d}')
                unchanged()
                write(args.out / 'user-submission-binding.json', binding)
            result = observe_official_record(api, binding, args.out / 'api-observations.jsonl')
        except Exception as exc:
            result = {'state': 'observation_incomplete',
                'submission_id': binding['submission_id'] if binding else args.submission_id,
                'verified_strict_public_record': False, 'goal_marked_complete': False, **safe_error(exc)}
            code = result['http_status']
            transient = retryable_observation(exc) or binding is None and code == 404
            write(args.out / f'observation-{cycle:06d}.json', result); write(args.out / 'progress.json', result)
            if not args.watch or not transient: raise
            time.sleep(60); cycle += 1; continue
        unchanged()
        write(args.out / f'observation-{cycle:06d}.json', result)
        write(args.out / 'progress.json', result)
        print(json.dumps({'state': result['state'], 'submission_id': binding['submission_id'],
            'verified_strict_public_record': result['verified_strict_public_record']}), flush=True)
        if not args.watch or result['state'] not in ['waiting_official_evaluation', 'waiting_public_leaderboard_identity']: return
        time.sleep(60); cycle += 1


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'state': 'error', **safe_error(exc)})); raise SystemExit(1)
