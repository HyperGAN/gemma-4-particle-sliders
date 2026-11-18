"""Capture the public record for the user's EMPF ZIP after the handoff is ready."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gemma_particle.conversion_handoff import read
from gemma_particle.dev_dispatch import live, process_binding
from gemma_particle.harness_identity import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--handoff-pid', type=int, required=True)
    parser.add_argument('--executor-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    handoff, out = args.handoff.resolve(), args.out.resolve()
    root = Path(__file__).resolve().parents[1]

    def check_checkout():
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if actual != args.executor_commit:
            raise ValueError('Expected the frozen user submission observer')
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)

    check_checkout()
    plan = handoff / 'plan.json'
    progress = handoff / 'progress.json'
    declaration = read(plan)
    if declaration.get('schema') != 'gemma4-empf-handoff-plan-v1' or declaration.get('account_writes') is not False:
        raise ValueError('Expected the registered EMPF handoff')
    plan_sha = sha256_file(plan)
    try:
        binding = process_binding(args.handoff_pid, 'run_empf_handoff.py', handoff)
    except (FileNotFoundError, ValueError):
        if read(progress)['state'] != 'handoff_ready':
            raise
        binding = None
    out.mkdir(parents=True, mode=0o700)
    (out / 'plan.json').write_text(json.dumps({'schema': 'gemma4-user-handoff-reference-wait-v1',
        'handoff': str(handoff), 'handoff_plan_sha256': plan_sha, 'handoff_binding': binding,
        'executor_commit': args.executor_commit, 'account_mutations': 0}, indent=2) + '\n')

    def publish(state, **values):
        status = {'state': state, 'as_of': datetime.now(timezone.utc).isoformat(), 'account_mutations': 0, **values}
        temporary = out / 'progress.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'progress.json')
        print(json.dumps(status), flush=True)

    if args.plan_only:
        publish('prepared', account_reads_started=False)
        return
    publish('waiting_for_completed_handoff', account_reads_started=False)
    try:
        while True:
            active = binding is not None and live(binding)
            status = read(progress)
            if status['state'] == 'no_eligible_dev_choice':
                publish('no_eligible_dev_choice', account_reads_started=False)
                return
            if status['state'] == 'error' or not active and status['state'] != 'handoff_ready':
                raise RuntimeError('EMPF handoff stopped before packaging completed')
            if not active:
                break
            time.sleep(15)
        check_checkout()
        readiness = handoff / 'handoff/readiness.json'
        archive = handoff / 'handoff/submission.zip'
        if (sha256_file(plan) != plan_sha or Path(status['readiness']).resolve() != readiness
                or Path(status['submission_zip']).resolve() != archive
                or sha256_file(readiness) != status['readiness_sha256']
                or sha256_file(archive) != status['artifact_sha256']):
            raise ValueError('The EMPF handoff differs from its verified ready receipt')
        pair = handoff / 'pair'
        reference_dir = out / 'reference-observation'
        command = [sys.executable, str(root / 'scripts/monitor_selected_submission.py'),
            '--capture-user-reference', '--watch', '--readiness', str(readiness),
            '--access', str(handoff / 'access/access.json'),
            '--environments', str(handoff / 'environments/qualification.json'),
            '--pair', read(pair / 'plan.json')['pair_claim'], '--comparison', str(pair / 'comparison.json'),
            '--out', str(reference_dir)]
        publish('capturing_reference', observation=str(reference_dir))
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='')
        with (out / 'observer.log').open('x') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = read(reference_dir / 'progress.json')
        reference = reference_dir / 'user-reference.json'
        if (result['state'] != 'user_reference_ready' or result['artifact_sha256'] != status['artifact_sha256']
                or sha256_file(reference) != result['reference_sha256']):
            raise ValueError('Public reference differs from the exact EMPF ZIP')
        publish('user_reference_ready', reference=str(reference), reference_sha256=sha256_file(reference),
            artifact_sha256=result['artifact_sha256'], best_score=result['best_score'])
    except Exception as exc:
        publish('error', error_type=type(exc).__name__)
        raise


if __name__ == '__main__':
    main()
