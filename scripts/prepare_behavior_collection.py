"""Declare a fresh raw TRAIN quartet under the frozen Jev study.

The existing collector also emits its legacy binary contexts. Those are an
unused auxiliary output here; only separately graded Jev contexts can be fit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request
from urllib.parse import urlsplit

from gemma_particle.behavior_protocol import load_behavior_protocol
from gemma_particle.budgets import OFFICIAL_BUDGET, budget_values
from gemma_particle.capture import CONTEXT_CAPTURE_PROTOCOL
from gemma_particle.curation import instruction_set
from gemma_particle.data import read_jsonl
from gemma_particle.data import digest
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.harness_identity import sha256_file, wheel_identity
from gemma_particle.history import OFFICIAL_COMPACTION, history_contract
from gemma_particle.paired_collection import PAIR, QUARTET, load_collection_pilot
from gemma_particle.prompt_profiles import DEFAULT_PROMPT_PROFILE,PROMPT_PROFILES,prompt_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--executor-commit', required=True)
    p.add_argument('--server-identity', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--prompt-profile', choices=PROMPT_PROFILES, default=DEFAULT_PROMPT_PROFILE)
    p.add_argument('--protocol', type=Path, default=Path('research/jev-evidence-loop-v1.json'))
    p.add_argument('--qualification', type=Path, default=Path('reports/train-environment-qualification-v2.json'))
    p.add_argument('--visible-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--collection-pilot', type=Path)
    p.add_argument('--collection-pilot-sha256')
    p.add_argument('--image', default='gemma-particle-sandbox:fastapi-deps-v2')
    args = p.parse_args()
    protocol = load_behavior_protocol(args.protocol)
    executor = args.executor.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=executor, text=True).strip()
    if commit != args.executor_commit:
        raise ValueError('frozen executor commit differs')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=executor, check=True)
    if sha256_file(executor / 'research/jev-evidence-loop-v1.json') != sha256_file(args.protocol):
        raise ValueError('prospective protocol differs from frozen executor')
    tasks = read_jsonl(args.visible_tasks)
    validate_agent_tasks(tasks)
    if any(t['split'] != 'train' for t in tasks) or args.task_id not in {t['instance_id'] for t in tasks}:
        raise ValueError('sanitized TRAIN task required')
    if args.task_id in protocol['cohort_policy']['excluded_fit_tasks']:
        raise ValueError('reserved conversion task cannot supply fitting data')
    arms = QUARTET
    if bool(args.collection_pilot) != bool(args.collection_pilot_sha256):
        raise ValueError('collection pilot requires its path and pinned hash')
    if args.collection_pilot:
        rows = load_collection_pilot(args.collection_pilot, args.collection_pilot_sha256,
            args.visible_tasks, args.qualification)
        task_arms = {r['instance_id']: tuple(r['planned_arms']) for r in rows}
        if args.task_id not in task_arms:
            raise ValueError('task is outside the declared TRAIN pilot')
        arms = task_arms[args.task_id]
    qualification = json.loads(args.qualification.read_text())
    row = next(r for r in qualification['controls'] if r['instance_id'] == args.task_id)
    receipt = Path(row['receipt']).resolve()
    wheels = Path(row['wheels']).resolve()
    control = json.loads(receipt.read_text())
    image = args.image
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    if (row['status'] != 'passed' or sha256_file(receipt) != row['receipt_sha256']
            or control['status'] != 'passed' or image_id != qualification['image_id']
            or control['image_id'] != image_id
            or sha256_file('data/kaggle-official/tasks.jsonl') != qualification['source_sha256']
            or control['tasks_sha256'] != qualification['source_sha256']
            or sha256_file(Path('data/kaggle-official/snapshots') / (args.task_id + '.tgz')) != control['snapshot_sha256']
            or sha256_file(wheels / 'receipt.json') != control['wheel_selection_sha256']
            or wheel_identity(wheels) != json.loads((wheels / 'receipt.json').read_text())['wheels']):
        raise ValueError('qualified TRAIN environment changed')
    server = json.loads(args.server_identity.read_text())
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1')
            or endpoint.path.rstrip('/') != '/v1' or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment):
        raise ValueError('explicit local OpenAI endpoint ending in /v1 required')
    with urllib.request.urlopen(args.endpoint.rstrip('/').removesuffix('/v1') + '/health', timeout=5) as response:
        health = json.load(response)
    if health.get('identity_sha256') != digest(server):
        raise ValueError('live server identity differs')
    config = json.loads((executor / 'configs/31b-ct-pilot.json').read_text())
    if (server.get('kind') != 'gemma4' or server.get('pretrained_inference') is not True
            or server.get('strength') != 0 or server.get('native_checkpoint') is not None
            or server['model_identity']['repo'] != config['model']
            or server['model_identity']['revision'] != config['model_revision']):
        raise ValueError('pinned frozen-base server required')
    sources = ['src/gemma_particle/curation.py', 'scripts/build_submission.py',
        'scripts/run_trajectories.py', 'scripts/collect_arms.py', 'src/gemma_particle/budgets.py',
        'src/gemma_particle/capture.py', 'src/gemma_particle/context_budget.py',
        'scripts/verify_trajectories.py', 'src/gemma_particle/history.py', 'src/gemma_particle/adk_capture.py']
    if args.prompt_profile != DEFAULT_PROMPT_PROFILE:
        sources.extend(['src/gemma_particle/prompt_profiles.py','submission/prompts/system.md'])
    if args.collection_pilot:
        sources.append('src/gemma_particle/paired_collection.py')
    plan = {'schema': 'gemma4-curation-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'task_ids': [args.task_id], 'arms': list(arms),
        'server_identity': server, 'target_instructions': instruction_set('tools-v2'),
        'endpoint': args.endpoint.rstrip('/'),
        'source_git': commit, 'source_checkout': str(executor),
        'execution_sources': {s: sha256_file(executor / s) for s in sources},
        'environment_control_sha256': row['receipt_sha256'],
        'teacher_policy': 'paired-verify' if arms == PAIR else 'all-four',
        'image':image, 'image_id':image_id, 'visible_tasks_sha256':sha256_file(args.visible_tasks),
        'instruction_protocol': 'tools-v2', 'capture_protocol': CONTEXT_CAPTURE_PROTOCOL,
        'history_profile': OFFICIAL_COMPACTION, 'history': history_contract(OFFICIAL_COMPACTION),
        'budget_profile': OFFICIAL_BUDGET, 'budget': budget_values(OFFICIAL_BUDGET),
        'behavior_study': protocol['name'], 'behavior_protocol_sha256': sha256_file(args.protocol),
        'preparer_sha256': sha256_file(Path(__file__)),
        'legacy_contexts': 'unused auxiliary output; not admitted Jev fitting data',
        'scope': 'fresh prospective TRAIN raw collection; no grades, fitting targets or benchmark score'}
    if args.collection_pilot:
        plan['collection_pilot'] = {'path': str(args.collection_pilot.resolve()), 'sha256': args.collection_pilot_sha256}
    if args.prompt_profile != DEFAULT_PROMPT_PROFILE:
        plan['prompt_profile'] = args.prompt_profile
        plan['common_prompt'] = prompt_identity((executor/'submission/prompts/system.md').read_text(),args.prompt_profile)
    args.root.mkdir(parents=True, exist_ok=False, mode=0o700)
    (args.root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (args.root / 'behavior-protocol.json').write_bytes(args.protocol.read_bytes())
    print(json.dumps({'plan': str(args.root / 'plan.json'), 'wheels': str(wheels), 'image': image,
        'protocol_sha256': plan['behavior_protocol_sha256'], 'task_id': args.task_id}))


if __name__ == '__main__':
    main()
