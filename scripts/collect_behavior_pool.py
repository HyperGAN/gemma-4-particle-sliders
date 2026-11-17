"""Collect fresh fixed-order TRAIN quartets until two initial targets qualify.

Raw collection and Jev curation run from independently frozen checkouts. An
already-running first quartet can be adopted by exact process/root identity;
it is never restarted. Every failed arm/judgment remains in its original run.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

from collect_train_pool import select_tasks, stop_owned_server, write_json
from gemma_particle.behavior_protocol import load_behavior_protocol
from gemma_particle.data import read_jsonl, write_jsonl
from gemma_particle.harness_identity import sha256_file
from gemma_particle.prompt_profiles import DEFAULT_PROMPT_PROFILE,PROMPT_PROFILES


def checked_checkout(path, expected):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    if actual != expected:
        raise ValueError('frozen executor commit changed')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=path, check=True)


def collector_identity(pid, cohort):
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return None
    args = [os.fsdecode(s) for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
    if (not any(s.endswith('/collect_arms.py') for s in args) or '--root' not in args
            or Path(args[args.index('--root')+1]).resolve() != cohort):
        raise ValueError('adopted PID is not this exact raw collection')
    return (proc/'stat').read_text().split(') ', 1)[1].split()[19]


def wait_existing(cohort, pid, birth):
    while True:
        progress = cohort/'progress.json'
        if progress.exists() and json.loads(progress.read_text()).get('state') == 'complete':
            return
        if collector_identity(pid, cohort) != birth:
            raise RuntimeError('adopted collection exited without a complete quartet; preserve and reconcile')
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--executor-commit', required=True)
    p.add_argument('--curator', type=Path, required=True)
    p.add_argument('--curator-commit', required=True)
    p.add_argument('--server-identity', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--prompt-profile',choices=PROMPT_PROFILES,default=DEFAULT_PROMPT_PROFILE)
    p.add_argument('--key-file', type=Path)
    p.add_argument('--first-cohort', type=Path)
    p.add_argument('--first-collector-pid', type=int)
    p.add_argument('--stop-server-pid', type=int)
    p.add_argument('--protocol', type=Path, default=Path('research/jev-evidence-loop-v1.json'))
    p.add_argument('--qualification', type=Path, default=Path('reports/train-environment-qualification-v2.json'))
    p.add_argument('--visible-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
    args = p.parse_args()
    for name in ('root', 'executor', 'curator', 'server_identity', 'protocol', 'qualification', 'visible_tasks'):
        setattr(args, name, getattr(args, name).resolve())
    if args.first_cohort:
        args.first_cohort = args.first_cohort.resolve()
    if bool(args.first_cohort) != bool(args.first_collector_pid):
        raise ValueError('adoption requires both the existing cohort and its exact PID')
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('localhost','127.0.0.1','::1')
            or endpoint.path.rstrip('/') != '/v1' or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment):
        raise ValueError('explicit local OpenAI endpoint ending in /v1 required')
    checked_checkout(args.executor, args.executor_commit)
    checked_checkout(args.curator, args.curator_commit)
    if Path(__file__).resolve().parent.parent != args.curator:
        raise ValueError('pool supervisor must run from the frozen curator checkout')
    protocol = load_behavior_protocol(args.protocol)
    if sha256_file(args.curator/'research/jev-evidence-loop-v1.json') != sha256_file(args.protocol):
        raise ValueError('curator protocol differs')
    qualification = json.loads(args.qualification.read_text())
    selected = select_tasks(read_jsonl(args.visible_tasks), qualification,
        protocol['cohort_policy']['excluded_fit_tasks'])
    raw_tasks = args.visible_tasks.parent.parent/'kaggle-official/tasks.jsonl'
    if sha256_file(raw_tasks) != qualification['source_sha256']:
        raise ValueError('qualified TRAIN source changed')
    birth = None
    server = json.loads(args.server_identity.read_text())
    if args.first_cohort:
        first = json.loads((args.first_cohort/'plan.json').read_text())
        if (first['task_ids'] != [selected[0]['instance_id']] or first['server_identity'] != server
                or first['source_git'] != args.executor_commit
                or first.get('behavior_protocol_sha256') != sha256_file(args.protocol)
                or first.get('prompt_profile',DEFAULT_PROMPT_PROFILE) != args.prompt_profile
                or first.get('endpoint') != args.endpoint.rstrip('/')):
            raise ValueError('existing first quartet differs from this fixed-order study')
        birth = collector_identity(args.first_collector_pid, args.first_cohort)
        progress = args.first_cohort/'progress.json'
        complete = progress.exists() and json.loads(progress.read_text()).get('state') == 'complete'
        if birth is None and not complete:
            raise RuntimeError('first collector is gone without a complete quartet')
    if args.root.exists():
        raise FileExistsError('pool run is immutable; do not restart incomplete collection')
    args.root.mkdir(parents=True, mode=0o700); (args.root/'logs').mkdir()
    plan = {'schema':'gemma4-behavior-pool-plan-v1', 'created_at':datetime.now(timezone.utc).isoformat(),
        'behavior_study':protocol['name'], 'protocol_sha256':sha256_file(args.protocol),
        'task_ids':[r['instance_id'] for r in selected],
        'excluded_tasks':protocol['cohort_policy']['excluded_fit_tasks'],
        'qualification_sha256':sha256_file(args.qualification), 'visible_tasks_sha256':sha256_file(args.visible_tasks),
        'raw_tasks_sha256':qualification['source_sha256'], 'server_identity':server,
        'endpoint':args.endpoint.rstrip('/'), 'executor':str(args.executor), 'executor_commit':args.executor_commit,
        'prompt_profile':args.prompt_profile,
        'curator':str(args.curator), 'curator_commit':args.curator_commit,
        'supervisor_sha256':sha256_file(Path(__file__)),
        'first_existing_cohort':str(args.first_cohort) if args.first_cohort else None,
        'first_collector_pid':args.first_collector_pid, 'first_collector_birth_ticks':birth,
        'timing_disclosure':'pool order frozen after first raw quartet started, before any real trajectory grade',
        'stop_rule':'first completed quartet bringing admitted initial contexts to at least two distinct TRAIN tasks',
        'scope':'adaptive TRAIN collection only; preserve all failures; no DEV/holdout or official score'}
    write_json(args.root/'plan.json', plan)
    env = dict(os.environ, PYTHONPATH=str(args.curator/'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    python = args.curator/'.harness-venv/bin/python'
    def execute(checkout, interpreter, command, log):
        child_env = dict(env, PYTHONPATH=str(checkout/'src'))
        with log.open('x') as stream:
            r = subprocess.run([str(interpreter), *command], cwd=checkout, env=child_env,
                stdout=stream, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'pipeline exit {r.returncode}; preserve and reconcile {log}')
    contexts, finished = [], []
    for index, record in enumerate(selected):
        checked_checkout(args.executor, args.executor_commit)
        checked_checkout(args.curator, args.curator_commit)
        task_id = record['instance_id']
        cohort = args.first_cohort if index == 0 and args.first_cohort else args.root/task_id
        if index == 0 and args.first_cohort:
            wait_existing(cohort, args.first_collector_pid, birth)
        else:
            execute(args.curator, python, ['scripts/prepare_behavior_collection.py', '--root', str(cohort),
                '--executor', str(args.executor), '--executor-commit', args.executor_commit,
                '--server-identity', str(args.server_identity), '--endpoint', args.endpoint,
                '--task-id', task_id, '--protocol', str(args.protocol), '--qualification', str(args.qualification),
                '--prompt-profile',args.prompt_profile],
                args.root/'logs'/f'{task_id}-prepare.log')
            inputs = raw_tasks.parent
            execute(args.executor, args.executor/'.harness-venv/bin/python', ['scripts/collect_arms.py',
                '--root', str(cohort), '--visible-tasks', str(args.visible_tasks), '--raw-tasks', str(raw_tasks),
                '--inputs', str(inputs), '--wheels', str(Path(record['wheels']).resolve()),
                '--server-identity', str(args.server_identity), '--endpoint', args.endpoint],
                args.root/'logs'/f'{task_id}-collect.log')
        grades = args.root/(task_id+'-grades')
        command = ['scripts/grade_behavior_cohort.py', '--cohort', str(cohort), '--visible-tasks', str(args.visible_tasks),
            '--protocol', str(args.protocol), '--out', str(grades)]
        if args.key_file:
            command.extend(['--key-file', str(args.key_file.expanduser().resolve())])
        execute(args.curator, python, command, args.root/'logs'/f'{task_id}-grade.log')
        curated = args.root/(task_id+'-curated')
        execute(args.curator, python, ['scripts/curate_behavior_cohort.py', '--cohort', str(cohort),
            '--grades', str(grades), '--visible-tasks', str(args.visible_tasks),
            '--protocol', str(args.protocol), '--out', str(curated)], args.root/'logs'/f'{task_id}-curate.log')
        if sha256_file(grades/'jev-contexts.jsonl') != sha256_file(curated/'jev-contexts.jsonl'):
            raise ValueError('offline curation differs from the original one-pass selection')
        admitted = read_jsonl(curated/'jev-contexts.jsonl')
        if len(admitted)>1 or any(r['instance_id']!=task_id or r['decision']!=0 for r in admitted):
            raise ValueError('pool requires one genuine initial target per distinct TRAIN task')
        contexts.extend(admitted)
        finished.append({'instance_id':task_id, 'cohort':str(cohort), 'grades':str(grades),
            'grade_result_sha256':sha256_file(grades/'result.json'),
            'contexts_sha256':sha256_file(curated/'jev-contexts.jsonl'), 'admitted_contexts':len(admitted)})
        write_json(args.root/'progress.json', {'state':'collecting', 'finished':finished,
            'planned':len(selected), 'admitted_contexts':len(contexts)})
        print(json.dumps(finished[-1]), flush=True)
        if len(contexts)>=2:
            if len({r['instance_id'] for r in contexts}) != len(contexts):
                raise ValueError('distinct initial target tasks required')
            context_path = args.root/'jev-contexts.jsonl'; write_jsonl(context_path, contexts)
            transport = args.root/'prompt-transport.json'
            execute(args.curator, args.curator/'.train-venv/bin/python', ['scripts/tokenize_contexts.py',
                '--contexts', str(context_path), '--server-identity', str(args.server_identity),
                '--endpoint', args.endpoint, '--out', str(transport)], args.root/'logs'/'tokenize.log')
            result = {'state':'target_ready', 'finished':finished, 'admitted_contexts':len(contexts),
                'contexts':str(context_path), 'prompt_transport':str(transport),
                'prompt_transport_sha256':sha256_file(transport), 'scope':plan['scope']}
            if args.stop_server_pid:
                result['server_stop'] = stop_owned_server(args.stop_server_pid, args.server_identity)
            write_json(args.root/'progress.json', result); print(json.dumps(result), flush=True)
            return
    write_json(args.root/'progress.json', {'state':'exhausted_without_targets', 'finished':finished,
        'admitted_contexts':len(contexts), 'scope':plan['scope']})


if __name__ == '__main__':
    main()
