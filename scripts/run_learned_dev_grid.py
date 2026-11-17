"""Wait for actual conversions and controls, run all18 learned DEV conditions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

from gemma_particle.conversion_handoff import read, validate_training_outputs
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import load_packets, load_control_results, candidate_preflight, process_binding, live
from gemma_particle.dev_evidence import canonical_contract
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import sha256_file


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['conversion', 'controls', 'executor', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--conversion-pid', type=int)
    parser.add_argument('--controls-pid', type=int)
    parser.add_argument('--executor-commit', required=True)
    parser.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    parser.add_argument('--probe-contexts', type=Path, default=Path('data/conversion-probe-v1/contexts.json'))
    parser.add_argument('--probe-qualification', type=Path, default=Path('reports/conversion-probe-preparation-v1.json'))
    parser.add_argument('--port', type=int, default=8031)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    controller = Path(__file__).resolve().parents[1]
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.out.exists():
        raise FileExistsError('learned DEV runs are immutable; reconcile partial evidence')
    if not 1024 <= args.port <= 65535:
        raise ValueError('explicit unprivileged loopback port required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.executor, text=True).strip() != args.executor_commit:
        raise ValueError('exact baseline DEV executor required')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=args.executor, check=True)
    selection = load_selection_plan(args.selection_plan)
    plan_sha = sha256_file(args.selection_plan)
    conversion_plan = read(args.conversion / 'plan.json')
    controls_plan = read(args.controls / 'plan.json')
    training = Path(conversion_plan['training'])
    training_plan_path = training / 'plan.json'
    pilot = Path(read(training_plan_path)['pilot'])
    config = args.executor / 'configs/31b-ct-shared-positional-pilot.json'
    baseline_path = Path(controls_plan['server_identity'])
    baseline = read(baseline_path)
    if (conversion_plan['selection_plan_sha256'] != plan_sha
            or controls_plan['selection_plan_sha256'] != plan_sha
            or baseline['device'] != 'cuda:0' or baseline['memory_fraction'] != .6
            or baseline.get('base_only') is not True or baseline.get('adapters')
            or controls_plan['source_sha256'] != sha256_file(args.executor / 'scripts/run_dev_controls.py')
            or conversion_plan['training_plan_sha256'] != sha256_file(training_plan_path)
            or conversion_plan['config_sha256'] != sha256_file(config)
            or conversion_plan['probe_sha256'] != sha256_file(args.probe_contexts)
            or conversion_plan['probe_qualification_sha256'] != sha256_file(args.probe_qualification)):
        raise ValueError('same fixed comparison, baseline executor and conversion probes required')
    baseline_contract = canonical_contract({'model_identity': baseline})
    producers = []
    for directory, pid, script, completed in [
        (args.conversion, args.conversion_pid, 'prepare_learned_sweep.py', 'prepared_complete_learned_grid'),
        (args.controls, args.controls_pid, 'run_dev_controls.py', 'complete')]:
        path = directory / 'progress.json'
        done = path.exists() and read(path).get('state') == completed
        binding = None
        if not args.plan_only and not done:
            if not pid:
                raise ValueError('live producer PID required while waiting: ' + script)
            binding = process_binding(pid, script, directory)
        producers.append({'directory': str(directory), 'completed_state': completed, 'binding': binding})
    executor_files = subprocess.check_output(['git', 'ls-files', 'scripts', 'src', 'configs', 'submission', 'sources.lock.json'],
        cwd=args.executor, text=True).splitlines()
    sources = {name: sha256_file(args.executor / name) for name in executor_files}
    controller_names = ['scripts/run_learned_dev_grid.py', 'src/gemma_particle/dev_dispatch.py',
        'src/gemma_particle/conversion_handoff.py', 'src/gemma_particle/dev_evidence.py',
        'src/gemma_particle/dev_selection.py', 'src/gemma_particle/data.py',
        'src/gemma_particle/harness_identity.py', 'src/gemma_particle/adapters.py', 'src/gemma_particle/lora_serving.py']
    own_sources = {name: sha256_file(controller / name) for name in controller_names}
    pins = {str(p): sha256_file(p) for p in [args.selection_plan, args.probe_contexts,
        args.probe_qualification, args.conversion / 'plan.json', args.controls / 'plan.json', baseline_path,
        training_plan_path]}
    args.out.mkdir(parents=True)
    (args.out / 'logs').mkdir()
    declaration = {'schema': 'gemma4-learned-dev-grid-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'executor': str(args.executor), 'executor_commit': args.executor_commit, 'producers': producers,
        'selection_plan_sha256': plan_sha, 'device': 'cuda:0', 'memory_fraction': .6, 'startup_headroom_GiB': 8,
        'port': args.port, 'condition_ids': [c['id'] for c in selection['conditions'] if c['role'] == 'learned'],
        'tasks_per_condition': 55, 'sources': sources, 'controller_sources': own_sources, 'declaration_pins': pins,
        'scope': 'complete learned DEV grid and offline choice only; no holdout or official submission'}
    write(args.out / 'plan.json', declaration)
    status = {'state': 'prepared' if args.plan_only else 'waiting_actual_inputs', 'finished': [],
        'planned_conditions': 18, 'model_dev_inference': False, 'holdout_accessed': False}

    def publish(state, **values):
        status.update(state=state, as_of=datetime.now(timezone.utc).isoformat(), **values)
        write(args.out / 'progress.json', status)
        print(json.dumps(status), flush=True)

    publish(status['state'])
    if args.plan_only:
        return

    def unchanged():
        if (any(sha256_file(p) != value for p, value in pins.items())
                or {n: sha256_file(args.executor / n) for n in sources} != sources
                or {n: sha256_file(controller / n) for n in own_sources} != own_sources):
            raise ValueError('frozen DEV sources or producer declarations changed')

    server = None
    try:
        while True:
            pending = []
            for producer in producers:
                path = Path(producer['directory']) / 'progress.json'
                current = read(path) if path.exists() else {}
                if current.get('state') == 'error':
                    raise RuntimeError('actual input producer failed: ' + producer['directory'])
                binding = producer['binding']
                if current.get('state') != producer['completed_state']:
                    if binding is None or not live(binding):
                        raise RuntimeError('producer exited without full completion: ' + producer['directory'])
                    pending.append(producer['directory'])
                elif binding is not None and live(binding):
                    pending.append(producer['directory'] + ':awaiting_process_release')
            if not pending:
                break
            if status.get('waiting_for') != pending:
                publish('waiting_actual_inputs', waiting_for=pending)
            time.sleep(10)
        prior_server = {'pid': controls_plan['server_pid'], 'birth_ticks': controls_plan['server_birth_ticks']}
        publish('waiting_baseline_server_release')
        while live(prior_server):
            time.sleep(1)
        unchanged()
        conversion_declaration = read(args.conversion / 'plan.json')
        jobs = validate_training_outputs(training, pilot / 'cache', pilot, config,
            selection, args.probe_contexts, args.probe_qualification)
        packets = load_packets(args.conversion, selection, plan_sha, jobs, conversion_declaration['probe_sha256'])
        control_results = load_control_results(args.controls, selection, plan_sha)
        write(args.out / 'validated-inputs.json', {'candidate_packets_sha256': sha256_file(args.conversion / 'candidate-packets.json'),
            'control_results': {str(p): sha256_file(p) for p in control_results}})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.executor / 'src'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
        root = f'http://127.0.0.1:{args.port}'
        finished = []
        for packet in packets:
            condition = packet['condition']; name = condition['id']
            publish('waiting_shared_gpu0_headroom', current_condition=name)
            while True:
                output = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.total',
                    '--format=csv,noheader,nounits'], text=True)
                free, total = (int(v.strip()) * 1024**2 for v in output.strip().split(','))
                if free >= total * .6 + 8 * 1024**3:
                    break
                time.sleep(10)
            unchanged()
            load_packets(args.conversion, selection, plan_sha, jobs, conversion_declaration['probe_sha256'])
            with socket.socket() as check:
                check.bind(('127.0.0.1', args.port))
            serving = args.out / (name + '-server')
            command = [str(args.executor / '.vllm-venv/bin/python'), 'scripts/serve_vllm_lora.py',
                '--adapter', packet['adapter'], '--conversion-report', packet['conversion_report'],
                '--device', 'cuda:0', '--memory-fraction', '.60', '--port', str(args.port), '--out', str(serving)]
            publish('starting_candidate_server', current_condition=name)
            with (args.out / 'logs' / (name + '-server.log')).open('x') as stream:
                server = subprocess.Popen(command, cwd=args.executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
            identity_path = serving / 'server-identity.json'
            while not identity_path.exists():
                if server.poll() is not None:
                    raise RuntimeError(f'candidate server exited {server.returncode}; retain actual startup log')
                time.sleep(1)
            identity = read(identity_path)
            if (identity.get('base_only') is not False or identity['adapters'] != packet['adapters']
                    or canonical_contract({'model_identity': identity}) != baseline_contract):
                raise ValueError('candidate server differs from exact matched baseline contract')
            attempts = args.out / (name + '-preflight-http.jsonl')

            def call(client, base, path, payload=None):
                with attempts.open('a') as stream:
                    stream.write(json.dumps({'phase': 'attempt', 'path': path, 'request': payload}) + '\n')
                response = client.get(base + path) if payload is None else client.post(base + path, json=payload)
                value = response.json()
                with attempts.open('a') as stream:
                    stream.write(json.dumps({'phase': 'response', 'path': path,
                        'status_code': response.status_code, 'response': value}) + '\n')
                response.raise_for_status()
                return value

            with httpx.Client(timeout=600, headers={'X-Gemma-Identity': digest(identity)}) as client:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'candidate server exited {server.returncode}; retain actual startup log')
                    try:
                        response = client.get(root + '/health', timeout=5)
                        if response.status_code == 200:
                            break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    time.sleep(5)
                preflight = candidate_preflight(client, root, identity, call)
            preflight.update(server_identity_sha256=digest(identity), http_journal_sha256=sha256_file(attempts))
            write(args.out / (name + '-preflight.json'), preflight)
            publish('running_full_dev_condition', model_dev_inference=True, current_condition=name)
            destination = args.out / name
            command = [str(args.executor / '.harness-venv/bin/python'), 'scripts/run_dev_condition.py',
                '--selection-plan', str(args.selection_plan), '--condition', name, '--endpoint', root + '/v1',
                '--server-identity', str(identity_path), '--out', str(destination)]
            for option in ['adapter', 'conversion_report', 'native', 'cache']:
                command.extend(['--' + option.replace('_', '-'), packet[option]])
            with (args.out / 'logs' / (name + '-dev.log')).open('x') as stream:
                result = subprocess.run(command, cwd=args.executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'actual DEV condition exited {result.returncode}; preserve every completed task')
            result_path = destination / 'result.json'; receipt = read(result_path)
            if (receipt['status'] != 'complete' or receipt['condition'] != condition
                    or receipt['primary']['total'] != 42 or receipt['secondary']['total'] != 13):
                raise ValueError('actual complete fixed DEV panel required')
            server.send_signal(signal.SIGINT)
            publish('waiting_candidate_server_release', current_condition=name)
            server.wait(); server = None
            finished.append({'condition_id': name, 'result': str(result_path), 'result_sha256': sha256_file(result_path),
                'preflight_sha256': sha256_file(args.out / (name + '-preflight.json'))})
            publish('condition_complete', finished=finished)
        unchanged()
        load_control_results(args.controls, selection, plan_sha)
        command = [str(args.executor / '.harness-venv/bin/python'), 'scripts/compare_dev_conditions.py',
            '--selection-plan', str(args.selection_plan), '--out', str(args.out / 'choice')]
        for result_path in [*control_results, *(Path(r['result']) for r in finished)]:
            command.extend(['--condition-result', str(result_path)])
        with (args.out / 'logs' / 'compare.log').open('x') as stream:
            result = subprocess.run(command, cwd=args.executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('full-grid offline audit failed; no choice inferred')
        choice = args.out / 'choice/selection.json'
        publish('complete_dev_choice', finished=finished, selection_sha256=sha256_file(choice),
            selected=read(choice)['selected'])
    except Exception as exc:
        publish('error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            server.wait()


if __name__ == '__main__':
    main()
