"""Run the original full learned DEV grid after actual two-slot qualification.

Preparation uses CPU metadata only. Actual dispatch requires the prior waiter
already stopped before any learned task, complete released input producers,
the fixed same-adapter numerical control, and an audited resource supplement.
"""
import argparse
import ast
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
from gemma_particle.dev_selection import baseline_selection_sha256, load_selection_plan
from gemma_particle.harness_identity import sha256_file
from gemma_particle.learned_effect import prepare_effect_probes, learned_effect_preflight
from gemma_particle.resource_dispatch import pending_producers, stopped_prior_grid, observe_slot, slot_record, stop_owned, wait_wave,task_audit_bridge
from gemma_particle.resource_supplement import (validate_resource_declaration, prepare_resource_supplement,
    load_resource_supplement, normalize_qualified_contract, audit_condition_slot)


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def frozen_sources(root, commit):
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != commit:
        raise ValueError('exact frozen executor commit required')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    names = subprocess.check_output(['git', 'ls-files', 'scripts', 'src', 'configs', 'submission', 'sources.lock.json'],
        cwd=root, text=True).splitlines()
    return {str(root / name): sha256_file(root / name) for name in names}


def scoring_source(path):
    tree = ast.parse(path.read_text())
    names = {'wilson_interval', 'summarize_verified_panel', '_summarize_rows', 'paired_comparison', 'choose_learned_candidate'}
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}


def input_producer(directory, expected, pid, script):
    progress = directory / 'progress.json'
    complete = progress.exists() and read(progress).get('state') == expected
    if not complete and not pid:
        raise ValueError(f'exact live {script} producer required')
    binding = process_binding(pid, script, directory) if pid else None
    return {'directory': str(directory), 'completed_state': expected, 'binding': binding}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prior-grid', 'conversion', 'controls', 'executor', 'qualifier', 'auditor', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['executor', 'qualifier', 'auditor']:
        p.add_argument('--' + name + '-commit', required=True)
    p.add_argument('--prior-pid', type=int, required=True)
    p.add_argument('--prior-birth-ticks', required=True)
    p.add_argument('--conversion-pid', type=int)
    p.add_argument('--controls-pid', type=int)
    p.add_argument('--task-executor',type=Path,help='separate frozen checkout with only DEV evidence-reader corrections')
    p.add_argument('--task-executor-commit')
    p.add_argument('--declaration', type=Path, default=Path('research/dev-resource-declaration-v1.json'))
    p.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    p.add_argument('--proposal', type=Path, default=Path('research/dev-hardware-resource-proposal-v1.json'))
    p.add_argument('--probe-inputs', type=Path, default=Path('research/dev-hardware-probe-inputs-v1.json'))
    p.add_argument('--probe-contexts', type=Path, default=Path('data/conversion-probe-v1/contexts.json'))
    p.add_argument('--probe-qualification', type=Path, default=Path('reports/conversion-probe-preparation-v1.json'))
    p.add_argument('--effect-protocol', type=Path, default=Path('research/learned-effect-protocol-v1.json'))
    p.add_argument('--left-port', type=int, default=8032)
    p.add_argument('--right-port', type=int, default=8033)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if bool(args.task_executor)!=bool(args.task_executor_commit):
        raise ValueError('task audit executor requires its exact commit')
    args.task_executor=args.task_executor or args.executor
    args.task_executor_commit=args.task_executor_commit or args.executor_commit
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    root = Path(__file__).resolve().parents[1]
    if args.out.exists(): raise FileExistsError('resource DEV runs are immutable; reconcile partial evidence')
    ports = [args.left_port, args.right_port]
    if len(set(ports)) != 2 or any(not 1024 <= port <= 65535 for port in ports):
        raise ValueError('two distinct explicit unprivileged loopback ports required')
    selection = load_selection_plan(args.selection_plan); plan_sha = sha256_file(args.selection_plan)
    scope = selection.get('selection_scope')
    original_sha = baseline_selection_sha256(selection, plan_sha)
    execution_plan = Path(scope['original_plan']).resolve() if scope else args.selection_plan
    learned_ids = [c['id'] for c in selection['conditions'] if c['role'] == 'learned']
    resource_declaration = validate_resource_declaration(args.declaration, args.selection_plan)
    effect_protocol, effect_cases = prepare_effect_probes(args.effect_protocol, args.probe_contexts,
        args.probe_qualification, selection, plan_sha)
    prior = read(args.prior_grid / 'plan.json'); prior_binding = {'pid': args.prior_pid, 'birth_ticks': args.prior_birth_ticks}
    controls_plan = read(args.controls / 'plan.json'); conversion_plan = read(args.conversion / 'plan.json')
    training = Path(conversion_plan['training']); training_plan_path = training / 'plan.json'
    pilot = Path(read(training_plan_path)['pilot']); config = args.executor / 'configs/31b-ct-shared-positional-pilot.json'
    baseline_path = Path(controls_plan['server_identity']); baseline = read(baseline_path)
    if scope:
        original_ids = [c['id'] for c in load_selection_plan(execution_plan)['conditions'] if c['role'] == 'learned']
        producers = [input_producer(args.conversion, 'prepared_complete_learned_grid', args.conversion_pid, 'prepare_learned_sweep.py'),
                     input_producer(args.controls, 'complete', args.controls_pid, 'run_dev_controls.py')]
    else:
        original_ids, producers = learned_ids, prior['producers']
    if (prior['selection_plan_sha256'] != original_sha or controls_plan['selection_plan_sha256'] != original_sha
            or conversion_plan['selection_plan_sha256'] != plan_sha
            or Path(prior['executor']).resolve() != args.executor or prior['executor_commit'] != args.executor_commit
            or args.executor_commit != resource_declaration['serving_executor_commit']
            or prior['condition_ids'] != original_ids
            or prior['tasks_per_condition'] != 55
            or prior['learned_effect_protocol_sha256'] != sha256_file(args.effect_protocol)
            or [item['directory'] for item in producers] != [str(args.conversion), str(args.controls)]
            or [item['completed_state'] for item in producers] != ['prepared_complete_learned_grid', 'complete']
            or baseline['device'] != 'cuda:0' or baseline['memory_fraction'] != .6
            or baseline.get('base_only') is not True or baseline.get('adapters')
            or controls_plan['source_sha256'] != sha256_file(args.executor / 'scripts/run_dev_controls.py')
            or conversion_plan['training_plan_sha256'] != sha256_file(training_plan_path)
            or conversion_plan['config_sha256'] != sha256_file(config)
            or conversion_plan['probe_sha256'] != sha256_file(args.probe_contexts)
            or conversion_plan['probe_qualification_sha256'] != sha256_file(args.probe_qualification)):
        raise ValueError('same complete core grid,original baseline,native inputs and numerical effect protocol required')
    pins = {}
    source_sets={}
    for executor, commit in [(args.executor, args.executor_commit),(args.task_executor,args.task_executor_commit), (args.qualifier, args.qualifier_commit),
            (args.auditor, args.auditor_commit)]:
        source_sets[executor]=frozen_sources(executor,commit);pins.update(source_sets[executor])
    audit_bridge=task_audit_bridge(args.executor,args.task_executor,source_sets[args.executor],source_sets[args.task_executor])
    task_audit = 'src/gemma_particle/dev_evidence.py'
    task_selection = 'src/gemma_particle/dev_selection.py'
    if (sha256_file(args.task_executor / task_audit) != sha256_file(args.auditor / task_audit)
            or (scope and (len(scoring_source(args.task_executor / task_selection)) != 5
                           or scoring_source(args.task_executor / task_selection) != scoring_source(args.auditor / task_selection)))
            or (not scope and sha256_file(args.task_executor / task_selection) != sha256_file(args.auditor / task_selection))):
        raise ValueError('original generation task audit,canonicalizer,statistics and selection required')
    for name, value in resource_declaration['qualification_sources_sha256'].items():
        if sha256_file(args.qualifier / name) != value: raise ValueError('original declared hardware qualifier required')
    for name, value in resource_declaration['serving_sources_sha256'].items():
        if sha256_file(args.executor / name) != value: raise ValueError('original declared serving sources required')
    if (sha256_file(args.proposal) != resource_declaration['qualification_proposal_sha256']
            or sha256_file(args.probe_inputs) != resource_declaration['qualification_probe_inputs_sha256']):
        raise ValueError('original numerical qualification declaration required')
    own_names = subprocess.check_output(['git', 'ls-files', 'scripts', 'src'], cwd=root, text=True).splitlines()
    own_names = sorted(set(own_names) | {'scripts/run_resource_dev_grid.py', 'src/gemma_particle/resource_dispatch.py'})
    sources = {str(root / name): sha256_file(root / name) for name in own_names}
    for path in [args.selection_plan, execution_plan, args.declaration, args.proposal, args.probe_inputs, args.probe_contexts,
            args.probe_qualification, args.effect_protocol, args.prior_grid / 'plan.json', args.prior_grid / 'progress.json',
            args.conversion / 'plan.json', args.controls / 'plan.json', training_plan_path, baseline_path]:
        pins[str(path)] = sha256_file(path)
    if not args.plan_only:
        if os.getpriority(os.PRIO_PROCESS, 0) < 10: raise ValueError('run the controller at nice10 or greater')
        stopped_prior_grid(args.prior_grid, prior_binding)
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    args.out.mkdir(parents=True); (args.out / 'logs').mkdir()
    producer_plan = None
    if scope:
        producer_plan = args.out / 'producer-plan.json'
        write(producer_plan, {'schema': 'gemma4-dev-qualification-inputs-v1', 'selection_plan_sha256': plan_sha,
                              'producers': producers})
        pins[str(producer_plan)] = sha256_file(producer_plan)
    declaration = {'schema': 'gemma4-learned-dev-grid-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'executor': str(args.executor), 'executor_commit': args.executor_commit, 'producers': producers,
        'selection_plan_sha256': plan_sha, 'condition_ids': learned_ids, 'tasks_per_condition': 55,
        'memory_fraction': .6, 'startup_headroom_GiB': 8, 'device_policy': 'explicit qualified physical slots0/1',
        'ports': ports, 'declaration_pins': pins, 'controller_sources': sources,
        'prior_grid': str(args.prior_grid), 'stopped_prior_dispatcher': prior_binding,
        'prospective_resource_declaration': {'path': str(args.declaration), 'sha256': sha256_file(args.declaration)},
        'qualifier': str(args.qualifier), 'qualifier_commit': args.qualifier_commit,
        'offline_auditor': str(args.auditor), 'offline_auditor_commit': args.auditor_commit,
        'task_evidence_executor':str(args.task_executor),'task_evidence_executor_commit':args.task_executor_commit,
        'task_audit_bridge':audit_bridge,
        'learned_effect_protocol_sha256': sha256_file(args.effect_protocol),
        'learned_effect_probe_task_ids': effect_protocol['probe_task_ids'],
        'resource_supplement': None, 'activation': 'only after actual numerical qualification and offline audit',
        'scope': 'complete declared learned DEV grid and controlled choice only'}
    if scope:
        declaration['execution_selection_plan'] = {'path': str(execution_plan), 'sha256': original_sha}
    write(args.out / 'plan.json', declaration)
    status = {'state': 'prepared' if args.plan_only else 'waiting_actual_inputs', 'finished': [],
        'planned_conditions': len(learned_ids), 'model_dev_inference': False, 'holdout_accessed': False, 'resource_activated': False}

    def publish(state, **values):
        status.update(state=state, as_of=datetime.now(timezone.utc).isoformat(), **values)
        write(args.out / 'progress.json', status); print(json.dumps(status), flush=True)

    publish(status['state'])
    if args.plan_only: return

    def unchanged():
        if any(sha256_file(path) != value for path, value in {**pins, **sources}.items()):
            raise ValueError('frozen dispatch source or input declaration changed')
        stopped_prior_grid(args.prior_grid, prior_binding)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(args.executor / 'src'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    servers = []; children = []

    def stopped(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)

    def wait_headroom(index):
        publish('waiting_shared_headroom', physical_index=index)
        while True:
            unchanged()
            raw = subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=memory.free,memory.total',
                '--format=csv,noheader,nounits'], text=True)
            free, total = [int(v.strip()) * 1024**2 for v in raw.strip().split(',')]
            if free >= total * .6 + 8 * 1024**3: return
            time.sleep(10)

    def start_server(packet, index, name):
        wait_headroom(index)
        with socket.socket() as check: check.bind(('127.0.0.1', ports[index]))
        serving = args.out / (name + '-server'); identity_path = serving / 'server-identity.json'
        command = [str(args.executor / '.vllm-venv/bin/python'), 'scripts/serve_vllm_lora.py',
            '--adapter', packet['adapter'], '--conversion-report', packet['conversion_report'],
            '--device', f'cuda:{index}', '--memory-fraction', '.60', '--port', str(ports[index]), '--out', str(serving)]
        publish('starting_owned_server', current_condition=name, physical_index=index)
        with (args.out / 'logs' / (name + '-server.log')).open('x') as stream:
            process = subprocess.Popen(command, cwd=args.executor, env=env, stdout=stream, stderr=subprocess.STDOUT)
        # Register ownership immediately, including a startup that fails before identity publication.
        binding = {'pid': process.pid, 'birth_ticks': Path('/proc', str(process.pid), 'stat').read_text().split(') ', 1)[1].split()[19]}
        owner = {'process': process, 'binding': binding, 'output': serving, 'identity_path': identity_path,
            'root': f'http://127.0.0.1:{ports[index]}', 'index': index,
            'command': {'cwd': str(args.executor), 'argv': command, 'identity_path': str(identity_path)}}
        servers.append(owner)
        write(args.out / (name + '-owner.json'), {k: v for k, v in owner.items() if k not in ['process', 'output', 'identity_path']} |
            {'output': str(serving), 'identity_path': str(identity_path)})
        while True:
            if process.poll() is not None: raise RuntimeError(f'owned server exited {process.returncode};preserve startup')
            if identity_path.exists():
                try:
                    identity = read(identity_path)
                    break
                except json.JSONDecodeError:
                    # The frozen launcher publishes with write_text. A partial
                    # read observes that same startup; it is not a new attempt.
                    pass
            time.sleep(1)
        if process_binding(process.pid, 'serve_vllm_lora.py', serving) != binding:
            raise ValueError('owned server binding changed')
        owner['identity'] = identity
        with httpx.Client(timeout=5, headers={'X-Gemma-Identity': digest(identity)}) as client:
            while True:
                unchanged()
                if process.poll() is not None: raise RuntimeError(f'owned server exited {process.returncode};preserve startup')
                try:
                    response = client.get(owner['root'] + '/health')
                    if response.status_code == 200:
                        if response.json().get('identity_sha256') != digest(identity):
                            raise ValueError('ready endpoint returned a different serving identity')
                        break
                except (httpx.ConnectError, httpx.TimeoutException): pass
                time.sleep(5)
        return owner

    def release_servers():
        while servers:
            owner = servers[-1]
            code = stop_owned(owner['process'], owner['binding'], 'serve_vllm_lora.py', owner['output'])
            write(owner['output'].parent / (owner['output'].name + '-release.json'),
                {'process_binding': owner['binding'], 'returncode': code, 'observed_exited': not live(owner['binding'])})
            servers.pop()

    def run_cpu(command, executor, logfile):
        child_env = dict(env, PYTHONPATH=str(executor / 'src'))
        with logfile.open('x') as stream:
            child = subprocess.Popen(command, cwd=executor, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        children.append(child)
        while child.poll() is None:
            unchanged(); time.sleep(1)
        children.remove(child)
        if child.returncode: raise RuntimeError(f'actual child exited {child.returncode};preserve {logfile.name}')

    try:
        while True:
            unchanged(); pending = pending_producers(producers)
            if not pending: break
            if status.get('waiting_for') != pending: publish('waiting_actual_inputs', waiting_for=pending)
            time.sleep(10)
        publish('waiting_baseline_server_release')
        while live({'pid': controls_plan['server_pid'], 'birth_ticks': controls_plan['server_birth_ticks']}):
            unchanged(); time.sleep(1)
        jobs = validate_training_outputs(training, pilot / 'cache', pilot, config, selection,
            args.probe_contexts, args.probe_qualification)
        packets = load_packets(args.conversion, selection, plan_sha, jobs, conversion_plan['probe_sha256'])
        control_results = load_control_results(args.controls, selection, plan_sha)
        write(args.out / 'validated-inputs.json', {'candidate_packets_sha256': sha256_file(args.conversion / 'candidate-packets.json'),
            'control_results': {str(path): sha256_file(path) for path in control_results}})
        probe = next(packet for packet in packets if all(packet['condition'][k] == read(args.proposal)['probe_adapter'][k]
            for k in ['preset', 'completed_steps', 'strength']))
        left = start_server(probe, 0, 'hardware-left'); right = start_server(probe, 1, 'hardware-right')
        command = [str(args.qualifier / '.harness-venv/bin/python'), 'scripts/qualify_hardware_pair.py',
            '--dev-grid', str(args.prior_grid), '--conversion', str(args.conversion), '--controls', str(args.controls),
            '--executor', str(args.executor), '--executor-commit', args.executor_commit,
            '--dev-dispatch-pid', str(args.prior_pid), '--dev-dispatch-birth-ticks', args.prior_birth_ticks,
            '--selection-plan', str(args.selection_plan), '--proposal', str(args.proposal), '--probe-inputs', str(args.probe_inputs),
            '--probe-contexts', str(args.probe_contexts), '--probe-qualification', str(args.probe_qualification),
            '--out', str(args.out / 'hardware-qualification')]
        if producer_plan is not None:
            command.extend(['--producer-plan', str(producer_plan)])
        for side, owner in [('left', left), ('right', right)]:
            command.extend(['--' + side + '-endpoint', owner['root'], '--' + side + '-identity', str(owner['identity_path']),
                '--' + side + '-pid', str(owner['process'].pid)])
        publish('measuring_hardware_pair')
        run_cpu(command, args.qualifier, args.out / 'logs/hardware-qualification.log')
        # Full offline raw-journal replay is mandatory even if the child exits zero.
        policy = prepare_resource_supplement(args.declaration, args.selection_plan, args.out / 'hardware-qualification')
        policy_path = args.out / 'qualified-resource.json'; write(policy_path, policy)
        load_resource_supplement(policy_path)
        release_servers()
        declaration['resource_supplement'] = {'path': str(policy_path), 'sha256': sha256_file(policy_path)}
        # Finalize this field before the first learned DEV request; old grid stays immutable.
        write(args.out / 'plan.json', declaration)
        pins[str(args.out / 'plan.json')] = sha256_file(args.out / 'plan.json')
        pins[str(policy_path)] = sha256_file(policy_path)
        publish('qualified_resource_ready', resource_activated=True)
        by_name = {packet['condition']['id']: packet for packet in packets}; finished = []
        for wave_index, names in enumerate(resource_declaration['learned_waves'], 1):
            unchanged()
            load_packets(args.conversion, selection, plan_sha, jobs, conversion_plan['probe_sha256'])
            load_resource_supplement(policy_path)
            wave = []
            for index, name in enumerate(names):
                packet = by_name[name]; owner = start_server(packet, index, name)
                normalized = normalize_qualified_contract(canonical_contract({'model_identity': owner['identity']}), name, policy)
                if normalized != canonical_contract({'model_identity': baseline}):
                    raise ValueError('candidate serving contract differs from original baseline')
                attempts = args.out / (name + '-preflight-http.jsonl')

                def call(client, base, path, payload=None):
                    unchanged()
                    with attempts.open('a') as stream:
                        stream.write(json.dumps({'phase': 'attempt', 'path': path, 'request': payload}) + '\n')
                    response = client.get(base + path) if payload is None else client.post(base + path, json=payload)
                    value = response.json()
                    with attempts.open('a') as stream:
                        stream.write(json.dumps({'phase': 'response', 'path': path,
                            'status_code': response.status_code, 'response': value}) + '\n')
                    response.raise_for_status(); return value

                with httpx.Client(timeout=600, headers={'X-Gemma-Identity': digest(owner['identity'])}) as client:
                    preflight = candidate_preflight(client, owner['root'], owner['identity'], call)
                    effect = learned_effect_preflight(client, owner['root'], owner['identity'], packet, effect_cases, call)
                preflight.update(server_identity_sha256=digest(owner['identity']), http_journal_sha256=sha256_file(attempts))
                write(args.out / (name + '-preflight.json'), preflight)
                effect.update(protocol_sha256=sha256_file(args.effect_protocol), http_journal_sha256=sha256_file(attempts))
                write(args.out / (name + '-effect.json'), effect)
                if effect['status'] != 'passed': raise RuntimeError('actual learned effect ' + effect['status'] + ';preserve observations')
                owner.update(packet=packet, name=name, observations=[])
                wave.append(owner)
            # Both preflights pass before either task panel is launched.
            for owner in wave:
                owner['observations'].append(observe_slot(owner['binding'], owner['index'],
                    policy['hardware'][owner['index']], 'before_first_task'))
            publish('launching_full_dev_wave', wave=wave_index, current_conditions=names, model_dev_inference=True)
            for owner in wave:
                name = owner['name']; packet = owner['packet']; index = owner['index']; destination = args.out / name
                command = [str(args.task_executor / '.harness-venv/bin/python'), 'scripts/run_dev_condition.py',
                    '--selection-plan', str(execution_plan), '--condition', name, '--endpoint', owner['root'] + '/v1',
                    '--server-identity', str(owner['identity_path']), '--out', str(destination)]
                for option in ['adapter', 'conversion_report', 'native', 'cache']:
                    command.extend(['--' + option.replace('_', '-'), packet[option]])
                write(args.out / (name + '-before-task-slot.json'), slot_record(packet, policy_path, policy,
                    owner['identity_path'], owner['binding'], owner['command'], owner['observations']))
                with (args.out / 'logs' / (name + '-dev.log')).open('x') as stream:
                    child = subprocess.Popen(command, cwd=args.task_executor,
                        env=dict(env,PYTHONPATH=str(args.task_executor/'src')), stdout=stream, stderr=subprocess.STDOUT)
                children.append(child); owner['dev_child'] = child
            publish('running_full_dev_wave', wave=wave_index, current_conditions=names, model_dev_inference=True)
            codes = wait_wave([owner['dev_child'] for owner in wave], observe=unchanged)
            for owner in wave: children.remove(owner['dev_child'])
            failures = []
            for owner, code in zip(wave, codes):
                name = owner['name']; result_path = args.out / name / 'result.json'
                if code:
                    failures.append(f'{name}:exit{code}'); continue
                receipt = read(result_path)
                if (receipt.get('status') != 'complete' or receipt['condition'] != owner['packet']['condition']
                        or receipt['primary']['total'] != 42 or receipt['secondary']['total'] != 13):
                    raise ValueError('actual complete fixed55-task DEV condition required')
                owner['observations'].append(observe_slot(owner['binding'], owner['index'], policy['hardware'][owner['index']], 'after_final_task'))
                evidence = slot_record(owner['packet'], policy_path, policy, owner['identity_path'], owner['binding'],
                    owner['command'], owner['observations'])
                write(result_path.parent / 'resource.json', evidence)
                audit_condition_slot(result_path, receipt, policy_path, policy)
                finished.append({'condition_id': name, 'result': str(result_path), 'result_sha256': sha256_file(result_path),
                    'learned_effect_sha256': sha256_file(args.out / (name + '-effect.json')),
                    'preflight_sha256': sha256_file(args.out / (name + '-preflight.json')),
                    'resource_evidence_sha256': sha256_file(result_path.parent / 'resource.json')})
                publish('condition_complete', finished=finished)
            release_servers()
            if failures: raise RuntimeError('actual DEV child failure;no reroll or further wave: ' + ','.join(failures))
        unchanged(); load_control_results(args.controls, selection, plan_sha)
        command = [str(args.auditor / '.harness-venv/bin/python'), 'scripts/compare_dev_conditions.py',
            '--selection-plan', str(args.selection_plan), '--resource-supplement', str(policy_path), '--out', str(args.out / 'choice')]
        for path in [*control_results, *(Path(record['result']) for record in finished)]:
            command.extend(['--condition-result', str(path)])
        publish('auditing_complete_grid')
        run_cpu(command, args.auditor, args.out / 'logs/compare.log')
        choice = args.out / 'choice/selection.json'
        publish('complete_dev_choice', finished=finished, selection_sha256=sha256_file(choice), selected=read(choice)['selected'])
    except BaseException as exc:
        publish('error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Explicit termination is not a retry. Preserve partial task directories.
        for child in children:
            if child.poll() is None: child.send_signal(signal.SIGINT)
        for child in children: child.wait()
        release_servers()


if __name__ == '__main__': main()
