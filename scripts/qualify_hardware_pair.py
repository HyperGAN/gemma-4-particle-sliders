"""Measure the predeclared same-adapter parity on two already-owned servers.

Never launches/stops servers, changes DEV contracts, or opens DEV/holdout tasks.
Plan-only performs no HTTP request, nvidia-smi invocation, model or CUDA access.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

import httpx

from gemma_particle.conversion_handoff import read, validate_training_outputs
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import live, process_binding, load_packets, load_control_results
from gemma_particle.dev_evidence import SERVER_SOURCES
from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.hardware_parity import (prepare_probes, qualify_server_contracts,
    validate_device_facts, compare_probe_responses)
from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import prepare_lora


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def descendants(pid):
    parents = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: parents[int(proc.name)] = int((proc / 'stat').read_text().split(') ', 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    owned = {pid}
    while True:
        expanded = owned | {p for p, parent in parents.items() if parent in owned}
        if expanded == owned: return owned
        owned = expanded


def active_dispatchers(output):
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if not any(a.endswith(b'/run_learned_dev_grid.py') for a in argv) or b'--out' not in argv: continue
            target = Path(os.fsdecode(argv[argv.index(b'--out') + 1]))
            if not target.is_absolute(): target = proc.joinpath('cwd').resolve() / target
            fields = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
            if target.resolve() == output and fields[0] != 'Z':
                matches.append({'pid': int(proc.name), 'birth_ticks': fields[19]})
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    return matches


def observe_hardware(bindings):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,compute_cap,memory.total,driver_version,uuid',
        '--format=csv,noheader,nounits'], text=True)
    slots = {}
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        index, name, capability, total, driver, uuid = row
        slots[int(index)] = {'physical_index': int(index), 'name': name, 'compute_capability': capability,
            'total_MiB': int(total), 'driver': driver, 'uuid': uuid}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    applications = [{'pid': int(row[0]), 'gpu_uuid': row[1]} for row in csv.reader(raw.splitlines(), skipinitialspace=True)]
    owned_pids = [descendants(b['pid']) for b in bindings]
    owned_apps = [[a for a in applications if a['pid'] in group] for group in owned_pids]
    return [slots[0], slots[1]], owned_apps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dev-grid', 'conversion', 'controls', 'executor', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--executor-commit', required=True)
    p.add_argument('--proposal', type=Path, default=Path('research/dev-hardware-resource-proposal-v1.json'))
    p.add_argument('--probe-inputs', type=Path, default=Path('research/dev-hardware-probe-inputs-v1.json'))
    p.add_argument('--selection-plan', type=Path, default=Path('research/dev-selection-v1.json'))
    p.add_argument('--probe-contexts', type=Path, default=Path('data/conversion-probe-v1/contexts.json'))
    p.add_argument('--probe-qualification', type=Path, default=Path('reports/conversion-probe-preparation-v1.json'))
    p.add_argument('--dev-dispatch-pid', type=int)
    p.add_argument('--dev-dispatch-birth-ticks')
    for side, port in [('left', 8032), ('right', 8033)]:
        p.add_argument('--' + side + '-endpoint', default=f'http://127.0.0.1:{port}')
        p.add_argument('--' + side + '-identity', type=Path)
        p.add_argument('--' + side + '-pid', type=int)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    if args.out.exists(): raise FileExistsError('hardware qualification evidence is immutable')
    endpoints = [args.left_endpoint, args.right_endpoint]
    for endpoint in endpoints:
        u = urlsplit(endpoint)
        if (u.scheme != 'http' or u.hostname not in ['127.0.0.1', 'localhost', '::1'] or not u.port
                or u.username or u.password or u.path not in ['', '/'] or u.query or u.fragment):
            raise ValueError('explicit local root endpoints required')
    if urlsplit(endpoints[0]).port == urlsplit(endpoints[1]).port: raise ValueError('distinct loopback ports required')
    selection = load_selection_plan(args.selection_plan); selection_sha = sha256_file(args.selection_plan)
    inputs, resource, cases = prepare_probes(args.probe_inputs, args.proposal, selection_sha)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.executor, text=True).strip() != args.executor_commit:
        raise ValueError('exact frozen baseline serving executor required')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=args.executor, check=True)
    grid_plan, conversion_plan, controls_plan = [read(d / 'plan.json') for d in [args.dev_grid, args.conversion, args.controls]]
    if (any(d['selection_plan_sha256'] != selection_sha for d in [grid_plan, conversion_plan, controls_plan])
            or grid_plan['executor_commit'] != args.executor_commit
            or Path(grid_plan['executor']).resolve() != args.executor
            or sha256_file(args.probe_contexts) != conversion_plan['probe_sha256']
            or sha256_file(args.probe_qualification) != conversion_plan['probe_qualification_sha256']):
        raise ValueError('unchanged actual full-grid/native/baseline declarations required')
    root = Path(__file__).resolve().parents[1]
    pins = {str(path): sha256_file(path) for path in [args.selection_plan, args.proposal, args.probe_inputs,
        args.dev_grid / 'plan.json', args.conversion / 'plan.json', args.controls / 'plan.json',
        root / 'scripts/qualify_hardware_pair.py', root / 'src/gemma_particle/hardware_parity.py',
        Path(controls_plan['server_identity']), args.probe_contexts, args.probe_qualification,
        *[args.executor / name for name in SERVER_SOURCES]]}
    args.out.mkdir(parents=True)
    declaration = {'schema': 'gemma4-hardware-parity-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'executor_commit': args.executor_commit, 'declaration_pins': pins, 'selection_plan_sha256': selection_sha,
        'cases': [{k: c[k] for k in ['length', 'alias', 'request_sha256']} for c in cases], 'endpoints': endpoints,
        'probe_adapter': resource['probe_adapter'], 'server_mutations': False,
        'scope': 'synthetic numerical qualification only; no resource activation or DEV/holdout task'}
    write(args.out / 'plan.json', declaration)
    receipt = {'schema': 'gemma4-hardware-parity-result-v1', 'status': 'prepared' if args.plan_only else 'starting',
        'plan_sha256': sha256_file(args.out / 'plan.json'), 'pairs': [], 'requests_attempted': 0, 'requests_completed': 0,
        'completion_requests_attempted': 0, 'completion_http200_responses': 0,
        'resource_supplement_activated': False, 'model_task_inference': False, 'servers_mutated': False}
    def save(): write(args.out / 'result.json', receipt)
    save()
    if args.plan_only:
        print(json.dumps({'status': 'prepared', 'cases': len(cases), 'model_or_http_access': False})); return 0
    try:
        status = read(args.dev_grid / 'progress.json')
        if (status.get('model_dev_inference') is not False or status.get('finished')
                or not args.dev_dispatch_pid or not args.dev_dispatch_birth_ticks
                or live({'pid': args.dev_dispatch_pid, 'birth_ticks': args.dev_dispatch_birth_ticks})
                or active_dispatchers(args.dev_grid)):
            raise ValueError('owned waiting DEV dispatcher must be observed stopped before any learned DEV task')
        for producer in grid_plan['producers']:
            current = read(Path(producer['directory']) / 'progress.json')
            if (current.get('state') != producer['completed_state']
                    or producer['binding'] is not None and live(producer['binding'])):
                raise ValueError('complete actual input producers and observed release required')
        if live({'pid': controls_plan['server_pid'], 'birth_ticks': controls_plan['server_birth_ticks']}):
            raise ValueError('original full-control serving allocation must have released')
        load_control_results(args.controls, selection, selection_sha)
        training = Path(conversion_plan['training']); pilot = Path(read(training / 'plan.json')['pilot'])
        config = args.executor / 'configs/31b-ct-shared-positional-pilot.json'
        if sha256_file(config) != conversion_plan['config_sha256']: raise ValueError('native config differs')
        jobs = validate_training_outputs(training, pilot / 'cache', pilot, config, selection,
            args.probe_contexts, args.probe_qualification)
        packets = load_packets(args.conversion, selection, selection_sha, jobs, conversion_plan['probe_sha256'])
        packet = next(c for c in packets if all(c['condition'][k] == resource['probe_adapter'][k]
            for k in ['preset', 'completed_steps', 'strength']))
        spec = prepare_lora(read(args.executor / 'configs/31b-ct-pilot.json'),
            adapter=packet['adapter'], conversion_report=packet['conversion_report'])
        if args.left_identity is None or args.right_identity is None:
            raise ValueError('both actual serving identity files required')
        identities = [read(args.left_identity), read(args.right_identity)]
        baseline = read(controls_plan['server_identity'])
        receipt['qualification_contract_sha256'] = qualify_server_contracts(identities, baseline, spec['adapters'])
        bindings = []
        for index, (pid, identity_path, endpoint) in enumerate(zip([args.left_pid, args.right_pid],
                [args.left_identity, args.right_identity], endpoints)):
            if not pid: raise ValueError('both exact owned serving launcher PIDs required')
            binding = process_binding(pid, 'serve_vllm_lora.py', identity_path.parent)
            proc = Path('/proc', str(pid)); argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if (proc.joinpath('cwd').resolve() != args.executor
                    or argv[argv.index(b'--device') + 1] != f'cuda:{index}'.encode()
                    or argv[argv.index(b'--port') + 1] != str(urlsplit(endpoint).port).encode()
                    or Path(os.fsdecode(argv[argv.index(b'--adapter') + 1])).resolve() != Path(packet['adapter']).resolve()):
                raise ValueError('actual owned server command differs from fixed slot/packet/endpoint')
            for name in SERVER_SOURCES:
                values = [v for k, v in identities[index]['sources'].items() if k == name or k.endswith('/' + name)]
                if values != [sha256_file(args.executor / name)]: raise ValueError('actual serving source bytes differ')
            bindings.append(binding)
        receipt.update(server_process_bindings=bindings, server_identity_sha256=[digest(i) for i in identities],
            selected_packet=packet)
        save()
        clients = [httpx.Client(timeout=600, headers={'X-Gemma-Identity': digest(i)}) for i in identities]
        try:
            def call(side, path, payload=None):
                if any(sha256_file(p) != value for p, value in pins.items()) or not all(live(b) for b in bindings):
                    raise ValueError('pinned qualification sources/declarations or owned servers changed')
                index = receipt['requests_attempted']; receipt['requests_attempted'] += 1
                if path == '/v1/completions': receipt['completion_requests_attempted'] += 1
                attempt = {'index': index, 'side': side, 'path': path, 'request': payload,
                    'request_sha256': digest(payload)}
                with (args.out / 'http-attempts.jsonl').open('a') as stream: stream.write(json.dumps(attempt) + '\n')
                receipt.update(phase=path, current_side=side); save()
                client, endpoint = clients[side], endpoints[side].rstrip('/')
                response = client.get(endpoint + path) if payload is None else client.post(endpoint + path, json=payload)
                value = response.json()
                with (args.out / 'http-responses.jsonl').open('a') as stream:
                    stream.write(json.dumps({'index': index, 'status_code': response.status_code, 'response': value}) + '\n')
                response.raise_for_status(); receipt['requests_completed'] += 1
                if path == '/v1/completions': receipt['completion_http200_responses'] += 1
                save(); return value
            for side, identity in enumerate(identities):
                if call(side, '/health').get('identity_sha256') != digest(identity): raise ValueError('actual health identity differs')
                if not set(inputs['aliases']) <= {r['id'] for r in call(side, '/v1/models')['data']}:
                    raise ValueError('actual advertised aliases differ')
                for alias in inputs['aliases']:
                    if call(side, '/tokenize', {'model': alias, 'prompt': inputs['synthetic_prompt']})['tokens'] != inputs['seed_token_ids']:
                        raise ValueError('actual fixed seed tokenization differs')
            facts, owned_apps = observe_hardware(bindings)
            validate_device_facts(facts, owned_apps, resource)
            receipt.update(hardware=facts, owned_gpu_applications=owned_apps); save()
            for case in cases:
                responses = [call(side, '/v1/completions', case['request']) for side in [0, 1]]
                receipt['pairs'].append(compare_probe_responses(case, *responses)); save()
            if any(sha256_file(p) != value for p, value in pins.items()) or not all(live(b) for b in bindings):
                raise ValueError('qualification declarations or owned servers changed before verdict')
            receipt['status'] = 'passed' if all(pair['passed'] for pair in receipt['pairs']) else 'failed'
        finally:
            for client in clients: client.close()
    except httpx.TransportError as exc:
        receipt.update(status='observation_incomplete', error_type=type(exc).__name__,
            note='no numerical verdict; server work is not inferred terminal and servers are not restarted')
    except Exception as exc:
        receipt.update(status='error', error_type=type(exc).__name__, error=str(exc))
    receipt.update(scope='numerical hardware observation only; actual versioned resource supplement still required',
        as_of=datetime.now(timezone.utc).isoformat()); save()
    print(json.dumps({'status': receipt['status'], 'pairs': len(receipt['pairs']),
        'requests_completed': receipt['requests_completed'], 'resource_supplement_activated': False}))
    return 0 if receipt['status'] == 'passed' else 3 if receipt['status'] == 'observation_incomplete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
