"""Reaudit recorded hardware qualification without activating a resource change.

Artifact consistency cannot authenticate fabricated execution. A finite synthetic
control does not establish universal numerical equivalence or task accuracy.
"""
import json
from pathlib import Path
from urllib.parse import urlsplit

from .conversion_handoff import read
from .data import digest
from .dev_evidence import SERVER_SOURCES
from .hardware_parity import (prepare_probes, qualify_server_contracts,
    validate_device_facts, compare_probe_responses)
from .harness_identity import sha256_file


def audit_hardware_pair(directory):
    directory = Path(directory); receipt = read(directory / 'result.json'); plan = read(directory / 'plan.json')
    if (receipt.get('schema') != 'gemma4-hardware-parity-result-v1' or receipt.get('status') != 'passed'
            or plan.get('schema') != 'gemma4-hardware-parity-plan-v1'
            or receipt.get('plan_sha256') != sha256_file(directory / 'plan.json')
            or receipt.get('resource_supplement_activated') is not False
            or receipt.get('model_task_inference') is not False or receipt.get('servers_mutated') is not False
            or any(sha256_file(p) != value for p, value in plan['declaration_pins'].items())
            or any(sha256_file(p) != value for p, value in receipt['prerequisite_progress_sha256'].items())):
        raise ValueError('complete unchanged numerical hardware observation required')
    inputs = plan['inputs']; executor = Path(inputs['executor'])
    packet, resource, cases = prepare_probes(inputs['probe_inputs'], inputs['proposal'], plan['selection_plan_sha256'])
    if (sha256_file(inputs['selection_plan']) != plan['selection_plan_sha256']
            or plan['cases'] != [{k: c[k] for k in ['length', 'alias', 'request_sha256']} for c in cases]
            or plan['probe_adapter'] != resource['probe_adapter'] or plan['server_mutations'] is not False):
        raise ValueError('original fixed resource protocol and all six cases required')
    grid = read(Path(inputs['dev_grid']) / 'plan.json')
    grid_status = read(Path(inputs['dev_grid']) / 'progress.json')
    if grid_status.get('model_dev_inference') is not False or grid_status.get('finished'):
        raise ValueError('original stopped grid must retain zero learned DEV attempts')
    expected_progress = {str(Path(inputs['dev_grid']) / 'progress.json')}
    for producer in grid['producers']:
        path = Path(producer['directory']) / 'progress.json'; expected_progress.add(str(path))
        if read(path).get('state') != producer['completed_state']:
            raise ValueError('actual completed prerequisite producer records required')
    if expected_progress != set(receipt['prerequisite_progress_sha256']):
        raise ValueError('all original prerequisite progress records required')
    identities = []
    commands = receipt['server_command_bindings']; bindings = receipt['server_process_bindings']
    if len(commands) != 2 or len(bindings) != 2 or len({b['pid'] for b in bindings}) != 2:
        raise ValueError('two recorded owned serving processes required')
    selected = receipt['selected_packet']; adapter = Path(selected['adapter']).resolve()
    if any(selected['condition'][k] != resource['probe_adapter'][k] for k in ['preset', 'completed_steps', 'strength']):
        raise ValueError('prospectively fixed parity adapter required')
    conversion = Path(inputs['conversion']); packets_path = conversion / 'candidate-packets.json'
    conversion_status = read(conversion / 'progress.json'); packets = read(packets_path)
    expected_conditions = [c for c in read(inputs['selection_plan'])['conditions'] if c['role'] == 'learned']
    if (conversion_status.get('state') != 'prepared_complete_learned_grid'
            or conversion_status.get('completed_unit_conversions') != 6 or conversion_status.get('candidate_packets') != 18
            or conversion_status['candidate_packets_sha256'] != sha256_file(packets_path)
            or packets['selection_plan_sha256'] != plan['selection_plan_sha256']
            or [p['condition'] for p in packets['packets']] != expected_conditions
            or packets['packets'].count(selected) != 1):
        raise ValueError('original complete measured packet manifest and fixed packet required')
    for index, command in enumerate(commands):
        argv = command['argv']; endpoint = urlsplit(command['endpoint'])
        if (command['endpoint'] != plan['endpoints'][index]
                or Path(command['cwd']).resolve() != executor.resolve()
                or sha256_file(command['identity_path']) != command['identity_file_sha256']
                or argv[argv.index('--device') + 1] != f'cuda:{index}'
                or argv[argv.index('--port') + 1] != str(endpoint.port)
                or Path(argv[argv.index('--adapter') + 1]).resolve() != adapter
                or not any(a.endswith('/serve_vllm_lora.py') for a in argv)):
            raise ValueError('original owned serving command,identity,endpoint and physical slot required')
        identity = read(command['identity_path']); identities.append(identity)
        if digest(identity) != receipt['server_identity_sha256'][index]:
            raise ValueError('original serving identity changed')
        for name in SERVER_SOURCES:
            values = [v for k, v in identity['sources'].items() if k == name or k.endswith('/' + name)]
            if values != [sha256_file(executor / name)]:
                raise ValueError('original frozen serving source bytes required')
    baseline = read(read(Path(inputs['controls']) / 'plan.json')['server_identity'])
    contract = qualify_server_contracts(identities, baseline, selected['adapters'])
    if contract != receipt['qualification_contract_sha256']:
        raise ValueError('recorded parity contract differs from serving facts')
    validate_device_facts(receipt['hardware'], receipt['owned_gpu_applications'], resource)
    descendants = receipt['server_descendant_pids']
    if len(descendants) != 2 or any(bindings[i]['pid'] not in descendants[i]
            or any(a['pid'] not in descendants[i] for a in receipt['owned_gpu_applications'][i]) for i in [0, 1]):
        raise ValueError('recorded GPU processes must belong to their owned serving descendants')
    attempts = [json.loads(line) for line in (directory / 'http-attempts.jsonl').read_text().splitlines()]
    responses = [json.loads(line) for line in (directory / 'http-responses.jsonl').read_text().splitlines()]
    if len(attempts) != 20 or len(responses) != 20:
        raise ValueError('all twenty original HTTP exchanges required')
    cursor = 0
    def replay(side, path, payload=None):
        nonlocal cursor
        attempt, response = attempts[cursor], responses[cursor]
        expected = {'index': cursor, 'side': side, 'path': path, 'request': payload, 'request_sha256': digest(payload)}
        if attempt != expected or response.get('index') != cursor or response.get('status_code') != 200:
            raise ValueError('original parity request/response order and bytes required')
        cursor += 1; return response['response']
    for side, identity in enumerate(identities):
        if replay(side, '/health').get('identity_sha256') != digest(identity):
            raise ValueError('recorded live health differs from serving identity')
        if not set(packet['aliases']) <= {r['id'] for r in replay(side, '/v1/models')['data']}:
            raise ValueError('recorded server aliases differ')
        for alias in packet['aliases']:
            if replay(side, '/tokenize', {'model': alias, 'prompt': packet['synthetic_prompt']})['tokens'] != packet['seed_token_ids']:
                raise ValueError('actual original seed tokenization required')
    measured = [compare_probe_responses(case, *[replay(side, '/v1/completions', case['request']) for side in [0, 1]]) for case in cases]
    if (cursor != 20 or measured != receipt['pairs'] or not all(r['passed'] for r in measured)
            or any(receipt.get(k) != n for k, n in [('requests_attempted', 20), ('requests_completed', 20),
                ('completion_requests_attempted', 12), ('completion_http200_responses', 12)])):
        raise ValueError('all six actual paired observations and exact request accounting required')
    return {'schema': 'gemma4-hardware-pair-audit-v1', 'status': 'passed',
        'result_sha256': sha256_file(directory / 'result.json'), 'plan_sha256': sha256_file(directory / 'plan.json'),
        'qualification_contract_sha256': contract, 'physical_servers': identities,
        'hardware': receipt['hardware'], 'pairs': measured, 'resource_supplement_activated': False,
        'scope': 'recorded numerical control only;no resource activation,universal parity or task-quality claim'}
