"""Synthetic CPU hardware artifacts; no actual server, GPU or parity result."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gemma_particle.data import digest
from gemma_particle.dev_evidence import SERVER_SOURCES, canonical_contract
from gemma_particle.hardware_evidence import audit_hardware_pair
from gemma_particle.hardware_parity import qualify_server_contracts, compare_probe_responses
from gemma_particle.harness_identity import sha256_file
from test_conversion_handoff import put
from test_hardware_parity import probes, servers, response

pytestmark = pytest.mark.usefixtures("archived_evidence")

ROOT = Path(__file__).resolve().parents[1]


def hardware_fixture(root):
    probe_inputs, resource, cases = probes(); pair, baseline, adapters = servers()
    executor = root / 'CPU-executor'
    sources = {}
    for name in SERVER_SOURCES:
        path = executor / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('CPU synthetic frozen source;not executable model code\n'); sources[name] = sha256_file(path)
    for identity in [*pair, baseline]: identity['sources'] = deepcopy(sources)
    baseline_path = root / 'baseline-server.json'; put(baseline_path, baseline)
    selection_path = ROOT / 'research/dev-selection-v1.json'; selection_sha = sha256_file(selection_path)
    selection = json.loads(selection_path.read_text())
    conversion, controls, grid = [root / p for p in ['conversion', 'controls', 'old-stopped-grid']]
    for p in [conversion, controls, grid]: p.mkdir()
    packets = [{'condition': c, 'adapter': str(root / 'CPU-adapter'), 'adapters': adapters}
        for c in selection['conditions'] if c['role'] == 'learned']
    selected = next(p for p in packets if all(p['condition'][k] == resource['probe_adapter'][k]
        for k in ['preset', 'completed_steps', 'strength']))
    put(conversion / 'candidate-packets.json', {'selection_plan_sha256': selection_sha, 'packets': packets})
    put(conversion / 'progress.json', {'state': 'prepared_complete_learned_grid',
        'completed_unit_conversions': 6, 'candidate_packets': 18,
        'candidate_packets_sha256': sha256_file(conversion / 'candidate-packets.json')})
    put(controls / 'progress.json', {'state': 'complete'})
    put(grid / 'progress.json', {'state': 'waiting_actual_inputs', 'model_dev_inference': False, 'finished': []})
    producers = [{'directory': str(p), 'completed_state': state} for p, state in [
        (conversion, 'prepared_complete_learned_grid'), (controls, 'complete')]]
    put(grid / 'plan.json', {'producers': producers}); put(conversion / 'plan.json', {'fixture': 'CPU'})
    put(controls / 'plan.json', {'server_identity': str(baseline_path)})
    commands = []; endpoints = ['http://127.0.0.1:8032', 'http://127.0.0.1:8033']
    for i, identity in enumerate(pair):
        path = root / f'CPU-server-{i}.json'; put(path, identity)
        commands.append({'cwd': str(executor), 'argv': ['/CPU/python', 'scripts/serve_vllm_lora.py',
            '--device', f'cuda:{i}', '--port', str(8032 + i), '--adapter', selected['adapter']],
            'endpoint': endpoints[i], 'identity_path': str(path), 'identity_file_sha256': sha256_file(path)})
    directory = root / 'parity'; directory.mkdir()
    plan = {'schema': 'gemma4-hardware-parity-plan-v1', 'selection_plan_sha256': selection_sha,
        'inputs': {'dev_grid': str(grid), 'conversion': str(conversion), 'controls': str(controls),
            'executor': str(executor), 'selection_plan': str(selection_path),
            'proposal': str(ROOT / 'research/dev-hardware-resource-proposal-v1.json'),
            'probe_inputs': str(ROOT / 'research/dev-hardware-probe-inputs-v1.json')},
        'cases': [{k: c[k] for k in ['length', 'alias', 'request_sha256']} for c in cases],
        'probe_adapter': resource['probe_adapter'], 'server_mutations': False, 'endpoints': endpoints,
        'declaration_pins': {str(path): sha256_file(path) for path in [baseline_path,
            *[executor / n for n in SERVER_SOURCES], selection_path, grid / 'plan.json',
            conversion / 'plan.json', controls / 'plan.json']}}
    put(directory / 'plan.json', plan)
    attempts = []; responses = []
    def exchange(side, path, value, payload=None):
        i = len(attempts)
        attempts.append({'index': i, 'side': side, 'path': path, 'request': payload, 'request_sha256': digest(payload)})
        responses.append({'index': i, 'status_code': 200, 'response': value})
        return value
    for side, identity in enumerate(pair):
        exchange(side, '/health', {'identity_sha256': digest(identity)})
        exchange(side, '/v1/models', {'data': [{'id': a} for a in probe_inputs['aliases']]})
        for alias in probe_inputs['aliases']:
            exchange(side, '/tokenize', {'tokens': probe_inputs['seed_token_ids']},
                {'model': alias, 'prompt': probe_inputs['synthetic_prompt']})
    measured = []
    for case in cases:
        values = [exchange(side, '/v1/completions', response(case), case['request']) for side in [0, 1]]
        measured.append(compare_probe_responses(case, *values))
    for name, rows in [('http-attempts.jsonl', attempts), ('http-responses.jsonl', responses)]:
        (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    facts = [{**slot, 'uuid': f'GPU-CPU-fixture-{i}'} for i, slot in enumerate(resource['observed_slots'])]
    receipt = {'schema': 'gemma4-hardware-parity-result-v1', 'status': 'passed',
        'plan_sha256': sha256_file(directory / 'plan.json'), 'resource_supplement_activated': False,
        'model_task_inference': False, 'servers_mutated': False, 'pairs': measured,
        'requests_attempted': 20, 'requests_completed': 20, 'completion_requests_attempted': 12, 'completion_http200_responses': 12,
        'prerequisite_progress_sha256': {str(p / 'progress.json'): sha256_file(p / 'progress.json') for p in [grid, conversion, controls]},
        'server_command_bindings': commands, 'server_process_bindings': [{'pid': 100 + i, 'birth_ticks': str(200 + i)} for i in [0, 1]],
        'server_identity_sha256': [digest(identity) for identity in pair], 'selected_packet': selected,
        'qualification_contract_sha256': qualify_server_contracts(pair, baseline, adapters), 'hardware': facts,
        'owned_gpu_applications': [[{'pid': 1000 + i, 'gpu_uuid': facts[i]['uuid']}] for i in [0, 1]],
        'server_descendant_pids': [[100 + i, 1000 + i] for i in [0, 1]],
        'fixture': 'CPU artifact structure only;no actual execution or hardware claim'}
    put(directory / 'result.json', receipt)
    return directory, pair


def test_scoped_hardware_audit_replays_original_probes_and_new_producers(tmp_path):
    directory, _ = hardware_fixture(tmp_path)
    plan_path, receipt_path = directory / 'plan.json', directory / 'result.json'
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    selection = ROOT / 'research/dev-selection-empf-v1.json'
    selection_sha = sha256_file(selection)
    grid_path = Path(plan['inputs']['dev_grid']) / 'plan.json'
    producer_path = tmp_path / 'new-producers.json'
    put(producer_path, {'schema': 'gemma4-dev-qualification-inputs-v1',
                        'selection_plan_sha256': selection_sha, 'producers': json.loads(grid_path.read_text())['producers']})
    put(grid_path, {'producers': []})
    plan['selection_plan_sha256'] = selection_sha
    plan['inputs'].update(selection_plan=str(selection), producer_plan=str(producer_path))
    plan['declaration_pins'].update({str(path): sha256_file(path) for path in [selection, producer_path, grid_path]})
    put(plan_path, plan)
    conversion = Path(plan['inputs']['conversion'])
    packet_path, status_path = conversion / 'candidate-packets.json', conversion / 'progress.json'
    packets = json.loads(packet_path.read_text())
    packets['selection_plan_sha256'] = selection_sha
    packets['packets'] = [packet for packet in packets['packets'] if packet['condition']['preset'] == 'empf']
    put(packet_path, packets)
    put(status_path, {'state': 'prepared_complete_learned_grid', 'completed_unit_conversions': 3,
                      'candidate_packets': 9, 'candidate_packets_sha256': sha256_file(packet_path)})
    receipt['prerequisite_progress_sha256'][str(status_path)] = sha256_file(status_path)
    receipt['plan_sha256'] = sha256_file(plan_path)
    put(receipt_path, receipt)
    assert len(audit_hardware_pair(directory)['pairs']) == 6
    put(producer_path, {'changed': True})
    with pytest.raises(ValueError, match='unchanged numerical hardware'):
        audit_hardware_pair(directory)


@pytest.mark.parametrize('damage', [None, 'status', 'source', 'progress', 'identity', 'command', 'request', 'response',
    'missing', 'pairs', 'counters', 'gpu', 'descendants', 'packet', 'matrix'])
def test_complete_original_hardware_journals_and_provenance_are_reaudited(tmp_path, damage):
    directory, pair = hardware_fixture(tmp_path)
    result_path = directory / 'result.json'; receipt = json.loads(result_path.read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if damage == 'status': receipt['status'] = 'observation_incomplete'
    elif damage == 'source': Path(next(p for p in plan['declaration_pins'] if p.endswith('serve_vllm.py'))).write_text('changed')
    elif damage == 'progress': Path(next(iter(receipt['prerequisite_progress_sha256']))).write_text('{}')
    elif damage == 'identity': Path(receipt['server_command_bindings'][0]['identity_path']).write_text('{}')
    elif damage == 'command': receipt['server_command_bindings'][1]['argv'][3] = 'cuda:0'
    elif damage in ['request', 'response', 'missing']:
        path = directory / ('http-attempts.jsonl' if damage == 'request' else 'http-responses.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == 'request': rows[-1]['request']['seed'] = 1
        elif damage == 'response': rows[-1]['response']['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += .01
        else: rows.pop()
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif damage == 'pairs': receipt['pairs'][0]['max_abs_logprob_delta'] = 1e-7
    elif damage == 'counters': receipt['completion_requests_attempted'] = 11
    elif damage == 'gpu': receipt['hardware'][1]['uuid'] = receipt['hardware'][0]['uuid']
    elif damage == 'descendants': receipt['server_descendant_pids'][1] = [101]
    elif damage == 'packet': receipt['selected_packet']['condition']['strength'] = .5
    elif damage == 'matrix':
        conversion = Path(plan['inputs']['conversion']); p = conversion / 'candidate-packets.json'
        value = json.loads(p.read_text()); value['packets'].pop(); put(p, value)
        status_path = conversion / 'progress.json'; value = json.loads(status_path.read_text()); value['candidate_packets_sha256'] = sha256_file(p); put(status_path, value)
        receipt['prerequisite_progress_sha256'][str(status_path)] = sha256_file(status_path)
    put(result_path, receipt)
    if damage:
        with pytest.raises(ValueError): audit_hardware_pair(directory)
    else:
        result = audit_hardware_pair(directory)
        assert result['status'] == 'passed' and len(result['pairs']) == 6
        assert result['resource_supplement_activated'] is False
        assert canonical_contract({'model_identity': pair[0]}) != canonical_contract({'model_identity': pair[1]})
