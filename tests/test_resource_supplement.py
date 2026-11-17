"""Prospective resource and physical-slot CPU fixtures; never actual GPU proof."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from gemma_particle.data import digest
from gemma_particle.dev_evidence import canonical_contract
from gemma_particle.harness_identity import sha256_file
from gemma_particle.resource_supplement import (validate_resource_declaration, prepare_resource_supplement,
    load_resource_supplement, normalize_qualified_contract, audit_condition_slot, audit_task_slot)
from test_conversion_handoff import put
from test_hardware_evidence import hardware_fixture

ROOT = Path(__file__).resolve().parents[1]


def resource_fixture(root):
    qualification, _ = hardware_fixture(root)
    plan_path = qualification / 'plan.json'; plan = json.loads(plan_path.read_text())
    declaration = json.loads((ROOT / 'research/dev-resource-declaration-v1.json').read_text())
    declaration['offline_hardware_auditor_sha256'] = sha256_file(ROOT / 'src/gemma_particle/hardware_evidence.py')
    declaration['qualification_sources_sha256'] = {name: sha256_file(ROOT / name)
                                                  for name in declaration['qualification_sources_sha256']}
    executor = Path(plan['inputs']['executor'])
    declaration['serving_sources_sha256'] = {name: sha256_file(executor / name) for name in declaration['serving_sources_sha256']}
    plan['executor_commit'] = declaration['serving_executor_commit']
    for name, value in declaration['qualification_sources_sha256'].items(): plan['declaration_pins'][str(ROOT / name)] = value
    put(plan_path, plan)
    result_path = qualification / 'result.json'; result = json.loads(result_path.read_text())
    conversion = Path(plan['inputs']['conversion']); packet_path = conversion / 'candidate-packets.json'
    matrix = json.loads(packet_path.read_text())
    for packet in matrix['packets']:
        packet.update(native_metadata_sha256='c' * 64, cache_index_sha256='d' * 64, conversion_report_sha256='e' * 64)
    put(packet_path, matrix)
    status_path = conversion / 'progress.json'; status = json.loads(status_path.read_text())
    status['candidate_packets_sha256'] = sha256_file(packet_path); put(status_path, status)
    result['selected_packet'] = next(p for p in matrix['packets'] if p['condition']['id'] == result['selected_packet']['condition']['id'])
    result['prerequisite_progress_sha256'][str(status_path)] = sha256_file(status_path)
    result['plan_sha256'] = sha256_file(plan_path); put(result_path, result)
    declaration_path = root / 'CPU-resource-declaration.json'; put(declaration_path, declaration)
    record = prepare_resource_supplement(declaration_path, ROOT / 'research/dev-selection-v1.json', qualification)
    record_path = root / 'CPU-qualified-resource.json'; put(record_path, record)
    return record_path, record, qualification


def test_public_assignment_precedes_learned_results_and_keeps_all_conditions_and_confirmation():
    d = validate_resource_declaration(ROOT / 'research/dev-resource-declaration-v1.json', ROOT / 'research/dev-selection-v1.json')
    assert len(d['assignments']) == 22 and len(d['learned_waves']) == 9
    assert [a['physical_index'] for a in d['assignments'][:4]] == [0] * 4
    assert [a['physical_index'] for a in d['assignments'][4:]] == [0, 1] * 9
    assert d['state'] == 'inactive_preparation' and d['confirmation_device_policy'] == 'strict original GPU0 serving contract'


def test_empf_only_assignment_has_one_terminal_single_condition_wave():
    declaration = validate_resource_declaration(ROOT / 'research/dev-resource-declaration-empf-v1.json',
                                                ROOT / 'research/dev-selection-empf-v1.json')
    assert [len(wave) for wave in declaration['learned_waves']] == [2, 2, 2, 2, 1]
    assert declaration['assignments'][-1]['physical_index'] == 0


@pytest.mark.parametrize('damage', ['order', 'waves', 'plan', 'memory', 'threads', 'confirmation'])
def test_changed_prospective_resource_policy_is_rejected(tmp_path, damage):
    d = json.loads((ROOT / 'research/dev-resource-declaration-v1.json').read_text())
    if damage == 'order': d['assignments'].reverse()
    elif damage == 'waves': d['learned_waves'][0].reverse()
    elif damage == 'plan': d['core_selection_plan_sha256'] = '0' * 64
    elif damage == 'memory': d['engine_memory_fraction'] = .5
    elif damage == 'threads': d['cpu_threads_per_slot'] = 8
    else: d['confirmation_device_policy'] = 'any device'
    path = tmp_path / 'declaration.json'; put(path, d)
    with pytest.raises(ValueError): validate_resource_declaration(path, ROOT / 'research/dev-selection-v1.json')


def test_qualified_policy_reaudits_and_normalizes_only_the_assigned_server_device(tmp_path):
    path, record, qualification = resource_fixture(tmp_path)
    assert load_resource_supplement(path) == record
    result = json.loads((qualification / 'result.json').read_text())
    servers = [json.loads(Path(c['identity_path']).read_text()) for c in result['server_command_bindings']]
    a, b = [canonical_contract({'model_identity': server, 'CPU-harness': {'unchanged': True}}) for server in servers]
    original = deepcopy(b); condition = record['learned_waves'][0][1]
    assert normalize_qualified_contract(b, condition, record) == a
    assert b == original and canonical_contract({'model_identity': servers[0]}) != canonical_contract({'model_identity': servers[1]})


@pytest.mark.parametrize('damage', ['unqualified', 'slot', 'packages', 'engine', 'unknown'])
def test_resource_normalization_cannot_hide_any_other_serving_change(tmp_path, damage):
    _, record, qualification = resource_fixture(tmp_path)
    result = json.loads((qualification / 'result.json').read_text())
    server = json.loads(Path(result['server_command_bindings'][1]['identity_path']).read_text())
    contract = canonical_contract({'model_identity': server}); condition = record['learned_waves'][0][1]
    if damage == 'unqualified': record['status'] = 'prepared'
    elif damage == 'slot': contract['model_identity']['device'] = 'cuda:0'
    elif damage == 'packages': contract['model_identity']['packages']['CPU-fixture'] = 'changed'
    elif damage == 'engine': contract['model_identity']['max_num_seqs'] += 1
    else: condition = 'undeclared'
    with pytest.raises(ValueError): normalize_qualified_contract(contract, condition, record)


def slot_fixture(root):
    path, resource, qualification = resource_fixture(root)
    measured = json.loads((qualification / 'result.json').read_text())
    packet = next(p for p in json.loads(Path(resource['packet_manifest']).read_text())['packets']
        if p['condition']['id'] == resource['learned_waves'][0][1])
    command = measured['server_command_bindings'][1]
    server = json.loads(Path(command['identity_path']).read_text()); condition = packet['condition']
    out = root / 'condition'; out.mkdir(); result_path = out / 'result.json'
    put(out / 'plan.json', {'server_identity': command['identity_path'], 'endpoint': command['endpoint'] + '/v1'})
    result = {'condition': condition, 'server_identity_sha256': digest(server), 'packet': packet}; put(result_path, result)
    evidence = {'schema': 'gemma4-dev-serving-slot-v1', 'status': 'complete', 'condition': condition,
        'supplement_sha256': sha256_file(path), 'server_identity_sha256': digest(server),
        'server_identity_file_sha256': sha256_file(command['identity_path']), 'command': command,
        'process_binding': measured['server_process_bindings'][1], 'observations': [
            {'phase': phase, 'hardware': resource['hardware'][1], 'process_binding': measured['server_process_bindings'][1],
             'nice': 10, 'cpu_threads': {'OMP_NUM_THREADS': 4, 'MKL_NUM_THREADS': 4},
             'descendant_pids': measured['server_descendant_pids'][1], 'owned_gpu_applications': measured['owned_gpu_applications'][1]}
            for phase in ['before_first_task', 'after_final_task']]}
    put(out / 'resource.json', evidence)
    return result_path, result, path, resource


@pytest.mark.parametrize('damage', [None, 'uuid', 'phase', 'pid', 'packet', 'identity', 'command', 'policy', 'priority'])
def test_every_learned_condition_needs_original_before_after_physical_slot_evidence(tmp_path, damage):
    args = slot_fixture(tmp_path); evidence_path = args[0].parent / 'resource.json'; e = json.loads(evidence_path.read_text())
    if damage == 'uuid': e['observations'][1]['hardware']['uuid'] = 'wrong'
    elif damage == 'phase': e['observations'].pop()
    elif damage == 'pid': e['observations'][0]['owned_gpu_applications'][0]['pid'] = 999
    elif damage == 'packet': args[1]['packet'] = deepcopy(args[1]['packet']); args[1]['packet']['cache_index_sha256'] = 'wrong'
    elif damage == 'identity': e['server_identity_file_sha256'] = 'wrong'
    elif damage == 'command': e['command']['argv'][3] = 'cuda:0'
    elif damage == 'policy': e['supplement_sha256'] = 'wrong'
    elif damage == 'priority': e['observations'][1]['nice'] = 0
    put(evidence_path, e)
    if damage:
        with pytest.raises(ValueError): audit_condition_slot(*args)
    else:
        evidence = audit_condition_slot(*args)
        assert evidence['physical_index'] == 1 and evidence['sha256'] == sha256_file(evidence_path)


@pytest.mark.parametrize('damage', [None, 'learned', 'undeclared', 'prior_plan', 'path',
    'contract', 'server', 'headroom', 'device'])
def test_recovered_base_task_binds_the_original_qualified_slot(tmp_path, damage):
    _, resource, _ = resource_fixture(tmp_path)
    baseline_path = Path(resource['baseline_server_identity'])
    current = json.loads(baseline_path.read_text())
    original = deepcopy(current)
    original['observed_free_bytes'] -= 1
    # The startup observation differs even when the execution contract matches.
    assert digest(original) != digest(current)
    if damage == 'headroom': original['observed_free_bytes'] = 0
    if damage == 'device': original['device'] = 'cuda:1'
    old = tmp_path / 'old'; new = tmp_path / 'new'
    condition = {'id': 'off', 'role': 'baseline'}
    task = 'fixture'; panel = 'primary'
    cohort = old / panel / (task + '-agent'); cohort.mkdir(parents=True)
    verification = old / panel / (task + '-verification'); verification.mkdir()
    original_path = old / 'server-identity.json'; put(original_path, original)
    plan = {'condition': condition, 'selection_plan_sha256': resource['selection_plan_sha256'],
        'packet': None, 'task_ids': {panel: [task]}, 'server_identity': str(original_path)}
    put(old / 'plan.json', plan)
    receipt = {'prior_condition': str(old), 'prior_plan_sha256': sha256_file(old / 'plan.json'),
        'task_ids': [task]}
    put(new / 'plan.json', dict(plan, server_identity=str(baseline_path), reused_evidence=receipt))
    result = {'condition': condition, 'reused_evidence': deepcopy(receipt)}
    record = {'instance_id': task, 'panel': panel, 'cohort': str(cohort),
        'verification': str(verification), 'reused_from': str(old)}
    manifest = {'server_identity': original, 'contract': {'model_identity': deepcopy(original)}}
    slot = {'server_identity_sha256': digest(current), 'baseline_control': True}
    if damage == 'learned': slot['baseline_control'] = False
    if damage == 'undeclared': result['reused_evidence']['task_ids'] = []
    if damage == 'prior_plan': put(old / 'plan.json', dict(plan, changed=True))
    if damage == 'path': record['cohort'] = str(tmp_path / 'other')
    if damage == 'contract': manifest['contract']['model_identity']['device'] = 'cuda:1'
    if damage == 'server': put(original_path, current)
    if damage:
        with pytest.raises(ValueError): audit_task_slot(new, result, record, manifest, slot, resource)
    else:
        audit_task_slot(new, result, record, manifest, slot, resource)
        record['reused_from'] = None
        with pytest.raises(ValueError, match='actual recorded condition slot'):
            audit_task_slot(new, result, record, manifest, slot, resource)


def test_actual_comparison_cli_rejects_unqualified_policy_before_result_or_task_reads(tmp_path):
    policy = tmp_path / 'incomplete.json'; put(policy, {'schema': 'CPU incomplete policy'})
    command = [str(ROOT / '.harness-venv/bin/python'), str(ROOT / 'scripts/compare_dev_conditions.py'),
        '--selection-plan', str(ROOT / 'research/dev-selection-v1.json'), '--resource-supplement', str(policy),
        '--condition-result', str(tmp_path / 'RESULT_MUST_NOT_BE_READ'), '--out', str(tmp_path / 'out')]
    # Build local synthetic calibration bindings so the public-source suite needs no private data.
    from test_dev_selection import fixtures
    fixtures(tmp_path)
    command[command.index('--selection-plan') + 1] = str(tmp_path / 'selection-plan.json')
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'src')))
    assert child.returncode and 'completed explicit qualified DEV resource record required' in child.stderr
    assert 'FileNotFoundError' not in child.stderr and not (tmp_path / 'out').exists()
