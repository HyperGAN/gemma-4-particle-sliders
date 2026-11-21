"""CPU numerical/protocol fixtures; no actual GPU parity is established here."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gemma_particle.data import digest
from gemma_particle.dev_evidence import canonical_contract
from gemma_particle.hardware_parity import (prepare_probes, qualify_server_contracts,
    validate_device_facts, compare_probe_responses)
from gemma_particle.harness_identity import sha256_file
from test_dev_evidence import contract

ROOT = Path(__file__).resolve().parents[1]


def probes():
    return prepare_probes(ROOT / 'research/dev-hardware-probe-inputs-v1.json',
        ROOT / 'research/dev-hardware-resource-proposal-v1.json',
        sha256_file(ROOT / 'research/dev-selection-v1.json'))


def test_fixed_probes_cover_full_declared_lengths_aliases_and_original_input_ids():
    inputs, _, cases = probes()
    assert [(c['length'], c['alias']) for c in cases] == [(n, a) for n in [32, 4096, 28672] for a in inputs['aliases']]
    for case in cases:
        tokens = case['request']['prompt']; assert len(tokens) == case['length']
        assert tokens[:14] == inputs['seed_token_ids'] and tokens[14:28] == inputs['seed_token_ids']
        assert digest(case['request']) == case['request_sha256']
        assert case['request']['max_tokens'] == 1 and case['request']['logprobs'] == 20


@pytest.mark.parametrize('damage', ['source_hash', 'negative_id', 'length', 'generation', 'recipe'])
def test_prospective_inputs_cannot_change_declared_qualification(tmp_path, damage):
    inputs, resource, _ = probes()
    if damage == 'negative_id': inputs['seed_token_ids'][0] = -1
    elif damage == 'length': inputs['lengths'][-1] = 1024
    elif damage == 'generation': resource['generation']['logprobs'] = 5
    elif damage == 'recipe': resource['probe_adapter']['completed_steps'] = 1024
    proposal = tmp_path / 'proposal.json'; proposal.write_text(json.dumps(resource))
    inputs['proposal_sha256'] = sha256_file(proposal) if damage != 'source_hash' else 'a' * 64
    path = tmp_path / 'inputs.json'; path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError, match='prospective fixed'):
        prepare_probes(path, proposal, resource['core_selection_plan_sha256'])


def servers():
    baseline = contract()['model_identity']
    baseline.update(base_only=True, adapters={}, total_device_bytes=49140 * 1024**2, observed_free_bytes=42000 * 1024**2)
    adapters = {'particle_lora': {'fixture': 'not actual model weights'}}
    pair = []
    for index in [0, 1]:
        server = deepcopy(baseline)
        server.update(device=f'cuda:{index}', base_only=False, adapters=deepcopy(adapters))
        server['launch'].extend(['--lora-modules', json.dumps({'name': 'particle_lora', 'path': '/CPU-fixture', 'base_model_name': 'fixture'})])
        pair.append(server)
    return pair, baseline, adapters


def test_numerical_comparison_does_not_activate_device_normalization_in_dev():
    pair, baseline, adapters = servers(); before = deepcopy(pair)
    assert canonical_contract({'model_identity': pair[0]}) != canonical_contract({'model_identity': pair[1]})
    assert len(qualify_server_contracts(pair, baseline, adapters)) == 64
    assert pair == before
    assert canonical_contract({'model_identity': pair[0]}) != canonical_contract({'model_identity': pair[1]})


@pytest.mark.parametrize('damage', ['device', 'memory', 'packages', 'weights', 'launch', 'adapters', 'startup', 'fixture'])
def test_actual_pair_must_match_original_serving_conditions(damage):
    pair, baseline, adapters = servers(); right = pair[1]
    if damage == 'device': right['device'] = 'cuda:0'
    elif damage == 'memory': right['memory_fraction'] = .5
    elif damage == 'packages': right['packages']['fixture'] = 'changed'
    elif damage == 'weights': right['model_identity']['revision'] = 'different'
    elif damage == 'launch': right['launch'].extend(['--max-model-len', '1024'])
    elif damage == 'adapters': right['adapters'] = {}
    elif damage == 'startup': right['observed_free_bytes'] = 1024
    else: right['qualification_only'] = True
    with pytest.raises(ValueError): qualify_server_contracts(pair, baseline, adapters)


@pytest.mark.parametrize('damage', [None, 'same_uuid', 'driver', 'wrong_process_gpu', 'no_owned_process'])
def test_actual_gpu_uuid_process_and_original_descriptor_bindings(damage):
    _, resource, _ = probes()
    facts = [{**slot, 'uuid': f'GPU-CPU-fixture-{i}'} for i, slot in enumerate(resource['observed_slots'])]
    apps = [[{'pid': 100 + i, 'gpu_uuid': facts[i]['uuid']}] for i in [0, 1]]
    if damage == 'same_uuid': facts[1]['uuid'] = facts[0]['uuid']
    elif damage == 'driver': facts[1]['driver'] = 'different'
    elif damage == 'wrong_process_gpu': apps[1][0]['gpu_uuid'] = facts[0]['uuid']
    elif damage == 'no_owned_process': apps[1] = []
    if damage:
        with pytest.raises(ValueError): validate_device_facts(facts, apps, resource)
    else: assert validate_device_facts(facts, apps, resource)


def response(case):
    return {'model': case['alias'], 'usage': {'prompt_tokens': case['length'], 'completion_tokens': 1},
        'choices': [{'logprobs': {'tokens': ['token_id:0'],
            'top_logprobs': [{f'token_id:{i}': -float(i + 1) for i in range(20)}]}}]}


def test_all_six_cases_require_exact_token_ids_and_predeclared_numerical_tolerance():
    _, _, cases = probes()
    for case in cases:
        left, right = response(case), response(case)
        right['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += 2e-6
        assert compare_probe_responses(case, left, right)['passed'] is True
        right['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += 1e-4
        measured = compare_probe_responses(case, left, right)
        assert measured['passed'] is False and measured['tolerance'] == 1e-5


@pytest.mark.parametrize('damage', ['generated_id', 'top_ids', 'alias', 'length', 'nonfinite', 'missing_top', 'request'])
def test_changed_observations_never_become_hardware_parity(damage):
    case = probes()[2][0]; left, right = response(case), response(case)
    if damage == 'generated_id': right['choices'][0]['logprobs']['tokens'] = ['token_id:1']
    elif damage == 'top_ids':
        top = right['choices'][0]['logprobs']['top_logprobs'][0]; top['token_id:99'] = top.pop('token_id:19')
    elif damage == 'alias': right['model'] = 'wrong'
    elif damage == 'length': right['usage']['prompt_tokens'] -= 1
    elif damage == 'nonfinite': right['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] = float('nan')
    elif damage == 'missing_top': right['choices'][0]['logprobs']['top_logprobs'][0].pop('token_id:19')
    else: case['request']['prompt'][0] += 1
    if damage in ['generated_id', 'top_ids']:
        assert compare_probe_responses(case, left, right)['passed'] is False
    else:
        with pytest.raises(ValueError): compare_probe_responses(case, left, right)


def test_active_owned_dispatcher_is_observed_through_actual_process_not_a_state_file(tmp_path):
    spec = importlib.util.spec_from_file_location('parity_cli', ROOT / 'scripts/qualify_hardware_pair.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    script = tmp_path / 'run_learned_dev_grid.py'
    script.write_text("import time\nprint('ready',flush=True)\ntime.sleep(60)\n")
    child = subprocess.Popen([sys.executable, str(script), '--out', str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        found = module.active_dispatchers(tmp_path)
        assert len(found) == 1 and found[0]['pid'] == child.pid
        assert child.pid in module.descendants(child.pid)
    finally:
        child.terminate(); child.wait()
    assert module.active_dispatchers(tmp_path) == []
