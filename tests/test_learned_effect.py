"""Synthetic numerical observations; no pretrained or task-quality results."""
from copy import deepcopy
import json

import httpx
import pytest

from gemma_particle.data import digest
from gemma_particle.harness_identity import sha256_file
from gemma_particle.learned_effect import GENERATION, TOLERANCE, prepare_effect_probes, learned_effect_preflight
from gemma_particle.lora_serving import BASE_ALIAS
from test_conversion_handoff import put


def protocol_fixture(root):
    contexts = root / 'contexts.json'; qualification = root / 'qualification.json'; protocol = root / 'protocol.json'
    entries = []
    for i, tokens in enumerate(([2, 3], [2, 4, 5])):
        request = {'messages': [{'role': 'system', 'content': 'CPU fixture'}, {'role': 'user', 'content': str(i)}]}
        entries.append({'instance_id': 'train-' + str(i), 'request': request, 'request_sha256': digest(request),
            'neutral_ids': tokens, 'neutral_ids_sha256': digest(tokens)})
    put(contexts, {'schema': 'gemma4-conversion-probe-v1',
        'purpose': 'conversion diagnostic only; forbidden as a training target', 'train_manifest_sha256': 'a' * 64,
        'entries': entries})
    ids = [e['instance_id'] for e in entries]; lengths = [len(e['neutral_ids']) for e in entries]
    put(qualification, {'status': 'passed', 'probe_sha256': sha256_file(contexts),
        'train_manifest_sha256': 'a' * 64, 'task_ids': ids, 'token_counts': lengths,
        'request_sha256': [e['request_sha256'] for e in entries]})
    put(protocol, {'schema': 'gemma4-learned-effect-protocol-v1', 'core_selection_plan_sha256': 'b' * 64,
        'probe_contexts_sha256': sha256_file(contexts), 'probe_qualification_sha256': sha256_file(qualification),
        'aliases': [BASE_ALIAS, 'particle_lora'], 'generation': GENERATION, 'tolerance': TOLERANCE,
        'repeats_per_alias': 2, 'required_effect_probes': 1, 'probe_task_ids': ids, 'probe_token_counts': lengths})
    return protocol, contexts, qualification, {'training': {'train_manifest_sha256': 'a' * 64}}, 'b' * 64


@pytest.mark.parametrize('damage', [None, 'tolerance', 'qualification', 'contexts', 'train', 'ids', 'generation', 'repeats'])
def test_fixed_train_effect_declaration_rejects_changed_inputs(tmp_path, damage):
    args = protocol_fixture(tmp_path)
    if damage in ['tolerance', 'ids', 'generation', 'repeats']:
        p = args[0]; value = json.loads(p.read_text())
        if damage == 'tolerance': value['tolerance'] = 1e-3
        elif damage == 'ids': value['probe_task_ids'].reverse()
        elif damage == 'generation': value['generation']['seed'] = 1
        else: value['repeats_per_alias'] = 1
        put(p, value)
    elif damage in ['qualification', 'contexts']:
        p = args[2 if damage == 'qualification' else 1]; p.write_text(p.read_text() + ' ')
    elif damage == 'train': args[3]['training']['train_manifest_sha256'] = 'c' * 64
    if damage:
        with pytest.raises(ValueError, match='fixed qualified TRAIN'): prepare_effect_probes(*args)
    else:
        protocol, cases = prepare_effect_probes(*args)
        assert [c['prompt_tokens'] for c in cases] == [2, 3]
        assert protocol['required_effect_probes'] == 1


def response(payload):
    return {'model': payload['model'], 'usage': {'prompt_tokens': len(payload['prompt']), 'completion_tokens': 1},
        'choices': [{'logprobs': {'tokens': ['token_id:0'],
            'top_logprobs': [{f'token_id:{i}': -float(i + 1) for i in range(20)}]}}]}


@pytest.mark.parametrize('mode', ['same', 'tiny', 'logprob', 'token', 'topset', 'unstable_base', 'unstable_learned'])
def test_actual_pair_effect_and_repeatability_are_distinct_observations(tmp_path, mode):
    _, cases = prepare_effect_probes(*protocol_fixture(tmp_path)); calls = []
    packet = {'condition': {'id': 'CPU-fixture'}, 'adapters': {'particle_lora': {'fixture': True}},
        'native_metadata_sha256': 'c' * 64, 'cache_index_sha256': 'd' * 64, 'conversion_report_sha256': 'e' * 64}
    identity = {'base_only': False, 'adapters': packet['adapters']}
    def call(client, root, path, payload):
        calls.append(deepcopy(payload)); value = response(payload); probs = value['choices'][0]['logprobs']
        if payload['model'] == 'particle_lora' and len(payload['prompt']) == 2:
            if mode in ['tiny', 'logprob']: probs['top_logprobs'][0]['token_id:0'] += 1e-7 if mode == 'tiny' else 1e-3
            elif mode == 'token': probs['tokens'] = ['token_id:1']
            elif mode == 'topset': probs['top_logprobs'][0]['token_id:30'] = probs['top_logprobs'][0].pop('token_id:19')
        if mode == 'unstable_base' and len(calls) == 2 or mode == 'unstable_learned' and len(calls) == 4:
            probs['top_logprobs'][0]['token_id:0'] += 1e-3
        return value
    result = learned_effect_preflight(None, 'CPU-fixture', identity, packet, cases, call)
    expected = 'unstable_observation' if mode.startswith('unstable') else 'no_observed_effect' if mode in ['same', 'tiny'] else 'passed'
    assert result['status'] == expected and result['probe_count'] == 2
    assert len(calls) == 8
    assert [p['model'] for p in calls] == [BASE_ALIAS, BASE_ALIAS, 'particle_lora', 'particle_lora'] * 2
    assert all({k: p[k] for k in GENERATION} == GENERATION for p in calls)
    assert result['native_metadata_sha256'] == packet['native_metadata_sha256']
    if expected == 'passed': assert result['effect_probes'] == 1


@pytest.mark.parametrize('damage', ['alias', 'length', 'nonfinite', 'missing_top', 'timeout', 'packet', 'prefix'])
def test_invalid_or_incomplete_observations_do_not_become_effects(tmp_path, damage):
    _, cases = prepare_effect_probes(*protocol_fixture(tmp_path)); calls = []
    packet = {'condition': {}, 'adapters': {'particle_lora': {'fixture': True}},
        'native_metadata_sha256': '', 'cache_index_sha256': '', 'conversion_report_sha256': ''}
    identity = {'base_only': False, 'adapters': deepcopy(packet['adapters'])}
    if damage == 'packet': packet['adapters'] = {'wrong': {}}
    if damage == 'prefix': cases[0]['prompt_token_ids'][0] = 100
    def call(client, root, path, payload):
        calls.append(payload)
        if damage == 'timeout': raise httpx.ReadTimeout('CPU observation timeout')
        value = response(payload)
        if damage == 'alias': value['model'] = 'wrong'
        elif damage == 'length': value['usage']['prompt_tokens'] -= 1
        elif damage == 'nonfinite': value['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] = float('nan')
        elif damage == 'missing_top': value['choices'][0]['logprobs']['top_logprobs'][0].pop('token_id:19')
        return value
    with pytest.raises(httpx.ReadTimeout if damage == 'timeout' else ValueError):
        learned_effect_preflight(None, 'CPU-fixture', identity, packet, cases, call)
    if damage in ['packet', 'prefix']: assert not calls
    if damage == 'timeout': assert len(calls) == 1
