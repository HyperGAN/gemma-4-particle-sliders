"""CPU release-boundary controls with synthetic files, never real candidates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from gemma_particle.adapters import adapter_identity
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import candidate_preflight
from gemma_particle.harness_identity import sha256_file
from gemma_particle.learned_effect import prepare_effect_probes, learned_effect_preflight, audit_learned_effect
from gemma_particle.lora_serving import BASE_ALIAS
from gemma_particle.selected_candidate import selected_evaluated_archive
from test_conversion_handoff import put
from test_kaggle_candidate import fixture, module
from test_learned_effect import protocol_fixture, response

ROOT = Path(__file__).resolve().parents[1]


def archive_fixture(root):
    artifact, _, _, _ = fixture(root)
    builder = module('selected_fixture_builder', 'build_submission.py')
    evaluated = root / 'confirmation/cohort/artifact'
    builder.build(evaluated, BASE_ALIAS, adapter=artifact / 'agent/adapters/particle_lora',
        instruction_protocol='tools-v2', prompt_profile='repair-finalize-v2')
    receipt = json.loads((evaluated / 'build.json').read_text())
    put(evaluated.parent / 'cohort.json', {'artifact': receipt})
    dev_cohort = root / 'dev/cohort'; (dev_cohort / 'artifact').mkdir(parents=True)
    put(dev_cohort / 'cohort.json', {'artifact': receipt}); put(dev_cohort / 'artifact/build.json', receipt)
    condition = {'id': 'fixture-selected', 'role': 'learned', 'completed_steps': 1, 'strength': 1}
    dev = root / 'dev/result.json'
    put(dev, {'condition': condition, 'status': 'complete', 'finished': [{'cohort': str(dev_cohort)}]})
    confirmation = root / 'confirmation/result.json'
    put(confirmation, {'finished': [{'cohort': str(evaluated.parent)}]})
    choice = root / 'choice.json'
    put(choice, {'condition_results': {condition['id']: {'path': str(dev), 'sha256': sha256_file(dev)}}})
    scope = SimpleNamespace(receipt={'selected_condition': condition, 'dev_choice': str(choice),
        'selected_packet': {'condition': condition, 'adapters': {'particle_lora': adapter_identity(evaluated / 'agent/adapters/particle_lora')}}},
        plan={'matching': json.loads((ROOT / 'research/dev-selection-v1.json').read_text())['matching']})
    comparison = {'condition_results': {condition['id']: {'path': str(confirmation), 'sha256': sha256_file(confirmation)}}}
    return scope, comparison, evaluated, dev_cohort


@pytest.mark.parametrize('damage', [None, 'archive', 'local_prompt', 'dev_artifact', 'profile', 'sampling', 'adapter', 'budget', 'prompt_hash'])
def test_only_the_exact_evaluated_selected_archive_can_be_prepared(tmp_path, damage):
    scope, comparison, artifact, dev = archive_fixture(tmp_path)
    if damage == 'archive':
        path = artifact / 'submission.zip'; path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'local_prompt': (artifact / 'agent/prompts/system.md').write_text('changed')
    elif damage == 'dev_artifact':
        path = dev / 'cohort.json'; value = json.loads(path.read_text()); value['artifact']['sha256'] = 'f' * 64; put(path, value)
    elif damage == 'profile': scope.plan['matching']['prompt_profile'] = 'repair-v1'
    elif damage == 'sampling': scope.plan['matching']['sampling_file_sha256'] = 'f' * 64
    elif damage == 'adapter': scope.receipt['selected_packet']['adapters']['particle_lora']['rank'] = 2
    elif damage == 'budget': scope.plan['matching']['budget_profile'] = 'pilot-5m-v1'
    elif damage == 'prompt_hash': scope.plan['matching']['common_system_sha256'] = 'f' * 64
    if damage:
        with pytest.raises(ValueError): selected_evaluated_archive(scope, comparison)
    else:
        selected, receipt = selected_evaluated_archive(scope, comparison)
        assert selected == artifact and receipt['sha256'] == sha256_file(artifact / 'submission.zip')


@pytest.mark.parametrize('damage', [None, 'request', 'response', 'extra', 'result', 'protocol'])
def test_selected_effect_reaudits_every_actual_http_exchange(tmp_path, damage):
    protocol, cases = prepare_effect_probes(*protocol_fixture(tmp_path))
    packet = {'condition': {'id': 'CPU-fixture'}, 'adapters': {'particle_lora': {'fixture': True}},
        'native_metadata_sha256': 'c' * 64, 'cache_index_sha256': 'd' * 64, 'conversion_report_sha256': 'e' * 64}
    identity = {'base_only': False, 'adapters': packet['adapters']}; exchanges = []
    def call(client, root, path, payload=None):
        exchanges.append({'phase': 'attempt', 'path': path, 'request': deepcopy(payload)})
        if path == '/health': value = {'identity_sha256': digest(identity)}
        elif path == '/v1/models': value = {'data': [{'id': BASE_ALIAS}, {'id': 'particle_lora'}]}
        elif path == '/tokenize': value = {'tokens': [2, 3]}
        else:
            value = response(payload)
            if payload['model'] == 'particle_lora':
                value['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += .01
        exchanges.append({'phase': 'response', 'path': path, 'status_code': 200, 'response': value})
        return value
    candidate_preflight(None, 'CPU-fixture', identity, call)
    effect = learned_effect_preflight(None, 'CPU-fixture', identity, packet, cases, call)
    journal = tmp_path / 'journal.jsonl'; result = tmp_path / 'effect.json'
    journal.write_text(''.join(json.dumps(e) + '\n' for e in exchanges))
    effect.update(protocol_sha256='f' * 64, http_journal_sha256=sha256_file(journal)); put(result, effect)
    if damage in ['request', 'response', 'extra']:
        if damage == 'request': exchanges[-2]['request']['seed'] = 1
        elif damage == 'response': exchanges[-1]['response']['model'] = 'wrong'
        else: exchanges.extend(deepcopy(exchanges[-2:]))
        journal.write_text(''.join(json.dumps(e) + '\n' for e in exchanges))
    elif damage == 'result': effect['effect_probes'] = 0; put(result, effect)
    expected_protocol = '0' * 64 if damage == 'protocol' else 'f' * 64
    if damage:
        with pytest.raises(ValueError): audit_learned_effect(result, journal, identity, packet, cases, expected_protocol)
    else:
        assert audit_learned_effect(result, journal, identity, packet, cases, expected_protocol)['status'] == 'passed'
        assert len(exchanges) == 24


def test_real_selected_preparation_cli_rejects_unreleased_access_before_other_inputs(tmp_path):
    access = tmp_path / 'access.json'; put(access, {'schema': 'CPU incomplete access'})
    command = [str(ROOT / '.harness-venv/bin/python'), str(ROOT / 'scripts/prepare_selected_candidate.py'),
        '--access', str(access), '--environments', str(tmp_path / 'CALIBRATION_MUST_NOT_BE_READ'),
        '--pair', str(tmp_path / 'CLAIM_MUST_NOT_BE_READ'), '--comparison', str(tmp_path / 'RESULTS_MUST_NOT_BE_READ'),
        '--owner', 'fixture-owner', '--notebook-slug', 'fixture-notebook', '--out', str(tmp_path / 'out')]
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'src')))
    assert child.returncode and 'completed explicit confirmation access receipt required' in child.stderr
    assert 'FileNotFoundError' not in child.stderr and not (tmp_path / 'out').exists()
