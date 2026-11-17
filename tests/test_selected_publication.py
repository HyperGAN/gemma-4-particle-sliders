"""Selected publication wiring with explicit CPU experiment-boundary fixtures."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
import shutil
import sys

import pytest

from gemma_particle.data import digest
from gemma_particle.dev_dispatch import candidate_preflight
from gemma_particle.harness_identity import sha256_file
from gemma_particle.learned_effect import prepare_effect_probes, learned_effect_preflight
from gemma_particle.lora_serving import BASE_ALIAS
from gemma_particle.selected_publication import audit_selected_publication
from test_conversion_handoff import put
from test_kaggle_candidate import module
from test_learned_effect import protocol_fixture, response
from test_selected_candidate import archive_fixture

ROOT = Path(__file__).resolve().parents[1]


def publication_fixture(root, monkeypatch):
    import gemma_particle.selected_publication as audit
    scope, comparison, artifact, _ = archive_fixture(root)
    preparer = module('selected_publication_payload_fixture', 'prepare_kaggle_candidate.py')
    payload = root / 'payload'
    conversion = root / 'fixture-conversion.json'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--artifact', str(artifact), '--native', str(root / 'fixture-native'),
        '--cache', str(root / 'fixture-cache'), '--conversion-report', str(conversion),
        '--owner', 'fixture-owner', '--notebook-slug', 'fixture-candidate', '--out', str(payload)])
    preparer.main(); prepared = json.loads((payload / 'preparation.json').read_text())
    packet = scope.receipt['selected_packet']
    packet.update({k: prepared[k] for k in ['native_metadata_sha256', 'cache_index_sha256', 'conversion_report_sha256']})
    protocol_args = protocol_fixture(root)
    protocol = root / 'learned-effect-protocol-v1.json'; qualification = root / 'conversion-probe-preparation-v1.json'
    shutil.copyfile(protocol_args[0], protocol); shutil.copyfile(protocol_args[2], qualification)
    contexts = protocol_args[1]
    _, cases = prepare_effect_probes(protocol, contexts, qualification, protocol_args[3], protocol_args[4])
    scope.plan.update(protocol_args[3]); scope.identity = {'selection_plan_sha256': protocol_args[4], 'CPU-fixture': True}
    grid = root / 'grid'; grid.mkdir()
    access_plan = root / 'access-plan.json'; put(access_plan, {'dev_grid': str(grid)})
    scope.receipt.update(access_plan=str(access_plan), permitted_conditions=['off', packet['condition']['id']])
    off = root / 'off-result.json'; put(off, {'CPU-fixture': True})
    comparison['condition_results']['off'] = {'path': str(off), 'sha256': sha256_file(off)}
    put(grid / 'plan.json', {'declaration_pins': {str(path): sha256_file(path) for path in [protocol, contexts, qualification]}})
    identity = {'base_only': False, 'adapters': packet['adapters']}
    exchanges = []
    def call(client, base, path, request=None):
        exchanges.append({'phase': 'attempt', 'path': path, 'request': deepcopy(request)})
        if path == '/health': value = {'identity_sha256': digest(identity)}
        elif path == '/v1/models': value = {'data': [{'id': BASE_ALIAS}, {'id': 'particle_lora'}]}
        elif path == '/tokenize': value = {'tokens': [2, 3]}
        else:
            value = response(request)
            if request['model'] == 'particle_lora': value['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += .01
        exchanges.append({'phase': 'response', 'path': path, 'status_code': 200, 'response': value})
        return value
    candidate_preflight(None, 'CPU fixture', identity, call)
    effect = learned_effect_preflight(None, 'CPU fixture', identity, packet, cases, call)
    name = packet['condition']['id']; journal = grid / (name + '-preflight-http.jsonl')
    journal.write_text(''.join(json.dumps(v) + '\n' for v in exchanges))
    effect.update(protocol_sha256=sha256_file(protocol), http_journal_sha256=sha256_file(journal))
    effect_path = grid / (name + '-effect.json'); put(effect_path, effect)
    put(grid / (name + '-server/server-identity.json'), identity)
    put(grid / 'progress.json', {'finished': [{'condition_id': name, 'learned_effect_sha256': sha256_file(effect_path)}]})
    access, environments, pair, comparison_path = [root / (name + '.json') for name in ['access', 'environments', 'claim', 'comparison']]
    for path in [access, environments, pair]: put(path, {'CPU-fixture': True})
    put(comparison_path, comparison)
    names = ['scripts/prepare_selected_candidate.py', 'scripts/prepare_kaggle_candidate.py',
        'src/gemma_particle/selected_candidate.py', 'src/gemma_particle/learned_effect.py',
        'src/gemma_particle/confirmation.py', 'src/gemma_particle/confirmation_results.py',
        'src/gemma_particle/confirmation_environments.py']
    readiness = root / 'readiness.json'
    put(readiness, {'schema': 'gemma4-selected-candidate-ready-v1', 'status': 'prepared', 'confirmation': scope.identity,
        'comparison_sha256': sha256_file(comparison_path), 'evaluated_artifact': str(artifact),
        'artifact_sha256': prepared['artifact_sha256'], 'fixed_dev_choice_retained': True, 'new_candidate_selected': False,
        'account_requests': False, 'model_inference': False, 'official_score': None,
        'sources': {str(ROOT / name): sha256_file(ROOT / name) for name in names}, 'payload': str(payload),
        'payload_preparation_sha256': sha256_file(payload / 'preparation.json'), 'learned_effect_sha256': sha256_file(effect_path)})
    # Only the existing access/environment/pair validators are supplied boundaries
    # here. Archive bytes, original24 HTTP exchanges and payload auditing are real.
    monkeypatch.setattr(audit, 'load_confirmation_access', lambda path: scope)
    monkeypatch.setattr(audit, 'load_confirmation_environments', lambda path, given: {'CPU-fixture': True})
    monkeypatch.setattr(audit, 'compare_confirmation_pair', lambda *args: comparison)
    return (readiness, access, environments, pair, comparison_path, ROOT, preparer.notebook_source), scope, grid


@pytest.mark.parametrize('damage', [None, 'readiness', 'source', 'effect', 'payload', 'packet', 'protocol'])
def test_publication_reaudits_selected_archive_effect_payload_and_readiness(tmp_path, monkeypatch, damage):
    args, scope, grid = publication_fixture(tmp_path, monkeypatch)
    if damage in ['readiness', 'source']:
        value = json.loads(args[0].read_text())
        if damage == 'readiness': value['new_candidate_selected'] = True
        else: value['sources'][str(ROOT / 'scripts/prepare_kaggle_candidate.py')] = 'a' * 64
        put(args[0], value)
    elif damage == 'effect':
        p = grid / (scope.receipt['selected_packet']['condition']['id'] + '-preflight-http.jsonl')
        rows = [json.loads(v) for v in p.read_text().splitlines()]; rows[-2]['request']['seed'] = 999
        p.write_text(''.join(json.dumps(v) + '\n' for v in rows))
    elif damage == 'payload': (tmp_path / 'payload/dataset/agent_bundle.bin').write_bytes(b'changed')
    elif damage == 'packet': scope.receipt['selected_packet']['condition']['strength'] = .5
    elif damage == 'protocol': (tmp_path / 'learned-effect-protocol-v1.json').write_text('{}')
    if damage:
        with pytest.raises(ValueError): audit_selected_publication(*args)
    else:
        result = audit_selected_publication(*args)
        assert result['artifact_sha256'] == sha256_file(Path(json.loads(args[0].read_text())['evaluated_artifact']) / 'submission.zip')
        assert result['experiment_pins'][str(args[0])] == sha256_file(args[0])


@pytest.mark.parametrize('plan_only', [False, True])
def test_actual_cli_gates_all_protected_inputs_before_any_kaggle_import(tmp_path, monkeypatch, plan_only):
    publisher = module('selected_publication_gate', 'publish_selected_candidate.py')
    monkeypatch.chdir(ROOT)
    access = tmp_path / 'access.json'; put(access, {'schema': 'CPU unreleased access'})
    out = tmp_path / 'out'; calls = []; original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'kaggle' or name.startswith('kaggle.') or name.startswith('kagglesdk'):
            calls.append(name); raise AssertionError('account import before experiment qualification')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    argv = ['publish', '--readiness', str(tmp_path / 'READINESS_MUST_NOT_BE_READ'), '--access', str(access),
        '--environments', str(tmp_path / 'ENVIRONMENTS_MUST_NOT_BE_READ'), '--pair', str(tmp_path / 'PAIR_MUST_NOT_BE_READ'),
        '--comparison', str(tmp_path / 'COMPARISON_MUST_NOT_BE_READ'), '--out', str(out)]
    if plan_only: argv.append('--plan-only')
    monkeypatch.setattr(sys, 'argv', argv)
    if plan_only:
        publisher.main()
        assert json.loads((out / 'progress.json').read_text())['account_requests'] == 0
    else:
        with pytest.raises(ValueError, match='completed explicit confirmation access'): publisher.main()
        assert not out.exists()
    assert not calls
