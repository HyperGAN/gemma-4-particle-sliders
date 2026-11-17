"""Synthetic CPU access fixtures; no real holdout or model results are read."""
import json
from pathlib import Path

import pytest

import gemma_particle.confirmation as confirmation
from gemma_particle.data import VISIBLE_FIELDS, write_jsonl
from gemma_particle.harness_identity import sha256_file
from test_conversion_handoff import put
from test_dev_dispatch import packet_fixture


def access_fixture(root):
    matrix, plan, plan_sha, _, _ = packet_fixture(root)
    controls = [{'id': name, 'role': 'off' if name == 'off' else 'prompt-control', 'arm': 'off'}
        for name in ['off', 'prompt-reproduce', 'prompt-localize', 'prompt-verify']]
    plan['conditions'] = controls + plan['conditions']
    manifest = root / 'synthetic-holdout.jsonl'
    rows = [{**{k: 'CPU synthetic fixture' for k in VISIBLE_FIELDS}, 'instance_id': f'fixture-holdout-{i}',
        'repo': 'Textualize/rich' if i < 48 else 'encode/httpx', 'split': 'holdout'} for i in range(49)]
    write_jsonl(manifest, rows)
    plan['confirmation'] = {'count': 49, 'manifest_sha256': sha256_file(manifest)}
    packets = json.loads((matrix / 'candidate-packets.json').read_text())['packets']
    by_id = {p['condition']['id']: p for p in packets}
    selected_id = packets[0]['condition']['id']
    grid = root / 'dev-grid'; files = {}; finished = []
    for condition in plan['conditions']:
        name = condition['id']; packet = by_id.get(name)
        bound = None
        if packet:
            bound = {k: packet[k] for k in ['adapters', 'native_metadata_sha256',
                'cache_index_sha256', 'conversion_report_sha256', 'conversion_report']}
            bound.update(native_metadata=str(Path(packet['native']) / 'native.json'),
                cache_index=str(Path(packet['cache']) / 'index.json'))
        path = grid / name / 'result.json'
        put(path, {'schema': 'gemma4-dev-condition-result-v1', 'status': 'complete',
            'condition': condition, 'selection_plan_sha256': plan_sha, 'packet': bound,
            'fixture': 'CPU structure only; no actual DEV outcome'})
        files[name] = {'path': str(path), 'sha256': sha256_file(path)}
        if packet:
            finished.append({'condition_id': name, 'result_sha256': sha256_file(path)})
    choice = {'schema': 'gemma4-locked-dev-selection-v1', 'selection_plan_sha256': plan_sha,
        'selected': selected_id, 'eligible': [selected_id], 'advance_to_locked_confirmation': True,
        'condition_results': files, 'model_dev_conditions': 22, 'holdout_accessed': False,
        'fixture': 'synthetic choice; never an execution authentication claim'}
    choice_path = grid / 'choice/selection.json'; put(choice_path, choice)
    put(grid / 'progress.json', {'state': 'complete_dev_choice', 'selected': selected_id,
        'planned_conditions': 18, 'finished': finished, 'selection_sha256': sha256_file(choice_path)})
    reaudit = root / 'synthetic-reaudit.json'; reaudit.write_bytes(choice_path.read_bytes())
    return grid, plan, plan_sha, reaudit, manifest, matrix


def test_exact_choice_and_selected_packet_release_only_sanitized_original_panel(tmp_path):
    args = access_fixture(tmp_path)
    access = confirmation.build_confirmation_access(*args)
    assert access['count'] == 49 and access['model_inference'] is False
    assert access['permitted_conditions'] == ['off', access['selected_condition']['id']]
    assert access['forbidden_training_and_selection'] is True
    assert access['confirmation_results'] is None
    assert access['dev_choice_sha256'] == access['reaudited_dev_choice_sha256']


@pytest.mark.parametrize('damage', ['running', 'missing_condition', 'changed_result', 'no_choice',
    'off_choice', 'reaudit', 'packet_order', 'adapter_bytes'])
def test_invalid_choice_or_packet_never_parses_holdout(tmp_path, monkeypatch, damage):
    args = access_fixture(tmp_path); grid, _, _, reaudit, _, matrix = args
    choice_path = grid / 'choice/selection.json'; choice = json.loads(choice_path.read_text())
    status_path = grid / 'progress.json'; status = json.loads(status_path.read_text())
    if damage == 'running': status['state'] = 'running_full_dev_condition'
    elif damage == 'missing_condition': choice['condition_results'].pop('off')
    elif damage == 'changed_result': Path(choice['condition_results']['off']['path']).write_text('changed CPU fixture')
    elif damage == 'no_choice':
        choice.update(selected=None, eligible=[], advance_to_locked_confirmation=False); status['selected'] = None
    elif damage == 'off_choice':
        choice.update(selected='off', eligible=['off']); status['selected'] = 'off'
    elif damage == 'reaudit': reaudit.write_text('different CPU fixture audit')
    elif damage == 'packet_order':
        path = matrix / 'candidate-packets.json'; value = json.loads(path.read_text())
        value['packets'][0], value['packets'][1] = value['packets'][1], value['packets'][0]; put(path, value)
        progress = json.loads((matrix / 'progress.json').read_text()); progress['candidate_packets_sha256'] = sha256_file(path)
        put(matrix / 'progress.json', progress)
    else:
        packet = json.loads((matrix / 'candidate-packets.json').read_text())['packets'][0]
        (Path(packet['adapter']) / 'adapter_model.safetensors').write_bytes(b'changed CPU fixture weights')
    put(choice_path, choice); status['selection_sha256'] = sha256_file(choice_path); put(status_path, status)
    if damage not in ['reaudit']: reaudit.write_bytes(choice_path.read_bytes())
    def forbidden(path):
        raise AssertionError('holdout content must remain unopened')
    monkeypatch.setattr(confirmation, 'read_jsonl', forbidden)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        confirmation.build_confirmation_access(*args)


@pytest.mark.parametrize('damage', ['reference_field', 'duplicate_id', 'train_repo', 'missing_field'])
def test_even_rebound_manifest_must_be_complete_original_visible_holdout(tmp_path, damage):
    args = access_fixture(tmp_path); plan, manifest = args[1], args[4]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if damage == 'reference_field': rows[0]['test_patch'] = 'CPU forbidden field'
    elif damage == 'duplicate_id': rows[1]['instance_id'] = rows[0]['instance_id']
    elif damage == 'train_repo': rows[0]['repo'] = 'fastapi/fastapi'
    else: rows[0].pop('instance_id')
    write_jsonl(manifest, rows); plan['confirmation']['manifest_sha256'] = sha256_file(manifest)
    with pytest.raises(ValueError, match='sanitized forty-nine'):
        confirmation.build_confirmation_access(*args)


def test_changed_locked_manifest_is_rejected_before_parsing(tmp_path, monkeypatch):
    args = access_fixture(tmp_path); args[4].write_text('changed synthetic manifest')
    def forbidden(path): raise AssertionError('changed manifest must not be parsed')
    monkeypatch.setattr(confirmation, 'read_jsonl', forbidden)
    with pytest.raises(ValueError, match='manifest changed'):
        confirmation.build_confirmation_access(*args)
