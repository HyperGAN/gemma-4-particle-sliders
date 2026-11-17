"""CPU confirmation access/one-run controls; no real holdout or model is used."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from gemma_particle.confirmation import (build_confirmation_access, load_confirmation_access,
    validate_confirmation_treatment, claim_confirmation_pair, reserve_confirmation_task,
    validate_confirmation_reservation, validate_confirmation_claim, confirmation_pair_identity,
    validate_confirmation_server)
from gemma_particle.capture import CONTEXT_CAPTURE_PROTOCOL
from gemma_particle.data import digest
from gemma_particle.dev import validate_agent_tasks
from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import BASE_ALIAS, BASE_REPO, BASE_REVISION
from test_confirmation import access_fixture
from test_conversion_handoff import put
from test_dev_evidence import contract

ROOT = Path(__file__).resolve().parents[1]


def execution_fixture(root):
    args = access_fixture(root); grid, plan, plan_sha, reaudit, manifest, matrix = args
    choice_path = grid / 'choice/selection.json'; choice = json.loads(choice_path.read_text())
    off = Path(choice['condition_results']['off']['path']); result = json.loads(off.read_text())
    baseline = contract()['model_identity']; baseline.update(base_only=True, adapters={},
        model_identity={'repo': BASE_REPO, 'revision': BASE_REVISION})
    baseline_path = root / 'CPU-baseline-server.json'; put(baseline_path, baseline)
    put(off.parent / 'plan.json', {'server_identity': str(baseline_path)})
    result['server_identity_sha256'] = digest(baseline); put(off, result)
    choice['condition_results']['off']['sha256'] = sha256_file(off); put(choice_path, choice)
    status_path = grid / 'progress.json'; status = json.loads(status_path.read_text())
    status['selection_sha256'] = sha256_file(choice_path); put(status_path, status)
    reaudit.write_bytes(choice_path.read_bytes())
    access = build_confirmation_access(*args)
    folder = root / 'access'; folder.mkdir(); log = folder / 'dev-reaudit.log'
    log.write_text('CPU synthetic fixture only; no actual model DEV audit\n')
    plan_path = folder / 'plan.json'; selection_path = root / 'selection-plan.json'
    put(plan_path, {'schema': 'gemma4-confirmation-access-plan-v1', 'selection_plan_sha256': plan_sha,
        'dev_executor_commit': 'CPU-fixture-not-real-executor', 'dev_grid': str(grid), 'packet_producer': str(matrix),
        'declaration_pins': {str(selection_path): plan_sha}})
    access.update(selection_plan=str(selection_path), access_plan=str(plan_path), access_plan_sha256=sha256_file(plan_path),
        offline_dev_auditor_executor_commit='CPU-fixture-not-real-executor',
        offline_dev_audit_log_sha256=sha256_file(log), offline_dev_auditor_returncode=0)
    path = folder / 'access.json'; put(path, access)
    return path, load_confirmation_access(path)


def profiles(scope):
    m = scope.plan['matching']
    return {**{k: m[k] for k in ['prompt_profile', 'instruction_protocol', 'budget_profile', 'history_profile']},
        'model': BASE_ALIAS, 'arm': 'off', 'capture_protocol': CONTEXT_CAPTURE_PROTOCOL, 'source_root': ROOT}


def test_only_full_explicit_holdout_scope_is_admitted_without_relabeling(tmp_path):
    _, scope = execution_fixture(tmp_path)
    with pytest.raises(ValueError): validate_agent_tasks(scope.rows)
    validate_agent_tasks(scope.rows, confirmation=scope)
    assert all(row['split'] == 'holdout' for row in scope.rows)
    with pytest.raises(ValueError, match='complete locked'): validate_agent_tasks(scope.rows[:1], confirmation=scope)
    with pytest.raises(ValueError, match='combined'): validate_agent_tasks(scope.rows, object(), confirmation=scope)
    changed = deepcopy(scope.rows); changed[0]['patch'] = 'forbidden CPU reference fixture'
    with pytest.raises(ValueError, match='allowlist'): validate_agent_tasks(changed, confirmation=scope)


@pytest.mark.parametrize('damage', ['choice', 'auditor_log', 'access_plan', 'packet'])
def test_changed_access_provenance_cannot_unlock_an_execution(tmp_path, damage):
    path, scope = execution_fixture(tmp_path)
    if damage == 'choice': Path(scope.receipt['dev_choice']).write_text('changed synthetic choice')
    elif damage == 'auditor_log': (path.parent / 'dev-reaudit.log').write_text('changed CPU audit')
    elif damage == 'access_plan': (path.parent / 'plan.json').write_text('changed CPU access plan')
    else: (Path(scope.receipt['selected_packet']['adapter']) / 'adapter_model.safetensors').write_bytes(b'changed CPU fixture')
    with pytest.raises(ValueError): load_confirmation_access(path)


def test_confirmation_preserves_locked_profiles_and_only_selected_adapter(tmp_path):
    _, scope = execution_fixture(tmp_path); options = profiles(scope)
    assert validate_confirmation_treatment(scope, adapter=None, **options) == 'off'
    adapter = Path(scope.receipt['selected_packet']['adapter'])
    assert validate_confirmation_treatment(scope, adapter=adapter, **options) == scope.receipt['selected_condition']['id']
    for field, value in [('arm', 'verify'), ('model', 'different'), ('prompt_profile', 'repair-v1'),
        ('budget_profile', 'pilot-5m-v1'), ('history_profile', 'none')]:
        with pytest.raises(ValueError, match='exact locked'):
            validate_confirmation_treatment(scope, adapter=None, **{**options, field: value})
    (adapter / 'adapter_model.safetensors').write_bytes(b'other CPU weights')
    with pytest.raises(ValueError, match='exact selected'):
        validate_confirmation_treatment(scope, adapter=adapter, **options)


def test_one_pair_and_every_case_reservation_are_exclusive_and_verifiable(tmp_path):
    _, scope = execution_fixture(tmp_path); out = tmp_path / 'pair'
    claim = claim_confirmation_pair(scope, out)
    with pytest.raises(FileExistsError): claim_confirmation_pair(scope, tmp_path / 'second-pair')
    another_choice = replace(scope, identity={**scope.identity, 'dev_choice_sha256': 'b' * 64})
    with pytest.raises(FileExistsError): claim_confirmation_pair(another_choice, tmp_path / 'another-choice-same-panel')
    identity = confirmation_pair_identity(scope, claim)
    assert identity['pair_claim_sha256'] == sha256_file(claim)
    for condition in scope.receipt['permitted_conditions']:
        for row in scope.rows:
            cohort = out / condition / (row['instance_id'] + '-agent')
            sha = reserve_confirmation_task(scope, claim, condition, row['instance_id'], cohort)
            assert validate_confirmation_reservation(scope, claim, condition, row['instance_id'], cohort) == sha
            with pytest.raises(FileExistsError):
                reserve_confirmation_task(scope, claim, condition, row['instance_id'], cohort)
    with pytest.raises(ValueError, match='fixed paired'):
        reserve_confirmation_task(scope, claim, 'different', scope.rows[0]['instance_id'], out / 'different')
    with pytest.raises(ValueError, match='fixed paired'):
        reserve_confirmation_task(scope, claim, 'off', 'unknown-task', out / 'unknown')
    copied = tmp_path / 'copied-claim.json'; copied.write_bytes(claim.read_bytes())
    with pytest.raises(ValueError, match='anchored'): validate_confirmation_claim(scope, copied)


def test_candidate_and_base_aliases_keep_the_original_baseline_engine_contract(tmp_path):
    _, scope = execution_fixture(tmp_path)
    choice = json.loads(Path(scope.receipt['dev_choice']).read_text())
    off = Path(choice['condition_results']['off']['path'])
    baseline = json.loads((tmp_path / 'CPU-baseline-server.json').read_text())
    selected = deepcopy(baseline); selected.update(base_only=False, adapters=scope.receipt['selected_packet']['adapters'])
    selected['launch'].extend(['--lora-modules', json.dumps({'name': 'particle_lora', 'path': '/CPU-selected', 'base_model_name': BASE_ALIAS})])
    validate_confirmation_server(scope, baseline, learned=False)
    validate_confirmation_server(scope, selected, learned=False)
    validate_confirmation_server(scope, selected, learned=True)
    with pytest.raises(ValueError): validate_confirmation_server(scope, baseline, learned=True)
    selected['device'] = 'cuda:1'
    with pytest.raises(ValueError, match='matched DEV'): validate_confirmation_server(scope, selected, learned=True)


@pytest.mark.parametrize('script', ['run_trajectories.py', 'verify_trajectories.py', 'verify_controls.py'])
def test_real_cli_rejects_incomplete_access_before_task_or_reference_reads(tmp_path, script):
    access = tmp_path / 'incomplete-access.json'; put(access, {'schema': 'CPU incomplete fixture'})
    forbidden = tmp_path / 'TASKS_MUST_NOT_BE_READ.jsonl'
    command = [str(ROOT / '.harness-venv/bin/python'), str(ROOT / 'scripts' / script),
        '--tasks', str(forbidden), '--wheels', str(tmp_path), '--out', str(tmp_path / 'output'),
        '--confirmation-access', str(access)]
    if script == 'run_trajectories.py':
        command += ['--task-id', 'fixture-held-task', '--arm', 'off', '--confirmation-pair', str(tmp_path / 'claim.json')]
    elif script == 'verify_trajectories.py':
        command += ['--cohort', str(tmp_path / 'COHORT_MUST_NOT_BE_READ'), '--image', 'CPU-unused-image',
            '--confirmation-pair', str(tmp_path / 'claim.json')]
    else: command += ['--split', 'holdout']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1')
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'completed explicit confirmation access receipt required' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    assert not (tmp_path / 'output').exists()
