import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from gemma_particle.harness_identity import sha256_file


script = Path(__file__).resolve().parents[1] / 'scripts/run_empf_handoff.py'
spec = importlib.util.spec_from_file_location('run_empf_handoff', script)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


@pytest.mark.parametrize('mode', ['pending', 'no_choice', 'fetch_error', 'wrong_zip', 'ready'])
def test_handoff_keeps_confirmation_gated_and_verifies_the_zip(tmp_path, monkeypatch, mode):
    grid = tmp_path / 'grid'
    packets = tmp_path / 'packets'
    out = tmp_path / 'workflow'
    grid.mkdir()
    packets.mkdir()
    policy = grid / 'qualified-resource.json'
    plan = tmp_path / 'selection.json'
    protected = tmp_path / 'protected-manifest-must-stay-absent'
    kaggle = tmp_path / 'kaggle/bin/python'
    handoff.write(plan, {'conditions': [
        {'id': f'empf-{i}', 'role': 'learned', 'preset': 'empf'} for i in range(9)
    ], 'confirmation': {'count': 49}})
    binding = {'pid': 123, 'birth_ticks': '456'}
    grid_plan = {
        'condition_ids': [row['id'] for row in handoff.read(plan)['conditions']],
        'selection_plan_sha256': sha256_file(plan), 'executor': 'dev', 'executor_commit': 'base',
        'offline_auditor': 'audit', 'offline_auditor_commit': 'audit-commit',
        'resource_supplement': None, 'producers': [{'directory': str(packets)}],
    }
    handoff.write(grid / 'plan.json', grid_plan)
    handoff.write(packets / 'plan.json', {'selection_plan_sha256': sha256_file(plan)})
    handoff.write(grid / 'progress.json', {'state': 'waiting_actual_inputs'})
    monkeypatch.setattr(handoff, 'require_checkout', lambda *_: None)
    monkeypatch.setattr(handoff, 'load_selection_plan', handoff.read)
    monkeypatch.setattr(handoff, 'process_binding', lambda *_: binding)
    active = True
    monkeypatch.setattr(handoff, 'live', lambda actual: active and actual == binding)
    commands = []

    def complete_dev(_):
        nonlocal active
        assert handoff.read(out / 'progress.json')['state'] == 'waiting_for_completed_dev'
        assert not commands
        active = False
        (grid / 'choice').mkdir()
        choice = grid / 'choice/selection.json'
        selected = None if mode == 'no_choice' else 'empf-4'
        handoff.write(choice, {'selected': selected})
        handoff.write(grid / 'progress.json', {
            'state': 'complete_dev_choice', 'selected': selected, 'selection_sha256': sha256_file(choice),
        })
        handoff.write(policy, {'measured': True})
        grid_plan['resource_supplement'] = {'path': str(policy), 'sha256': sha256_file(policy)}
        handoff.write(grid / 'plan.json', grid_plan)

    def stage(command, **options):
        commands.append(command[1].rsplit('/', 1)[-1])
        assert options['check'] is True
        assert not active
        if commands[-1] == 'fetch_official_inputs.py':
            assert command[0] == str(kaggle)
            if mode == 'fetch_error':
                raise subprocess.CalledProcessError(1, command)
        if commands[-1] == 'run_confirmation_pair.py':
            (out / 'pair').mkdir()
            handoff.write(out / 'pair/plan.json', {'pair_claim': str(tmp_path / 'anchored-claim.json')})
        if commands[-1] == 'prepare_selected_candidate.py':
            assert command[command.index('--pair') + 1] == str(tmp_path / 'anchored-claim.json')
            (out / 'handoff').mkdir()
            archive = out / 'handoff/submission.zip'
            archive.write_bytes(b'fixture archive from the selected packet')
            handoff.write(out / 'handoff/readiness.json', {
                'status': 'prepared', 'handoff_zip': str(archive), 'account_requests': False,
                'artifact_sha256': 'wrong' if mode == 'wrong_zip' else sha256_file(archive),
            })

    monkeypatch.setattr(handoff.time, 'sleep', complete_dev)
    monkeypatch.setattr(handoff.subprocess, 'run', stage)
    monkeypatch.setattr(sys, 'argv', [str(handoff.__file__), '--grid', str(grid), '--grid-pid', '123',
        '--packets', str(packets), '--selection-plan', str(plan), '--manifest', str(protected),
        '--wheels', str(tmp_path / 'wheels'), '--kaggle-python', str(kaggle), '--out', str(out),
        '--executor-commit', 'test', '--owner', 'test', '--notebook-slug', 'test',
        *(['--plan-only'] if mode == 'pending' else [])])
    if mode == 'fetch_error':
        with pytest.raises(subprocess.CalledProcessError):
            handoff.main()
    elif mode == 'wrong_zip':
        with pytest.raises(ValueError, match='differs from the evaluated artifact'):
            handoff.main()
    else:
        handoff.main()
    progress = handoff.read(out / 'progress.json')
    assert not protected.exists()
    if mode in {'pending', 'no_choice'}:
        assert not commands
        assert progress['state'] == ('prepared' if mode == 'pending' else 'no_eligible_dev_choice')
        assert progress['protected_inputs_opened'] is False
    elif mode == 'fetch_error':
        assert commands == ['unlock_resource_confirmation.py', 'fetch_official_inputs.py']
        assert progress['state'] == 'error'
        assert progress['stage'] == 'downloading_confirmation_inputs'
    else:
        assert commands == ['unlock_resource_confirmation.py', 'fetch_official_inputs.py',
            'verify_confirmation_environments.py', 'run_confirmation_pair.py', 'prepare_selected_candidate.py']
        assert progress['state'] == ('handoff_ready' if mode == 'ready' else 'error')
        if mode == 'ready':
            assert progress['artifact_sha256'] == sha256_file(out / 'handoff/submission.zip')
            assert progress['account_writes'] is False
