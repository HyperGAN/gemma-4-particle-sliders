"""CPU tests for the user's own submission and its exact official bytes."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from gemma_particle.harness_identity import sha256_file
from gemma_particle.official_results import COMPETITION
from gemma_particle.user_submission import (SubmissionNotVisible, bind_user_submission,
    capture_user_reference, load_user_reference)
from test_kaggle_candidate import module


class Account:
    def __init__(self, archive, *, score='0.35'):
        self.archive = archive
        self.score = score
        self.day = datetime.now(timezone.utc) + timedelta(minutes=1)
        self.calls = []
        self.visible = True
        self.completed = False

    def row(self):
        return NS(ref=701, status='COMPLETE' if self.completed else 'PENDING',
            description='User selected their own description', file_name='submission.zip',
            public_score=self.score if self.completed else '', error_description='',
            date=self.day.replace(tzinfo=None), team_name='Test team')

    def competition_submissions(self, competition):
        assert competition == COMPETITION
        self.calls.append(('list', competition))
        return [self.row()] if self.visible else []

    def competition_download_submission(self, identifier, path, *, quiet):
        assert identifier == 701 and quiet
        self.calls.append(('download', identifier))
        (Path(path) / '701.raw').write_bytes(self.archive.read_bytes())

    def competition_leaderboard_view(self, competition, *, page_size):
        assert competition == COMPETITION and page_size == 20
        self.calls.append(('leaderboard', competition))
        prior = NS(team_id=2, team_name='Other team', score='0.29', submission_date=self.day)
        own = NS(team_id=1, team_name='Test team', score=self.score, submission_date=self.day)
        return [own, prior] if self.completed else [prior]

    def get_submission(self, identifier):
        assert identifier == 701
        self.calls.append(('get', identifier))
        return self.row()

    def competition_team_submissions(self, team):
        assert team == 1
        self.calls.append(('membership', team))
        return [NS(id=701, public_score=self.score)]


def handoff(tmp_path):
    archive = tmp_path / 'submission.zip'
    archive.write_bytes(b'fixture represents exact evaluated ZIP bytes')
    candidate = {'artifact_sha256': sha256_file(archive), 'experiment_pins': {}}
    readiness = tmp_path / 'readiness.json'
    readiness.write_text(json.dumps({'handoff_zip': str(archive)}))
    return archive, candidate, readiness


@pytest.mark.parametrize('damage', [None, 'wrong_bytes', 'old_submission', 'wrong_id', 'reference'])
def test_user_binding_requires_real_competition_membership_and_byte_match(tmp_path, damage):
    archive, candidate, readiness = handoff(tmp_path)
    api = Account(archive)
    reference = tmp_path / 'reference.json'
    capture_user_reference(api, candidate, sha256_file(readiness), reference)
    if damage == 'reference':
        value = json.loads(reference.read_text())
        value['leaderboard']['best_score'] = '0.01'
        reference.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='exact local handoff'):
            load_user_reference(reference, candidate, sha256_file(readiness))
        return
    prior = load_user_reference(reference, candidate, sha256_file(readiness))
    identifier = 701
    if damage == 'wrong_bytes':
        api.archive = tmp_path / 'other.zip'
        api.archive.write_bytes(b'another file')
    elif damage == 'old_submission':
        api.day -= timedelta(days=1)
    elif damage == 'wrong_id':
        identifier = 702
    out = tmp_path / 'observation'
    if damage:
        error = SubmissionNotVisible if damage == 'wrong_id' else ValueError
        with pytest.raises(error):
            bind_user_submission(api, identifier, candidate, reference, prior, out)
        assert (('download', 701) in api.calls) is (damage == 'wrong_bytes')
    else:
        binding = bind_user_submission(api, identifier, candidate, reference, prior, out)
        assert binding['submission_id'] == identifier and binding['artifact_sha256'] == sha256_file(archive)
        assert binding['account_mutations'] == 0
        assert binding['message'] == api.row().description
        assert binding['publication_files'][str(out / 'downloaded/701.raw')] == sha256_file(archive)


def test_monitor_waits_for_user_id_then_uses_the_same_verified_archive(tmp_path, monkeypatch):
    monitor = module('official_user_monitor_fixture', 'monitor_selected_submission.py')
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    archive, candidate, readiness = handoff(tmp_path)
    reference = tmp_path / 'reference.json'
    api = Account(archive)
    capture_user_reference(api, candidate, sha256_file(readiness), reference)
    api.visible = False
    states = []
    def sleep(_):
        states.append(json.loads((out / 'progress.json').read_text())['state'])
        if api.visible:
            api.completed = True
        api.visible = True
    monkeypatch.setattr(monitor.time, 'sleep', sleep)
    monkeypatch.setattr(monitor, 'audit_selected_publication', lambda *args: candidate)
    monkeypatch.setattr(monitor, 'load_script', lambda path, name: NS(
        notebook_source=lambda *args: None, validate_runtime=lambda *args: ({'versions': {}}, {})))
    monkeypatch.setattr(monitor.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 0))
    monkeypatch.setitem(sys.modules, 'kaggle', NS(api=api))
    out = tmp_path / 'monitor'
    argv = ['monitor', '--submission-id', '701', '--prior-reference', str(reference),
        '--readiness', str(readiness), '--out', str(out), '--watch']
    for name in ('access', 'environments', 'pair', 'comparison'):
        argv.extend(['--' + name, str(tmp_path / name)])
    monkeypatch.setattr(sys, 'argv', argv)
    monitor.main()
    assert states == ['observation_incomplete', 'waiting_official_evaluation']
    assert api.calls.count(('download', 701)) == 1
    assert api.calls.count(('get', 701)) == 2
    result = json.loads((out / 'progress.json').read_text())
    assert result['state'] == 'verified_learned_public_record'
    assert result['artifact_sha256'] == sha256_file(archive)
    assert result['verified_strict_public_record'] and not result['goal_marked_complete']


@pytest.mark.parametrize('outcome', ['none', 'tampered', 'ready'])
def test_reference_watcher_releases_only_verified_handoff(tmp_path, monkeypatch, outcome):
    watcher = module('reference_watcher_fixture', 'wait_for_user_handoff_reference.py')
    workflow = tmp_path / 'handoff-workflow'
    watcher_out = tmp_path / 'reference'
    workflow.mkdir()
    (workflow / 'plan.json').write_text(json.dumps({'schema': 'gemma4-empf-handoff-plan-v1',
        'account_writes': False}))
    handoff_progress = workflow / 'progress.json'
    handoff_progress.write_text(json.dumps({'state': 'waiting_for_completed_dev'}))
    binding = {'pid': 99, 'birth_ticks': '42'}
    active = True
    monkeypatch.setattr(watcher, 'process_binding', lambda *_: binding)
    monkeypatch.setattr(watcher, 'live', lambda observed: active and observed == binding)
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **k: 'fixture\n')
    children = []
    def run(command, **options):
        if command[0] == 'git':
            return
        assert not active and options['check'] is True and '--capture-user-reference' in command
        assert command[command.index('--pair') + 1] == str(workflow / 'fixture-claim.json')
        children.append(command)
        reference = Path(command[command.index('--out') + 1])
        reference.mkdir()
        path = reference / 'user-reference.json'
        path.write_text('{}')
        (reference / 'progress.json').write_text(json.dumps({'state': 'user_reference_ready',
            'reference_sha256': sha256_file(path), 'artifact_sha256': sha256_file(archive),
            'best_score': '0.29'}))
    monkeypatch.setattr(watcher.subprocess, 'run', run)
    def complete(_):
        nonlocal active
        assert not children
        assert json.loads((watcher_out / 'progress.json').read_text())['account_reads_started'] is False
        active = False
        if outcome == 'none':
            handoff_progress.write_text(json.dumps({'state': 'no_eligible_dev_choice'}))
            return
        handoff_progress.write_text(json.dumps({'state': 'handoff_ready',
            'readiness': str(readiness), 'submission_zip': str(archive),
            'artifact_sha256': 'wrong digest' if outcome == 'tampered' else sha256_file(archive),
            'readiness_sha256': sha256_file(readiness)}))
    monkeypatch.setattr(watcher.time, 'sleep', complete)
    local = workflow / 'handoff'
    local.mkdir()
    archive, _, readiness = handoff(local)
    pair = workflow / 'pair'
    pair.mkdir()
    (pair / 'plan.json').write_text(json.dumps({'pair_claim': str(workflow / 'fixture-claim.json')}))
    monkeypatch.setattr(sys, 'argv', ['watcher', '--handoff', str(workflow), '--handoff-pid', '99',
        '--executor-commit', 'fixture', '--out', str(watcher_out)])
    if outcome == 'tampered':
        with pytest.raises(ValueError, match='verified ready receipt'):
            watcher.main()
    else:
        watcher.main()
    final = json.loads((watcher_out / 'progress.json').read_text())
    assert final['state'] == {'none': 'no_eligible_dev_choice', 'tampered': 'error',
        'ready': 'user_reference_ready'}[outcome]
    assert len(children) == (1 if outcome == 'ready' else 0)
