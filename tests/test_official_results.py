"""CPU official-record fixtures, never real accepted learned submissions."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from requests.exceptions import Timeout

from gemma_particle.harness_identity import sha256_file
from gemma_particle.kaggle_publication import publish_once
from gemma_particle.official_results import (score_value, leaderboard_snapshot, audit_publication,
    compare_official_record, observe_official_record, official_submission)
from test_kaggle_candidate import module
from test_kaggle_publication import prepared, FakeAPI

NOW = datetime.now(timezone.utc).isoformat()


def snapshot(values):
    return leaderboard_snapshot([NS(team_id=index, team_name='CPU team ' + str(index), score=score,
        submission_date=NOW) for index, score in values], NOW, NOW)


def binding():
    return {'kind': 'learned', 'submission_id': 701, 'artifact_sha256': 'a' * 64,
        'message': 'CPU learned submission fixture', 'prior_best_score': '0.29'}


def submission(score='0.35', status='COMPLETE'):
    return {'submission_id': 701, 'description': binding()['message'], 'file_name': 'submission.zip',
        'status': status, 'public_score': score, 'error_present': False, 'date': NOW, 'team_name': 'CPU team1'}


@pytest.mark.parametrize('mode,expected', [('unlisted', False), ('own_displayed', True), ('wrong_member', False),
    ('tie', False), ('new_other_record', False), ('old_stronger_reference', False), ('decimal_tie', False),
    ('control', None), ('wrong_id', None), ('error', False), ('pending', False), ('unknown', False), ('complete_error', False)])
def test_strict_official_record_cannot_be_a_tie_control_pending_or_wrong_submission(mode, expected):
    reference = binding(); row = submission(); current = snapshot([(2, '0.29'), (3, '0.24')]); members = {}
    if mode in ['own_displayed', 'wrong_member', 'tie']:
        current = snapshot([(1, '0.35'), (2, '0.35' if mode == 'tie' else '0.29')])
        members = {1: [{'submission_id': 702 if mode == 'wrong_member' else 701, 'public_score': '0.35'}]}
    elif mode == 'new_other_record': current = snapshot([(2, '0.40'), (3, '0.29')])
    elif mode == 'old_stronger_reference': reference['prior_best_score'] = '0.40'
    elif mode == 'decimal_tie': row['public_score'] = '0.290000000000'
    elif mode == 'control': reference['kind'] = 'control'
    elif mode == 'wrong_id': row['submission_id'] = 999
    elif mode == 'error': row['status'] = 'ERROR'; current = None
    elif mode in ['pending', 'unknown']: row['status'] = 'PENDING' if mode == 'pending' else 'NEW_UNKNOWN_STATUS'; row['public_score'] = ''; current = None
    elif mode == 'complete_error': row['error_present'] = True; current = None
    if expected is None:
        with pytest.raises(ValueError): compare_official_record(reference, row, current, members)
    else:
        result = compare_official_record(reference, row, current, members)
        assert result['verified_strict_public_record'] is expected and result['goal_marked_complete'] is False
        assert result['verified_private_record'] is False
        if mode == 'unlisted':
            assert result['state'] == 'waiting_public_leaderboard_identity'
            assert result['score_above_observed_public_values'] is True
        if mode == 'wrong_member':
            assert result['state'] == 'waiting_public_leaderboard_identity'
        if mode in ['pending', 'unknown']: assert result['state'] == 'waiting_official_evaluation'


@pytest.mark.parametrize('value', ['', 'NaN', 'Infinity', '-Infinity', 'not a score', 0.35])
def test_missing_or_nonfinite_official_scores_are_never_inferred(value):
    with pytest.raises(ValueError): score_value(value)


@pytest.mark.parametrize('damage', ['empty', 'order', 'duplicate', 'missing', 'too_many'])
def test_public_reference_is_a_real_ordered_first_page(damage):
    rows = [NS(team_id=1, team_name='a', score='0.29', submission_date=NOW), NS(team_id=2, team_name='b', score='0.24', submission_date=NOW)]
    if damage == 'empty': rows = []
    elif damage == 'order': rows.reverse()
    elif damage == 'duplicate': rows[1].team_id = 1
    elif damage == 'missing': rows[1] = None
    else: rows = rows * 11
    with pytest.raises(ValueError): leaderboard_snapshot(rows, NOW, NOW)


@pytest.mark.parametrize('damage', [None, 'response', 'reference', 'version', 'archive', 'calls', 'role'])
def test_monitor_requires_exact_created_publication_and_prior_reference(tmp_path, monkeypatch, damage):
    candidate, artifact, _ = prepared(tmp_path, monkeypatch); out = tmp_path / 'published'
    api = FakeAPI(candidate, artifact / 'submission.zip', out, 'complete')
    plan = {'schema': 'gemma4-learned-publication-plan-v1', 'readiness_sha256': 'b' * 64,
        'artifact_sha256': candidate['artifact_sha256'], 'dataset': candidate['dataset'], 'notebook': candidate['notebook'],
        'sources': {}}
    publish_once(api, candidate, out, lambda: None, sleep=lambda seconds: None, declaration=plan)
    name = 'submission-response.json' if damage == 'response' else 'leaderboard-before-submission.json' if damage == 'reference' else 'submission-intent.json' if damage == 'version' else 'progress.json' if damage == 'calls' else 'plan.json'
    path = out / name; value = json.loads(path.read_text())
    if damage == 'response': value['submission_id'] = 999
    elif damage == 'reference': value['best_score'] = '0.01'
    elif damage == 'version': value['version'] = 2
    elif damage == 'archive': (out / 'downloaded/submission.zip').write_bytes(b'changed')
    elif damage == 'calls': value['submission_calls'] = 2
    elif damage == 'role': value['schema'] = 'OFF control'
    path.write_text(json.dumps(value))
    if damage:
        with pytest.raises(ValueError): audit_publication(out, candidate, 'b' * 64)
    else:
        result = audit_publication(out, candidate, 'b' * 64)
        assert result['submission_id'] == 701 and result['kind'] == 'learned' and result['prior_best_score'] == '0.29'


@pytest.mark.parametrize('mode', ['pending', 'complete', 'read_timeout'])
def test_observer_only_reads_same_numeric_handle_and_preserves_attempts(tmp_path, mode):
    calls = []
    class API:
        def get_submission(self, identifier):
            calls.append(('get_submission', identifier)); assert identifier == 701
            if mode == 'read_timeout': raise Timeout('CPU observation expired')
            value = submission(status='PENDING' if mode == 'pending' else 'COMPLETE')
            return NS(ref=value['submission_id'], status=value['status'], description=value['description'],
                file_name=value['file_name'], public_score=value['public_score'], error_description='',
                date=NOW, team_name=value['team_name'])
        def competition_leaderboard_view(self, competition, **kwargs):
            calls.append(('leaderboard', competition))
            return [NS(team_id=1, team_name='CPU team1', score='0.35', submission_date=NOW),
                NS(team_id=2, team_name='CPU team2', score='0.29', submission_date=NOW)]
        def competition_team_submissions(self, team_id):
            calls.append(('public_membership', team_id)); return [NS(id=701, public_score='0.35')]
    journal = tmp_path / 'api.jsonl'
    if mode == 'read_timeout':
        with pytest.raises(Timeout): observe_official_record(API(), binding(), journal)
        assert len(calls) == 1 and 'observation_error' in journal.read_text()
    else:
        result = observe_official_record(API(), binding(), journal)
        assert result['verified_strict_public_record'] is (mode == 'complete')
        assert len(calls) == (1 if mode == 'pending' else 3)
        assert result['journal_sha256'] == sha256_file(journal)


@pytest.mark.parametrize('plan_only', [False, True])
def test_monitor_rejects_unreleased_access_before_account_or_publication_reads(tmp_path, monkeypatch, plan_only):
    monitor = module('official_monitor_gate', 'monitor_selected_submission.py')
    root = Path(__file__).resolve().parents[1]; monkeypatch.chdir(root)
    access = tmp_path / 'access.json'; access.write_text('{"schema":"CPU unreleased access"}')
    argv = ['monitor']
    for key in ['publication', 'readiness', 'environments', 'pair', 'comparison']:
        argv += ['--' + key, str(tmp_path / (key + '_MUST_NOT_BE_READ'))]
    out = tmp_path / 'out'; argv += ['--access', str(access), '--out', str(out)]
    if plan_only: argv += ['--plan-only']
    monkeypatch.setattr('sys.argv', argv)
    if plan_only:
        monitor.main()
        assert json.loads((out / 'progress.json').read_text())['account_requests'] == 0
    else:
        with pytest.raises(ValueError, match='completed explicit confirmation access'): monitor.main()
        assert not out.exists()


def test_private_score_cannot_supply_a_public_record():
    raw = NS(ref=701, status='COMPLETE', description=binding()['message'], file_name='submission.zip',
        public_score='0.29', private_score='0.99', error_description='', date=NOW, team_name='CPU team1')
    row = official_submission(raw)
    result = compare_official_record(binding(), row, snapshot([(1, '0.29'), (2, '0.24')]),
        {1: [{'submission_id': 701, 'public_score': '0.29'}]})
    assert 'private_score' not in row
    assert not result['verified_strict_public_record'] and not result['verified_private_record']


def test_watch_retains_same_submission_through_provisional_rank_publication(tmp_path, monkeypatch):
    """Real observer loop; science/publication/runtime validation are supplied boundaries."""
    import subprocess
    import sys
    observer = module('official_monitor_provisional_watch', 'monitor_selected_submission.py')
    root = Path(__file__).resolve().parents[1]; monkeypatch.chdir(root)
    reference = {**binding(), 'publication_files': {}}
    readiness = tmp_path / 'readiness.json'; readiness.write_text('CPU fixture')
    candidate = {'experiment_pins': {}}
    monkeypatch.setattr(observer, 'audit_selected_publication', lambda *a: candidate)
    monkeypatch.setattr(observer, 'audit_publication', lambda *a: reference)
    monkeypatch.setattr(observer, 'load_script', lambda path, name:
        NS(notebook_source=lambda *a: None) if 'payload' in name else
        NS(validate_runtime=lambda *a: ({'versions': {'CPU-fixture': 'not real SDK'}}, {})))
    monkeypatch.setattr(observer.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0))
    ids = []; pauses = []
    class API:
        def get_submission(self, identifier):
            ids.append(identifier); value = submission(status='PENDING' if len(ids) == 1 else 'COMPLETE')
            return NS(ref=identifier, status=value['status'], description=value['description'],
                file_name='submission.zip', public_score=value['public_score'], error_description='', date=NOW, team_name='CPU team1')
        def competition_leaderboard_view(self, *args, **kwargs):
            values = [(2, '0.29'), (3, '0.24')] if len(ids) == 2 else [(1, '0.35'), (2, '0.29')]
            return [NS(team_id=i, team_name='CPU team' + str(i), score=s, submission_date=NOW) for i, s in values]
        def competition_team_submissions(self, team_id):
            assert len(ids) == 3 and team_id == 1
            return [NS(id=701, public_score='0.35')]
    monkeypatch.setitem(sys.modules, 'kaggle', NS(api=API()))
    monkeypatch.setattr(observer.time, 'sleep', lambda seconds: pauses.append(seconds))
    out = tmp_path / 'out'; argv = ['observe']
    for name in ['publication', 'access', 'environments', 'pair', 'comparison']:
        argv += ['--' + name, str(tmp_path / name)]
    argv += ['--readiness', str(readiness), '--out', str(out), '--watch']
    monkeypatch.setattr(sys, 'argv', argv); observer.main()
    states = [json.loads(p.read_text()) for p in sorted(out.glob('observation-*.json'))]
    assert [r['state'] for r in states] == ['waiting_official_evaluation',
        'waiting_public_leaderboard_identity', 'verified_learned_public_record']
    assert [r['verified_strict_public_record'] for r in states] == [False, False, True]
    assert ids == [701, 701, 701] and pauses == [60, 60]
    assert all(r['goal_marked_complete'] is False and r['verified_private_record'] is False for r in states)
