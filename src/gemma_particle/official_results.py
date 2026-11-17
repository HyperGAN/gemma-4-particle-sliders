"""Exact official score and public-record comparisons, never model selection."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from .conversion_handoff import read
from .harness_identity import sha256_file

COMPETITION = 'gemma-4-developer-agent'


def score_value(value):
    if not isinstance(value, str) or not value:
        raise ValueError('actual nonempty decimal score string required')
    try: result = Decimal(value)
    except InvalidOperation as exc: raise ValueError('finite decimal score required') from exc
    if not result.is_finite(): raise ValueError('finite decimal score required')
    return result


def leaderboard_snapshot(rows, started_at, finished_at):
    if not rows or len(rows) > 20 or any(row is None for row in rows):
        raise ValueError('actual nonempty first public leaderboard page required')
    entries = [{'team_id': row.team_id, 'team': row.team_name, 'score': row.score,
        'submission_date': str(row.submission_date)} for row in rows]
    values = [score_value(row['score']) for row in entries]
    if (any(type(row['team_id']) is not int or row['team_id'] <= 0 for row in entries)
            or len({row['team_id'] for row in entries}) != len(entries)
            or values != sorted(values, reverse=True)):
        raise ValueError('distinct actual teams in descending public leaderboard order required')
    return {'schema': 'gemma4-public-leaderboard-observation-v1', 'competition': COMPETITION,
        'request': {'method': 'competition_leaderboard_view', 'page_size': 20, 'page_token': None},
        'started_at': started_at, 'finished_at': finished_at, 'entries': entries,
        'best_score': entries[0]['score'], 'scope': 'fresh first public leaderboard page;not a private score or model selection result'}


def audit_publication(directory, candidate, readiness_sha256):
    directory = Path(directory)
    plan = read(directory / 'plan.json'); status = read(directory / 'progress.json')
    intent = read(directory / 'submission-intent.json'); response = read(directory / 'submission-response.json')
    output = read(directory / 'verified-notebook-output.json')
    prior = read(directory / 'leaderboard-before-submission.json')
    expected_message = 'Particle Sliders learned; locked DEV; sha256=' + candidate['artifact_sha256']
    if (plan.get('schema') != 'gemma4-learned-publication-plan-v1'
            or plan['readiness_sha256'] != readiness_sha256 or plan['artifact_sha256'] != candidate['artifact_sha256']
            or plan['dataset'] != candidate['dataset'] or plan['notebook'] != candidate['notebook']
            or any(sha256_file(path) != value for path, value in plan.get('sources', {}).items())
            or status.get('state') != 'created'
            or any(status.get(name) != 1 for name in ['dataset_create_calls', 'notebook_push_calls', 'submission_calls'])
            or type(status.get('submission_id')) is not int or status['submission_id'] <= 0
            or response.get('submission_id') != status['submission_id']
            or intent['competition'] != COMPETITION or intent['kernel'] != candidate['notebook']
            or intent['version'] != 1 or intent['file_name'] != 'submission.zip'
            or intent['artifact_sha256'] != candidate['artifact_sha256'] or intent['message'] != expected_message
            or output['kernel'] != candidate['notebook'] or output['version'] != 1
            or output['handle'] != candidate['notebook'] + '/1'
            or output['artifact_sha256'] != candidate['artifact_sha256']
            or Path(output['path']).resolve() != (directory / 'downloaded/submission.zip').resolve()
            or sha256_file(output['path']) != candidate['artifact_sha256']
            or prior.get('schema') != 'gemma4-public-leaderboard-observation-v1'
            or prior['competition'] != COMPETITION
            or intent['public_reference_sha256'] != sha256_file(directory / 'leaderboard-before-submission.json')
            or prior['request'] != {'method': 'competition_leaderboard_view', 'page_size': 20, 'page_token': None}
            or score_value(prior['best_score']) != max(score_value(row['score']) for row in prior['entries'])
            or datetime.fromisoformat(prior['finished_at']) > datetime.fromisoformat(intent['as_of'])):
        raise ValueError('one actual created learned submission,exact evaluated output and prior public reference required')
    return {'kind': 'learned', 'submission_id': status['submission_id'], 'message': expected_message,
        'artifact_sha256': candidate['artifact_sha256'], 'prior_best_score': prior['best_score'],
        'publication': str(directory.resolve()),
        'publication_files': {str(path): sha256_file(path) for path in [
            directory / name for name in ['plan.json', 'progress.json', 'submission-intent.json',
                'submission-response.json', 'verified-notebook-output.json', 'leaderboard-before-submission.json']]}}


def official_submission(row):
    return {'submission_id': row.ref, 'status': str(getattr(row.status, 'name', row.status)).split('.')[-1],
        'description': row.description, 'file_name': row.file_name, 'public_score': row.public_score,
        'error_present': bool(row.error_description), 'date': str(row.date), 'team_name': row.team_name}


def compare_official_record(binding, submission, current, public_submissions):
    """A decimal tie never counts. A displayed own entry needs numeric-ID proof."""
    if (binding.get('kind') != 'learned' or submission['submission_id'] != binding['submission_id']
            or submission['description'] != binding['message'] or submission['file_name'] != 'submission.zip'):
        raise ValueError('actual bound learned competition submission required')
    result = {'schema': 'gemma4-official-learned-record-v1', 'competition': COMPETITION,
        'submission_id': submission['submission_id'], 'artifact_sha256': binding['artifact_sha256'],
        'submission': submission, 'prior_best_score': binding['prior_best_score'],
        'verified_strict_public_record': False, 'goal_marked_complete': False,
        'scope': 'official learned score versus observed public records only;full goal completion remains a separate audit'}
    if submission['status'] != 'COMPLETE':
        result['state'] = 'official_error' if submission['status'] == 'ERROR' else 'waiting_official_evaluation'
        return result
    if submission['error_present']:
        result['state'] = 'official_error'; return result
    score = score_value(submission['public_score']); before = score_value(binding['prior_best_score'])
    if current is None: raise ValueError('fresh public leaderboard observation required after score completion')
    if (current['competition'] != COMPETITION or current['schema'] != 'gemma4-public-leaderboard-observation-v1'
            or current['request'] != {'method': 'competition_leaderboard_view', 'page_size': 20, 'page_token': None}):
        raise ValueError('same competition public leaderboard required')
    entries = current['entries']; best = max(score_value(row['score']) for row in entries)
    if score_value(current['best_score']) != best: raise ValueError('actual public maximum required')
    contenders = [row for row in entries if score_value(row['score']) >= score]
    own_entry = None
    if score == best and len(contenders) == 1:
        team = contenders[0]
        members = public_submissions.get(team['team_id'], [])
        matching = [row for row in members if row['submission_id'] == binding['submission_id']]
        if len(matching) == 1 and score_value(matching[0]['public_score']) == score:
            own_entry = team['team_id']
    other_values = [score_value(row['score']) for row in entries if row['team_id'] != own_entry]
    current_threshold = max(other_values) if other_values else best
    # If the only listed entry is our candidate, do not invent an unobserved
    # other-team threshold; a real preceding reference must still be exceeded.
    strict = score > before and (score > best or own_entry is not None and bool(other_values) and score > current_threshold)
    result.update(state='verified_learned_public_record' if strict else 'complete_without_verified_strict_record',
        public_score=submission['public_score'], current_best_score=current['best_score'],
        public_leaderboard=current, public_team_submissions=public_submissions,
        displayed_candidate_team_id=own_entry, verified_strict_public_record=strict,
        threshold_policy='strictly above prior all-team best and fresh other-team best;displayed own entry requires numeric submission-ID evidence')
    return result


def observe_official_record(api, binding, journal):
    """Read one bound submission and fresh public evidence, without mutations."""
    if binding.get('kind') != 'learned': raise ValueError('qualified learned publication binding required')
    journal = Path(journal)
    def call(method, args, kwargs, project):
        started = datetime.now(timezone.utc).isoformat()
        with journal.open('a') as stream:
            stream.write(json.dumps({'phase': 'attempt', 'method': method, 'args': args,
                'kwargs': kwargs, 'started_at': started}) + '\n')
        try:
            value = getattr(api, method)(*args, **kwargs)
            result = project(value, started)
        except Exception as exc:
            with journal.open('a') as stream:
                stream.write(json.dumps({'phase': 'observation_error', 'method': method,
                    'error_type': type(exc).__name__,
                    'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}) + '\n')
            raise
        with journal.open('a') as stream:
            stream.write(json.dumps({'phase': 'observed', 'method': method, 'result': result,
                'finished_at': datetime.now(timezone.utc).isoformat()}) + '\n')
        return result
    submission = call('get_submission', [binding['submission_id']], {}, lambda value, start: official_submission(value))
    current = None; members = {}
    if submission['status'] == 'COMPLETE' and not submission['error_present']:
        score = score_value(submission['public_score'])
        current = call('competition_leaderboard_view', [COMPETITION], {'page_size': 20},
            lambda value, start: leaderboard_snapshot(value, start, datetime.now(timezone.utc).isoformat()))
        contenders = [row for row in current['entries'] if score_value(row['score']) >= score]
        if score == score_value(current['best_score']) and len(contenders) == 1:
            team = contenders[0]['team_id']
            def project(rows, start):
                if rows is None or any(row is None for row in rows):
                    raise ValueError('actual public submission identity observation required')
                return [{'submission_id': row.id, 'public_score': row.public_score} for row in rows]
            members[team] = call('competition_team_submissions', [team], {}, project)
    result = compare_official_record(binding, submission, current, members)
    result.update(observed_at=datetime.now(timezone.utc).isoformat(), journal_sha256=sha256_file(journal))
    return result
