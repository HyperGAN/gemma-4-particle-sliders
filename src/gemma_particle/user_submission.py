"""Bind a user-submitted Kaggle archive using the exact official submission ID."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .conversion_handoff import read
from .harness_identity import sha256_file
from .kaggle_publication import durable_json
from .official_results import COMPETITION, leaderboard_snapshot, official_submission


class SubmissionNotVisible(LookupError):
    """The authenticated competition submission list does not yet show the ID."""


def utc(value):
    stamp = datetime.fromisoformat(value)
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def capture_user_reference(api, candidate, readiness_sha, output):
    started = datetime.now(timezone.utc).isoformat()
    rows = api.competition_leaderboard_view(COMPETITION, page_size=20)
    finished = datetime.now(timezone.utc).isoformat()
    reference = {
        'schema': 'gemma4-user-submission-reference-v1', 'competition': COMPETITION,
        'artifact_sha256': candidate['artifact_sha256'], 'readiness_sha256': readiness_sha,
        'account_mutations': 0, 'leaderboard': leaderboard_snapshot(rows, started, finished),
    }
    durable_json(output, reference)
    return reference


def load_user_reference(path, candidate, readiness_sha):
    value = read(path)
    board = value['leaderboard']
    rows = [SimpleNamespace(team_id=row['team_id'], team_name=row['team'],
        score=row['score'], submission_date=row['submission_date']) for row in board['entries']]
    if (value.get('schema') != 'gemma4-user-submission-reference-v1'
            or value['competition'] != COMPETITION or value['account_mutations'] != 0
            or value['artifact_sha256'] != candidate['artifact_sha256'] or value['readiness_sha256'] != readiness_sha
            or board != leaderboard_snapshot(rows, board['started_at'], board['finished_at'])
            or utc(board['started_at']) > utc(board['finished_at'])):
        raise ValueError('Expected the public leaderboard reference for the exact local handoff')
    return value


def bind_user_submission(api, submission_id, candidate, reference_path, reference, out):
    if type(submission_id) is not int or submission_id <= 0:
        raise ValueError('Expected a positive numeric Kaggle submission ID')
    out = Path(out)
    out.mkdir(mode=0o700)
    rows = [row for row in api.competition_submissions(COMPETITION) if row.ref == submission_id]
    if not rows:
        raise SubmissionNotVisible('Waiting for the authenticated competition list to show this submission')
    if len(rows) != 1:
        raise ValueError('Expected one authenticated competition submission with this ID')
    submission = official_submission(rows[0])
    list_receipt = out / 'competition-submission.json'
    durable_json(list_receipt, {'method': 'competition_submissions', 'competition': COMPETITION,
        'observed_at': datetime.now(timezone.utc).isoformat(), 'submission': submission})
    leaderboard = reference['leaderboard']
    if submission['file_name'] != 'submission.zip' or utc(leaderboard['finished_at']) > utc(submission['date']):
        raise ValueError('Expected submission.zip created after the handoff reference')
    downloads = out / 'downloaded'
    downloads.mkdir(mode=0o700)
    api.competition_download_submission(submission_id, path=str(downloads), quiet=True)
    files = list(downloads.iterdir())
    if len(files) != 1 or files[0].is_symlink() or not files[0].is_file():
        raise ValueError('Kaggle must return a single regular submitted archive')
    archive = files[0]
    archive_sha = sha256_file(archive)
    download_receipt = out / 'submission-download.json'
    durable_json(download_receipt, {'method': 'competition_download_submission', 'submission_id': submission_id,
        'observed_at': datetime.now(timezone.utc).isoformat(), 'path': str(archive.resolve()),
        'bytes': archive.stat().st_size, 'sha256': archive_sha, 'account_mutations': 0})
    if archive_sha != candidate['artifact_sha256']:
        raise ValueError('Kaggle submitted bytes differ from the evaluated learned ZIP')
    paths = [reference_path, list_receipt, download_receipt, archive]
    return {'kind': 'learned', 'origin': 'user_submission', 'submission_id': submission_id,
        'message': submission['description'], 'artifact_sha256': archive_sha,
        'prior_best_score': leaderboard['best_score'], 'account_mutations': 0,
        'publication_files': {str(Path(path).resolve()): sha256_file(path) for path in paths}}
