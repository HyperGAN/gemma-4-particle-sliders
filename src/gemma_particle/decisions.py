"""NanoGPT's typed decision endpoint; independent of native training/scoring.

Credentials remain in memory. HTTP failures expose only their status code, and
redirects are refused so Authorization cannot be forwarded to another host.
"""
import json
import math
import os
from pathlib import Path
import re
import shlex
import urllib.error
import urllib.request

ENDPOINT = 'https://api.nano-gpt.com/api/v1/decisions'
DEFAULT_MODEL = 'typesafe/jev-latest'


def load_key(assignment_file=None):
    """Read env, or one literal assignment without executing a shell startup file.

    An explicit file takes precedence over a possibly stale inherited env value.
    Shell expansion and multiple assignments require the caller to export the key.
    """
    if assignment_file is None:
        key = os.environ.get('NANOGPT_API_KEY', '')
    else:
        assignments = []
        for line in Path(assignment_file).expanduser().read_text().splitlines():
            match = re.match(r'^\s*(?:export\s+)?NANOGPT_API_KEY\s*=\s*(.*?)\s*$', line)
            if match:
                assignments.append(match.group(1))
        if len(assignments) != 1:
            raise ValueError('exactly one literal NANOGPT_API_KEY assignment required')
        try:
            words = shlex.split(assignments[0], comments=True, posix=True)
        except ValueError:
            raise ValueError('literal NANOGPT_API_KEY assignment required') from None
        if len(words) != 1 or any(c in words[0] for c in '$`'):
            raise ValueError('literal NANOGPT_API_KEY assignment required')
        key = words[0]
    if not key or any(c.isspace() for c in key):
        raise ValueError('nonempty NANOGPT_API_KEY without whitespace required')
    return key


def _finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate_request(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), str) or not payload['model']:
        raise ValueError('explicit decision model required')
    if 'state' not in payload:
        raise ValueError('decision state required')
    questions = payload.get('questions')
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError('one to 64 named questions required')
    for name, question in questions.items():
        if not isinstance(name, str) or not name or not isinstance(question, dict) or not question.get('instructions'):
            raise ValueError('named question with instructions required')
        kind, criteria = question.get('type'), question.get('criteria')
        if kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError('score needs two to ten ordered descriptive levels')
        elif kind == 'choice':
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
                raise ValueError('choice needs two to 26 named criteria')
        elif kind != 'noul':
            raise ValueError('unsupported decision primitive')
    return json.dumps(payload, allow_nan=False, ensure_ascii=False).encode()


def validate_response(payload, response):
    """Fail on missing/refused answers, mismatched models or invalid probabilities.

    A refusal is an unavailable judgment, never a zero or an inferred success.
    Preserve raw distributions; a Score's number is an ordered level position.
    """
    requested = payload['model']
    accepted = {requested}
    # NanoGPT's contract also names this exact alias with a leading '~'. The
    # live latest route returns that spelling. Preserve it in the raw receipt.
    if requested == 'typesafe/jev-latest':
        accepted.add('~typesafe/jev-latest')
    if not isinstance(response, dict) or response.get('model') not in accepted:
        raise ValueError('decision response model differs from the requested model')
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(payload['questions']):
        raise ValueError('decision response must answer exactly the requested questions')
    for name, question in payload['questions'].items():
        answer = answers[name]
        kind = question['type']
        if not isinstance(answer, dict) or answer.get('type') != kind:
            raise ValueError('missing, declined or mismatched typed decision answer')
        if kind == 'noul':
            if not _finite(answer.get('noul'), 0, 1):
                raise ValueError('invalid yes/no probability')
            continue
        criteria = question['criteria']
        keys = set(criteria) if kind == 'choice' else {str(i) for i in range(len(criteria))}
        probs = answer.get('probabilities')
        if (not isinstance(probs, dict) or set(probs) != keys
                or not all(_finite(v, 0, 1) for v in probs.values())
                # The live endpoint rounds each probability to two decimals.
                or not math.isclose(sum(probs.values()), 1, abs_tol=.005 * len(keys) + 1e-8)
                or not _finite(answer.get('confidence'), 0, 1)):
            raise ValueError('invalid decision distribution')
        if kind == 'choice':
            if answer.get('choice') not in keys:
                raise ValueError('choice outside requested candidates')
        else:
            score = answer.get('score')
            expected = sum(int(k) * v for k, v in probs.items())
            rounding_bound = .005 * (1 + sum(range(len(criteria)))) + 1e-8
            if not _finite(score, 0, len(criteria)-1) or not math.isclose(score, expected, abs_tol=rounding_bound):
                raise ValueError('invalid ordered-level score')
    return response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def decide(payload, *, key=None, assignment_file=None, timeout=60, response_path=None):
    body = validate_request(payload)
    if len(body) > 1024 * 1024:
        raise ValueError('decision text request exceeds one MiB; no truncation applied')
    if key is None:
        key = load_key(assignment_file)
    request = urllib.request.Request(ENDPOINT, data=body, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as result:
            raw = result.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'NanoGPT decision endpoint returned HTTP {exc.code}') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError('NanoGPT decision transport failed; no retry performed') from None
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('decision response exceeds four MiB')
    try:
        response = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError('decision endpoint returned invalid JSON') from None
    if response_path is not None:
        with Path(response_path).open('x') as stream:
            json.dump(response, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return validate_response(payload, response)
