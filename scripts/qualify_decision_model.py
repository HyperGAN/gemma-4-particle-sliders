"""One paid typed-API control; no pretrained Gemma, targets or task score."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from gemma_particle.decisions import DEFAULT_MODEL, ENDPOINT, decide, validate_response


def fixture(model):
    levels = [
        'The trace contains no attempt to investigate or repair the reported failure.',
        'The trace proposes a repair without inspecting evidence of the reported failure.',
        'The trace investigates the failure and implements a related repair, without checking its effect.',
        'The trace reproduces the failure, implements a related repair, and observes a confirming test result.',
    ]
    return {'model': model, 'state': {
        'kind': 'synthetic API control, never a training target or coding result',
        'issue': 'A function returns None for an empty list, but must return zero.',
        'observed': [
            'Tool: test_empty fails: expected 0, got None.',
            'Tool: inspected function; the empty branch returns None.',
            'Tool: changed the empty branch to return 0.',
            'Tool: ran test_empty after the change; 1 passed.',
        ],
        'unverified': ['Assistant: I fixed it. I did not inspect code or run any tests.'],
    }, 'questions': {
        'observed_repair': {'type': 'score', 'instructions':
            'How thoroughly does `observed` ground its repair of `issue` in execution evidence?', 'criteria': levels},
        'unverified_repair': {'type': 'score', 'instructions':
            'How thoroughly does `unverified` ground its repair of `issue` in execution evidence?', 'criteria': levels},
        'test_executed': {'type': 'noul', 'instructions':
            'Does `observed` contain a tool observation from actually executing a test after the change?'},
        'unverified_stage': {'type': 'choice', 'instructions':
            'Which stage of repair is established by `unverified`?', 'criteria': {
                'claim_only': 'An unsupported claim of a repair without tool evidence.',
                'test_observed': 'A repair with an actual observed tool result from running a test.',
            }},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--replay-response', type=Path, help='Validate a retained response without another API call')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    request = fixture(args.model)
    (args.out/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    receipt = {'schema': 'gemma4-decision-api-control-v1', 'model': args.model, 'endpoint': ENDPOINT,
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'started', 'training_updates': 0,
        'scope': 'synthetic typed transport/ordering control; no target admission or real-task quality claim',
        'request_sha256': hashlib.sha256((args.out/'request.json').read_bytes()).hexdigest(),
        'api_calls': 0 if args.replay_response else 1,
        'replay_source': str(args.replay_response) if args.replay_response else None}
    start = time.perf_counter()
    try:
        if args.replay_response:
            raw = args.replay_response.read_bytes()
            (args.out/'response.json').write_bytes(raw)
            response = validate_response(request, json.loads(raw))
        else:
            response = decide(request, assignment_file=args.key_file, response_path=args.out/'response.json')
        answers = response['answers']
        checks = {
            'observed_above_unverified': answers['observed_repair']['score'] > answers['unverified_repair']['score'],
            'observed_test_likely': answers['test_executed']['noul'] > .5,
            'unsupported_claim_identified': answers['unverified_stage']['choice'] == 'claim_only',
        }
        receipt.update(status='passed' if all(checks.values()) else 'failed', checks=checks,
            response_sha256=hashlib.sha256((args.out/'response.json').read_bytes()).hexdigest(),
            usage=response.get('usage'), cost=response.get('cost'), response_model=response['model'])
    except Exception as exc:
        receipt.update(status='error', error_type=type(exc).__name__, error=str(exc))
    receipt['seconds'] = time.perf_counter()-start
    (args.out/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
