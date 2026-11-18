"""One immutable paid Jev pass over eligible fresh TRAIN episodes, then curate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from gemma_particle.behavior_curation import (eligible_arms, episode_payload, load_behavior_cohort,
    select_behavior_contexts, verified_grade)
from gemma_particle.data import digest, write_jsonl
from gemma_particle.decisions import decide
from gemma_particle.harness_identity import sha256_file


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--visible-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--protocol', type=Path, default=Path('research/jev-evidence-loop-v1.json'))
    p.add_argument('--key-file', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol, protocol_sha, tasks, outcomes = load_behavior_cohort(args.cohort, args.visible_tasks, args.protocol)
    _, eligible = eligible_arms(tasks, outcomes, protocol)
    lookup = {t['instance_id']: t for t in tasks}
    prepared = []
    for row in eligible:
        directory = args.cohort / row['arm'] / row['instance_id'] / 'requests'
        request, binding = episode_payload(lookup[row['instance_id']], row, directory, protocol)
        prepared.append((row, request, binding))
    # This cohort-wide exclusive claim prevents a second paid pass in another
    # output directory, including after a refusal, transport failure or crash.
    claim = {'schema': 'gemma4-behavior-grading-claim-v1', 'out': str(args.out.resolve()),
        'protocol_sha256': protocol_sha, 'outcomes_sha256': sha256_file(args.cohort / 'outcomes.jsonl'),
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha256_file(Path(__file__)),
        'eligible': [binding for _, _, binding in prepared], 'automatic_paid_retries': False}
    if 'collection' in protocol:
        claim['collection_plan_sha256'] = protocol['collection']['plan_sha256']
    if args.out.exists():
        raise FileExistsError('grading output is immutable')
    write_json(args.cohort / 'jev-grading-claim.json', claim)
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(args.out / 'claim.json', claim)
    grades = {}
    finished = []
    for row, request, binding in prepared:
        folder = args.out / row['instance_id'] / row['arm']
        folder.mkdir(parents=True)
        write_json(folder / 'request.json', request)
        receipt = {'schema': 'gemma4-behavior-grade-v1', 'status': 'started', **binding,
            'protocol_sha256': protocol_sha, 'request_file_sha256': sha256_file(folder / 'request.json'),
            'created_at': datetime.now(timezone.utc).isoformat(), 'api_calls_attempted': 1,
            'model': request['model'], 'resolved_checkpoint': None}
        write_json(folder / 'attempt.json', receipt)
        start = time.perf_counter()
        try:
            response = decide(request, assignment_file=args.key_file, timeout=300, response_path=folder / 'response.json')
            receipt.update(status='qualified', normalized_score=response['answers']['repair_evidence']['score'] / 4,
                decision_response_sha256=digest(response), response_model=response['model'],
                response_file_sha256=sha256_file(folder / 'response.json'), usage=response.get('usage'))
            verified_grade(receipt, request, response, binding, protocol_sha)
            grades[(row['instance_id'], row['arm'])] = receipt
        except Exception as exc:
            receipt.update(status='unqualified', error_type=type(exc).__name__, error=str(exc))
        if (folder / 'response.json').exists():
            receipt['response_file_sha256'] = sha256_file(folder / 'response.json')
        receipt['seconds'] = time.perf_counter() - start
        write_json(folder / 'result.json', receipt)
        finished.append(receipt)
        print(json.dumps({'task': row['instance_id'], 'arm': row['arm'], 'status': receipt['status'],
            'normalized_score': receipt.get('normalized_score')}), flush=True)
    contexts = select_behavior_contexts(tasks, outcomes, grades, protocol, protocol_sha)
    write_jsonl(args.out / 'jev-contexts.jsonl', contexts)
    result = {'schema': 'gemma4-behavior-cohort-grade-v1', 'state': 'complete',
        'protocol_sha256': protocol_sha, 'claim_sha256': sha256_file(args.cohort / 'jev-grading-claim.json'),
        'source_sha256': sha256_file(Path(__file__)), 'finished': finished, 'eligible_arms': len(eligible),
        'api_calls_attempted': len(prepared), 'qualified_grades': len(grades), 'admitted_contexts': len(contexts),
        'distinct_fit_tasks': len({r['instance_id'] for r in contexts}),
        'contexts_sha256': sha256_file(args.out / 'jev-contexts.jsonl'), 'training_updates': 0,
        'scope': 'TRAIN initial-boundary selection only; no model DEV/holdout or official score'}
    write_json(args.out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'finished'}), flush=True)


if __name__ == '__main__':
    main()
