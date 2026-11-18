"""Revalidate already captured Jev judgments offline; never make an API call."""
import argparse
import json
from pathlib import Path

from gemma_particle.behavior_curation import (eligible_arms, load_behavior_cohort, load_verified_grades,
    select_behavior_contexts)
from gemma_particle.data import write_jsonl
from gemma_particle.harness_identity import sha256_file
from gemma_particle.paired_collection import PAIR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--grades', type=Path, required=True)
    p.add_argument('--visible-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--protocol', type=Path, default=Path('research/jev-evidence-loop-v1.json'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol, protocol_sha, tasks, outcomes = load_behavior_cohort(args.cohort, args.visible_tasks, args.protocol)
    grades = load_verified_grades(args.cohort, args.grades, tasks, outcomes, protocol, protocol_sha)
    contexts = select_behavior_contexts(tasks, outcomes, grades, protocol, protocol_sha)
    if len({r['instance_id'] for r in contexts}) != len(contexts):
        raise ValueError('at most one initial target per task required')
    pilot_comparison = None
    if 'collection' in protocol:
        paired_protocol = {**protocol, 'collection': {**protocol['collection'], 'arms': list(PAIR)}}
        paired_rows = [row for row in outcomes if row['arm'] in PAIR]
        _, paired_eligible = eligible_arms(tasks, paired_rows, paired_protocol)
        paired_keys = {(row['instance_id'], row['arm']) for row in paired_eligible}
        paired_grades = {key: grade for key, grade in grades.items() if key in paired_keys}
        verify_contexts = select_behavior_contexts(tasks, paired_rows, paired_grades, paired_protocol, protocol_sha)
        pilot_comparison = {'verify_only_admitted': bool(verify_contexts),
            'declared_arms_admitted': bool(contexts), 'selected_arm': contexts[0]['target_arm'] if contexts else None,
            'qualified_grades': sorted(arm for _, arm in grades)}
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(args.out / 'jev-contexts.jsonl', contexts)
    result = {'schema': 'gemma4-behavior-offline-curation-v1', 'status': 'passed',
        'protocol_sha256': protocol_sha, 'grades_result_sha256': sha256_file(args.grades / 'result.json'),
        'contexts_sha256': sha256_file(args.out / 'jev-contexts.jsonl'), 'admitted_contexts': len(contexts),
        'api_calls': 0, 'training_updates': 0, 'scope': 'offline evidence binding and target selection only'}
    if pilot_comparison is not None:
        result['pilot_comparison'] = pilot_comparison
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
