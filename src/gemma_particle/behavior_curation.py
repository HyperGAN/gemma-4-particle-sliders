"""Prospective Jev admission, separate from the original binary selection rule."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .behavior_protocol import load_behavior_protocol
from .budgets import OFFICIAL_BUDGET, budget_values
from .capture import CONTEXT_CAPTURE_PROTOCOL, load_requests
from .curation import pair_contexts, validate_context_outcomes
from .data import digest, read_jsonl
from .decisions import validate_response
from .dev import validate_agent_tasks
from .history import OFFICIAL_COMPACTION, history_contract, is_assistant_decision
from .harness_identity import sha256_file
from .paired_collection import PAIR, QUARTET, load_collection_pilot
from .prompt_profiles import DEFAULT_PROMPT_PROFILE,PROMPT_PROFILES,prompt_identity


def load_behavior_cohort(root, visible_tasks, protocol_path):
    """Reject retrospective relabeling and unmatched raw collection contracts."""
    root, protocol_path = Path(root), Path(protocol_path)
    protocol = load_behavior_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    arms = tuple(plan.get('arms', []))
    pilot = plan.get('collection_pilot')
    if pilot:
        rows = load_collection_pilot(pilot['path'], pilot['sha256'], visible_tasks)
        declared = {row['instance_id']: tuple(row['planned_arms']) for row in rows}
        if len(plan['task_ids']) != 1 or declared.get(plan['task_ids'][0]) != arms:
            raise ValueError('behavior cohort differs from the fixed collection pilot')
        protocol['collection'] = {'arms': list(arms), 'pilot_sha256': pilot['sha256'],
            'plan_sha256': sha256_file(plan_path)}
    if (plan.get('behavior_study') != protocol['name'] or plan.get('behavior_protocol_sha256') != protocol_sha
            or sha256_file(root / 'behavior-protocol.json') != protocol_sha
            or plan.get('teacher_policy') != ('paired-verify' if arms == PAIR else 'all-four')
            or (not pilot and arms != QUARTET)
            or plan.get('budget_profile') != OFFICIAL_BUDGET
            or plan.get('budget') != budget_values(OFFICIAL_BUDGET)
            or plan.get('history') != history_contract(OFFICIAL_COMPACTION)):
        raise ValueError('fresh prospective collection declaration required')
    tasks = read_jsonl(visible_tasks)
    validate_agent_tasks(tasks)
    ids = plan['task_ids']
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('distinct declared TRAIN tasks required')
    tasks = [t for t in tasks if t['instance_id'] in ids]
    if {t['instance_id'] for t in tasks} != set(ids) or any(t['split'] != 'train' for t in tasks):
        raise ValueError('declared tasks differ from sanitized TRAIN data')
    if json.loads((root / 'progress.json').read_text()).get('state') != 'complete':
        raise ValueError('all raw arms must reach terminal state before grading')
    outcomes = read_jsonl(root / 'outcomes.jsonl')
    grouped, _ = eligible_arms(tasks, outcomes, protocol)
    if set(grouped) != set(ids):
        raise ValueError('terminal collection is missing declared tasks')
    executor = Path(plan['source_checkout'])
    profile = plan.get('prompt_profile',DEFAULT_PROMPT_PROFILE)
    if profile not in PROMPT_PROFILES:
        raise ValueError('unknown common prompt profile')
    common_prompt = None
    if profile != DEFAULT_PROMPT_PROFILE:
        common_prompt = prompt_identity((executor/'submission/prompts/system.md').read_text(),profile)
        if plan.get('common_prompt') != common_prompt:
            raise ValueError('declared common prompt changed')
    if (len(plan['source_git']) != 40 or not plan['execution_sources']
            or any(sha256_file(executor / name) != sha for name, sha in plan['execution_sources'].items())):
        raise ValueError('frozen raw collection sources changed')
    server = plan['server_identity']
    if (server.get('kind') != 'gemma4' or server.get('pretrained_inference') is not True
            or server.get('strength') != 0 or server.get('native_checkpoint') is not None):
        raise ValueError('fresh frozen-base Gemma episodes required')
    lookup = {t['instance_id']: t for t in tasks}
    for arm in plan['arms']:
        manifest = json.loads((root / arm / 'cohort.json').read_text())
        contract = manifest['contract']
        budget, harness = contract['budget'], contract['harness']
        if (manifest['arm'] != arm or manifest['server_identity'] != server
                or manifest.get('prompt_profile',DEFAULT_PROMPT_PROFILE) != profile
                or contract.get('common_prompt') != common_prompt
                or manifest['evaluation_contract_sha256'] != digest(contract)
                or budget['time_minutes'] != 60 or budget['tool_calls'] != 100 or budget['turns'] != 500
                or budget['cost_usd'] is not None or budget['total_tokens'] is not None
                or harness['command_timeout_seconds'] != 300
                or contract.get('history') != plan['history']):
            raise ValueError('actual collector model/budget/history contract differs')
        if pilot and read_jsonl(root / (arm + '-verification') / 'outcomes.jsonl') != [
                grouped[task_id][arm] for task_id in ids]:
            raise ValueError('paired pilot outcomes differ from the actual fresh verifier output')
        for task_id, arms in grouped.items():
            row = arms[arm]
            if (row['evaluation_contract_sha256'] != manifest['evaluation_contract_sha256']
                    or row.get('prompt_profile',DEFAULT_PROMPT_PROFILE) != profile
                    or row['model_identity'] != server['model_identity']
                    or row['visible_task_sha256'] != digest(lookup[task_id])
                    or row['verification']['image_id'] != manifest['image_id']):
                raise ValueError('verified episode differs from the declared raw collection')
    return protocol, protocol_sha, tasks, outcomes


def clean_capture(row):
    return (row.get('capture_complete') is True and row.get('agent_error') is None
        and row.get('capture_error') is None and row.get('task_budget_exhausted') is False
        and row.get('context_budget_proof') is None and bool(row.get('requests')))


def eligible_arms(tasks, outcomes, protocol):
    validate_agent_tasks(tasks)
    collection = protocol.get('collection')
    grouped = validate_context_outcomes(tasks, outcomes,
        paired_verify=bool(collection and tuple(collection['arms']) == PAIR))
    eligible = []
    for task_id, arms in sorted(grouped.items()):
        if task_id in protocol['cohort_policy']['excluded_fit_tasks']:
            raise ValueError('reserved conversion task cannot enter fitting')
        for row in arms.values():
            if (row.get('instruction_protocol') != 'tools-v2'
                    or row.get('budget_profile') != OFFICIAL_BUDGET
                    or row.get('capture_protocol') != CONTEXT_CAPTURE_PROTOCOL
                    or row.get('history_profile') != OFFICIAL_COMPACTION):
                raise ValueError('behavior study requires its declared collection profiles')
        baseline = arms['off']
        if not clean_capture(baseline):
            continue
        if baseline['requests'][0]['request'] != baseline['first_request']:
            raise ValueError('initial OFF request differs from its complete journal')
        candidates = []
        for arm, row in sorted(arms.items()):
            if arm == 'off' or not clean_capture(row) or row['resolved'] is not True:
                continue
            receipt = row['verification']
            if (receipt.get('error') is not None or receipt.get('test_exit_code') != 0
                    or receipt.get('evaluation_contract_sha256') != row['evaluation_contract_sha256']
                    or not receipt.get('origin_checks')
                    or not all(c.get('workspace') is True for c in receipt['origin_checks'])):
                raise ValueError('teacher success requires clean fresh tests against the edited workspace')
            # Validate the whole initial question/cue pairing before paid grading.
            pair_contexts(row, baseline, initial_only=True)
            candidates.append(row)
        if candidates:
            eligible.extend([baseline, *candidates])
    return grouped, eligible


def episode_payload(task, row, journal_directory, protocol):
    """Bind the terminal ordinary decision plus completion; omit judge labels."""
    validate_agent_tasks([task])
    if task['split'] != 'train' or task['instance_id'] != row['instance_id'] or not clean_capture(row):
        raise ValueError('a clean complete matching TRAIN episode is required')
    journal_directory = Path(journal_directory)
    packets = load_requests(journal_directory)
    if packets != row['requests']:
        raise ValueError('complete episode journal differs from verified requests')
    ordinary = [p for p in packets if is_assistant_decision(p, row['history_profile'])]
    if not ordinary:
        raise ValueError('terminal ordinary decision required')
    packet = ordinary[-1]
    completed = json.loads((journal_directory / f"{packet['decision']:04d}-response.json").read_text())
    if completed.get('schema') != 'gemma4-model-response-v1':
        raise ValueError('actual captured completion required')
    choices = completed['response']['choices']
    if len(choices) != 1 or choices[0]['message'].get('role') != 'assistant':
        raise ValueError('one actual terminal assistant completion required')
    messages = deepcopy([m for m in packet['request']['messages'] if m['role'] not in ('system', 'developer')])
    if any(m['role'] not in ('user', 'assistant', 'tool', 'function') for m in messages):
        raise ValueError('unknown episode message role')
    messages.append(deepcopy(choices[0]['message']))
    state = {'issue': {k: task[k] for k in ('problem_statement', 'hints_text')},
        'episode': messages, 'visibility': protocol['judge_input']['disclosure']}
    request = {'model': protocol['judge']['model'], 'state': state,
        'questions': {'repair_evidence': deepcopy(protocol['question'])}}
    binding = {'instance_id': task['instance_id'], 'arm': row['arm'], 'outcome_sha256': digest(row),
        'terminal_request_sha256': packet['request_sha256'],
        'terminal_response_sha256': completed['response_sha256'],
        'episode_sha256': digest(state), 'api_request_sha256': digest(request)}
    return request, binding


def verified_grade(receipt, request, response, binding, protocol_sha256):
    if (receipt.get('schema') != 'gemma4-behavior-grade-v1' or receipt.get('status') != 'qualified'
            or receipt.get('protocol_sha256') != protocol_sha256
            or any(receipt.get(k) != v for k, v in binding.items())
            or receipt.get('decision_response_sha256') != digest(response)):
        raise ValueError('behavior grade is not bound to the declared episode and protocol')
    validate_response(request, response)
    score = response['answers']['repair_evidence']['score'] / 4
    if receipt.get('normalized_score') != score:
        raise ValueError('behavior grade differs from its actual typed API response')
    return score


def load_verified_grades(cohort, grade_directory, tasks, outcomes, protocol, protocol_sha256):
    """Offline recheck of every qualified grade against its captured API bytes."""
    cohort, grade_directory = Path(cohort), Path(grade_directory)
    claim_path = cohort / 'jev-grading-claim.json'
    claim = json.loads(claim_path.read_text())
    result = json.loads((grade_directory / 'result.json').read_text())
    if (result.get('state') != 'complete' or result.get('protocol_sha256') != protocol_sha256
            or result.get('claim_sha256') != sha256_file(claim_path)
            or claim['out'] != str(grade_directory.resolve())
            or claim['protocol_sha256'] != protocol_sha256
            or claim.get('collection_plan_sha256') != protocol.get('collection', {}).get('plan_sha256')
            or claim['outcomes_sha256'] != sha256_file(cohort / 'outcomes.jsonl')
            or json.loads((grade_directory / 'claim.json').read_text()) != claim):
        raise ValueError('grading claim/result differs from the declared immutable pass')
    _, eligible = eligible_arms(tasks, outcomes, protocol)
    lookup = {t['instance_id']: t for t in tasks}
    grades, bindings, finished = {}, [], []
    for row in eligible:
        request, binding = episode_payload(lookup[row['instance_id']], row,
            cohort / row['arm'] / row['instance_id'] / 'requests', protocol)
        bindings.append(binding)
        folder = grade_directory / row['instance_id'] / row['arm']
        receipt = json.loads((folder / 'result.json').read_text())
        finished.append(receipt)
        if (json.loads((folder / 'request.json').read_text()) != request
                or receipt.get('request_file_sha256') != sha256_file(folder / 'request.json')
                or any(receipt.get(k) != v for k, v in binding.items())
                or receipt.get('protocol_sha256') != protocol_sha256):
            raise ValueError('grade request/episode binding changed')
        if receipt['status'] == 'qualified':
            response = json.loads((folder / 'response.json').read_text())
            if receipt.get('response_file_sha256') != sha256_file(folder / 'response.json'):
                raise ValueError('raw decision response bytes changed')
            verified_grade(receipt, request, response, binding, protocol_sha256)
            grades[(row['instance_id'], row['arm'])] = receipt
        elif receipt['status'] != 'unqualified':
            raise ValueError('unfinished or unknown grade cannot supply a target')
    if (bindings != claim['eligible'] or finished != result['finished']
            or result.get('qualified_grades') != len(grades)):
        raise ValueError('grading summary differs from its actual eligible arm receipts')
    return grades


def select_behavior_contexts(tasks, outcomes, grades, protocol, protocol_sha256):
    """Grades must already be checked against actual journal/API bytes."""
    grouped, eligible = eligible_arms(tasks, outcomes, protocol)
    eligible_keys = {(r['instance_id'], r['arm']) for r in eligible}
    if not set(grades) <= eligible_keys:
        raise ValueError('grade for an ineligible or unknown arm')
    rows = {(r['instance_id'], r['arm']): r for r in eligible}
    for key, grade in grades.items():
        score = grade.get('normalized_score')
        if (grade.get('status') != 'qualified' or grade.get('schema') != 'gemma4-behavior-grade-v1'
                or grade.get('protocol_sha256') != protocol_sha256
                or grade.get('outcome_sha256') != digest(rows[key])
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('only bound, qualified finite behavior grades can select targets')
    selected = []
    for task_id, arms in sorted(grouped.items()):
        off = grades.get((task_id, 'off'))
        if off is None:
            continue
        candidates = [r for r in arms.values() if (task_id, r['arm']) in eligible_keys
            and r['arm'] != 'off' and (task_id, r['arm']) in grades
            and grades[(task_id, r['arm'])]['normalized_score'] >= .75
            and grades[(task_id, r['arm'])]['normalized_score'] > off['normalized_score']]
        if not candidates:
            continue
        winner = min(candidates, key=lambda r: (-grades[(task_id, r['arm'])]['normalized_score'],
            r['tool_calls'], r['elapsed_seconds'], r['arm']))
        pair = pair_contexts(winner, arms['off'], initial_only=True)[0]
        teacher = grades[(task_id, winner['arm'])]
        pair.update(behavior_study=protocol['name'], behavior_protocol_sha256=protocol_sha256,
            boundary='initial-assistant-decision', quality_rubric=deepcopy(protocol['question']),
            behavior_grades={'off': deepcopy(off), 'teacher': deepcopy(teacher)})
        if 'collection' in protocol:
            pair['collection'] = deepcopy(protocol['collection'])
        selected.append(pair)
    return selected
