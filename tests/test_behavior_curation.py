"""Synthetic admission controls; these fixtures are never actual TRAIN targets."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import sys

import pytest

from gemma_particle import behavior_curation
from gemma_particle.behavior_curation import (eligible_arms, episode_payload, select_behavior_contexts,
    verified_grade, load_behavior_cohort, load_verified_grades)
from gemma_particle.behavior_protocol import load_behavior_protocol
from gemma_particle.budgets import OFFICIAL_BUDGET, budget_values
from gemma_particle.capture import CONTEXT_CAPTURE_PROTOCOL, RequestJournal
from gemma_particle.curation import VERIFY_SUMMARY_INSTRUCTION_PROTOCOL
from gemma_particle.data import digest, write_jsonl
from gemma_particle.harness_identity import sha256_file
from gemma_particle.history import OFFICIAL_COMPACTION, history_contract
from gemma_particle.paired_collection import PAIR, PILOT_QUARTET
from test_data import arms


def study_rows(*, passing_off=True, instruction_protocol='tools-v2'):
    task = dict(instance_id='task', repo='fastapi/fastapi', base_commit='a'*40,
        problem_statement='Return zero for an empty list.', hints_text='', created_at='2026-01-01', split='train')
    rows = arms(off=passing_off, protocol=instruction_protocol)
    for row in rows:
        row.update(schema='gemma4-verified-trajectory-v4', agent_error=None, capture_error=None,
            capture_protocol=CONTEXT_CAPTURE_PROTOCOL, context_budget_proof=None,
            history_profile=OFFICIAL_COMPACTION, budget_profile=OFFICIAL_BUDGET)
        row['verification'].update(capture_protocol=CONTEXT_CAPTURE_PROTOCOL,
            context_budget_proof_sha256=digest(None), history_profile=OFFICIAL_COMPACTION,
            test_exit_code=0 if row['resolved'] else 1, error=None,
            evaluation_contract_sha256=row['evaluation_contract_sha256'], origin_checks=[{'workspace': True}])
    protocol = load_behavior_protocol(Path(__file__).resolve().parents[1]/'research/jev-evidence-loop-v1.json')
    return [task], rows, protocol


def grade(row, score):
    return dict(schema='gemma4-behavior-grade-v1', status='qualified',
        normalized_score=score, protocol_sha256='protocol', outcome_sha256=digest(row))


def test_behavior_selection_can_use_passing_off_but_requires_success_and_strict_gain():
    tasks, rows, protocol = study_rows()
    grades = {('task', 'off'): grade(rows[0], .75), ('task', 'verify'): grade(rows[1], .9)}
    pairs = select_behavior_contexts(tasks, rows, grades, protocol, 'protocol')
    assert len(pairs) == 1 and pairs[0]['decision'] == 0
    assert pairs[0]['boundary'] == 'initial-assistant-decision'
    assert pairs[0]['neutral_messages'] == rows[0]['first_request']['messages']
    assert all(m['role'] in ('system', 'user') for m in pairs[0]['neutral_messages'])
    assert pairs[0]['behavior_grades']['off']['normalized_score'] == .75
    for off, teacher in [(.9, .9), (.5, .7475)]:
        grades = {('task', 'off'): grade(rows[0], off), ('task', 'verify'): grade(rows[1], teacher)}
        assert select_behavior_contexts(tasks, rows, grades, protocol, 'protocol') == []
    assert select_behavior_contexts(tasks, rows, {}, protocol, 'protocol') == []


def test_behavior_rejects_failed_incomplete_and_error_teachers_and_dirty_off():
    tasks, rows, protocol = study_rows()
    _, eligible = eligible_arms(tasks, rows, protocol)
    assert [r['arm'] for r in eligible] == ['off', 'verify']
    with pytest.raises(ValueError, match='ineligible'):
        select_behavior_contexts(tasks, rows, {('task', 'reproduce'): grade(rows[2], 1)}, protocol, 'protocol')
    for field, value in [('agent_error', 'timeout'), ('capture_error', 'missing response'), ('capture_complete', False)]:
        damaged = deepcopy(rows); damaged[1][field] = value
        assert eligible_arms(tasks, damaged, protocol)[1] == []
    damaged = deepcopy(rows); damaged[0]['agent_error'] = 'timeout'
    assert eligible_arms(tasks, damaged, protocol)[1] == []
    damaged = deepcopy(rows); damaged[1]['verification']['origin_checks'] = [{'workspace': False}]
    with pytest.raises(ValueError, match='edited workspace'):
        eligible_arms(tasks, damaged, protocol)
    with pytest.raises(ValueError, match='four declared'):
        eligible_arms(tasks, rows[:-1], protocol)


def test_initial_boundary_cannot_crop_or_change_history_and_grades_cannot_be_nonfinite():
    tasks, rows, protocol = study_rows()
    winner = rows[1]
    winner['first_request']['messages'].append({'role': 'assistant', 'content': 'Earlier action'})
    winner['requests'][0]['request_sha256'] = digest(winner['first_request'])
    winner['verification'].update(requests_sha256=digest(winner['requests']), first_request_sha256=digest(winner['first_request']))
    with pytest.raises(ValueError, match='initial target'):
        eligible_arms(tasks, rows, protocol)
    tasks, rows, protocol = study_rows()
    grades = {('task', 'off'): grade(rows[0], .5), ('task', 'verify'): grade(rows[1], float('nan'))}
    with pytest.raises(ValueError, match='finite'):
        select_behavior_contexts(tasks, rows, grades, protocol, 'protocol')
    rows[0]['instance_id'] = tasks[0]['instance_id'] = 'fastapi_13207'
    # Unknown/mismatched request identities fail before reserved-task selection.
    with pytest.raises(ValueError):
        eligible_arms(tasks, rows, protocol)


def test_real_journal_shape_blinds_judge_and_binds_response_and_grade(tmp_path):
    tasks, rows, protocol = study_rows()
    row = rows[1]
    journal = RequestJournal(tmp_path/'requests', instance_id='task', arm='verify')
    initial = row['first_request']
    first = journal.begin(model=initial['model'], messages=initial['messages'], tools=None, options=initial['generation'])
    journal.complete(first, {'choices': [{'message': {'role': 'assistant', 'content': 'I will investigate.'}}]})
    messages = deepcopy(initial['messages']) + [
        {'role': 'assistant', 'content': 'Run the focused check.'},
        {'role': 'tool', 'content': 'Before: expected zero, got None. After: zero; tests pass.', 'tool_call_id': 'call'}]
    last = journal.begin(model=initial['model'], messages=messages, tools=None, options=initial['generation'])
    journal.complete(last, {'choices': [{'message': {'role': 'assistant', 'content': 'The focused check passed.'}}]})
    row['requests'] = [first, last]
    request, binding = episode_payload(tasks[0], row, journal.directory, protocol)
    assert request['state']['issue'] == {k: tasks[0][k] for k in ('problem_statement', 'hints_text')}
    assert all(m['role'] not in ('system', 'developer') for m in request['state']['episode'])
    assert request['state']['episode'][-1]['content'] == 'The focused check passed.'
    assert 'verification' not in request['state'] and 'arm' not in request['state']
    response = {'model': '~typesafe/jev-latest', 'answers': {'repair_evidence': {'type': 'score',
        'score': 3.6, 'confidence': .8, 'probabilities': {'0': 0, '1': 0, '2': 0, '3': .4, '4': .6}}}}
    receipt = dict(schema='gemma4-behavior-grade-v1', status='qualified', **binding,
        protocol_sha256='protocol', normalized_score=.9, decision_response_sha256=digest(response))
    assert verified_grade(receipt, request, response, binding, 'protocol') == .9
    receipt['normalized_score'] = .95
    with pytest.raises(ValueError, match='actual typed'):
        verified_grade(receipt, request, response, binding, 'protocol')
    path = journal.directory/'0001-response.json'
    data = json.loads(path.read_text()); data['response']['choices'][0]['message']['content'] = 'altered'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='completion journal'):
        episode_payload(tasks[0], row, journal.directory, protocol)


def test_behavior_tie_break_uses_score_before_cost_and_rejects_changed_grade_binding():
    tasks, rows, protocol = study_rows()
    candidate = rows[2]
    candidate['resolved'] = candidate['verification']['resolved'] = True
    candidate['verification']['test_exit_code'] = 0
    candidate['tool_calls'] = 100
    grades = {('task', 'off'): grade(rows[0], .5), ('task', 'verify'): grade(rows[1], .8),
        ('task', 'reproduce'): grade(candidate, .9)}
    assert select_behavior_contexts(tasks, rows, grades, protocol, 'protocol')[0]['target_arm'] == 'reproduce'
    grades[('task', 'reproduce')]['outcome_sha256'] = 'changed'
    with pytest.raises(ValueError, match='bound'):
        select_behavior_contexts(tasks, rows, grades, protocol, 'protocol')


def synthetic_cohort(tmp_path, *, pilot_arms=None):
    tasks, rows, protocol = study_rows()
    root = tmp_path/'cohort'; root.mkdir()
    protocol_path = Path(__file__).resolve().parents[1]/'research/jev-evidence-loop-v1.json'
    (root/'behavior-protocol.json').write_bytes(protocol_path.read_bytes())
    executor = tmp_path/'frozen'; executor.mkdir(); (executor/'source.py').write_text('# synthetic source identity fixture\n')
    server = dict(kind='gemma4', pretrained_inference=True, strength=0, native_checkpoint=None,
        model_identity=rows[0]['model_identity'])
    plan = dict(behavior_study=protocol['name'], behavior_protocol_sha256=sha256_file(protocol_path),
        teacher_policy='all-four', arms=['off','reproduce','localize','verify'], task_ids=['task'],
        source_checkout=str(executor), source_git='a'*40, execution_sources={'source.py':sha256_file(executor/'source.py')},
        budget_profile=OFFICIAL_BUDGET, budget=budget_values(OFFICIAL_BUDGET),
        server_identity=server, history=history_contract(OFFICIAL_COMPACTION))
    if pilot_arms:
        plan.update(arms=list(pilot_arms), teacher_policy='paired-verify' if pilot_arms == PAIR else 'all-four',
            collection_pilot={'path': str(tmp_path/'pilot.json'), 'sha256': 'synthetic declaration'})
        rows = sorted((row for row in rows if row['arm'] in pilot_arms),
            key=lambda row: pilot_arms.index(row['arm']))
        if pilot_arms == PILOT_QUARTET:
            for row in rows:
                if row['arm'] in ('reproduce', 'verify'):
                    success = row['arm'] == 'reproduce'
                    row['resolved'] = row['verification']['resolved'] = success
                    row['verification']['test_exit_code'] = 0 if success else 1
    contract = dict(budget=dict(time_minutes=60, tool_calls=100, turns=500, cost_usd=None, total_tokens=None),
        harness=dict(command_timeout_seconds=300), history=history_contract(OFFICIAL_COMPACTION))
    for row in rows:
        arm = root/row['arm']; arm.mkdir()
        journal = RequestJournal(arm/'task/requests', instance_id='task', arm=row['arm'])
        first = row['first_request']
        packet = journal.begin(model=first['model'], messages=first['messages'], tools=first['tools'], options=first['generation'])
        journal.complete(packet, {'choices':[{'message':{'role':'assistant','content':'Synthetic completion for admission control.'}}]})
        row.update(requests=[packet], first_request=packet['request'], visible_task_sha256=digest(tasks[0]),
            evaluation_contract_sha256=digest(contract))
        row['verification'].update(requests_sha256=digest(row['requests']), first_request_sha256=digest(row['first_request']),
            evaluation_contract_sha256=digest(contract), image_id='synthetic-image')
        (arm/'cohort.json').write_text(json.dumps(dict(arm=row['arm'], server_identity=server,
            contract=contract, evaluation_contract_sha256=digest(contract), image_id='synthetic-image')))
        if pilot_arms:
            verified = root/(row['arm']+'-verification')
            verified.mkdir()
            write_jsonl(verified/'outcomes.jsonl', [row])
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'progress.json').write_text(json.dumps({'state':'complete'}))
    write_jsonl(root/'outcomes.jsonl', rows)
    visible = tmp_path/'visible.jsonl'; write_jsonl(visible, tasks)
    return root, visible, protocol_path


@pytest.mark.parametrize('pilot_arms', [PAIR, PILOT_QUARTET])
def test_paired_pilot_grading_and_offline_comparison(tmp_path, monkeypatch, pilot_arms):
    root, visible, protocol_path = synthetic_cohort(tmp_path, pilot_arms=pilot_arms)
    monkeypatch.setattr(behavior_curation, 'load_collection_pilot',
        lambda *_: [{'instance_id': 'task', 'planned_arms': list(pilot_arms)}])
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    def load(name):
        spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    grade_script = load('grade_behavior_cohort')
    curator = load('curate_behavior_cohort')
    calls = []
    def synthetic_decide(request, **options):
        calls.append(request)
        score = 2 if len(calls) == 1 else 4
        response = dict(model='~typesafe/jev-latest', answers={'repair_evidence':dict(type='score',
            score=score, confidence=.8, probabilities={str(i):float(i == score) for i in range(5)})})
        Path(options['response_path']).write_text(json.dumps(response))
        return response
    monkeypatch.setattr(grade_script, 'decide', synthetic_decide)
    common = ['--cohort',str(root),'--visible-tasks',str(visible),'--protocol',str(protocol_path)]
    grades = tmp_path/'graded'
    monkeypatch.setattr(sys, 'argv', ['grader',*common,'--out',str(grades)])
    grade_script.main()
    curated = tmp_path/'curated'
    monkeypatch.setattr(sys, 'argv', ['curator',*common,'--grades',str(grades),'--out',str(curated)])
    curator.main()
    assert len(calls) == 2
    assert (curated/'jev-contexts.jsonl').read_bytes() == (grades/'jev-contexts.jsonl').read_bytes()
    result = json.loads((curated/'result.json').read_text())
    assert result['admitted_contexts'] == 1
    assert result['pilot_comparison']['verify_only_admitted'] is (pilot_arms == PAIR)
    assert result['pilot_comparison']['selected_arm'] == ('verify' if pilot_arms == PAIR else 'reproduce')
    (root/'verify-verification/outcomes.jsonl').write_text('')
    with pytest.raises(ValueError, match='actual fresh verifier'):
        load_behavior_cohort(root, visible, protocol_path)


def test_grading_script_single_pass_offline_revalidation_and_tamper_rejection(tmp_path, monkeypatch):
    root, visible, protocol_path = synthetic_cohort(tmp_path)
    script = Path(__file__).resolve().parents[1]/'scripts/grade_behavior_cohort.py'
    spec = importlib.util.spec_from_file_location('behavior_grader_under_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    def synthetic_decide(request, **kwargs):
        calls.append(request)
        score = 2 if len(calls)==1 else 4
        response = dict(model='~typesafe/jev-latest', answers={'repair_evidence':dict(type='score',
            score=score, confidence=.8, probabilities={str(i):float(i==score) for i in range(5)})})
        Path(kwargs['response_path']).write_text(json.dumps(response))
        return response
    monkeypatch.setattr(module, 'decide', synthetic_decide)
    out = tmp_path/'grades'
    argv = ['grader','--cohort',str(root),'--visible-tasks',str(visible),
        '--protocol',str(protocol_path),'--out',str(out)]
    monkeypatch.setattr(sys, 'argv', argv); module.main()
    assert len(calls)==2
    result = json.loads((out/'result.json').read_text())
    assert result['admitted_contexts']==1 and result['distinct_fit_tasks']==1
    protocol, protocol_sha, tasks, outcomes = load_behavior_cohort(root, visible, protocol_path)
    grades = load_verified_grades(root, out, tasks, outcomes, protocol, protocol_sha)
    replay = select_behavior_contexts(tasks, outcomes, grades, protocol, protocol_sha)
    assert replay == [json.loads(line) for line in (out/'jev-contexts.jsonl').read_text().splitlines()]
    # A new directory cannot silently purchase another pass on the same quartet.
    monkeypatch.setattr(sys, 'argv', argv[:-1]+[str(tmp_path/'reroll')])
    with pytest.raises(FileExistsError): module.main()
    assert len(calls)==2
    response_path = out/'task/verify/response.json'
    response_path.write_text(response_path.read_text()+'\n')
    with pytest.raises(ValueError, match='raw decision response bytes'):
        load_verified_grades(root, out, tasks, outcomes, protocol, protocol_sha)


def test_verify_summary_requires_prospective_protocol_and_preserves_off():
    tasks, original, protocol = study_rows(passing_off=False)
    _, variant, _ = study_rows(passing_off=False, instruction_protocol=VERIFY_SUMMARY_INSTRUCTION_PROTOCOL)
    assert original[0]['first_request'] == variant[0]['first_request']
    assert original[2]['first_request'] == variant[2]['first_request']
    assert original[3]['first_request'] == variant[3]['first_request']
    assert 'record the exact commands' in variant[1]['first_request']['messages'][0]['content']
    with pytest.raises(ValueError, match='declared collection profiles'):
        eligible_arms(tasks, variant, protocol)
    protocol['instruction_protocol'] = VERIFY_SUMMARY_INSTRUCTION_PROTOCOL
    assert [row['arm'] for row in eligible_arms(tasks, variant, protocol)[1]] == ['off', 'verify']
    with pytest.raises(ValueError, match='unmatched instruction protocols'):
        eligible_arms(tasks, [original[0], *variant[1:]], protocol)


def test_full_tool_trace_recovers_observed_results_outside_terminal_history(tmp_path):
    tasks, rows, terminal = study_rows(passing_off=False)
    row = rows[1]
    journal = RequestJournal(tmp_path / 'requests', instance_id='task', arm='verify')
    initial = row['first_request']
    start = journal.begin(model=initial['model'], messages=initial['messages'], tools=None, options=initial['generation'])
    call = {'id': 'check', 'function': {'name': 'run_command', 'arguments': '{"command":"pytest -q"}'}}
    journal.complete(start, {'choices': [{'message': {'role': 'assistant', 'tool_calls': [call]}}]})
    reply = {'role': 'tool', 'tool_call_id': 'check', 'content': 'Observed failing regression'}
    midway = journal.begin(model=initial['model'], messages=[*initial['messages'], reply], tools=None,
                           options=initial['generation'])
    missing = {'id': 'not-observed', 'function': {'name': 'get_status', 'arguments': '{'}}
    journal.complete(midway, {'choices': [{'message': {'role': 'assistant', 'tool_calls': [missing]}}]})
    latter = journal.begin(model=initial['model'], messages=initial['messages'], tools=None, options=initial['generation'])
    journal.complete(latter, {'choices': [{'message': {'role': 'assistant', 'content': 'Final reply'}}]})
    row['requests'] = [start, midway, latter]

    protocol = load_behavior_protocol(Path(__file__).resolve().parents[1] / 'research/jev-full-tool-trace-v1.json')
    request, binding = episode_payload(tasks[0], row, journal.directory, protocol)
    episode = request['state']['episode']
    assert episode['completed_tools'] == [{'decision': 0, 'tool': 'run_command',
        'arguments': {'command': 'pytest -q'}, 'observed_result': reply['content']}]
    assert episode['tool_proposals_without_confirmed_result'] == [
        {'decision': 1, 'tool': 'get_status', 'arguments': '{'}]
    assert episode['final_reply'] == 'Final reply' and 'trace_responses_sha256' in binding
    assert request['questions']['repair_evidence'] == terminal['question']
    assert protocol['admission'] == terminal['admission']
    ordinary_request, ordinary_binding = episode_payload(tasks[0], row, journal.directory, terminal)
    assert not any(message['role'] == 'tool' for message in ordinary_request['state']['episode'])
    assert 'trace_responses_sha256' not in ordinary_binding
