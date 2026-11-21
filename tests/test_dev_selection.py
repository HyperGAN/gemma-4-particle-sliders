"""CPU selection fixtures; never real DEV scores or model executions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gemma_particle.data import digest
from gemma_particle.harness_identity import sha256_file
from gemma_particle.dev_selection import (empf_only_selection,load_selection_plan,summarize_verified_panel,
    paired_comparison,choose_learned_candidate,wilson_interval,EMPTY_PATCH_SHA256)

pytestmark = pytest.mark.usefixtures("archived_evidence")


def fixture_row(task,resolved,*,tools=5,error=None):
    request={'model':'openai/fixture','messages':[{'role':'user','content':'CPU fixture'}],
        'tools':None,'generation':{'seed':0}}
    packets=[{'instance_id':task['instance_id'],'arm':'off','decision':0,'request':request,'request_sha256':digest(request)}]
    receipt={'stage':'fresh-container-tests','instance_id':task['instance_id'],'patch_sha256':'a'*64,
        'requests_sha256':digest(packets),'first_request_sha256':digest(request),'resolved':resolved,
        'evaluation_contract_sha256':'fixture-contract','error':None,'test_exit_code':0 if resolved else 1,
        'origin_checks':[{'workspace':True}]}
    return {'schema':'gemma4-verified-trajectory-v4','generator_kind':'gemma4','instance_id':task['instance_id'],
        'split':'dev','resolved':resolved,'capture_complete':True,'agent_error':error,'tool_calls':tools,
        'elapsed_seconds':2.,'patch_sha256':'a'*64,'requests':packets,'first_request':request,
        'visible_task_sha256':digest(task),'evaluation_contract_sha256':'fixture-contract','verification':receipt}


def fixtures(tmp_path):
    # Keep the declared grid, but bind only local synthetic panel files. CPU
    # controls must not require private Kaggle downloads or qualification runs.
    plan=json.loads((Path(__file__).resolve().parents[1]/'research/dev-selection-v1.json').read_text())
    tasks={}
    for panel,n in [('primary',42),('secondary',13)]:
        tasks[panel]=[dict(instance_id=f'{panel}-fixture-{i}',split='dev',repo='fixture/repository') for i in range(n)]
        plan[panel]['task_ids']=[t['instance_id'] for t in tasks[panel]]
        manifest=tmp_path/(panel+'-manifest.json');qualification=tmp_path/(panel+'-qualification.json')
        manifest.write_text(json.dumps({'fixture':'CPU synthetic panel','task_ids':plan[panel]['task_ids']}))
        qualification.write_text(json.dumps({'fixture':'CPU receipt; not an actual environment qualification'}))
        plan[panel].update(manifest=str(manifest),manifest_sha256=sha256_file(manifest),
            qualification=str(qualification),qualification_sha256=sha256_file(qualification))
    plan['secondary']['calibrated_task_ids']=plan['secondary']['task_ids'][:5]
    packet=tmp_path/'selection-plan.json';packet.write_text(json.dumps(plan));plan=load_selection_plan(packet)
    result={'status':'complete'}
    for panel in tasks:
        calibrated=plan['secondary']['calibrated_task_ids'] if panel=='secondary' else ()
        result[panel]=summarize_verified_panel(tasks[panel],[fixture_row(t,i<2) for i,t in enumerate(tasks[panel])],calibrated_ids=calibrated)
    results={c['id']:deepcopy(result) for c in plan['conditions']}
    return plan,tasks,results


def test_empf_only_scope_preserves_controls_and_original_policy(tmp_path):
    original, _, _ = fixtures(tmp_path)
    origin = tmp_path / 'selection-plan.json'
    path = tmp_path / 'empf-only.json'
    selected = empf_only_selection(original, origin)
    path.write_text(json.dumps(selected))
    assert load_selection_plan(path) == selected
    assert len(selected['conditions']) == 13
    assert sum(row['role']=='learned' for row in selected['conditions']) == 9
    assert selected['conditions'][:4] == original['conditions'][:4]
    assert selected['confirmation'] == original['confirmation']
    selected['promotion']['primary_strict_gain'] = 'changed'
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='may change only the recipe grid'):
        load_selection_plan(path)


def test_complete_denominators_keep_errors_and_reject_grades_training_or_changed_receipts():
    tasks=[dict(instance_id='fixture-a',split='dev',repo='fixture/repo'),dict(instance_id='fixture-b',split='dev',repo='fixture/repo')]
    rows=[fixture_row(tasks[0],True),fixture_row(tasks[1],False,error='timeout')]
    summary=summarize_verified_panel(tasks,rows)
    assert summary['total']==2 and summary['resolved']==1 and summary['agent_errors']==1
    with pytest.raises(ValueError,match='every fixed'):summarize_verified_panel(tasks,rows[:1])
    with pytest.raises(ValueError,match='duplicate'):summarize_verified_panel(tasks,rows+[rows[0]])
    damaged=deepcopy(rows);damaged[0]['resolved']=.99
    with pytest.raises(ValueError,match='actual verified'):summarize_verified_panel(tasks,damaged)
    damaged=deepcopy(rows);damaged[0]['verification']['patch_sha256']='changed'
    with pytest.raises(ValueError,match='verified patch'):summarize_verified_panel(tasks,damaged)
    with pytest.raises(ValueError,match='only the fixed DEV'):
        summarize_verified_panel([dict(tasks[0],split='train')],[rows[0]])


def test_full_grid_and_primary_gain_secondary_floor_control_promotion(tmp_path):
    plan,tasks,results=fixtures(tmp_path);candidate=next(c for c in plan['conditions'] if c['role']=='learned')
    assert choose_learned_candidate(plan,results)['selected'] is None
    # Elapsed time alone cannot promote a condition with identical outcomes/tools.
    rows=[fixture_row(t,i<2) for i,t in enumerate(tasks['primary'])]
    for row in rows:row['elapsed_seconds']=.1
    results[candidate['id']]['primary']=summarize_verified_panel(tasks['primary'],rows)
    assert choose_learned_candidate(plan,results)['selected'] is None
    results[candidate['id']]['primary']=summarize_verified_panel(tasks['primary'],
        [fixture_row(t,i<3) for i,t in enumerate(tasks['primary'])])
    selected=choose_learned_candidate(plan,results)
    assert selected['selected']==candidate['id']
    assert selected['primary_paired_vs_off'][candidate['id']]['wins']==1
    results[candidate['id']]['secondary']=summarize_verified_panel(tasks['secondary'],
        [fixture_row(t,i<1) for i,t in enumerate(tasks['secondary'])],calibrated_ids=plan['secondary']['calibrated_task_ids'])
    assert choose_learned_candidate(plan,results)['selected'] is None
    incomplete=deepcopy(results);incomplete.pop(plan['conditions'][-1]['id'])
    with pytest.raises(ValueError,match='all declared'):choose_learned_candidate(plan,incomplete)


def test_aggregate_tampering_and_nonterminal_conditions_cannot_select(tmp_path):
    plan,_,results=fixtures(tmp_path);candidate=next(c['id'] for c in plan['conditions'] if c['role']=='learned')
    results[candidate]['primary']['resolved']=42
    with pytest.raises(ValueError,match='aggregate differs'):choose_learned_candidate(plan,results)
    plan,_,results=fixtures(tmp_path);results[candidate]={'status':'running'}
    with pytest.raises(ValueError,match='unfinished'):choose_learned_candidate(plan,results)
    results[candidate]={'status':'unqualified'}
    with pytest.raises(ValueError,match='preflight error evidence'):choose_learned_candidate(plan,results)


def test_paired_uncertainty_and_interval_boundaries():
    tasks=[dict(instance_id=str(i),split='dev',repo='fixture/repo') for i in range(4)]
    off=summarize_verified_panel(tasks,[fixture_row(t,False) for t in tasks])
    candidate=summarize_verified_panel(tasks,[fixture_row(t,True) for t in tasks])
    paired=paired_comparison(candidate,off)
    assert paired=={'wins':4,'losses':0,'ties':0,'exact_discordance_p':.125}
    assert wilson_interval(0,42)[0]==0 and wilson_interval(42,42)[1]==1
    with pytest.raises(ValueError):wilson_interval(43,42)


def test_agent_error_with_no_patch_preserves_verifier_result_but_mirrors_official_failure():
    task=dict(instance_id='fixture',split='dev',repo='fixture/repo')
    row=fixture_row(task,True,error='failed before patch')
    row['patch_sha256']=row['verification']['patch_sha256']=EMPTY_PATCH_SHA256
    result=summarize_verified_panel([task],[row])
    assert result['resolved']==0 and result['agent_errors']==1
    assert result['tasks']['fixture']['fresh_test_resolved'] is True
    assert result['tasks']['fixture']['resolved'] is False
