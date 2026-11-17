"""CPU recovery policy controls; no fixture is an actual training target."""
from copy import deepcopy
import json

import pytest

from gemma_particle.harness_identity import sha256_file
from gemma_particle.pool_recovery import reconcile_pool_prefix


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    previous=tmp_path/'old';curator=tmp_path/'curator';grade=previous/'task-a-grades'
    write(grade/'result.json',{'fixture':'not a real judgment'})
    (grade/'jev-contexts.jsonl').write_text('CPU fixture, not a fitting target\n')
    plan={'behavior_study':'fixture','protocol_sha256':'protocol','task_ids':['task-a','task-b','task-c'],
        'excluded_tasks':['probe'],'qualification_sha256':'qualification','visible_tasks_sha256':'visible',
        'raw_tasks_sha256':'raw','image':'fixture','image_id':'fixture-image','endpoint':'http://127.0.0.1:9999/v1',
        'executor':str(tmp_path/'executor'),'executor_commit':'original','prompt_profile':'repair-finalize-v2',
        'curator':str(curator),'curator_commit':'original','stop_rule':'two distinct tasks',
        'server_identity':{'device':'cuda:1','memory_fraction':.6,'observed_free_bytes':100,'model_identity':'original'}}
    write(previous/'plan.json',plan)
    finished=[{'instance_id':'task-a','grades':str(grade),'cohort':str(previous/'task-a'),
        'grade_result_sha256':sha256_file(grade/'result.json'),
        'contexts_sha256':sha256_file(grade/'jev-contexts.jsonl'),'admitted_contexts':1}]
    write(previous/'progress.json',{'state':'collecting','finished':finished,'admitted_contexts':1})
    write(previous/'task-b/off/cohort.json',{'task_ids':['task-b'],'arm':'off','runs':[]})
    write(previous/'task-b/off/task-b/requests/0000-request.json',{'fixture':'interrupted'})
    receipt={'schema':'gemma4-runtime-interruption-observation-v1','historical_supervisors':[
        {'role':'collect_behavior_pool.py','proc_exists':False,'historical_binding':
            {'pid':2**30,'birth_ticks':'not-a-live-birth','cwd':str(curator)}}],
        'interrupted_train_task':'task-b','interrupted_train_arm':'off','train_capture_terminal':False,
        'unattempted_train_arms':['reproduce','localize','verify']}
    write(tmp_path/'interruption.json',receipt)
    return previous,deepcopy(plan),tmp_path/'interruption.json',finished


def test_recovery_reuses_exact_prefix_and_continues_only_unattempted_suffix(tmp_path):
    previous,plan,receipt,expected=fixture(tmp_path)
    before={str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()}
    plan['server_identity']['observed_free_bytes']=80
    finished,recovery=reconcile_pool_prefix(previous,plan,receipt)
    assert finished==expected
    assert recovery['interrupted_task']=='task-b' and recovery['unattempted_suffix']==['task-c']
    assert 'off/task-b/requests/0000-request.json' in recovery['interrupted_artifact_sha256']
    assert {str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()}==before


def test_legacy_pool_image_defaults_are_bound_to_original_raw_image(tmp_path):
    previous,plan,receipt,_=fixture(tmp_path)
    original=json.loads((previous/'plan.json').read_text())
    original.pop('image');original.pop('image_id');write(previous/'plan.json',original)
    plan['image']='gemma-particle-sandbox:fastapi-deps-v2'
    raw=previous/'task-b/off/cohort.json';row=json.loads(raw.read_text())
    row['image_id']=plan['image_id'];write(raw,row)
    _,recovery=reconcile_pool_prefix(previous,plan,receipt)
    assert set(recovery['legacy_image_field_sources'])=={'image','image_id'}
    row['image_id']='changed-image';write(raw,row)
    with pytest.raises(ValueError,match='original fixed study'):
        reconcile_pool_prefix(previous,plan,receipt)


@pytest.mark.parametrize('change',['reorder','model','memory','grade','contexts','terminal',
    'teacher_attempt','suffix_attempt','prefix_gap','missing_receipt','target_count'])
def test_recovery_rejects_changed_contracts_evidence_or_attempted_suffix(tmp_path,change):
    previous,plan,receipt,_=fixture(tmp_path)
    if change=='reorder':plan['task_ids']=['task-a','task-c','task-b']
    elif change=='model':plan['server_identity']['model_identity']='different'
    elif change=='memory':plan['server_identity']['memory_fraction']=.8
    elif change=='grade':write(previous/'task-a-grades/result.json',{'changed':True})
    elif change=='contexts':(previous/'task-a-grades/jev-contexts.jsonl').write_text('changed')
    elif change=='terminal':write(previous/'task-b/off/task-b/trajectory.json',{})
    elif change=='teacher_attempt':(previous/'task-b/verify').mkdir()
    elif change=='suffix_attempt':(previous/'task-c').mkdir()
    elif change in ('prefix_gap','target_count'):
        p=previous/'progress.json';r=json.loads(p.read_text())
        if change=='prefix_gap':r['finished'][0]['instance_id']='task-c'
        else:r['admitted_contexts']=2
        write(p,r)
    elif change=='missing_receipt':write(receipt,{'schema':'unrelated'})
    with pytest.raises(ValueError):reconcile_pool_prefix(previous,plan,receipt)
