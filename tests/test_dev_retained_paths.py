"""CPU receipt-path controls, without any model or scorer fixtures."""
from copy import deepcopy
import json

import pytest

from gemma_particle.dev_evidence import retained_case_paths,journal_digest
from gemma_particle.harness_identity import sha256_file


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path):
    first=tmp_path/'first';next_run=tmp_path/'next'
    c=first/'primary/task-agent';v=first/'primary/task-verification'
    write(c/'task/requests/0000-request.json',{'fixture':'unscored journal'})
    write(v/'outcomes.jsonl',{'fixture':'unscored outcome'})
    plan={'condition':{'id':'off'},'selection_plan_sha256':'selection','packet':None,
        'task_ids':{'primary':['task']},'sources':{'scripts/run_dev_condition.py':'before',
            'src/gemma_particle/dev_evidence.py':'before','scripts/run_trajectories.py':'same'},'reused_evidence':None}
    write(first/'plan.json',plan);next_plan=deepcopy(plan)
    next_plan['sources']['src/gemma_particle/dev_evidence.py']='after'
    next_plan['reused_evidence']={'prior_condition':str(first),'prior_plan_sha256':sha256_file(first/'plan.json'),'task_ids':['task']}
    write(next_run/'plan.json',next_plan)
    record={'instance_id':'task','panel':'primary','cohort':str(c),'verification':str(v),
        'reused_from':str(first),'outcomes_sha256':sha256_file(v/'outcomes.jsonl'),'model_journal_sha256':journal_digest(c,'task')}
    write(next_run/'progress.json',{'finished':[record]})
    return first,next_run,c,v


def test_prior_reused_evidence_retains_actual_paths_and_never_copies_or_reruns(tmp_path):
    first,next_run,c,v=fixture(tmp_path)
    assert retained_case_paths(next_run,'primary','task')==(c,v)
    assert not (next_run/'primary').exists()
    # Any partial direct work remains visible for explicit reconciliation.
    partial=next_run/'primary/task-agent';partial.mkdir(parents=True)
    assert retained_case_paths(next_run,'primary','task')[0]==partial


@pytest.mark.parametrize('change',['plan','outcome','journal','path','duplicate','generation'])
def test_changed_prior_evidence_or_unbound_paths_are_rejected(tmp_path,change):
    first,next_run,c,v=fixture(tmp_path)
    if change=='plan':write(first/'plan.json',{'changed':True})
    elif change=='outcome':(v/'outcomes.jsonl').write_text('changed')
    elif change=='journal':write(c/'task/requests/0000-request.json',{'changed':True})
    elif change in ('path','duplicate'):
        p=next_run/'progress.json';r=json.loads(p.read_text())
        if change=='path':r['finished'][0]['cohort']=str(tmp_path/'other')
        else:r['finished'].append(deepcopy(r['finished'][0]))
        write(p,r)
    elif change=='generation':
        p=next_run/'plan.json';r=json.loads(p.read_text());r['sources']['scripts/run_trajectories.py']='changed';write(p,r)
    with pytest.raises(ValueError):retained_case_paths(next_run,'primary','task')
