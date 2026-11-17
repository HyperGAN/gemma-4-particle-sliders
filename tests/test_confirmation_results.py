"""Complete synthetic CPU pair artifacts; never real model/holdout outcomes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from gemma_particle.capture import RequestJournal,load_requests
from gemma_particle.confirmation import (claim_confirmation_pair,confirmation_pair_identity,reserve_confirmation_task)
from gemma_particle.confirmation_environments import load_confirmation_environments
from gemma_particle.confirmation_results import (summarize_confirmation_rows,compare_confirmation_pair)
from gemma_particle.curation import instruction_set
from gemma_particle.data import digest
from gemma_particle.dev_evidence import canonical_contract,journal_digest
from gemma_particle.harness_identity import sha256_file
from gemma_particle.history import OFFICIAL_COMPACTION,history_contract
from gemma_particle.lora_serving import BASE_ALIAS
from test_confirmation_environments import environment_fixture
from test_conversion_handoff import put

ROOT=Path(__file__).resolve().parents[1]


def pair_fixture(root):
    env_path,scope=environment_fixture(root)
    result=json.loads(env_path.read_text());plan_path=Path(result['plan']);plan=json.loads(plan_path.read_text())
    plan['harness']={'fixture':'CPU packages, no actual execution'};put(plan_path,plan)
    result['plan_sha256']=sha256_file(plan_path);put(env_path,result)
    env=load_confirmation_environments(env_path,scope);env['qualification_sha256']=sha256_file(env_path)
    out=root/'pair';claim=claim_confirmation_pair(scope,out);identity=confirmation_pair_identity(scope,claim)
    baseline=json.loads((root/'CPU-baseline-server.json').read_text())
    server=deepcopy(baseline);server.update(base_only=False,adapters=scope.receipt['selected_packet']['adapters'])
    server['launch'].extend(['--lora-modules',json.dumps({'name':'particle_lora','path':'/CPU-fixture','base_model_name':BASE_ALIAS})])
    server_path=out/'CPU-server.json';put(server_path,server)
    m=scope.plan['matching'];sources={str(root/'CPU-controller.py'):sha256_file(root/'CPU-controller.py')}
    paths=[]
    for condition in scope.receipt['permitted_conditions']:
        learned=condition!='off';finished=[];rows=[];contract_digest=None
        for i,task in enumerate(scope.rows):
            task_id=task['instance_id'];cohort=out/condition/(task_id+'-agent');folder=cohort/task_id;folder.mkdir(parents=True)
            verification=out/condition/(task_id+'-verification');dest=verification/task_id;dest.mkdir(parents=True)
            reservation=reserve_confirmation_task(scope,claim,condition,task_id,cohort)
            alias='particle_lora' if learned else BASE_ALIAS
            journal=RequestJournal(folder/'requests',instance_id=task_id,arm='off')
            packet=journal.begin(model='openai/'+alias,messages=[],tools=None,options={})
            journal.complete(packet,{'model':alias,'choices':[{'message':{'role':'assistant','content':'CPU fixture'}}]})
            requests=load_requests(folder/'requests');first=packet['request']
            error='CPU fixture agent error' if learned and i==3 else None
            fresh=(i%4==0 or learned and i in (1,2,3)) and not (learned and i==4)
            patch='' if error else 'CPU fixture patch, not actual code'
            (folder/'agent.patch').write_text(patch);(dest/'test-output.txt').write_text('CPU fixture tests, not real results')
            contract={'budget':m['budget'],'harness':{'command_timeout_seconds':m['command_timeout_seconds']},
                'sampling_sha256':m['sampling_file_sha256'],'common_prompt':{'profile':m['prompt_profile'],'sha256':m['common_system_sha256']},
                'instruction_protocol':m['instruction_protocol'],'instruction_set_sha256':digest(instruction_set(m['instruction_protocol'])),
                'history':history_contract(OFFICIAL_COMPACTION),'model_identity':server,'confirmation':identity,
                'image_id':plan['image_id'],'environment':{'packages':plan['harness'],'wheels':plan['wheels'],
                    'sandbox_setup_sha256':plan['input_files']['sandbox/setup.py']}}
            if learned:contract['candidate_adapters']=scope.receipt['selected_packet']['adapters']
            captured={'schema':'gemma4-captured-trajectory-v1','instance_id':task_id,'split':'holdout','arm':'off',
                'generator_kind':'gemma4','capture_complete':True,'frozen_base_generator':not learned,
                'patch_sha256':sha256_file(folder/'agent.patch'),'requests_sha256':digest(requests),
                'visible_task_sha256':digest(task),'snapshot_sha256':plan['input_files'][f'snapshots/{task_id}.tgz'],
                'evaluation_contract_sha256':digest(contract),'confirmation':identity,'confirmation_task_reservation_sha256':reservation,
                'agent_error':error,'tool_calls':0,'elapsed_seconds':1.0,'fixture':'CPU structure only, no model execution'}
            if learned:captured['candidate_adapters']=scope.receipt['selected_packet']['adapters']
            manifest={'generator_kind':'gemma4','arm':'off','task_ids':[task_id],'runs':[captured],
                'contract':contract,'evaluation_contract_sha256':digest(contract),'server_identity':server}
            put(cohort/'cohort.json',manifest);put(folder/'trajectory.json',captured)
            receipt={'stage':'fresh-container-tests','instance_id':task_id,'patch_sha256':captured['patch_sha256'],
                'requests_sha256':digest(requests),'first_request_sha256':digest(first),'resolved':fresh,
                'evaluation_contract_sha256':digest(contract),'error':None,'test_exit_code':0 if fresh else 1,
                'origin_checks':[{'workspace':True,'origin':'/workspace/CPU-fixture.py'}],
                'test_output_sha256':sha256_file(dest/'test-output.txt')}
            put(dest/'verification.json',receipt)
            row={**captured,'schema':'gemma4-verified-trajectory-v4','requests':requests,'first_request':first,
                'resolved':fresh,'verification':receipt}
            (verification/'outcomes.jsonl').write_text(json.dumps(row)+'\n')
            rows.append(row);contract_digest=digest(canonical_contract(contract))
            finished.append({'instance_id':task_id,'cohort':str(cohort),'verification':str(verification),
                'outcomes_sha256':sha256_file(verification/'outcomes.jsonl'),'model_journal_sha256':journal_digest(cohort,task_id)})
        path=out/condition/'result.json'
        put(path,{'schema':'gemma4-confirmation-condition-v1','status':'complete','condition_id':condition,
            'confirmation':identity,'environment_qualification_sha256':env['qualification_sha256'],'sources':sources,
            'server_identity':str(server_path),'server_identity_sha256':digest(server),'finished':finished,
            'canonical_contract_sha256':contract_digest,
            'summary':summarize_confirmation_rows(scope.rows,rows,qualified_ids=env['qualified_task_ids'])})
        paths.append(path)
    return paths,scope,claim,env,env_path


def test_complete_pair_reaudits_all98_and_preserves_official_error_accounting(tmp_path):
    paths,scope,claim,env,_=pair_fixture(tmp_path)
    result=compare_confirmation_pair(paths,scope,claim,env)
    assert result['declared_total']==49 and result['off']['total']==result['selected']['total']==49
    assert result['off']['resolved']==13 and result['selected']['resolved']==14
    assert result['paired']['wins']==2 and result['paired']['losses']==1 and result['paired']['ties']==46
    assert result['selected']['agent_errors']==1
    task=scope.rows[3]['instance_id'];row=result['selected']['tasks'][task]
    assert row['fresh_test_resolved'] is True and row['resolved'] is False
    assert result['new_candidate_selected'] is False and result['official_score'] is None


@pytest.mark.parametrize('damage',['drop','reorder','patch','journal','test_output','summary','claim'])
def test_partial_changed_or_rebound_evidence_cannot_complete_confirmation(tmp_path,damage):
    paths,scope,claim,env,_=pair_fixture(tmp_path);path=paths[1];result=json.loads(path.read_text());record=result['finished'][0]
    if damage=='drop':result['finished'].pop()
    elif damage=='reorder':result['finished'][0],result['finished'][1]=result['finished'][1],result['finished'][0]
    elif damage=='patch':(Path(record['cohort'])/record['instance_id']/'agent.patch').write_text('changed CPU fixture')
    elif damage=='journal':
        p=Path(record['cohort'])/record['instance_id']/'requests/0000-response.json';p.write_text('changed CPU response')
    elif damage=='test_output':(Path(record['verification'])/record['instance_id']/'test-output.txt').write_text('changed CPU test output')
    elif damage=='summary':result['summary']['resolved']=49
    else:claim.write_text('changed CPU claim')
    put(path,result)
    with pytest.raises(ValueError):compare_confirmation_pair(paths,scope,claim,env)


@pytest.mark.parametrize('script',['run_confirmation_pair.py','compare_confirmation.py'])
def test_real_pair_clis_keep_unreleased_tasks_and_environments_unopened(tmp_path,script):
    access=tmp_path/'incomplete.json';put(access,{'schema':'CPU incomplete access'})
    command=[str(ROOT/'.harness-venv/bin/python'),str(ROOT/'scripts'/script),'--access',str(access),
        '--environments',str(tmp_path/'ENVIRONMENT_MUST_NOT_BE_READ'),'--out',str(tmp_path/'out')]
    if script=='run_confirmation_pair.py':command+=['--executor',str(tmp_path/'EXECUTOR_MUST_NOT_BE_READ'),'--executor-commit','CPU-fixture']
    else:command+=['--pair',str(tmp_path/'CLAIM_MUST_NOT_BE_READ'),'--condition-result',str(tmp_path/'RESULT_MUST_NOT_BE_READ')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1')
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'completed explicit confirmation access receipt required' in result.stderr
    assert 'FileNotFoundError' not in result.stderr and not (tmp_path/'out').exists()
