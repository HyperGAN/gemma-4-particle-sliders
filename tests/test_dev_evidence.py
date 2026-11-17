"""CPU canonical-contract controls; no serving environment or model score."""
from copy import deepcopy
import json

import pytest

from gemma_particle.dev_evidence import canonical_contract,SERVER_SOURCES,validate_matching_contract
from gemma_particle.curation import instruction_set
from gemma_particle.data import digest
from gemma_particle.history import history_contract,OFFICIAL_COMPACTION
from gemma_particle.capture import RequestJournal,load_requests
from gemma_particle.dev_evidence import audit_verified_task
from gemma_particle.harness_identity import sha256_file


def contract():
    server=dict(kind='gemma4',qualification_only=False,pretrained_inference=True,runtime_adapter_updates=False,
        strength=0,native_checkpoint=None,model_identity={'fixture':'not actual model'},device='cuda:0',
        memory_fraction=.6,max_context=32768,tensor_parallel_size=1,max_num_seqs=1,adapter_dtype='auto',
        official_sources={'fixture':'official wheel hashes'},packages={'fixture':'packages'},
        sources={name:'a'*64 for name in SERVER_SOURCES},
        launch=['vllm','serve','fixture','--port','8029','--enable-lora','--max-loras','1','--max-lora-rank','8'])
    return {'model_identity':server,'budget':{'time_minutes':60},'sampling_sha256':'fixture'}


def test_only_adapter_packet_and_bound_port_differences_are_canonicalized():
    off=contract();learned=deepcopy(off)
    learned['candidate_adapters']={'particle_lora':{'fixture':'weights'}}
    learned['model_identity']['launch'][4]='8030'
    learned['model_identity']['launch'].extend(['--lora-modules','{"name":"particle_lora","path":"/fixture/path","base_model_name":"fixture"}'])
    learned['model_identity']['sources']['/fixture/conversion.json']='b'*64
    assert canonical_contract(off)==canonical_contract(learned)
    changed=deepcopy(learned);changed['model_identity']['launch'].extend(['--revision','different'])
    assert canonical_contract(off)!=canonical_contract(changed)
    learned['budget']['time_minutes']=5
    assert canonical_contract(off)!=canonical_contract(learned)
    learned=contract();learned['model_identity']['max_num_seqs']=2
    assert canonical_contract(off)!=canonical_contract(learned)
    learned=contract();learned['model_identity']['qualification_only']=True
    with pytest.raises(ValueError,match='actual immutable'):canonical_contract(learned)
    learned=contract();learned['model_identity']['launch'].remove('--enable-lora')
    with pytest.raises(ValueError,match='LoRA-engine'):canonical_contract(learned)


def test_declared_budget_sampling_and_history_cannot_silently_drift():
    matching={'budget':{'time_minutes':60},'command_timeout_seconds':300,'sampling_file_sha256':'sampling',
        'prompt_profile':'repair-finalize-v2','common_system_sha256':'prompt','instruction_protocol':'tools-v2'}
    c={'budget':matching['budget'],'harness':{'command_timeout_seconds':300},'sampling_sha256':'sampling',
        'common_prompt':{'profile':'repair-finalize-v2','sha256':'prompt'},'instruction_protocol':'tools-v2',
        'instruction_set_sha256':digest(instruction_set('tools-v2')),'history':history_contract(OFFICIAL_COMPACTION)}
    validate_matching_contract(c,{'matching':matching})
    changed=deepcopy(c);changed['sampling_sha256']='changed'
    with pytest.raises(ValueError,match='declared comparison'):validate_matching_contract(changed,{'matching':matching})


def test_capture_and_verification_have_distinct_schemas_but_identical_copied_fields(tmp_path):
    # Structural CPU journal fixture, not an actual model or benchmark result.
    cohort=tmp_path/'cohort';folder=cohort/'fixture-task';folder.mkdir(parents=True)
    verification=tmp_path/'verification';destination=verification/'fixture-task';destination.mkdir(parents=True)
    journal=RequestJournal(folder/'requests',instance_id='fixture-task',arm='off')
    packet=journal.begin(model='openai/fixture',messages=[],tools=None,options={})
    journal.complete(packet,{'model':'fixture','choices':[]})
    (folder/'agent.patch').write_text('CPU fixture patch')
    (destination/'test-output.txt').write_text('CPU fixture test output')
    c=contract();captured={'schema':'gemma4-captured-trajectory-v1','instance_id':'fixture-task','arm':'off',
        'capture_complete':True,'frozen_base_generator':True,'patch_sha256':sha256_file(folder/'agent.patch'),'tool_calls':0}
    manifest={'generator_kind':'gemma4','arm':'off','task_ids':['fixture-task'],'runs':[captured],
        'server_identity':c['model_identity'],'contract':c,'evaluation_contract_sha256':digest(c)}
    receipt={'test_output_sha256':sha256_file(destination/'test-output.txt')}
    row={**captured,'schema':'gemma4-verified-trajectory-v4','requests':load_requests(folder/'requests'),'verification':receipt}
    (cohort/'cohort.json').write_text(json.dumps(manifest));(folder/'trajectory.json').write_text(json.dumps(captured))
    (destination/'verification.json').write_text(json.dumps(receipt))
    outcomes=verification/'outcomes.jsonl';outcomes.write_text(json.dumps(row)+'\n')
    actual,_=audit_verified_task(cohort,verification,'fixture-task',expected_arm='off',expected_alias='fixture')
    assert actual['schema']=='gemma4-verified-trajectory-v4'
    row['tool_calls']=1;outcomes.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='DEV row differs'):
        audit_verified_task(cohort,verification,'fixture-task',expected_arm='off',expected_alias='fixture')
