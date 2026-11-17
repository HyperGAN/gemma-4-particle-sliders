"""CPU canonical-contract controls; no serving environment or model score."""
from copy import deepcopy

import pytest

from gemma_particle.dev_evidence import canonical_contract,SERVER_SOURCES,validate_matching_contract
from gemma_particle.curation import instruction_set
from gemma_particle.data import digest
from gemma_particle.history import history_contract,OFFICIAL_COMPACTION


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
