from copy import deepcopy

import pytest

from gemma_particle.data import digest
from gemma_particle.transport import validate_transport


def fixture():
    generator={'repo':'google/gemma-4-31B-it-qat-w4a16-ct','revision':'52f3f65bc7a02d555763bc923bd1d9094898219d',
        'dtype':'bfloat16','weight_format':'ct-w4g32','attn_implementation':'vllm:auto'}
    response={**generator,'attn_implementation':'sdpa'}
    server={'model_identity':generator,'strength':0,'native_checkpoint':None}
    contexts=[{'instance_id':'train-fixture','decision':0,'model_identity':generator}]
    entry={'instance_id':'train-fixture','decision':0,'context_sha256':digest(contexts[0])}
    for side,ids in [('neutral',[2,3]),('positive',[2,4,3])]:
        entry[side+'_ids']=ids;entry[side+'_ids_sha256']=digest(ids)
    transport={'schema':'gemma4-prompt-transport-v1','generator_model_identity':generator,
        'response_model_identity':response,'contexts_sha256':'fixture-file-sha','contexts_digest':digest(contexts),
        'server_identity':server,'server_identity_sha256':digest(server),'entries':[entry]}
    return transport,contexts,response


def test_prompt_transport_keeps_generator_and_response_backends_distinct():
    t,c,r=fixture()
    assert validate_transport(t,c,contexts_sha256='fixture-file-sha',response_identity=r)==t['entries']
    assert t['generator_model_identity']['attn_implementation']!=r['attn_implementation']
    for edit in ('weight_revision','response_runtime','contexts','tokens','order','trained_generator'):
        bad=deepcopy(t)
        if edit=='weight_revision':bad['generator_model_identity']['revision']='a'*40
        elif edit=='response_runtime':bad['response_model_identity']['attn_implementation']='eager'
        elif edit=='contexts':bad['contexts_digest']='changed'
        elif edit=='tokens':bad['entries'][0]['positive_ids'][0]=99
        elif edit=='order':bad['entries'][0]['decision']=1
        elif edit=='trained_generator':
            bad['server_identity']['strength']=1
            bad['server_identity_sha256']=digest(bad['server_identity'])
        with pytest.raises(ValueError):validate_transport(bad,c,contexts_sha256='fixture-file-sha',response_identity=r)


def test_prompt_transport_refuses_wrong_weights_even_with_consistent_receipt():
    t,c,r=fixture()
    # Rehashing an artifact cannot turn another checkpoint into the response model.
    t['generator_model_identity']['revision']='b'*40
    t['server_identity_sha256']=digest(t['server_identity'])
    t['contexts_digest']=digest(c);t['entries'][0]['context_sha256']=digest(c[0])
    with pytest.raises(ValueError,match='matching pinned'):
        validate_transport(t,c,contexts_sha256='fixture-file-sha',response_identity=r)
