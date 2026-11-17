from copy import deepcopy

import pytest
import torch

from gemma_particle.conversion_probe import initial_off_request,validate_probe
from gemma_particle.data import digest
from gemma_particle.export import collect_activations
from gemma_particle.model import Native,ContextRow
from test_gemma_binding import tiny_model


def fixture():
    request={'model':'fixture','messages':[{'role':'system','content':'repair'},{'role':'user','content':'issue'}],
             'tools':[],'generation':{}}
    requests=[{'request':request}]
    outcome={'schema':'gemma4-verified-trajectory-v4','instance_id':'train-probe','split':'train','arm':'off',
        'generator_kind':'gemma4','frozen_base_generator':True,'resolved':True,'capture_complete':True,
        'first_request':request,'requests':requests,'verification':{'stage':'fresh-container-tests',
        'instance_id':'train-probe','resolved':True,'first_request_sha256':digest(request),'requests_sha256':digest(requests)}}
    response={'repo':'fixture','revision':'a'*40,'dtype':'bfloat16','attn_implementation':'sdpa'}
    server={'model_identity':dict(response,attn_implementation='vllm:auto'),'strength':0,
            'native_checkpoint':None,'kind':'gemma4','pretrained_inference':True}
    ids=[2,3,4]
    packet={'schema':'gemma4-conversion-probe-v1','purpose':'conversion diagnostic only; forbidden as a training target',
        'server_identity':server,'server_identity_sha256':digest(server),'response_model_identity':response,
        'entries':[{'instance_id':'train-probe','request':request,'request_sha256':digest(request),
                    'neutral_ids':ids,'neutral_ids_sha256':digest(ids)}]}
    return outcome,packet,response


def test_issue_only_probe_accepts_bound_off_and_excludes_fit_dev_and_tool_history():
    outcome,packet,response=fixture()
    assert initial_off_request(outcome,{'train-probe'})==outcome['first_request']
    rows=validate_probe(packet,response_identity=response,train_ids={'train-probe'},fit_ids={'train-fit'})
    assert rows[0].neutral_ids==(2,3,4) and rows[0].positive_ids==()
    for train,fit in [(set(),set()),({'train-probe'},{'train-probe'})]:
        with pytest.raises(ValueError):validate_probe(packet,response_identity=response,train_ids=train,fit_ids=fit)
    broken=deepcopy(packet);broken['entries'][0]['neutral_ids'][0]=9
    with pytest.raises(ValueError):validate_probe(broken,response_identity=response,train_ids={'train-probe'},fit_ids=set())
    changed=deepcopy(outcome);changed['first_request']['messages'].append({'role':'tool','content':'later history'})
    with pytest.raises(ValueError):initial_off_request(changed,{'train-probe'})
    changed=deepcopy(outcome);changed['verification']['first_request_sha256']='0'*64
    with pytest.raises(ValueError):initial_off_request(changed,{'train-probe'})


def test_activation_calibration_observes_late_history_and_each_row():
    native=Native(tiny_model(),identity={'fixture':'conversion-sampling'})
    try:
        a=ContextRow((2,3,4,5,6,7),(),'fit-a',0)
        b=ContextRow((2,3,4,5,6,15),(),'fit-b',0)
        both=collect_activations(native,[a,b],tokens_per_layer=4)
        first=collect_activations(native,[a],tokens_per_layer=2)
        second=collect_activations(native,[b],tokens_per_layer=2)
        for name in both:
            assert torch.equal(both[name][:2],first[name])
            assert torch.equal(both[name][2:],second[name])
        # Shared prefix alone would make these identical. The last issue token
        # must enter calibration even under a two-token per-row budget.
        name=next(iter(both))
        assert not torch.equal(first[name][-1],second[name][-1])
    finally:native.close()
