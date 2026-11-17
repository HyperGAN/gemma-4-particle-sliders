from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from gemma_particle.generation import NativeGenerator,openai_message
from gemma_particle.official_thinking import OfficialThinkingBudget,DEFAULT_WHEEL


def test_native_tool_response_preserves_reasoning_and_object_arguments():
    message=openai_message({'thinking':'inspect first','tool_calls':[{'function':{
        'name':'read_file','arguments':{'filepath':'a.py','start_line':1}}}]},'request')
    assert message['reasoning_content']=='inspect first'
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'filepath':'a.py','start_line':1}
    assert message['tool_calls'][0]['id']=='call_request_0'


def test_native_sampling_disables_hidden_top_k_default_and_rejects_dropped_controls():
    observed={}
    class Model:
        generation_config=SimpleNamespace(eos_token_id=[1])
        def generate(self,**kwargs):
            observed.update(kwargs)
            return torch.tensor([[2,3,4,1]])
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [2,3,4]
        def parse_response(self,ids,**kwargs):return {'content':'ok'}
        def decode(self,*args,**kwargs):return 'ok<turn|>'
    native=SimpleNamespace(trunk=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1))),
        model=Model(),network=SimpleNamespace(scaled=lambda strength:nullcontext()))
    generator=NativeGenerator(native,Tokenizer(),served_model='native')
    request={'model':'native','messages':[{'role':'user','content':'issue'}],
        'temperature':.2,'top_p':.95,'seed':0,'max_tokens':4}
    response,audit=generator.complete(request)
    assert observed['top_k']==0 and observed['temperature']==.2 and observed['max_new_tokens']==4
    assert response['choices'][0]['finish_reason']=='stop' and audit['strength']==0
    with pytest.raises(ValueError,match='nonzero token penalties'):
        generator.complete({**request,'presence_penalty':1})
    with pytest.raises(ValueError,match='unsupported request fields'):
        generator.complete({**request,'unknown_control':True})


@pytest.mark.skipif(not DEFAULT_WHEEL.exists(),reason='official wheel is a separate downloaded input')
def test_unchanged_official_thinking_countdown_and_forced_end():
    processor=OfficialThinkingBudget([2,5,9],start_ids=[5],end_ids=[6],budget=3)
    outputs=[];forced=[]
    for i in range(6):
        scores=processor(torch.tensor([[2,5,9,*outputs]]),torch.zeros(1,16))
        token=int(scores.argmax()) if scores.max()>1e8 else 10
        if scores.max()>1e8:forced.append(i)
        outputs.append(token)
    # The official countdown checks after three generated tokens even though
    # the prompt already has one token after the thinking start delimiter.
    assert forced==[3] and outputs[3]==6 and processor.forced_tokens==1
    no_thinking=OfficialThinkingBudget([2,5,9,6],start_ids=[5],end_ids=[6],budget=3)
    assert no_thinking(torch.tensor([[2,5,9,6,10,10,10]]),torch.zeros(1,16)).max()==0
