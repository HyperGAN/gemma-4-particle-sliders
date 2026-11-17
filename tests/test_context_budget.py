from copy import deepcopy

import pytest

from gemma_particle.capture import RequestJournal,load_terminal_request,CONTEXT_CAPTURE_PROTOCOL
from gemma_particle.context_budget import prove_context_exhaustion
from gemma_particle.data import digest
from gemma_particle.curation import select_contexts
from test_data import arms


ERROR="ContextWindowExceededError: This model's maximum context length is 32768 tokens. However, you requested 4096 output tokens and your prompt contains at least 28673 input tokens"


def journal(tmp_path):
    j=RequestJournal(tmp_path/'requests',instance_id='task',arm='off')
    p=j.begin(model='openai/gemma',messages=[{'role':'user','content':'issue'}],tools=None,options={'max_tokens':4096})
    j.complete(p,{'choices':[{'message':{'role':'assistant','content':'inspect'},'finish_reason':'stop'}]})
    pending=j.begin(model='openai/gemma',messages=[{'role':'user','content':'issue'},{'role':'assistant','content':'inspect'}],tools=None,options={'max_tokens':4096})
    return j,pending


class Connection:
    def __init__(self,identity,tokens):self.identity=identity;self.tokens=tokens;self.calls=[]
    def get(self,url):
        self.calls.append(url);return self.Response({'identity_sha256':self.identity})
    def post(self,url,*,json):
        self.calls.append((url,json));return self.Response({'tokens':[1]*self.tokens})
    class Response:
        def __init__(self,value):self.value=value
        def raise_for_status(self):pass
        def json(self):return self.value


def test_exact_bound_token_count_proves_terminal_context_exhaustion(tmp_path):
    _,pending=journal(tmp_path)
    server={'served_model':'gemma','max_context':32768}
    captured=dict(instance_id='task',arm='off',capture_complete=False,agent_error=ERROR)
    c=Connection(digest(server),28673)
    proof=prove_context_exhaustion(tmp_path,captured,server,'http://127.0.0.1:8026/v1',client=c)
    assert proof['prompt_tokens']+proof['requested_output_tokens']==32769
    assert proof['terminal_request_sha256']==pending['request_sha256']
    assert proof['model_calls_completed']==1 and proof['model_calls_attempted']==2
    assert len(c.calls)==3
    assert c.calls[1][1]['messages']==pending['request']['messages']


@pytest.mark.parametrize('change',['within_window','identity','wrong_output','transport'])
def test_unproven_context_or_transport_failure_cannot_qualify(tmp_path,change):
    journal(tmp_path);server={'served_model':'gemma','max_context':32768}
    captured=dict(instance_id='task',arm='off',capture_complete=False,agent_error=ERROR)
    c=Connection(digest(server),28673)
    if change=='within_window':c.tokens=28672
    elif change=='identity':c.identity='another-model'
    elif change=='wrong_output':captured['agent_error']=ERROR.replace('4096','8192')
    elif change=='transport':
        captured['agent_error']='ConnectionError: unreachable'
        assert prove_context_exhaustion(tmp_path,captured,server,None,client=c) is None
        assert not c.calls
        return
    with pytest.raises(ValueError):prove_context_exhaustion(tmp_path,captured,server,'http://127.0.0.1:8026/v1',client=c)


def test_incomplete_prefix_and_answered_terminal_request_are_rejected(tmp_path):
    j,pending=journal(tmp_path)
    response=tmp_path/'requests/0000-response.json';saved=response.read_bytes();response.unlink()
    with pytest.raises(ValueError,match='incomplete prefix'):load_terminal_request(tmp_path/'requests')
    response.write_bytes(saved);j.complete(pending,{'choices':[]})
    with pytest.raises(ValueError,match='already has a response'):load_terminal_request(tmp_path/'requests')


def test_only_declared_v4_off_proof_can_admit_a_complete_verified_teacher():
    tasks=[dict(instance_id='task',split='train')]
    rows=arms();baseline=rows[0]
    for row in rows:
        row.update(schema='gemma4-verified-trajectory-v4',capture_protocol=CONTEXT_CAPTURE_PROTOCOL,context_budget_proof=None)
        row['verification'].update(capture_protocol=CONTEXT_CAPTURE_PROTOCOL,context_budget_proof_sha256=digest(None))
    proof=dict(schema='gemma4-context-budget-proof-v1',context_limit=32768,prompt_tokens=28673,requested_output_tokens=4096)
    baseline.update(capture_complete=False,requests=[],context_budget_proof=proof)
    baseline['verification'].update(requests_sha256=digest([]),context_budget_proof_sha256=digest(proof))
    assert len(select_contexts(tasks,rows))==1
    incomplete=deepcopy(rows);incomplete[1]['capture_complete']=False
    assert select_contexts(tasks,incomplete)==[]
    damaged=deepcopy(rows);damaged[0]['context_budget_proof']['prompt_tokens']=28672
    with pytest.raises(ValueError,match='bound'):select_contexts(tasks,damaged)
    damaged=deepcopy(rows);damaged[0]['context_budget_proof'].update(prompt_tokens=32769,requested_output_tokens=0)
    damaged[0]['verification']['context_budget_proof_sha256']=digest(damaged[0]['context_budget_proof'])
    with pytest.raises(ValueError,match='invalid'):select_contexts(tasks,damaged)
    legacy=arms();legacy[0].update(capture_complete=False,requests=[],context_budget_proof=proof)
    legacy[0]['verification']['requests_sha256']=digest([])
    assert select_contexts(tasks,legacy)==[]
