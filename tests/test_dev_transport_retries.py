"""CPU evidence-reader controls; these fixtures are never model results."""
from copy import deepcopy
import json

import pytest

from gemma_particle.capture import RequestJournal
from gemma_particle.data import digest
from gemma_particle.dev_evidence import audit_incomplete_requests


def retry_fixture(tmp_path):
    directory=tmp_path/'requests'
    journal=RequestJournal(directory,instance_id='fixture-task',arm='off')
    packet=journal.begin(model='openai/fixture',messages=[],tools=None,options={})
    journal.complete(packet,{'model':'fixture','choices':[]})
    for _ in range(6):
        journal.begin(model='openai/fixture',messages=[{'role':'user','content':'same failed call'}],
            tools=None,options={'seed':0})
    captured={'instance_id':'fixture-task','arm':'off','capture_complete':False,
        'capture_error':'incomplete model request','agent_error':'Sandbox execution error: '
        'litellm.InternalServerError: InternalServerError: OpenAIException - Connection error. '
        'LiteLLM Retried: 5 times'}
    return directory,captured


def test_identical_failed_retries_are_retained_without_altering_evidence(tmp_path):
    directory,captured=retry_fixture(tmp_path)
    before={p.name:p.read_bytes() for p in directory.iterdir()};original=deepcopy(captured)
    packets=audit_incomplete_requests(directory,captured)
    assert [p['decision'] for p in packets]==list(range(7))
    assert len({p['request_sha256'] for p in packets[1:]})==1
    assert {p.name:p.read_bytes() for p in directory.iterdir()}==before
    assert captured==original and captured['capture_complete'] is False


@pytest.mark.parametrize('change',['body','purpose','task','arm','order','late_response',
    'extra_response','count','error','capture_complete','capture_error','prefix_hash'])
def test_mixed_forged_or_misclassified_retry_tails_are_rejected(tmp_path,change):
    directory,captured=retry_fixture(tmp_path)
    path=directory/'0003-request.json';packet=json.loads(path.read_text())
    if change=='body':
        packet['request']['messages'][0]['content']='different failed call'
        packet['request_sha256']=digest(packet['request'])
    elif change=='purpose':packet['purpose']='history_summary'
    elif change=='task':packet['instance_id']='other-task'
    elif change=='arm':packet['arm']='verify'
    elif change=='order':packet['decision']=1
    elif change in ('late_response','extra_response'):
        response={'request_sha256':packet['request_sha256'],'response':{'choices':[]},
            'response_sha256':digest({'choices':[]})}
        (directory/('0003-response.json' if change=='late_response' else '9999-response.json')).write_text(json.dumps(response))
    elif change=='count':captured['agent_error']=captured['agent_error'].replace('5 times','4 times')
    elif change=='error':captured['agent_error']='Agent exceeded session timeout'
    elif change=='capture_complete':captured['capture_complete']=True
    elif change=='capture_error':captured['capture_error']='different error'
    elif change=='prefix_hash':
        prefix=directory/'0000-response.json';value=json.loads(prefix.read_text())
        value['response']['model']='forged';prefix.write_text(json.dumps(value))
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):audit_incomplete_requests(directory,captured)
