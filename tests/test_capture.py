import json
from copy import deepcopy
import pytest
from gemma_particle.capture import RequestJournal,load_requests


def test_exact_inputs_with_no_transport_credentials(tmp_path):
    journal = RequestJournal(tmp_path/'capture',instance_id='task',arm='off')
    messages = [{'role':'system','content':'precise\nbytes'}, {'role':'user','content':'issue'}]
    tools = [{'type':'function','function':{'name':'read_file','parameters':{'type':'object'}}}]
    packet = journal.begin(model='gemma',messages=messages,tools=tools,
        options={'api_key':'PRIVATE-SECRET','headers':{'Authorization':'PRIVATE-SECRET'},'seed':0,'temperature':.2})
    journal.complete(packet,{'choices':[{'message':{'role':'assistant','content':'answer'},'finish_reason':'stop'}],
                             'usage':{'total_tokens':3},'_hidden_params':{'api_key':'PRIVATE-SECRET'}})
    records = load_requests(tmp_path/'capture')
    assert records[0]['request']['messages'] == messages
    assert records[0]['request']['tools'] == tools
    assert records[0]['request']['generation'] == {'seed':0,'temperature':.2}
    assert all('PRIVATE-SECRET' not in p.read_text() for p in (tmp_path/'capture').iterdir())
    with pytest.raises(FileExistsError):
        RequestJournal(tmp_path/'capture',instance_id='task',arm='off')


def test_incomplete_and_tampered_capture_rejected(tmp_path):
    journal = RequestJournal(tmp_path/'capture',instance_id='task',arm='off')
    packet = journal.begin(model='gemma',messages=[],tools=None,options={})
    with pytest.raises(ValueError,match='incomplete'):
        load_requests(tmp_path/'capture')
    journal.complete(packet,{'choices':[]})
    file = tmp_path/'capture/0000-request.json'
    value = json.loads(file.read_text());value['request']['messages']=[{'role':'user','content':'changed'}]
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='identity'):
        load_requests(tmp_path/'capture')


def test_cloned_counter_cannot_overwrite_an_existing_request(tmp_path):
    journal=RequestJournal(tmp_path/'capture',instance_id='task',arm='off')
    cloned=deepcopy(journal)
    packet=journal.begin(model='gemma',messages=[{'role':'user','content':'original'}],tools=None,options={})
    journal.complete(packet,{'choices':[]})
    before={p.name:p.read_bytes() for p in (tmp_path/'capture').iterdir()}
    with pytest.raises(FileExistsError):
        cloned.begin(model='gemma',messages=[{'role':'user','content':'overwritten summary'}],tools=None,options={})
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'capture').iterdir()}
    assert len(load_requests(tmp_path/'capture'))==1
