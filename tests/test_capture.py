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


def test_returned_model_identity_is_preserved_and_hash_bound(tmp_path):
    journal=RequestJournal(tmp_path/'capture',instance_id='task',arm='off')
    packet=journal.begin(model='openai/particle_lora',messages=[],tools=None,options={})
    journal.complete(packet,{'model':'particle_lora','choices':[],
        'headers':{'Authorization':'PRIVATE'},'_hidden_params':{'api_key':'PRIVATE'}})
    path=tmp_path/'capture/0000-response.json'
    value=json.loads(path.read_text())
    assert value['response']['model']=='particle_lora'
    assert 'PRIVATE' not in path.read_text()
    load_requests(tmp_path/'capture')
    value['response']['model']='gemma-4-31b-it-qat-w4a16-ct'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completion journal identity'):
        load_requests(tmp_path/'capture')
