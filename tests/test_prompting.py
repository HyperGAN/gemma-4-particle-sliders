import pytest
from gemma_particle.prompting import template_options,tokenize_context,native_messages


def test_preserves_captured_thinking_and_tool_schema():
    captured={}
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            captured.update(kwargs);return [1,2]
    tools=[{'type':'function','function':{'name':'read_file'}}]
    assert tokenize_context(Tokenizer(),[{'role':'user','content':'issue'}],tools,
        {'extra_body':{'chat_template_kwargs':{'enable_thinking':True}}})==[1,2]
    assert captured['enable_thinking'] is True and captured['tools']==tools
    assert captured['return_dict'] is False
    with pytest.raises(ValueError,match='contradictory'):
        template_options({'extra_body':{'chat_template_kwargs':{'enable_thinking':True}},'chat_template_kwargs':{'enable_thinking':False}})


def test_native_tool_arguments_decode_without_mutating_wire_capture():
    messages=[{'role':'assistant','tool_calls':[{'id':'call_1','type':'function',
        'function':{'name':'read_file','arguments':'{"filepath":"a.py","start_line":1}'}}]}]
    converted=native_messages(messages)
    assert converted[0]['tool_calls'][0]['function']['arguments']=={'filepath':'a.py','start_line':1}
    assert isinstance(messages[0]['tool_calls'][0]['function']['arguments'],str)
    messages[0]['tool_calls'][0]['function']['arguments']='[]'
    with pytest.raises(ValueError,match='object'):native_messages(messages)
