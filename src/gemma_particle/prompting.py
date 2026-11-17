"""Replay the captured chat-template controls, including Gemma thinking mode."""
from copy import deepcopy
import json


def native_messages(messages):
    """Adapt OpenAI's JSON-string arguments to Gemma's native object schema.

    Keep the captured wire messages unchanged. The pinned Gemma template
    explicitly requires a mapping and serializes that mapping itself.
    """
    result=deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls') or []:
            function=call.get('function',call)
            arguments=function.get('arguments')
            if isinstance(arguments,str):
                arguments=json.loads(arguments)
                if not isinstance(arguments,dict):
                    raise ValueError('native tool arguments must decode to an object')
                function['arguments']=arguments
    return result


def template_options(generation):
    result=dict(generation.get('extra_body',{}).get('chat_template_kwargs',{}))
    direct=generation.get('chat_template_kwargs',{})
    if any(k in result and result[k]!=v for k,v in direct.items()):
        raise ValueError('contradictory chat-template controls')
    result.update(direct)
    if set(result) & {'tokenize','add_generation_prompt','return_tensors','return_dict','tools'}:
        raise ValueError('reserved chat-template control')
    return result


def tokenize_context(tokenizer,messages,tools,generation):
    return tokenizer.apply_chat_template(native_messages(messages),tokenize=True,add_generation_prompt=True,return_dict=False,
                                         tools=tools,**template_options(generation))
