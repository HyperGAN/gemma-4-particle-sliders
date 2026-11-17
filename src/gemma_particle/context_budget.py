"""Prove fixed-window exhaustion using exact tokens from the bound server.

This evidence may qualify a failed OFF control in a declared new protocol.
It cannot promote an old result or an incomplete successful teacher.
"""
import re
from urllib.parse import urlsplit

import httpx

from .capture import load_terminal_request
from .data import digest
from .prompting import template_options


def prove_context_exhaustion(folder,captured,server_identity,endpoint,*,client=None):
    error=str(captured.get('agent_error',''))
    limit=server_identity.get('max_context')
    if (captured.get('capture_complete') is not False or 'ContextWindowExceededError' not in error
            or limit!=32768 or f'maximum context length is {limit} tokens' not in error):
        return None
    parsed=urlsplit(endpoint or '')
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
        raise ValueError('explicit local bound server required for context-budget proof')
    prefix,pending=load_terminal_request(folder/'requests')
    if pending['instance_id']!=captured['instance_id'] or pending['arm']!=captured['arm']:
        raise ValueError('terminal request differs from the captured task/arm')
    request=pending['request'];generation=request['generation']
    outputs=[generation[k] for k in ('max_tokens','max_completion_tokens') if generation.get(k) is not None]
    if not outputs or len(set(outputs))!=1 or type(outputs[0]) is not int or outputs[0]<=0:
        raise ValueError('one exact requested output-token allowance required')
    output=outputs[0]
    if not re.search(rf'requested {output} output tokens\b',error):
        raise ValueError('context error differs from the actual requested output allowance')
    if request['model'].removeprefix('openai/')!=server_identity['served_model']:
        raise ValueError('terminal request uses a different served model')
    body={'model':server_identity['served_model'],'messages':request['messages'],'tools':request['tools'],
          'chat_template_kwargs':template_options(generation)}
    base=endpoint.rstrip('/').removesuffix('/v1');identity=digest(server_identity)
    def check(connection):
        health=connection.get(base+'/health');health.raise_for_status()
        if health.json().get('identity_sha256')!=identity:
            raise ValueError('context-budget tokenizer server identity differs')
    def measure(connection):
        check(connection)
        response=connection.post(base+'/tokenize',json=body);response.raise_for_status();ids=response.json()['tokens']
        check(connection)
        if not ids or any(type(token) is not int for token in ids) or len(ids)+output<=limit:
            raise ValueError('exact prompt tokens do not exhaust the declared context window')
        return {'schema':'gemma4-context-budget-proof-v1','context_limit':limit,'prompt_tokens':len(ids),
            'requested_output_tokens':output,'token_ids_sha256':digest(ids),
            'tokenize_request_sha256':digest(body),'server_identity_sha256':identity,
            'terminal_request_sha256':pending['request_sha256'],'completed_prefix_sha256':digest(prefix),
            'model_calls_completed':len(prefix),'model_calls_attempted':len(prefix)+1}
    if client is not None:return measure(client)
    with httpx.Client(headers={'X-Gemma-Identity':identity},timeout=60) as connection:return measure(connection)
