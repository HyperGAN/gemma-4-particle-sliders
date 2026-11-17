"""Fixed synthetic control of final text after a successful submit_patch.

Tool observations below are declared fixtures. No tool is executed, no patch
is repaired, and these episodes are forbidden as fitting/benchmark targets.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from gemma_particle.data import digest
from gemma_particle.harness_identity import sha256_file

FINALIZATION_CUE = ('After submit_patch returns status ok, finish with one short final text response. '
    'Do not call submit_patch again unless you made another code change after the previous submission.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--server-identity', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('localhost','127.0.0.1','::1')
            or endpoint.path.rstrip('/') != '/v1' or endpoint.username or endpoint.password):
        raise ValueError('local OpenAI /v1 endpoint required')
    identity = json.loads(args.server_identity.read_text())
    if identity.get('kind') != 'gemma4' or identity.get('strength') != 0 or identity.get('native_checkpoint') is not None:
        raise ValueError('frozen-base Gemma server required')
    with urllib.request.urlopen(args.endpoint.rstrip('/').removesuffix('/v1')+'/health', timeout=5) as response:
        health = json.load(response)
    if health.get('identity_sha256') != digest(identity):
        raise ValueError('live server identity differs')
    system = Path('submission/prompts/system.md').read_text().rstrip()
    tools = [{'type':'function','function':{'name':'submit_patch',
        'description':"Capture the current working tree modifications as the agent's submission. The agent can submit multiple times; only the final submission is evaluated.",
        'parameters':{'type':'object','properties':{}}}}]
    cases = []
    for repeats in (1,12):
        messages = [{'role':'system','content':system}, {'role':'user','content':
            'Synthetic finalization control: these tool observations are fixtures, not an actual repair. '
            'The toy issue was to return zero for an empty list. Prior observed checks: '
            'before the change the focused check failed with None; after the change it returned zero '
            'and all12 relevant checks passed. The working tree has no further changes after submission.'}]
        for index in range(repeats):
            messages.extend([{'role':'assistant','content':None,'tool_calls':[{'id':f'call_{index}','type':'function',
                'function':{'name':'submit_patch','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':f'call_{index}','content':json.dumps({'status':'ok','patch_size':100,'files_changed':1})}])
        for cue in (False,True):
            observed = deepcopy(messages)
            if cue:
                observed[0]['content'] += '\n\n'+FINALIZATION_CUE
            body = {'model':identity['served_model'],'messages':observed,'tools':tools,
                'temperature':.2,'top_p':.95,'seed':0,'max_completion_tokens':256,
                'chat_template_kwargs':{'enable_thinking':True},'thinking_token_budget':1024}
            cases.append({'id':f'submits_{repeats}_'+('explicit_finish' if cue else 'current_prompt'),
                'request':body,'request_sha256':digest(body)})
    args.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    plan = {'schema':'gemma4-submission-finalization-control-v1', 'created_at':datetime.now(timezone.utc).isoformat(),
        'server_identity_sha256':digest(identity),'source_sha256':sha256_file(Path(__file__)),
        'system_prompt_sha256':sha256_file('submission/prompts/system.md'),
        'cue':FINALIZATION_CUE,'cases':cases,'automatic_retries':False,
        'scope':'fixed synthetic termination control; no actual tool execution, task repair, fitting target or score'}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    finished = []
    for index,case in enumerate(cases):
        row = {'id':case['id'],'request_sha256':case['request_sha256'],'status':'attempted'}
        path = args.out/f'{index:02d}-attempt.json'
        path.write_text(json.dumps(row,indent=2)+'\n')
        start = time.perf_counter()
        try:
            request = urllib.request.Request(args.endpoint.rstrip('/')+'/chat/completions',
                data=json.dumps(case['request']).encode(),headers={'Content-Type':'application/json','X-Gemma-Identity':digest(identity)})
            with urllib.request.urlopen(request,timeout=300) as response:
                raw = response.read()
            response_path = args.out/f'{index:02d}-response.json'; response_path.write_bytes(raw)
            data = json.loads(raw); choices = data['choices']
            if data['model'] != identity['served_model'] or len(choices)!=1:
                raise ValueError('declared model and one completion required')
            choice = choices[0]; message = choice['message']
            final = bool((message.get('content') or '').strip()) and not message.get('tool_calls') and choice['finish_reason']=='stop'
            row.update(status='completed',final_text=final,finish_reason=choice['finish_reason'],
                response_sha256=sha256_file(response_path),tool_calls=len(message.get('tool_calls') or []),usage=data.get('usage'))
        except Exception as exc:
            row.update(status='error',error_type=type(exc).__name__)
        row['seconds'] = time.perf_counter()-start
        (args.out/f'{index:02d}-result.json').write_text(json.dumps(row,indent=2)+'\n')
        finished.append(row); print(json.dumps(row),flush=True)
    result = {k:v for k,v in plan.items() if k!='cases'}
    result.update(state='complete',plan_sha256=sha256_file(args.out/'plan.json'),finished=finished,
        training_updates=0,actual_task_results=0)
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
