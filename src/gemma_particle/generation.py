"""Single-sequence native HF inference with explicit sampling and tool grammar."""
import json
import time
import uuid

import torch
from transformers import LogitsProcessorList

from .data import digest
from .official_thinking import OfficialThinkingBudget,DEFAULT_WHEEL
from .prompting import tokenize_context


def openai_message(parsed,request_id):
    message={'role':'assistant','content':parsed.get('content') or None}
    if parsed.get('thinking'):
        message['reasoning_content']=parsed['thinking']
    calls=[]
    for i,call in enumerate(parsed.get('tool_calls') or []):
        function=call['function']
        if not isinstance(function['arguments'],dict):
            raise ValueError('native parser must produce JSON-object tool arguments')
        calls.append({'id':f'call_{request_id}_{i}','type':'function','function':{
            'name':function['name'],'arguments':json.dumps(function['arguments'],ensure_ascii=False)}})
    if calls:message['tool_calls']=calls
    return message


class NativeGenerator:
    def __init__(self,native,tokenizer,*,served_model,strength=0,wheel=DEFAULT_WHEEL,max_context=32768,max_seconds=120):
        self.native,self.tokenizer=native,tokenizer
        self.served_model,self.strength=served_model,float(strength)
        self.wheel,self.max_context,self.max_seconds=wheel,max_context,max_seconds

    def complete(self,body):
        allowed={'model','messages','tools','stream','n','temperature','top_p','top_k','seed','max_tokens',
            'max_completion_tokens','stop','response_format','presence_penalty','frequency_penalty','tool_choice',
            'parallel_tool_calls','chat_template_kwargs','thinking_token_budget'}
        if set(body)-allowed:raise ValueError('unsupported request fields: '+','.join(sorted(set(body)-allowed)))
        if body['model']!=self.served_model or body.get('stream') or body.get('n',1)!=1:
            raise ValueError('declared model and one nonstreaming sequence required')
        if body.get('response_format') not in (None,{'type':'text'}):raise ValueError('structured response format unsupported')
        if body.get('tool_choice') not in (None,'auto'):raise ValueError('only automatic tool choice is qualified')
        if body.get('presence_penalty',0) or body.get('frequency_penalty',0):raise ValueError('nonzero token penalties unsupported')
        if 'max_tokens' in body and 'max_completion_tokens' in body and body['max_tokens']!=body['max_completion_tokens']:
            raise ValueError('conflicting output token budgets')
        maximum=body.get('max_completion_tokens',body.get('max_tokens',4096))
        if type(maximum) is not int or not 1<=maximum<=32768:raise ValueError('invalid output budget')
        temperature=float(body.get('temperature',1));top_p=float(body.get('top_p',1));top_k=body.get('top_k',0)
        if temperature<0 or not 0<top_p<=1 or type(top_k) is not int or top_k<0:
            raise ValueError('invalid sampling controls')
        generation={'chat_template_kwargs':body.get('chat_template_kwargs',{})}
        prompt=tokenize_context(self.tokenizer,body['messages'],body.get('tools'),generation)
        if not prompt or len(prompt)+maximum>self.max_context:
            raise ValueError('request exceeds the declared context window; no truncation applied')
        device=self.native.trunk.embed_tokens.weight.device
        tokens=torch.tensor([prompt],device=device)
        processors=LogitsProcessorList()
        budget=body.get('thinking_token_budget')
        if budget is not None:
            if not generation['chat_template_kwargs'].get('enable_thinking',False):
                raise ValueError('thinking budget supplied while thinking mode is disabled')
            processors.append(OfficialThinkingBudget(prompt,start_ids=self.tokenizer.encode('<|channel>',add_special_tokens=False),
                end_ids=self.tokenizer.encode('<channel|>',add_special_tokens=False),budget=budget,wheel=self.wheel))
        options={'max_new_tokens':maximum,'max_time':self.max_seconds,'do_sample':temperature>0,
            'repetition_penalty':1.,'logits_processor':processors,'logits_to_keep':1,
            'use_cache':True,'return_dict_in_generate':False}
        if temperature>0:options.update(temperature=temperature,top_p=top_p,top_k=top_k)
        if body.get('stop'):
            options.update(stop_strings=body['stop'],tokenizer=self.tokenizer)
        seed=body.get('seed',0)
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('nonnegative integer seed required')
        start=time.perf_counter()
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            with torch.inference_mode(),self.native.network.scaled(self.strength):
                output=self.native.model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),**options)
        ids=output[0,len(prompt):].tolist()
        parsed=self.tokenizer.parse_response(ids,prefix=prompt)
        request_id=uuid.uuid4().hex
        message=openai_message(parsed,request_id)
        if body.get('parallel_tool_calls') is False and len(message.get('tool_calls',[]))>1:
            raise ValueError('model emitted parallel calls while the request disallowed them')
        eos=self.native.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        finish='tool_calls' if message.get('tool_calls') else ('stop' if ids and ids[-1] in eos else 'length')
        response={'id':'chatcmpl-'+request_id,'object':'chat.completion','created':int(time.time()),'model':self.served_model,
            'choices':[{'index':0,'message':message,'finish_reason':finish}],
            'usage':{'prompt_tokens':len(prompt),'completion_tokens':len(ids),'total_tokens':len(prompt)+len(ids)}}
        audit={'request_sha256':digest(body),'prompt_ids_sha256':digest(prompt),'output_ids':ids,
            'raw_output':self.tokenizer.decode(ids,skip_special_tokens=False),'response':response,
            'seconds':time.perf_counter()-start,'strength':self.strength,
            'cuda_peak_memory_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'thinking_forced_tokens':processors[0].forced_tokens if processors else 0}
        return response,audit
