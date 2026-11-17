"""Run the official vLLM thinking state machine under a single HF sequence.

The three state methods are compiled unchanged from a hash-pinned official
wheel. vLLM itself is not imported and its unrelated modules are not executed.
This preserves the actual policy, including its countdown behavior. Native HF
sampling remains a distinct cohort from production vLLM sampling.
"""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any
import zipfile

from transformers import LogitsProcessor

WHEEL_SHA256='73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310'
SOURCE='vllm/v1/sample/logits_processor/builtin.py'
SOURCE_SHA256='769038f9f91a613f2157556e74413466218e1ca9a5e9e166487a3660809d6e28'
DEFAULT_WHEEL=Path('data/kaggle-official/wheelhouse/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl')


@lru_cache(maxsize=2)
def state_machine(wheel=DEFAULT_WHEEL):
    wheel=Path(wheel)
    with wheel.open('rb') as f:
        observed=hashlib.file_digest(f,'sha256').hexdigest()
    if observed!=WHEEL_SHA256:raise ValueError('official vLLM wheel hash differs')
    with zipfile.ZipFile(wheel) as z:source=z.read(SOURCE)
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:
        raise ValueError('official thinking processor source differs')
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ThinkingTokenBudgetLogitsProcessor')
    wanted={'_find_last_sequence_index','_init_state_entry','_update_think_state'}
    methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if {n.name for n in methods}!=wanted:raise ValueError('official state methods missing')
    original.name='OfficialThinkingState'
    original.bases=[];original.keywords=[];original.decorator_list=[];original.body=methods
    module=ast.fix_missing_locations(ast.Module(body=[original],type_ignores=[]))
    namespace={'Any':Any}
    exec(compile(module,SOURCE,'exec'),namespace)
    return namespace['OfficialThinkingState']


class OfficialThinkingBudget(LogitsProcessor):
    def __init__(self,prompt_ids,*,start_ids,end_ids,budget,wheel=DEFAULT_WHEEL):
        if type(budget) is not int or budget<0 or not start_ids or not end_ids:
            raise ValueError('nonnegative integer thinking budget and delimiters required')
        self.machine=state_machine(wheel)()
        self.machine.reasoning_start_token_ids=list(start_ids)
        self.machine.reasoning_end_token_ids=list(end_ids)
        self.prompt_length=len(prompt_ids)
        self.state=self.machine._init_state_entry(list(prompt_ids),budget)
        self.forced_tokens=0

    def __call__(self,input_ids,scores):
        if input_ids.shape[0]!=1:raise ValueError('native cohort serves one sequence at a time')
        self.state['output_tok_ids']=input_ids[0,self.prompt_length:].tolist()
        self.machine._update_think_state(self.state)
        if self.state['in_end']:
            token=self.machine.reasoning_end_token_ids[self.state['end_count']]
            # Same 1e9 forcing rule as the official processor's apply().
            scores[0,token]=1e9
            self.forced_tokens+=1
        return scores
