"""Scripted CPU transport probe. It is forbidden as behavioral training data."""
import json
from google.adk.models.lite_llm import LiteLLMClient
from litellm.types.utils import ModelResponse


class ProtocolFixtureClient(LiteLLMClient):
    def __init__(self):
        self.calls = 0

    async def acompletion(self, model, messages, tools, **kwargs):
        if self.calls == 0:
            name, args = 'run_command', {'command':"printf 'CPU protocol fixture\n' > particle_protocol_probe.txt"}
        elif self.calls == 1:
            name, args = 'submit_patch', {}
        else:
            self.calls += 1
            return ModelResponse(model=model, choices=[{'index':0,'message':{'role':'assistant','content':'CPU protocol probe complete.'},'finish_reason':'stop'}])
        call_id = f'cpu-protocol-{self.calls}'
        self.calls += 1
        return ModelResponse(model=model,choices=[{'index':0,'message':{'role':'assistant','content':None,
            'tool_calls':[{'id':call_id,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]},'finish_reason':'tool_calls'}])
