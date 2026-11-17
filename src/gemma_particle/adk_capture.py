"""Small adapter around the unchanged Google ADK LiteLLM client."""
from google.adk.models.lite_llm import LiteLLMClient,LiteLlm
from .history import CALL_PURPOSE


class RecordingClient(LiteLLMClient):
    def __init__(self, journal, *, delegate=None):
        self.journal = journal
        self.delegate = delegate or LiteLLMClient()
        self.bound_model = None

    def __copy__(self):
        return self

    def __deepcopy__(self,memo):
        # One live transport and journal belong to one task. The official
        # compiler clones model options, not this external I/O resource.
        memo[id(self)]=self
        return self

    async def acompletion(self, model, messages, tools, **kwargs):
        packet = self.journal.begin(model=model,messages=messages,tools=tools,options=kwargs,purpose=CALL_PURPOSE.get())
        response = await self.delegate.acompletion(model=model,messages=messages,tools=tools,**kwargs)
        self.journal.complete(packet,response)
        return response


class RecordingLlm(LiteLlm):
    async def generate_content_async(self,llm_request,stream=False):
        if CALL_PURPOSE.get() is None:
            self.llm_client.bound_model=self
        async for response in super().generate_content_async(llm_request,stream=stream):
            yield response
