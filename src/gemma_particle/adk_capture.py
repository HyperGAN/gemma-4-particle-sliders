"""Small adapter around the unchanged Google ADK LiteLLM client."""
from google.adk.models.lite_llm import LiteLLMClient
from .history import CALL_PURPOSE


class RecordingClient(LiteLLMClient):
    def __init__(self, journal, *, delegate=None):
        self.journal = journal
        self.delegate = delegate or LiteLLMClient()

    async def acompletion(self, model, messages, tools, **kwargs):
        packet = self.journal.begin(model=model,messages=messages,tools=tools,options=kwargs,purpose=CALL_PURPOSE.get())
        response = await self.delegate.acompletion(model=model,messages=messages,tools=tools,**kwargs)
        self.journal.complete(packet,response)
        return response
