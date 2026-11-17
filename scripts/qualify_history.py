"""Run a real ADK compaction loop with a scripted CPU transport, never targets."""
import argparse
import asyncio
import json
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm,LiteLLMClient
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from litellm.types.utils import ModelResponse

from gemma_particle.adk_capture import RecordingClient
from gemma_particle.capture import RequestJournal,load_requests
from gemma_particle.history import OFFICIAL_COMPACTION,CALL_PURPOSE,eval_history_options,is_assistant_decision


class Fixture(LiteLLMClient):
    def __init__(self):self.decisions=0;self.summaries=0

    async def acompletion(self,model,messages,tools,**kwargs):
        if CALL_PURPOSE.get()=='history_summary':
            self.summaries+=1;content='CPU compacted history: preserve the original issue and completed work.'
        else:
            self.decisions+=1;content=f'CPU fixture agent response {self.decisions}.'
        # Explicit scripted usage triggers ADK's actual token-threshold gate.
        # This is software protocol evidence, not measured token/model behavior.
        return ModelResponse(model=model,choices=[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
            usage={'prompt_tokens':16000,'completion_tokens':10,'total_tokens':16010})


async def run(out):
    if out.exists():raise FileExistsError('history qualification is immutable')
    out.mkdir(parents=True)
    fixture=Fixture();journal=RequestJournal(out/'requests',instance_id='cpu-history-fixture',arm='off')
    llm=LiteLlm(model='openai/cpu-history-fixture',api_key='local',llm_client=RecordingClient(journal,delegate=fixture))
    options=eval_history_options(OFFICIAL_COMPACTION,llm)
    agent=LlmAgent(name='cpu_history_fixture',model=llm,instruction='Preserve the issue and report progress.')
    app=App(name='history_fixture',root_agent=agent,**options)
    sessions=InMemorySessionService();session=await sessions.create_session(app_name=app.name,user_id='fixture')
    runner=Runner(app=app,session_service=sessions)
    for i in range(8):
        message=types.Content(role='user',parts=[types.Part(text=f'Original issue: fixture repair. Turn {i}.')])
        async for _ in runner.run_async(user_id='fixture',session_id=session.id,new_message=message):pass
        assert CALL_PURPOSE.get() is None
    records=load_requests(out/'requests')
    auxiliary=[p for p in records if not is_assistant_decision(p,OFFICIAL_COMPACTION)]
    decisions=[p for p in records if is_assistant_decision(p,OFFICIAL_COMPACTION)]
    assert fixture.summaries>0 and len(auxiliary)==fixture.summaries and len(decisions)==8
    assert any('CPU compacted history' in json.dumps(p['request']['messages']) for p in decisions)
    receipt={'schema':'gemma4-history-cpu-fixture-v1','status':'passed','pretrained_inference':False,
        'profile':OFFICIAL_COMPACTION,'assistant_decisions':len(decisions),'summary_calls':len(auxiliary),
        'all_calls_journaled':len(records),'summary_seen_in_subsequent_assistant_history':True,
        'purpose_scope_restored':True,'generation':'scripted transport with declared synthetic usage metadata',
        'scope':'actual ADK Runner and public token-threshold compaction APIs; forbidden as training targets or task-quality evidence'}
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();asyncio.run(run(args.out))
