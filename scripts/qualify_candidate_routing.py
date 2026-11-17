"""Real official compiler/ADK alias control with synthetic CPU adapter files.

No PEFT effect, vLLM inference, task result or training targets are claimed here.
"""
import argparse
import asyncio
import json
from pathlib import Path
import struct

from adk_submission import compile_submission,discover_adapters,ModelRegistry,ToolRegistry
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from build_submission import build
import validate_submission as declarations
from qualify_history import Fixture
from gemma_particle.adapters import adapter_identity,register_packaged_adapter
from gemma_particle.adk_capture import RecordingClient,RecordingLlm
from gemma_particle.capture import RequestJournal,load_requests
from gemma_particle.history import OFFICIAL_COMPACTION,eval_history_options


async def run(out):
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    fixture_dir=out/'synthetic-routing-only-peft';fixture_dir.mkdir()
    model_name='gemma-4-31b-it-qat-w4a16-ct'
    config={'peft_type':'LORA','r':1,'lora_alpha':1,'target_modules':['q_proj'],
        'base_model_name_or_path':'google/gemma-4-31B-it',
        'fixture_scope':'tiny synthetic tensors for compiler/alias transport only; not loadable 31B weights'}
    (fixture_dir/'adapter_config.json').write_text(json.dumps(config)+'\n')
    header={
        '__metadata__':{'scope':'synthetic CPU compiler routing control; never model-quality evidence'},
        'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':{'dtype':'F32','shape':[1,2],'data_offsets':[0,8]},
        'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':{'dtype':'F32','shape':[2,1],'data_offsets':[8,16]},
    }
    raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
    (fixture_dir/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*16)
    build(out/'artifact',model_name,adapter=fixture_dir)
    artifact=out/'artifact/agent'
    manifest=discover_adapters(artifact,adapter_extensions=frozenset({'.safetensors'}))
    identity={'particle_lora':adapter_identity(fixture_dir)}
    fixture=Fixture();journal=RequestJournal(out/'requests',instance_id='cpu-candidate-alias',arm='off')
    llm=RecordingLlm(model='openai/'+model_name,api_key='local',llm_client=RecordingClient(journal,delegate=fixture))
    models=ModelRegistry();models.register(model_name,llm)
    bound=register_packaged_adapter(models,manifest,llm,server_adapters=identity)
    tools=ToolRegistry()
    for name in ['run_command','read_file','edit_file','write_file','get_status','submit_patch',
                 'get_code_neighbors','search_similar_code','get_code_subgraph']:
        tools.register(name,getattr(declarations,name))
    agent=compile_submission(submission_dir=artifact,model_registry=models,tool_registry=tools,adapter_manifest=manifest)
    assert agent.canonical_model.model=='openai/particle_lora'
    assert agent.canonical_model.llm_client is llm.llm_client
    app=App(name='candidate_fixture',root_agent=agent,**eval_history_options(OFFICIAL_COMPACTION,llm))
    sessions=InMemorySessionService();session=await sessions.create_session(app_name=app.name,user_id='fixture')
    runner=Runner(app=app,session_service=sessions)
    for i in range(3):
        message=types.Content(role='user',parts=[types.Part(text=f'CPU alias fixture turn {i}.')])
        async for _ in runner.run_async(user_id='fixture',session_id=session.id,new_message=message):pass
    records=load_requests(out/'requests')
    assert records and fixture.summaries>0
    assert all(p['request']['model']=='openai/particle_lora' for p in records)
    assert all(p['request']['generation']['seed']==0 and
               p['request']['generation']['extra_body']['thinking_token_budget']==1024 for p in records)
    changed_files=dict(identity['particle_lora']['files'],**{'adapter_model.safetensors':'0'*64})
    for wrong in [{},dict(identity,particle_lora=dict(identity['particle_lora'],rank=2)),
                  dict(identity,particle_lora=dict(identity['particle_lora'],files=changed_files))]:
        try:register_packaged_adapter(ModelRegistry(),manifest,llm,server_adapters=wrong)
        except ValueError:pass
        else:raise AssertionError('server/adapter mismatch was accepted')
    receipt={'schema':'gemma4-candidate-alias-cpu-control-v1','status':'passed','pretrained_inference':False,
        'synthetic_adapter':True,'adapter_identities':bound,'official_compiler_selected_adapter_alias':True,
        'all_captured_calls_use_exact_adapter_alias':True,'summary_uses_adapter_alias':True,
        'seed_and_thinking_controls_preserved':True,
        'server_alias_and_bytes_mismatches_rejected':True,'model_calls':len(records),'summary_calls':fixture.summaries,
        'wire_alias_inference':'unmeasured; capture is before LiteLLM provider dispatch',
        'scope':'actual official compiler and ADK model-boundary routing only; no PEFT/vLLM effect or task-quality evidence; forbidden targets'}
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    asyncio.run(run(p.parse_args().out))
