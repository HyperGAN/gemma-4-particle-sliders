"""Run the official agent loop with immutable model-boundary request capture.

Only sanitized TRAIN/DEV tasks enter phase A. Verification is a separate process
that alone reads the reference-bearing task data. Nonfixture inference is local.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import httpx
from urllib.parse import urlsplit

from adk_submission import ModelRegistry
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import EvalConfig
from swegemma.harness.agent_runner import run_agent_sandbox
from swegemma.models import Task
from swegemma.sandbox import ContainerConfig, ContainerManager

from gemma_particle.adk_capture import RecordingClient
from gemma_particle.capture import RequestJournal,load_requests
from gemma_particle.curation import TARGET_INSTRUCTIONS,INSTRUCTION_PROTOCOLS,DEFAULT_INSTRUCTION_PROTOCOL,instruction_set
from gemma_particle.data import read_jsonl,digest
from gemma_particle.sandbox import isolate_sandbox_cache
from gemma_particle.harness_identity import package_identity,wheel_identity,sha256_file
from build_submission import build,MODELS


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--task-id',action='append',required=True)
    p.add_argument('--arm',choices=['off',*TARGET_INSTRUCTIONS],required=True)
    p.add_argument('--instruction-protocol',choices=INSTRUCTION_PROTOCOLS,default=DEFAULT_INSTRUCTION_PROTOCOL)
    p.add_argument('--model',choices=MODELS,default=MODELS[0])
    p.add_argument('--endpoint',default='http://127.0.0.1:8000/v1')
    p.add_argument('--server-identity',type=Path)
    p.add_argument('--cpu-protocol-fixture',action='store_true')
    p.add_argument('--inputs',type=Path,default=Path('data/control-inputs/root/.cache/kagglehub/competitions/gemma-4-developer-agent'))
    p.add_argument('--image',default='gemma-particle-sandbox:fastapi-deps-v2')
    p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('trajectory cohort is immutable')
    endpoint=urlsplit(args.endpoint)
    if endpoint.scheme!='http' or endpoint.hostname not in ('localhost','127.0.0.1','::1') or endpoint.username or endpoint.password:
        raise ValueError('an explicit local model endpoint is required')
    rows=read_jsonl(args.tasks)
    if any(r.get('split') not in ('train','dev') or 'patch' in r or 'test_patch' in r for r in rows):
        raise ValueError('agent phase requires sanitized TRAIN/DEV tasks')
    lookup={r['instance_id']:r for r in rows}
    if len(set(args.task_id))!=len(args.task_id) or any(i not in lookup for i in args.task_id):
        raise ValueError('selected task IDs must be unique and in the visible manifest')
    server_identity={'kind':'cpu_protocol_fixture','pretrained_inference':False}
    if not args.cpu_protocol_fixture:
        if not args.server_identity: raise ValueError('source-bound server identity file required')
        server_identity=json.loads(args.server_identity.read_text())
        if server_identity.get('kind')!='gemma4' or server_identity.get('served_model')!=args.model:
            raise ValueError('server identity does not match the declared model')
        identity=server_identity.get('model_identity',{})
        if not identity.get('repo') or len(identity.get('revision',''))!=40 or not identity.get('dtype'):
            raise ValueError('pinned model revision and precision required')
        health=httpx.get(args.endpoint.rstrip('/').removesuffix('/v1')+'/health',timeout=5).json()
        if health.get('identity_sha256')!=digest(server_identity):
            raise ValueError('live endpoint differs from the declared server identity')
    os.environ['KAGGLE_SANDBOX_DIR']=str((args.inputs/'sandbox').resolve())
    args.out.mkdir(parents=True,mode=0o700)
    artifact=build(args.out/'artifact',args.model,instruction=None if args.arm=='off' else args.arm,instruction_protocol=args.instruction_protocol)
    manifest={'schema':'gemma4-trajectory-cohort-v1','generator_kind':server_identity['kind'],
        'server_identity':server_identity,'model':args.model,'arm':args.arm,'task_ids':args.task_id,
        'visible_tasks_sha256':hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        'artifact':artifact,'image_id':None,'runs':[]}
    if args.instruction_protocol!=DEFAULT_INSTRUCTION_PROTOCOL:manifest['instruction_protocol']=args.instruction_protocol
    manager=ContainerManager(ContainerConfig(image=args.image,network_mode='none',mem_limit='4g',cpu_quota=200000))
    manifest['image_id']=manager.client.images.get(args.image).id
    manifest['environment']={'packages':package_identity(),'wheels':wheel_identity(args.wheels),
        'sandbox_setup_sha256':sha256_file(args.inputs/'sandbox/setup.py'),
        'network_mode':'none','mem_limit':'4g','cpu_quota':200000}
    (args.out/'cohort.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for task_id in args.task_id:
        row=lookup[task_id]
        visible={k:v for k,v in row.items() if k!='split'}
        task=Task(**visible)
        assert not task.patch and not task.test_patch
        folder=args.out/task_id;folder.mkdir(mode=0o700)
        isolate_sandbox_cache(folder/'tmp')
        (folder/'visible-task.json').write_text(json.dumps(row,indent=2)+'\n')
        journal=RequestJournal(folder/'requests',instance_id=task_id,arm=args.arm)
        delegate=None
        if args.cpu_protocol_fixture:
            from gemma_particle.protocol_fixture import ProtocolFixtureClient
            delegate=ProtocolFixtureClient()
        client=RecordingClient(journal,delegate=delegate)
        models=ModelRegistry()
        models.register(args.model,LiteLlm(model='openai/'+args.model,api_base=args.endpoint,api_key='local',
                                         extra_headers={'X-Gemma-Identity':digest(server_identity)} if not args.cpu_protocol_fixture else {},
                                         num_retries=5,llm_client=client))
        config=EvalConfig(tasks_path=args.tasks,snapshots_dir=args.inputs/'snapshots',results_dir=folder,
            submission_dir=args.out/'artifact/agent',models=models,wheels_dir=args.wheels,
            graph_dir=str(args.inputs/'graphs'),embeddings_dir=str(args.inputs/'embeddings'),
            image=args.image,max_tool_calls=40,max_time_minutes=5,max_turns=70,timeout_seconds=60,
            display_mode='quiet')
        manifest.setdefault('contract',{'budget':asdict(config.budget),'harness':asdict(config.harness),
            'environment':manifest['environment'],'image_id':manifest['image_id'],
            'model_identity':server_identity,'served_model':args.model,
            'sampling_sha256':sha256_file(args.out/'artifact/agent/configs/sampling.yaml')})
        if args.instruction_protocol!=DEFAULT_INSTRUCTION_PROTOCOL:
            manifest['contract'].update(instruction_protocol=args.instruction_protocol,
                instruction_set_sha256=digest(instruction_set(args.instruction_protocol)))
        manifest['evaluation_contract_sha256']=digest(manifest['contract'])
        start=time.perf_counter()
        patch,error,trace=await run_agent_sandbox(manager,config,task,args.inputs/'snapshots'/f'{task_id}.tgz')
        (folder/'agent.patch').write_text(patch)
        capture_error=None
        try:
            requests=load_requests(folder/'requests')
        except ValueError as exc:
            # Preserve inference failures in the denominator and the incomplete
            # journal on disk. Such runs cannot supply training contexts.
            requests=[]
            capture_error=str(exc)
        result={'schema':'gemma4-captured-trajectory-v1','generator_kind':server_identity['kind'],
            'instance_id':task_id,'split':row['split'],'arm':args.arm,'agent_error':error,
            'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'requests_sha256':digest(requests),
            'model_calls':len(requests),'tool_calls':sum(len(r['request']['messages'][-1].get('tool_calls',[])) for r in requests),
            'elapsed_seconds':time.perf_counter()-start}
        result['visible_task_sha256']=digest(row)
        result['snapshot_sha256']=sha256_file(args.inputs/'snapshots'/f'{task_id}.tgz')
        result['model_identity']=server_identity.get('model_identity')
        if args.instruction_protocol!=DEFAULT_INSTRUCTION_PROTOCOL:result['instruction_protocol']=args.instruction_protocol
        result['frozen_base_generator']=server_identity.get('strength')==0 and server_identity.get('native_checkpoint') is None
        result['evaluation_contract_sha256']=manifest['evaluation_contract_sha256']
        result['capture_complete']=capture_error is None
        result['capture_error']=capture_error
        # Tool count is authoritative from the actual emitted completions.
        result['tool_calls']=sum(len(c.get('message',{}).get('tool_calls') or [])
            for path in (folder/'requests').glob('*-response.json')
            for c in json.loads(path.read_text())['response']['choices'])
        (folder/'trajectory.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest['runs'].append(result)
        (args.out/'cohort.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__': asyncio.run(main())
