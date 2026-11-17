"""Read local source-bound DEV artifacts and preserve active model evidence."""
from copy import deepcopy
import json
from pathlib import Path

from .capture import load_requests,load_terminal_request
from .data import digest,read_jsonl
from .harness_identity import sha256_file
from .history import history_contract,OFFICIAL_COMPACTION
from .curation import instruction_set

SERVER_SOURCES=('scripts/serve_vllm_lora.py','scripts/serve_vllm.py','configs/31b-ct-pilot.json',
    'src/gemma_particle/lora_serving.py','src/gemma_particle/adapters.py','src/gemma_particle/vllm_identity.py')


def canonical_contract(contract):
    result=deepcopy(contract);result.pop('candidate_adapters',None)
    server=result['model_identity']
    if (server.get('kind')!='gemma4' or server.get('qualification_only') is not False
            or server.get('pretrained_inference') is not True or server.get('runtime_adapter_updates') is not False
            or server.get('strength')!=0 or server.get('native_checkpoint') is not None):
        raise ValueError('actual immutable Gemma LoRA-enabled server required')
    launch=list(server['launch'])
    if '--enable-lora' not in launch:raise ValueError('matched base/candidate LoRA-engine settings required')
    if '--lora-modules' in launch:
        start=launch.index('--lora-modules');end=start+1
        while end<len(launch) and not launch[end].startswith('--'):
            entry=json.loads(launch[end])
            if set(entry)!={'name','path','base_model_name'}:raise ValueError('unknown adapter launch fields')
            end+=1
        launch=launch[:start]+launch[end:]
    if '--port' not in launch:raise ValueError('explicit loopback port required')
    launch[launch.index('--port')+1]='<bound-loopback-port>'
    sources={}
    for name in SERVER_SOURCES:
        matches=[value for path,value in server['sources'].items() if path==name or path.endswith('/'+name)]
        if len(matches)!=1:raise ValueError('complete matching serving implementation sources required')
        sources[name]=matches[0]
    keys=('model_identity','device','memory_fraction','max_context','tensor_parallel_size','max_num_seqs',
        'adapter_dtype','runtime_adapter_updates','official_sources','packages')
    result['model_identity']={key:server[key] for key in keys}
    result['model_identity'].update(launch=launch,sources=sources)
    return result


def validate_matching_contract(contract,plan):
    matching=plan['matching']
    if (contract['budget']!=matching['budget']
            or contract['harness']['command_timeout_seconds']!=matching['command_timeout_seconds']
            or contract['sampling_sha256']!=matching['sampling_file_sha256']
            or contract.get('common_prompt')!={'profile':matching['prompt_profile'],'sha256':matching['common_system_sha256']}
            or contract.get('instruction_protocol')!=matching['instruction_protocol']
            or contract.get('instruction_set_sha256')!=digest(instruction_set(matching['instruction_protocol']))
            or contract.get('history')!=history_contract(OFFICIAL_COMPACTION)):
        raise ValueError('actual DEV prompt/sampling/budget/history differs from declared comparison')


def journal_digest(cohort,task_id):
    directory=Path(cohort)/task_id/'requests'
    return digest({p.name:sha256_file(p) for p in sorted(directory.glob('*.json'))})


def audit_verified_task(cohort,verification,task_id,*,expected_arm,expected_alias):
    cohort,verification=Path(cohort),Path(verification)
    manifest=json.loads((cohort/'cohort.json').read_text())
    if (manifest['generator_kind']!='gemma4' or manifest['arm']!=expected_arm
            or manifest['task_ids']!=[task_id] or len(manifest['runs'])!=1
            or manifest['evaluation_contract_sha256']!=digest(manifest['contract'])):
        raise ValueError('complete actual single-task DEV cohort required')
    rows=read_jsonl(verification/'outcomes.jsonl')
    if len(rows)!=1 or rows[0]['instance_id']!=task_id:raise ValueError('complete matching DEV verification required')
    row=rows[0];folder=cohort/task_id
    learned=expected_alias=='particle_lora'
    if (row.get('frozen_base_generator') is not (not learned)
            or learned and row.get('candidate_adapters')!=manifest['server_identity']['adapters']
            or not learned and row.get('candidate_adapters')):
        raise ValueError('DEV active adapter identity differs from its declared treatment')
    captured=json.loads((folder/'trajectory.json').read_text())
    if (captured.get('schema')!='gemma4-captured-trajectory-v1'
            or row.get('schema') not in ('gemma4-verified-trajectory-v3','gemma4-verified-trajectory-v4')
            or manifest['runs']!=[captured] or any(row.get(k)!=v for k,v in captured.items() if k!='schema')
            or row['verification']!=json.loads((verification/task_id/'verification.json').read_text())
            or row['patch_sha256']!=sha256_file(folder/'agent.patch')
            or row['verification']['test_output_sha256']!=sha256_file(verification/task_id/'test-output.txt')):
        raise ValueError('DEV row differs from captured patch or actual fresh verification')
    journal=folder/'requests'
    if row['capture_complete']:
        packets=load_requests(journal)
        if packets!=row['requests']:raise ValueError('verified DEV requests changed')
    else:
        paths=list(journal.glob('*-request.json'))
        if paths:
            prefix,terminal=load_terminal_request(journal);packets=[*prefix,terminal]
        else:packets=[]
    allowed={expected_alias,'openai/'+expected_alias}
    for packet in packets:
        if packet['request']['model'] not in allowed:raise ValueError('DEV requested model alias differs')
        response_path=journal/f"{packet['decision']:04d}-response.json"
        if response_path.exists():
            response=json.loads(response_path.read_text())
            if response['response'].get('model') not in allowed:
                raise ValueError('returned DEV model alias missing or different')
    return row,canonical_contract(manifest['contract'])
