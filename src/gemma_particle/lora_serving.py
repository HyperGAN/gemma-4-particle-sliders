"""Prepare immutable LoRA aliases before starting the pinned vLLM server."""
import json
import math
from pathlib import Path
import re

from .adapters import adapter_identity

BASE_ALIAS='gemma-4-31b-it-qat-w4a16-ct'
BASE_REPO='google/gemma-4-31B-it-qat-w4a16-ct'
BASE_REVISION='52f3f65bc7a02d555763bc923bd1d9094898219d'


def prepare_lora(config,*,adapter=None,fixtures=None,conversion_report=None):
    if config.get('model')!=BASE_REPO or config.get('model_revision')!=BASE_REVISION:
        raise ValueError('pinned official 31B CT base required')
    if (adapter is None)==(fixtures is None):
        raise ValueError('choose exactly one measured adapter or synthetic fixture manifest')
    directories={};provenance={}
    if fixtures is not None:
        path=Path(fixtures);manifest=json.loads(path.read_text())
        if manifest.get('kind')!='synthetic_lora_transport' or manifest.get('model_identity')!={'repo':BASE_REPO,'revision':BASE_REVISION}:
            raise ValueError('synthetic fixture kind and pinned model must match')
        if set(manifest.get('adapters',{}))!={'particle_zero','particle_probe'}:
            raise ValueError('both fixed zero and nonzero fixture aliases required')
        for name,row in manifest['adapters'].items():
            relative=Path(row['directory'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('fixture directories must stay under their manifest')
            directory=path.parent/relative
            if adapter_identity(directory)!=row['identity']:
                raise ValueError('synthetic fixture bytes changed')
            directories[name]=directory.resolve()
        kind='gemma4_lora_transport_fixture'
        provenance={'fixture_manifest':str(path.resolve()),'training_updates':0}
    else:
        directory=Path(adapter)
        identity=adapter_identity(directory)
        report=Path(conversion_report) if conversion_report else directory.with_name(directory.name+'-conversion.json')
        receipt=json.loads(report.read_text())
        if (receipt.get('schema')!='gemma4-routed-to-peft-v1' or receipt.get('exact_merge') is not False
                or receipt.get('adapter_sha256')!=identity['files']['adapter_model.safetensors']):
            raise ValueError('measured conversion receipt does not match the adapter')
        bindings=receipt.get('provenance',{})
        if any(not isinstance(bindings.get(k),str) or not re.fullmatch('[0-9a-f]{64}',bindings[k])
               for k in ('native_metadata_sha256','cache_index_sha256')):
            raise ValueError('native checkpoint/cache bindings required for a learned candidate')
        directories={'particle_lora':directory.resolve()};kind='gemma4'
        provenance={'conversion_report':str(report.resolve()),'conversion_provenance':bindings,
                    'export_strength':receipt.get('strength')}
    identities={name:adapter_identity(path) for name,path in directories.items()}
    allowed={BASE_REPO.lower(),BASE_REPO.removesuffix('-qat-w4a16-ct').lower()}
    if any(row['base_model_name_or_path'].lower() not in allowed for row in identities.values()):
        raise ValueError('adapter base differs from the selected 31B model')
    return {'kind':kind,'directories':{k:str(v) for k,v in directories.items()},'adapters':identities,
            'provenance':provenance,'qualification_only':fixtures is not None}


def lora_launch(config,spec,*,port,memory_fraction):
    if not 0<memory_fraction<=.65 or not 1024<=port<=65535:
        raise ValueError('shared memory fraction <=.65 and an explicit user port required')
    rank=max(row['rank'] for row in spec['adapters'].values())
    max_rank=next(r for r in (8,16,32,64,128) if r>=rank)
    launch=['vllm','serve',config['model'],'--revision',config['model_revision'],
        '--tokenizer-revision',config['model_revision'],'--served-model-name',BASE_ALIAS,
        '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--tensor-parallel-size','1',
        '--gpu-memory-utilization',str(memory_fraction),'--max-model-len','32768','--max-num-seqs','1',
        '--max-num-batched-tokens','2048','--enforce-eager','--language-model-only',
        '--no-scheduler-reserve-full-isl','--enable-auto-tool-choice','--tool-call-parser','gemma4',
        '--reasoning-parser','gemma4','--reasoning-config',json.dumps({'reasoning_start_str':'<|channel>','reasoning_end_str':'<channel|>'}),
        '--middleware','gemma_particle.vllm_identity.guard','--enable-lora','--lora-dtype','auto',
        '--max-lora-rank',str(max_rank),'--max-loras',str(len(spec['adapters'])),
        '--max-cpu-loras',str(len(spec['adapters'])),'--lora-modules']
    launch.extend(json.dumps({'name':name,'path':path,'base_model_name':BASE_ALIAS})
                  for name,path in spec['directories'].items())
    return launch


def compare_alias_effects(responses,*,tolerance=1e-5):
    """Require observed numerical zero parity and a nonzero adapter effect."""
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('finite nonnegative tolerance required')
    summaries={}
    for name in (BASE_ALIAS,'particle_zero','particle_probe'):
        response=responses[name]
        if response.get('model')!=name:raise ValueError('response model alias differs')
        choice=response['choices'][0];probs=choice['logprobs']
        if len(probs['tokens'])!=1 or not re.fullmatch(r'token_id:\d+',probs['tokens'][0]):
            raise ValueError('one generated token represented by its exact ID required')
        values=probs['top_logprobs'][0]
        if (not values or probs['tokens'][0] not in values
                or any(not re.fullmatch(r'token_id:\d+',key) or type(value) not in (int,float)
                       or not math.isfinite(value) for key,value in values.items())):
            raise ValueError('finite token-ID logprob observations required')
        summaries[name]={'token':probs['tokens'][0],'top':values}
    base,zero,probe=(summaries[name] for name in (BASE_ALIAS,'particle_zero','particle_probe'))
    zero_same_ids=base['top'].keys()==zero['top'].keys()
    zero_delta=max((abs(base['top'][key]-zero['top'][key]) for key in base['top'].keys()&zero['top'].keys()),default=None)
    common=base['top'].keys()&probe['top'].keys()
    probe_delta=max((abs(base['top'][key]-probe['top'][key]) for key in common),default=None)
    checks={'zero_token_equal':base['token']==zero['token'],
        'zero_top_ids_equal':zero_same_ids,'zero_logprobs_within_tolerance':zero_delta is not None and zero_delta<=tolerance,
        'nonzero_effect_observed':base['token']!=probe['token'] or base['top'].keys()!=probe['top'].keys()
                                  or (probe_delta is not None and probe_delta>tolerance)}
    return {'checks':checks,'passed':all(checks.values()),'zero_max_logprob_delta':zero_delta,
            'probe_max_common_logprob_delta':probe_delta,'tolerance':tolerance}
