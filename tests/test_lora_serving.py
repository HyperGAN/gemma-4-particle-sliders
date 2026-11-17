import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
from safetensors.torch import save_file

from gemma_particle.adapters import adapter_identity
from gemma_particle.lora_serving import BASE_REPO,BASE_REVISION,BASE_ALIAS,prepare_lora,lora_launch,compare_alias_effects


def fixture_packet(tmp_path):
    rows={}
    for alias,folder,value in [('particle_zero','zero',0.),('particle_probe','probe',.05)]:
        path=tmp_path/folder;path.mkdir()
        (path/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':1,'base_model_name_or_path':BASE_REPO}))
        save_file({'a.lora_A.weight':torch.ones(1,2),'a.lora_B.weight':torch.full((2,1),value)},path/'adapter_model.safetensors')
        rows[alias]={'directory':folder,'identity':adapter_identity(path)}
    packet={'kind':'synthetic_lora_transport','model_identity':{'repo':BASE_REPO,'revision':BASE_REVISION},'adapters':rows}
    path=tmp_path/'fixtures.json';path.write_text(json.dumps(packet))
    return path


def test_cpu_plan_preserves_fixture_isolation_and_all_declared_aliases(tmp_path):
    root=Path(__file__).resolve().parents[1];packet=fixture_packet(tmp_path)
    out=tmp_path/'plan';env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([str(root/'.harness-venv/bin/python'),'scripts/serve_vllm_lora.py',
        '--synthetic-fixtures',str(packet),'--device','cuda:1','--out',str(out),'--plan-only'],
        cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr[-3000:]
    plan=json.loads((out/'launch-plan.json').read_text())
    assert not plan['gpu_used'] and not plan['inference_executed'] and not plan['environment_verified']
    assert plan['spec']['kind']=='gemma4_lora_transport_fixture' and plan['spec']['qualification_only']
    assert set(plan['spec']['adapters'])=={'particle_zero','particle_probe'}
    assert not (out/'server-identity.json').exists()
    entries=plan['launch'][plan['launch'].index('--lora-modules')+1:]
    assert {json.loads(x)['name'] for x in entries}==set(plan['spec']['adapters'])
    assert all(json.loads(x)['base_model_name']==BASE_ALIAS for x in entries)
    assert plan['launch'][plan['launch'].index('--max-model-len')+1]=='32768'


def test_changed_fixture_or_conversion_bytes_fail_before_launch(tmp_path):
    config={'model':BASE_REPO,'model_revision':BASE_REVISION};packet=fixture_packet(tmp_path)
    spec=prepare_lora(config,fixtures=packet)
    assert spec['qualification_only']
    weights=tmp_path/'probe/adapter_model.safetensors'
    weights.write_bytes(weights.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='bytes changed'):prepare_lora(config,fixtures=packet)
    report=tmp_path/'conversion.json'
    report.write_text(json.dumps({'schema':'gemma4-routed-to-peft-v1','exact_merge':False,'adapter_sha256':'0'*64}))
    with pytest.raises(ValueError,match='does not match'):prepare_lora(config,adapter=tmp_path/'zero',conversion_report=report)


def test_real_official_vllm_lora_parser_accepts_prepared_aliases_without_cuda(tmp_path):
    root=Path(__file__).resolve().parents[1];packet=fixture_packet(tmp_path)
    config={'model':BASE_REPO,'model_revision':BASE_REVISION}
    launch=lora_launch(config,prepare_lora(config,fixtures=packet),port=8027,memory_fraction=.60)
    entries=launch[launch.index('--lora-modules')+1:]
    code='''import argparse,json,sys
from vllm.entrypoints.openai.cli_args import LoRAParserAction
p=argparse.ArgumentParser();p.add_argument('--lora-modules',nargs='+',action=LoRAParserAction)
args=p.parse_args(['--lora-modules',*sys.argv[1:]])
assert {x.name for x in args.lora_modules}=={'particle_zero','particle_probe'}
assert all(x.base_model_name=='gemma-4-31b-it-qat-w4a16-ct' for x in args.lora_modules)
print(json.dumps({'official_parser':'passed','aliases':[x.name for x in args.lora_modules]}))
'''
    result=subprocess.run([str(root/'.vllm-venv/bin/python'),'-c',code,*entries],
        cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),text=True,capture_output=True)
    assert result.returncode==0,result.stderr[-4000:]
    assert '"official_parser": "passed"' in result.stdout


def test_inert_or_nonzero_zero_adapter_is_never_promoted_as_a_passing_effect_control():
    from copy import deepcopy
    def response(name,offset=0):
        return {'model':name,'choices':[{'logprobs':{'tokens':['token_id:7'],
            'top_logprobs':[{'token_id:7':-.1+offset,'token_id:9':-2.}]}}]}
    rows={name:response(name,.01 if name=='particle_probe' else 0) for name in [BASE_ALIAS,'particle_zero','particle_probe']}
    assert compare_alias_effects(rows)['passed']
    inert=deepcopy(rows);inert['particle_probe']=response('particle_probe')
    assert not compare_alias_effects(inert)['passed']
    changed_zero=deepcopy(rows);changed_zero['particle_zero']=response('particle_zero',.01)
    assert not compare_alias_effects(changed_zero)['passed']
    nonfinite=deepcopy(rows);nonfinite['particle_probe']['choices'][0]['logprobs']['top_logprobs'][0]['token_id:7']=float('nan')
    with pytest.raises(ValueError,match='finite'):compare_alias_effects(nonfinite)


def test_true_base_control_has_no_synthetic_weights_and_preserves_lora_engine_shape(tmp_path):
    config={'model':BASE_REPO,'model_revision':BASE_REVISION}
    spec=prepare_lora(config,base_only=True)
    assert spec['kind']=='gemma4' and spec['base_only'] and not spec['qualification_only']
    assert spec['directories']=={} and spec['adapters']=={}
    launch=lora_launch(config,spec,port=8029,memory_fraction=.60)
    assert '--enable-lora' in launch and '--lora-modules' not in launch
    assert launch[launch.index('--max-loras')+1]=='1'
    assert launch[launch.index('--max-cpu-loras')+1]=='1'
    assert launch[launch.index('--max-lora-rank')+1]=='8'
    with pytest.raises(ValueError,match='exactly one'):
        prepare_lora(config,base_only=True,fixtures=fixture_packet(tmp_path))
