"""CPU receipt fixtures; never real checkpoints, targets or model-quality data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gemma_particle.conversion_handoff import validate_training_outputs,validate_conversion,projection_names
from gemma_particle.harness_identity import sha256_file
from gemma_particle.model import Native,ContextRow,make_game
from test_gemma_binding import tiny_model


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fake_finished_run(root):
    config=root/'config.json';put(config,{'model':'CPU-fixture','model_revision':'f'*40,
        'attn_implementation':'sdpa','frozen_quantization':'ct-w4g32',
        'rank':8,'alpha':8,'particles':128,'particle_width':4})
    identity={'repo':'CPU-fixture','revision':'f'*40,'dtype':'bfloat16',
        'attn_implementation':'sdpa','weight_format':'ct-w4g32'}
    train=root/'train.jsonl';train.parent.mkdir(parents=True,exist_ok=True)
    train.write_text(''.join(json.dumps({'instance_id':i,'split':'train'})+'\n' for i in ['fit-a','fit-b','probe-a','probe-b']))
    selection={'training':{'train_manifest':str(train),'train_manifest_sha256':sha256_file(train),
        'presets':['empf','rhbi_optimized'],'checkpoints':[256,512,1024],'source_cohort':{'fixture':'not actual PG source'}},
        'conditions':[{'role':'learned','preset':p,'completed_steps':s} for p in ['empf','rhbi_optimized'] for s in [256,512,1024]]}
    cache=root/'cache';cache.mkdir();(cache/'responses.safetensors').write_bytes(b'CPU fixture only; not actual tensors')
    put(cache/'index.json',{'schema':'gemma4-matched-response-cache-v1','metadata':{'model_identity':identity,
        'train_sha256':sha256_file(train)},'responses_sha256':sha256_file(cache/'responses.safetensors'),
        'rows':[{'instance_id':'fit-a'},{'instance_id':'fit-b'}]})
    cache_sha=sha256_file(cache/'index.json');pilot=root/'pilot'
    put(pilot/'progress.json',{'state':'completed_one_update','training_updates':1,'cache_index_sha256':cache_sha})
    probe=root/'probe.json';put(probe,{'train_manifest_sha256':sha256_file(train),'fixture':'CPU probe only'})
    qualification=root/'probe-qualification.json';put(qualification,{'status':'passed','probe_sha256':sha256_file(probe),
        'train_manifest_sha256':sha256_file(train),'task_ids':['probe-a','probe-b']})
    training=root/'training';put(training/'plan.json',{'config_sha256':sha256_file(config)});records=[]
    names=[f'adapters.fixture-layer-{i}-q_proj.lora_down.weight' for i in range(230)]
    for preset in selection['training']['presets']:
        folder=training/preset;folder.mkdir()
        lines=[json.dumps({'step':step,'g_loss':.1,'d_loss':.2})+'\n' for step in range(1,1025)]
        (folder/'updates.jsonl').write_text(''.join(lines))
        for step in [256,512,1024]:
            full=folder/f'FULL-{step:04d}.pt';full.write_bytes(f'CPU fixture full {preset}{step}'.encode())
            put(full.with_suffix('.json'),{'steps':step,'sha256':sha256_file(full)})
            native=folder/f'RAW-{step:04d}';native.mkdir();(native/'native.safetensors').write_bytes(b'CPU fixture raw only')
            signature={'recipe':{'name':preset,'total_steps':1024,'batch_size':2},'response_shape':[5376],
                'response_domain':'continuous_vector','penalty_input_domain':'continuous','row_count':2,
                'rows_sha256':'a'*64,'conditions_sha256':'b'*64,
                'backend':{'source_cohort':selection['training']['source_cohort'],'adapter_names':names}}
            put(native/'native.json',{'schema':'gemma4-native-slider-v1','completed_steps':step,'weights':'raw','ema_decay':None,
                'rank':8,'alpha':8,'bank_shape':[128,4],
                'model_identity':identity,'sha256':sha256_file(native/'native.safetensors'),'game_signature':signature})
            records.append({'preset':preset,'completed_steps':step,'full_checkpoint_sha256':sha256_file(full),
                'raw_metadata_sha256':sha256_file(native/'native.json'),
                'updates_sha256':hashlib.sha256(''.join(lines[:step]).encode()).hexdigest()})
    put(training/'progress.json',{'state':'completed_both_recipes','training_updates':2048,
        'cache_index_sha256':cache_sha,'finished':records})
    return training,cache,pilot,config,selection,probe,qualification


def test_complete_actual_receipt_shape_and_append_only_metric_prefixes(tmp_path):
    args=fake_finished_run(tmp_path);jobs=validate_training_outputs(*args)
    assert len(jobs)==6 and len(jobs[0]['projection_names'])==230
    # Later metrics append to the file; the old checkpoint binds its exact prefix.
    metrics=args[0]/'empf/updates.jsonl';lines=metrics.read_text().splitlines()
    damaged=json.loads(lines[0]);damaged['g_loss']=.3;lines[0]=json.dumps(damaged);metrics.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='journal prefix'):validate_training_outputs(*args)


@pytest.mark.parametrize('damage',['raw','cache','missing_checkpoint','probe_overlap','objective'])
def test_invalid_inputs_reject_before_any_conversion(tmp_path,damage):
    args=fake_finished_run(tmp_path);training,cache,_,_,_,_,qualification=args
    if damage=='raw':(training/'empf/RAW-0256/native.safetensors').write_bytes(b'changed')
    elif damage=='cache':(cache/'responses.safetensors').write_bytes(b'changed')
    elif damage=='missing_checkpoint':
        p=training/'progress.json';j=json.loads(p.read_text());j['finished'].pop();put(p,j)
    elif damage=='probe_overlap':
        j=json.loads(qualification.read_text());j['task_ids']=['fit-a'];put(qualification,j)
    else:
        p=training/'empf/RAW-0256/native.json';j=json.loads(p.read_text());j['game_signature']['recipe']['total_steps']=256;put(p,j)
        p2=training/'progress.json';q=json.loads(p2.read_text());q['finished'][0]['raw_metadata_sha256']=sha256_file(p);put(p2,q)
    with pytest.raises(ValueError):validate_training_outputs(*args)


def test_projection_names_match_actual_random_native_host_tree():
    native=Native(tiny_model(),identity={'fixture':'CPU random Gemma4'})
    try:
        rows=[native.prepare(ContextRow((2,3,4+i),(2,8,9,3,4+i),f'fixture-{i}',0)) for i in range(2)]
        for preset in ['empf','rhbi_optimized']:
            game=make_game(native,rows,preset=preset,steps=1024,batch=2)
            signature=game.signature
            assert projection_names(signature)=={name for name,_,_ in native.slider.hosts}
            assert signature['backend']['source_cohort']==native.source_cohort
            assert signature['recipe']['batch_size']==2 and signature['recipe']['total_steps']==1024
            assert signature['row_count']==2
    finally:native.close()


def test_conversion_packet_keeps_actual_file_native_cache_and_probe_bindings(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir()
    put(adapter/'adapter_config.json',{'peft_type':'LORA','r':8,'base_model_name_or_path':'CPU-fixture'})
    save_file({'fixture.lora_A.weight':torch.ones(8,16),'fixture.lora_B.weight':torch.ones(16,8)},adapter/'adapter_model.safetensors')
    job={'native_metadata_sha256':'a'*64,'cache_index_sha256':'b'*64,
        'fit_task_ids':['fit-a','fit-b'],'probe_task_ids':['probe-a','probe-b'],'projection_names':['fixture']}
    receipt={'schema':'gemma4-routed-to-peft-v1','exact_merge':False,'strength':1,
        'adapter_sha256':sha256_file(adapter/'adapter_model.safetensors'),
        'provenance':{**{k:job[k] for k in ['native_metadata_sha256','cache_index_sha256','fit_task_ids','probe_task_ids']},
            'probe_contexts_sha256':'c'*64},
        'layer_metrics':{'fixture':{'fit_tokens':256,'probe_tokens':256,'ridge':.001,
            'delta_rmse':.1,'delta_rms':1.,'relative_delta_rmse':.1}}}
    report=tmp_path/'conversion.json';put(report,receipt)
    identity=validate_conversion(adapter,report,job,strength=1,probe_sha256='c'*64)
    assert identity['rank']==8
    for field in ['native','probe','nonfinite','missing_host']:
        damaged=deepcopy(receipt)
        if field=='native':damaged['provenance']['native_metadata_sha256']='d'*64
        elif field=='probe':damaged['provenance']['probe_task_ids']=['fit-a']
        elif field=='nonfinite':damaged['layer_metrics']['fixture']['delta_rmse']=float('nan')
        else:damaged['layer_metrics']={}
        put(report,damaged)
        with pytest.raises(ValueError):validate_conversion(adapter,report,job,strength=1,probe_sha256='c'*64)
