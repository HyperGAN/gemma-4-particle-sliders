"""Validate actual training receipts before spending GPU time on conversion.

These checks bind local files and their declared source cohort. They cannot
authenticate fabricated JSON as a real training run.
"""
import hashlib
import json
import math
from pathlib import Path

from .adapters import adapter_identity
from .data import read_jsonl
from .harness_identity import sha256_file


def read(path):return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True


def projection_names(signature):
    return {n.removeprefix('adapters.').removesuffix('.lora_down.weight').replace('-','.')
        for n in signature['backend']['adapter_names'] if n.startswith('adapters.') and n.endswith('.lora_down.weight')}


def completed_training_state(selection):
    presets=set(selection['training']['presets'])
    if presets=={'empf'}:return 'completed_empf_only'
    if presets=={'empf','rhbi_optimized'}:return 'completed_both_recipes'
    raise ValueError('unsupported native recipe scope')


def validate_training_outputs(training,cache,pilot,config_path,selection,probe_contexts,probe_qualification,*,require_complete=True):
    training,cache,pilot=map(Path,(training,cache,pilot));config=read(config_path)
    status=read(training/'progress.json');declaration=read(training/'plan.json');first=read(pilot/'progress.json')
    presets=selection['training']['presets']
    complete=status.get('state')==completed_training_state(selection)
    running=status.get('state') in {'waiting_adopted_empf_checkpoint','waiting_for_shared_headroom','training','checkpoint_verified'}
    if ((not complete and (require_complete or not running))
            or (complete and status.get('training_updates')!=1024*len(presets))
            or set(declaration['presets'])!=set(presets)
            or first.get('state')!='completed_one_update' or first.get('training_updates')!=1
            or declaration['config_sha256']!=sha256_file(config_path)):
        raise ValueError('qualified native training state and unchanged pilot/config required')
    cache_sha=sha256_file(cache/'index.json');index=read(cache/'index.json')
    identity={'repo':config['model'],'revision':config['model_revision'],'dtype':'bfloat16',
        'attn_implementation':config['attn_implementation'],'weight_format':config['frozen_quantization']}
    if (status.get('cache_index_sha256')!=cache_sha or first.get('cache_index_sha256')!=cache_sha
            or index['schema']!='gemma4-matched-response-cache-v1' or index['metadata']['model_identity']!=identity
            or index['metadata']['train_sha256']!=selection['training']['train_manifest_sha256']
            or index['responses_sha256']!=sha256_file(cache/'responses.safetensors')):
        raise ValueError('one exact actual response cache required for the declared recipes')
    train=Path(selection['training']['train_manifest']);tasks=read_jsonl(train)
    if (sha256_file(train)!=selection['training']['train_manifest_sha256']
            or not tasks or any(t['split']!='train' for t in tasks)):
        raise ValueError('unchanged original TRAIN manifest required')
    ids=[r['instance_id'] for r in index['rows']]
    if len(ids)<2 or len(set(ids))!=len(ids) or not set(ids)<={t['instance_id'] for t in tasks}:
        raise ValueError('distinct admitted TRAIN initial questions required')
    probe=read(probe_contexts);qualification=read(probe_qualification)
    if (qualification.get('status')!='passed' or qualification['probe_sha256']!=sha256_file(probe_contexts)
            or qualification['train_manifest_sha256']!=sha256_file(train)
            or set(qualification['task_ids'])&set(ids)
            or probe['train_manifest_sha256']!=sha256_file(train)):
        raise ValueError('fixed qualified conversion probes must remain disjoint from all fitting questions')
    expected={(c['preset'],c['completed_steps']) for c in selection['conditions'] if c['role']=='learned'}
    records={(r['preset'],r['completed_steps']):r for r in status.get('finished',[])}
    if (len(records)!=len(status.get('finished',[])) or not records.keys()<=expected
            or (complete and records.keys()!=expected)):
        raise ValueError('unique actual receipts for the declared recipe checkpoints required')
    if not complete:
        verified=status['completed_steps_by_preset']
        if (any(verified[preset]<step for preset,step in records)
                or sum(verified.values())!=status['training_updates']):
            raise ValueError('verified checkpoint exceeds published training progress')
    jobs=[];data_signatures=set();preset_signatures={}
    for preset in selection['training']['presets']:
        checkpoints=selection['training']['checkpoints']
        available=[step for step in checkpoints if (preset,step) in records]
        if not available:
            continue
        if available!=checkpoints[:len(available)]:
            raise ValueError('verified checkpoints must form a contiguous declared prefix')
        directory=training/preset;lines=(directory/'updates.jsonl').read_bytes().splitlines(keepends=True)
        last=available[-1]
        if complete and len(lines)!=1024:
            raise ValueError('contiguous finite actual metric journal required')
        lines=lines[:last]
        metrics=[json.loads(line) for line in lines]
        if (len(lines)!=last or [m['step'] for m in metrics]!=list(range(1,last+1)) or not finite(metrics)
                or any(type(m.get(k)) not in (int,float) for m in metrics for k in ('g_loss','d_loss'))):
            raise ValueError('contiguous finite actual metric journal required')
        for step in available:
            record=records[preset,step]
            native=directory/f'RAW-{step:04d}';metadata=read(native/'native.json');pin=read(directory/f'FULL-{step:04d}.json')
            signature=metadata['game_signature'];recipe=signature['recipe']
            prefix_sha=hashlib.sha256(b''.join(lines[:step])).hexdigest()
            if (record['full_checkpoint_sha256']!=sha256_file(directory/f'FULL-{step:04d}.pt')
                    or pin['sha256']!=record['full_checkpoint_sha256'] or pin['steps']!=step
                    or record['raw_metadata_sha256']!=sha256_file(native/'native.json')
                    or record['updates_sha256']!=prefix_sha
                    or metadata['schema']!='gemma4-native-slider-v1' or metadata['completed_steps']!=step
                    or metadata['weights']!='raw' or metadata['ema_decay'] is not None
                    or metadata['rank']!=config['rank'] or metadata['alpha']!=config['alpha']
                    or metadata['bank_shape']!=[config['particles'],config['particle_width']]
                    or metadata['model_identity']!=identity
                    or metadata['sha256']!=sha256_file(native/'native.safetensors')
                    or recipe['name']!=preset or recipe['total_steps']!=1024 or recipe['batch_size']!=2
                    or signature['response_shape']!=[5376] or signature['response_domain']!='continuous_vector'
                    or signature['penalty_input_domain']!='continuous' or signature['row_count']!=len(ids)
                    or signature['backend']['source_cohort']!=selection['training']['source_cohort']):
                raise ValueError('actual raw/full checkpoint, journal prefix or native objective binding differs')
            if preset in preset_signatures and signature!=preset_signatures[preset]:
                raise ValueError('full objective/source signature changes within one recipe trajectory')
            preset_signatures[preset]=signature
            data_signatures.add((signature['rows_sha256'],signature['conditions_sha256'],signature['row_count']))
            hosts=projection_names(signature)
            if len(hosts)!=230:raise ValueError('all230 actual Gemma31B attention hosts required')
            jobs.append({'preset':preset,'completed_steps':step,'native':str(native.resolve()),
                'native_metadata_sha256':record['raw_metadata_sha256'],'full_checkpoint_sha256':record['full_checkpoint_sha256'],
                'projection_names':sorted(hosts),'cache_index_sha256':cache_sha,'fit_task_ids':sorted(ids),
                'probe_task_ids':sorted(qualification['task_ids'])})
    if len(data_signatures)>1:raise ValueError('training response/condition identities differ across recipes or checkpoints')
    return jobs


def validate_conversion(adapter,report,job,*,strength,probe_sha256):
    identity=adapter_identity(adapter);receipt=read(report);provenance=receipt.get('provenance',{})
    if (receipt.get('schema')!='gemma4-routed-to-peft-v1' or receipt.get('exact_merge') is not False
            or receipt.get('strength')!=strength or not finite(receipt)
            or receipt['adapter_sha256']!=identity['files']['adapter_model.safetensors']
            or provenance.get('native_metadata_sha256')!=job['native_metadata_sha256']
            or provenance.get('cache_index_sha256')!=job['cache_index_sha256']
            or provenance.get('probe_contexts_sha256')!=probe_sha256
            or sorted(provenance.get('fit_task_ids',[]))!=job['fit_task_ids']
            or sorted(provenance.get('probe_task_ids',[]))!=job['probe_task_ids']
            or set(receipt.get('layer_metrics',{}))!=set(job['projection_names'])):
        raise ValueError('measured conversion differs from actual native/cache/probe/strength bindings')
    for row in receipt['layer_metrics'].values():
        relative=row.get('relative_delta_rmse')
        if (row.get('fit_tokens')!=256 or row.get('probe_tokens')!=256
                or any(type(row.get(k)) not in (int,float) or row[k]<0 for k in ('ridge','delta_rmse','delta_rms'))
                or relative is None and row['delta_rms']!=0
                or relative is not None and (type(relative) not in (int,float) or relative<0)):
            raise ValueError('complete finite full-position fit/probe diagnostics required')
    return identity
