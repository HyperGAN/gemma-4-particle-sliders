"""Execute every fixed DEV task for one declared model condition, then audit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from gemma_particle.data import digest,read_jsonl
from gemma_particle.dev_evidence import audit_verified_task,validate_matching_contract,journal_digest,retained_case_paths
from gemma_particle.dev_selection import load_selection_plan,summarize_verified_panel
from gemma_particle.harness_identity import sha256_file,wheel_identity
from gemma_particle.lora_serving import BASE_ALIAS,prepare_lora


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-plan',type=Path,default=Path('research/dev-selection-v1.json'))
    p.add_argument('--condition',required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--primary-inputs',type=Path,default=Path('data/freshswe/selected'))
    p.add_argument('--primary-wheels',type=Path,default=Path('runs/external-dev-controls-v1/empty-wheels'))
    p.add_argument('--secondary-inputs',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--secondary-wheels',type=Path,default=Path('runs/original-dev-environments-v1/wheels'))
    p.add_argument('--adapter',type=Path)
    p.add_argument('--conversion-report',type=Path)
    p.add_argument('--native',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--reuse-completed-from',type=Path,help='reaudit every complete contiguous task prefix; preserve completed inference, never reroll it')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if Path.cwd().resolve()!=root:raise ValueError('run inside the frozen DEV executor checkout')
    if args.out.exists():raise FileExistsError('DEV condition is immutable; incomplete trials require reconciliation')
    plan=load_selection_plan(args.selection_plan)
    found=[c for c in plan['conditions'] if c['id']==args.condition]
    if len(found)!=1:raise ValueError('condition outside the declared search grid')
    condition=found[0];endpoint=urlsplit(args.endpoint)
    if (endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','localhost','::1')
            or endpoint.path.rstrip('/')!='/v1' or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('explicit local /v1 endpoint required')
    packet=None
    if condition['role']=='learned':
        if any(getattr(args,k) is None for k in ['adapter','conversion_report','native','cache']):
            raise ValueError('learned condition requires measured native/cache/conversion inputs')
        config=json.loads((root/'configs/31b-ct-pilot.json').read_text())
        spec=prepare_lora(config,adapter=args.adapter,conversion_report=args.conversion_report)
        native=json.loads((args.native/'native.json').read_text())
        cache=json.loads((args.cache/'index.json').read_text())
        bindings=spec['provenance']['conversion_provenance']
        signature=native['game_signature'];source=plan['training']['source_cohort']
        if (native['completed_steps']!=condition['completed_steps'] or native['weights']!='raw' or native['ema_decay'] is not None
                or native['game_signature']['recipe']['name']!=condition['preset']
                or native['game_signature']['recipe']['total_steps']!=plan['training']['horizon']
                or bindings['native_metadata_sha256']!=sha256_file(args.native/'native.json')
                or native['sha256']!=sha256_file(args.native/'native.safetensors')
                or bindings['cache_index_sha256']!=sha256_file(args.cache/'index.json')
                or spec['provenance']['export_strength']!=condition['strength']
                or signature['response_shape']!=[plan['training']['response_width']]
                or signature['response_domain']!='continuous_vector' or signature['penalty_input_domain']!='continuous'
                or any(signature.get(k)!=source[k] for k in ('pg_version','pg_source_commit','pg_sources_sha256'))
                or cache['metadata']['train_sha256']!=plan['training']['train_manifest_sha256']
                or cache['responses_sha256']!=sha256_file(args.cache/'responses.safetensors')):
            raise ValueError('condition differs from actual trained checkpoint/cache/strength')
        packet={'adapters':spec['adapters'],'cache_index_sha256':bindings['cache_index_sha256'],
            'native_metadata_sha256':bindings['native_metadata_sha256'],'conversion_report_sha256':sha256_file(args.conversion_report),
            'native_metadata':str((args.native/'native.json').resolve()),'cache_index':str((args.cache/'index.json').resolve()),
            'conversion_report':str(args.conversion_report.resolve())}
    elif any(getattr(args,k) is not None for k in ['adapter','conversion_report','native','cache']):
        raise ValueError('true base/prompt controls cannot include adapter packets')
    panels={}
    for name in ('primary','secondary'):
        panel=plan[name];qualification=json.loads(Path(panel['qualification']).read_text())
        image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',panel['image_id']],text=True).strip()
        if image_id!=panel['image_id']:raise ValueError('qualified immutable DEV image unavailable')
        if name=='primary':
            visible=Path(panel['manifest']).parent/'dev.jsonl';raw=Path('data/freshswe/raw/tasks.jsonl')
            inputs=args.primary_inputs;wheels=args.primary_wheels
        else:
            visible=Path(panel['manifest']);raw=Path('data/kaggle-official/tasks.jsonl')
            inputs=args.secondary_inputs;wheels=args.secondary_wheels
            if sha256_file(wheels/'receipt.json')!=qualification['wheel_profile_sha256']:
                raise ValueError('original DEV wheel receipt differs')
            if wheel_identity(wheels)!=json.loads((wheels/'receipt.json').read_text())['wheels']:
                raise ValueError('original DEV wheel bytes differ')
            for control in qualification['controls']:
                receipt_path=Path(control['receipt'])
                if sha256_file(receipt_path)!=control['receipt_sha256']:raise ValueError('original DEV calibration receipt changed')
                receipt=json.loads(receipt_path.read_text())
                if (receipt['snapshot_sha256']!=sha256_file(inputs/'snapshots'/(control['instance_id']+'.tgz'))
                        or receipt['image_id']!=image_id):
                    raise ValueError('original DEV snapshot/image changed after calibration')
            calibration_root=Path(qualification['controls'][0]['receipt']).parent.parent
            calibration_plan=calibration_root/'plan.json'
            if (sha256_file(calibration_plan)!=qualification['plan_sha256']
                    or json.loads(calibration_plan.read_text())['sandbox_setup_sha256']!=sha256_file(inputs/'sandbox/setup.py')):
                raise ValueError('original DEV setup differs from calibration')
        tasks=read_jsonl(visible)
        if [t['instance_id'] for t in tasks]!=panel['task_ids']:raise ValueError('complete fixed DEV order changed')
        if sha256_file(raw)!=qualification['source_sha256']:raise ValueError('verifier source dataset changed')
        panels[name]={'tasks':tasks,'visible':visible,'raw':raw,'inputs':inputs,'wheels':wheels,'image':image_id}
    source_files=['scripts/run_dev_condition.py','scripts/run_trajectories.py','scripts/verify_trajectories.py',
        'scripts/build_submission.py','src/gemma_particle/capture.py','src/gemma_particle/adk_capture.py',
        'src/gemma_particle/history.py','src/gemma_particle/dev_evidence.py','src/gemma_particle/dev_selection.py',
        'src/gemma_particle/prompt_profiles.py','src/gemma_particle/dev.py','src/gemma_particle/sandbox.py',
        'src/gemma_particle/budgets.py','src/gemma_particle/curation.py','src/gemma_particle/context_budget.py',
        'src/gemma_particle/adapters.py','src/gemma_particle/lora_serving.py',
        'submission/prompts/system.md','submission/configs/sampling.yaml']
    sources={name:sha256_file(root/name) for name in source_files}
    reuse={};reuse_receipt=None
    if args.reuse_completed_from:
        previous=args.reuse_completed_from;previous_plan=previous/'plan.json'
        original=json.loads(previous_plan.read_text())
        audit_changes={'scripts/run_dev_condition.py','src/gemma_particle/dev_evidence.py'}
        if (original['condition']!=condition or original.get('packet')!=packet
                or original['selection_plan_sha256']!=sha256_file(args.selection_plan)
                or set(original['sources'])!=set(sources)
                or any(original['sources'][k]!=v for k,v in sources.items() if k not in audit_changes)):
            raise ValueError('reuse requires unchanged actual generation/verification implementations and declared condition')
        gap=False
        for name,panel in panels.items():
            for task in panel['tasks']:
                task_id=task['instance_id'];cohort,verification=retained_case_paths(previous,name,task_id)
                if not cohort.exists() and not verification.exists():gap=True;continue
                if gap or not (cohort/task_id/'trajectory.json').is_file() or not (verification/'outcomes.jsonl').is_file():
                    raise ValueError('reuse requires the entire complete contiguous prefix; partial tasks require reconciliation')
                row,contract=audit_verified_task(cohort,verification,task_id,expected_arm=condition['arm'],
                    expected_alias='particle_lora' if packet else BASE_ALIAS)
                validate_matching_contract(contract,plan)
                reuse[(name,task_id)]=(cohort,verification)
        if not reuse:raise ValueError('no complete actual prefix available for evidence reuse')
        reuse_receipt={'prior_condition':str(previous.resolve()),'prior_plan_sha256':sha256_file(previous_plan),
            'task_ids':[task_id for _,task_id in reuse],'audit_only_source_changes':sorted(audit_changes),
            'generation_and_verification_sources':'unchanged','policy':'all complete contiguous evidence retained regardless of outcome; no repeated model inference'}
    if not args.plan_only:
        subprocess.run(['git','diff','--quiet','HEAD'],check=True)
        server=json.loads(args.server_identity.read_text())
        if (server.get('kind')!='gemma4' or server.get('qualification_only') is not False
                or server.get('pretrained_inference') is not True
                or server['model_identity']['repo']!=plan['model'] or server['model_identity']['revision']!=plan['model_revision']):
            raise ValueError('actual pinned Gemma server required')
        if packet is not None:
            if server['adapters']!=packet['adapters'] or server.get('base_only'):raise ValueError('live candidate packet differs')
        elif server.get('base_only') is not True or server.get('adapters'):
            raise ValueError('controls require true unadapted LoRA-enabled base server')
    args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    declaration={'schema':'gemma4-dev-condition-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'condition':condition,'selection_plan_sha256':sha256_file(args.selection_plan),'sources':sources,
        'packet':packet,'reused_evidence':reuse_receipt,'task_ids':{n:[t['instance_id'] for t in p['tasks']] for n,p in panels.items()},
        'server_identity':str(args.server_identity.resolve()),'endpoint':args.endpoint,'model_inference':False,
        'status':'prepared' if args.plan_only else 'starting','scope':'full42+13 DEV condition; no selection or holdout access'}
    write(args.out/'plan.json',declaration)
    if args.plan_only:print(json.dumps({'prepared':str(args.out),'tasks':55,'model_inference':False}));return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def execute(command,log):
        if {name:sha256_file(root/name) for name in sources}!=sources:raise ValueError('frozen DEV executor sources changed')
        with log.open('x') as stream:r=subprocess.run([sys.executable,*command],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'pipeline exit{r.returncode}; preserve {log}')
    finished=[];metrics={};canonical={}
    expected_alias='particle_lora' if packet else BASE_ALIAS
    for panel_name,panel in panels.items():
        records=[];panel_root=args.out/panel_name;panel_root.mkdir()
        for task in panel['tasks']:
            task_id=task['instance_id'];cohort=panel_root/(task_id+'-agent');verification=panel_root/(task_id+'-verification')
            command=['scripts/run_trajectories.py','--tasks',str(panel['visible']),'--task-id',task_id,
                '--arm',condition['arm'],'--model',BASE_ALIAS,'--endpoint',args.endpoint,
                '--server-identity',str(args.server_identity),'--inputs',str(panel['inputs']),
                '--wheels',str(panel['wheels']),'--image',panel['image'],'--instruction-protocol','tools-v2',
                '--prompt-profile','repair-finalize-v2','--budget-profile','official-default-60m-v1',
                '--history-profile','official-compaction-v2','--capture-protocol','context-budget-off-v1','--out',str(cohort)]
            if packet:command.extend(['--adapter',str(args.adapter)])
            extra=[]
            if panel_name=='primary':extra=['--external-dev-manifest',plan['primary']['manifest'],
                '--external-dev-qualification',plan['primary']['qualification']]
            if (panel_name,task_id) in reuse:
                cohort,verification=reuse[(panel_name,task_id)]
            else:
                execute(command+extra,args.out/'logs'/(task_id+'-agent.log'))
                execute(['scripts/verify_trajectories.py','--cohort',str(cohort),'--tasks',str(panel['raw']),
                    '--inputs',str(panel['inputs']),'--wheels',str(panel['wheels']),'--image',panel['image'],
                    '--endpoint',args.endpoint,'--out',str(verification)]+extra,args.out/'logs'/(task_id+'-verify.log'))
            row,contract=audit_verified_task(cohort,verification,task_id,expected_arm=condition['arm'],expected_alias=expected_alias)
            validate_matching_contract(contract,plan)
            fingerprint=digest(contract)
            if panel_name in canonical and canonical[panel_name]!=fingerprint:raise ValueError('unmatched within-panel DEV contracts')
            canonical[panel_name]=fingerprint
            record={'instance_id':task_id,'panel':panel_name,'cohort':str(cohort),'verification':str(verification),
                'outcomes_sha256':sha256_file(verification/'outcomes.jsonl'),'resolved':row['resolved'],
                'agent_error':row['agent_error'],'capture_complete':row['capture_complete'],
                'model_journal_sha256':journal_digest(cohort,task_id),
                'reused_from':str(args.reuse_completed_from.resolve()) if (panel_name,task_id) in reuse else None}
            records.append(record);finished.append(record)
            write(args.out/'progress.json',{'state':'running','condition':args.condition,'finished':finished,
                'planned':55,'model_inference':True})
            print(json.dumps(record),flush=True)
        def outcomes():
            for record in records:
                if journal_digest(record['cohort'],record['instance_id'])!=record['model_journal_sha256']:
                    raise ValueError('DEV completion journal changed after verification')
                row,_=audit_verified_task(record['cohort'],record['verification'],record['instance_id'],
                    expected_arm=condition['arm'],expected_alias=expected_alias)
                yield row
        calibrated=plan['secondary']['calibrated_task_ids'] if panel_name=='secondary' else ()
        metrics[panel_name]=summarize_verified_panel(panel['tasks'],outcomes(),calibrated_ids=calibrated)
    result={'schema':'gemma4-dev-condition-result-v1','status':'complete','condition':condition,
        'selection_plan_sha256':declaration['selection_plan_sha256'],'canonical_contracts':canonical,
        'sources':sources,'server_identity_sha256':digest(server),'packet':packet,'reused_evidence':reuse_receipt,'finished':finished,
        **metrics,'scope':'one complete DEV condition only; no locked confirmation or official SOTA claim'}
    write(args.out/'result.json',result);print(json.dumps({'condition':args.condition,'status':'complete',
        'primary_resolved':metrics['primary']['resolved'],'secondary_resolved':metrics['secondary']['resolved']}),flush=True)


if __name__=='__main__':main()
