"""Measure verified native checkpoint conversions and stage the declared DEV packets."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

from gemma_particle.conversion_handoff import completed_training_state,validate_training_outputs,validate_conversion,read
from gemma_particle.dev_dispatch import live,load_control_results,process_binding
from gemma_particle.dev_selection import baseline_selection_sha256,load_selection_plan
from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import prepare_lora


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--training-pid',type=int)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--executor',type=Path,required=True)
    p.add_argument('--executor-commit',required=True)
    p.add_argument('--selection-plan',type=Path,default=Path('research/dev-selection-v1.json'))
    p.add_argument('--probe-contexts',type=Path,default=Path('data/conversion-probe-v1/contexts.json'))
    p.add_argument('--probe-qualification',type=Path,default=Path('reports/conversion-probe-preparation-v1.json'))
    p.add_argument('--device',choices=['cuda:0','cuda:1'],required=True)
    p.add_argument('--staged-controls',type=Path,help='baseline control root for distinct-slot staged conversion')
    p.add_argument('--staged-controls-pid',type=int)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    controller=Path(__file__).resolve().parents[1]
    for name in ['training','pilot','executor','selection_plan','probe_contexts','probe_qualification','out']:
        setattr(args,name,getattr(args,name).resolve())
    if args.out.exists():raise FileExistsError('sweep preparation is immutable; reconcile incomplete outputs')
    selection=load_selection_plan(args.selection_plan);config=args.executor/'configs/31b-ct-shared-positional-pilot.json'
    selection_sha=sha256_file(args.selection_plan)
    control_sha=baseline_selection_sha256(selection,selection_sha)
    learned=[condition for condition in selection['conditions'] if condition['role']=='learned']
    expected_jobs={(condition['preset'],condition['completed_steps']) for condition in learned}
    complete_state=completed_training_state(selection)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()!=args.executor_commit:
        raise ValueError('exact qualified native executor required')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=args.executor,check=True)
    training_plan=read(args.training/'plan.json');qualification=read(args.probe_qualification)
    same_device=training_plan['device']==args.device
    expect_same_device=args.staged_controls is None
    if (training_plan['executor_commit']!=args.executor_commit or same_device!=expect_same_device
            or set(training_plan['presets'])!=set(selection['training']['presets'])
            or training_plan['config_sha256']!=sha256_file(config)
            or qualification.get('status')!='passed' or qualification['probe_sha256']!=sha256_file(args.probe_contexts)
            or qualification['train_manifest_sha256']!=selection['training']['train_manifest_sha256']):
        raise ValueError('fixed native producer and pretraining TRAIN conversion probes required')
    sources={name:sha256_file(args.executor/name) for name in ['scripts/convert_to_peft.py','scripts/derive_peft_strength.py',
        'src/gemma_particle/model.py','src/gemma_particle/runtime.py','src/gemma_particle/export.py',
        'src/gemma_particle/conversion_probe.py','src/gemma_particle/lora_strengths.py']}
    controller_sources={name:sha256_file(controller/name) for name in ['scripts/prepare_learned_sweep.py',
        'src/gemma_particle/conversion_handoff.py','src/gemma_particle/dev_selection.py','src/gemma_particle/dev_dispatch.py',
        'src/gemma_particle/lora_serving.py','src/gemma_particle/harness_identity.py','src/gemma_particle/adapters.py','src/gemma_particle/data.py']}
    staged=None;training_binding=None
    if args.staged_controls:
        args.staged_controls=args.staged_controls.resolve()
        plan_path=args.staged_controls/'plan.json'
        controls_plan=read(plan_path)
        identity_path=Path(controls_plan['server_identity'])
        baseline=read(identity_path)
        if (controls_plan.get('schema')!='gemma4-full-dev-controls-plan-v1'
                or controls_plan['selection_plan_sha256']!=control_sha
                or baseline.get('device')!=args.device or baseline.get('base_only') is not True
                or baseline.get('adapters') or type(controls_plan.get('server_pid')) is not int
                or not controls_plan.get('server_birth_ticks')):
            raise ValueError('staged conversion requires the declared distinct baseline GPU and exact server binding')
        controls_progress=args.staged_controls/'progress.json'
        controls_complete=controls_progress.exists() and read(controls_progress).get('state')=='complete'
        producer=None
        if not args.plan_only and not controls_complete:
            if not args.staged_controls_pid:raise ValueError('owned live staged baseline producer required')
            producer=process_binding(args.staged_controls_pid,'run_dev_controls.py',args.staged_controls)
        staged={'plan_sha256':sha256_file(plan_path),'baseline_identity_sha256':sha256_file(identity_path),
            'directory':str(args.staged_controls),'baseline_identity':str(identity_path),'producer':producer,
            'server':{'pid':controls_plan['server_pid'],'birth_ticks':controls_plan['server_birth_ticks']}}
    elif args.staged_controls_pid:
        raise ValueError('--staged-controls-pid requires --staged-controls')
    already_complete=read(args.training/'progress.json').get('state')==complete_state
    if not args.plan_only and not already_complete:
        if not args.training_pid:raise ValueError('owned live training supervisor required for this waiting handoff')
        script='continue_empf_only.py' if complete_state=='completed_empf_only' else 'continue_full_training.py'
        training_binding=process_binding(args.training_pid,script,args.training)
    args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    declaration={'schema':'gemma4-learned-sweep-preparation-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'training':str(args.training),'training_plan_sha256':sha256_file(args.training/'plan.json'),
        'training_pid':args.training_pid,'training_birth_ticks':training_binding['birth_ticks'] if training_binding else None,
        'executor_commit':args.executor_commit,'selection_plan_sha256':selection_sha,'probe_sha256':sha256_file(args.probe_contexts),
        'probe_qualification_sha256':sha256_file(args.probe_qualification),'config_sha256':sha256_file(config),
        'sources':sources,'controller_sources':controller_sources,'source_sha256':sha256_file(Path(__file__)),'device':args.device,'memory_fraction':.5,
        'expected_unit_conversions':len(expected_jobs),'expected_candidate_packets':len(learned),
        'scope':'conversion/packet preparation only; no model DEV,holdout or official submission'}
    if staged:declaration['staged_controls']=staged
    initial_state='waiting_staged_baseline_release' if staged else 'waiting_actual_full_training'
    write(args.out/'plan.json',declaration);status={'state':'prepared' if args.plan_only else initial_state,
        'completed_unit_conversions':0,'candidate_packets':0,'model_dev_inference':False}
    def publish(state,**values):
        status.update(state=state,as_of=datetime.now(timezone.utc).isoformat(),**values)
        write(args.out/'progress.json',status);print(json.dumps(status),flush=True)
    publish(status['state'])
    if args.plan_only:return
    try:
        def training_status():
            status=read(args.training/'progress.json')
            if status['state']=='error':
                raise RuntimeError('actual native training failed; conversion cannot substitute checkpoints')
            if status['state']!=complete_state and not (training_binding and live(training_binding)):
                raise RuntimeError('native supervisor exited without qualified full completion')
            return status

        def check_declarations():
            if (sha256_file(args.training/'plan.json')!=declaration['training_plan_sha256']
                    or sha256_file(args.selection_plan)!=declaration['selection_plan_sha256']
                    or sha256_file(args.probe_qualification)!=declaration['probe_qualification_sha256']
                    or sha256_file(config)!=declaration['config_sha256']):
                raise ValueError('pinned producer/selection/probe/config declaration changed')
            if {k:sha256_file(controller/k) for k in controller_sources}!=controller_sources:
                raise ValueError('frozen conversion supervisor/auditor sources changed')
            if staged and (sha256_file(args.staged_controls/'plan.json')!=staged['plan_sha256']
                           or sha256_file(staged['baseline_identity'])!=staged['baseline_identity_sha256']):
                raise ValueError('pinned staged baseline producer or server identity changed')

        if staged:
            controls_progress=args.staged_controls/'progress.json'
            while True:
                training_status()
                control_state=read(controls_progress).get('state') if controls_progress.exists() else None
                if control_state=='error':raise RuntimeError('staged baseline producer failed')
                if control_state=='complete':break
                if not (staged['producer'] and live(staged['producer'])):
                    raise RuntimeError('staged baseline producer exited before completion')
                time.sleep(10)
            while live(staged['server']):
                training_status();time.sleep(1)
            check_declarations()
            controls=load_control_results(args.staged_controls,selection,selection_sha)
            write(args.out/'staged-baseline-release.json',{
                'server':staged['server'],'observed_exited':True,
                'as_of':datetime.now(timezone.utc).isoformat(),
                'control_results':{str(path):sha256_file(path) for path in controls}})
            (args.out/'staged-checkpoints').mkdir()
        else:
            while training_status()['state']!=complete_state:time.sleep(10)
            # The ordinary converter shares the trainer's physical GPU.
            while training_binding and live(training_binding):time.sleep(1)
            check_declarations()
        cache=args.pilot/'cache'

        def verified_jobs(*,require_complete):
            return validate_training_outputs(args.training,cache,args.pilot,config,selection,args.probe_contexts,
                                             args.probe_qualification,require_complete=require_complete)

        if not staged:
            write(args.out/'validated-native-jobs.json',verified_jobs(require_complete=True))

        env=dict(os.environ,PYTHONPATH=str(args.executor/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        def headroom():
            raw=subprocess.check_output(['nvidia-smi',f'--id={args.device[-1]}','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
            free,total=(int(v.strip())*1024**2 for v in raw.strip().split(','))
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            bound=read(args.executor/'reports/long-history-gradient-qualification-v1.json')['peak_host_rss_bytes']
            return free>=total*.5+8*1024**3 and available>=bound+8*1024**3
        def execute(command,log,cpu=False):
            check_declarations()
            if {k:sha256_file(args.executor/k) for k in sources}!=sources:raise ValueError('frozen conversion sources changed')
            if sha256_file(args.probe_contexts)!=declaration['probe_sha256']:raise ValueError('fixed conversion probes changed')
            child_env=dict(env)
            if cpu:child_env['CUDA_VISIBLE_DEVICES']=''
            else:child_env.pop('CUDA_VISIBLE_DEVICES',None)
            with log.open('x') as stream:
                result=subprocess.run([str(args.executor/'.train-venv/bin/python'),*command],cwd=args.executor,
                    env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'conversion pipeline exit{result.returncode}; retain {log}')
        exports={};completed=set()
        while True:
            current=training_status()
            recorded={(row['preset'],row['completed_steps']) for row in current.get('finished',[])}
            if not recorded-completed:
                if current['state']==complete_state:
                    if staged:write(args.out/'validated-native-jobs.json',verified_jobs(require_complete=True))
                    break
                if status['state']!='waiting_verified_checkpoint':publish('waiting_verified_checkpoint')
                time.sleep(10)
                continue
            check_declarations()
            job=next(row for row in verified_jobs(require_complete=not staged)
                     if (row['preset'],row['completed_steps']) not in completed)
            key=job['preset'],job['completed_steps']
            label=f"{job['preset']}-step{job['completed_steps']}";unit=args.out/(label+'-s1')
            publish('waiting_shared_headroom',current_checkpoint=label)
            while not headroom():
                training_status();time.sleep(10)
            # Revalidate immutable checkpoints/cache after any resource wait.
            check_declarations()
            if job not in verified_jobs(require_complete=not staged):
                raise ValueError('verified checkpoint changed before conversion')
            publish('measuring_unit_conversion',current_checkpoint=label)
            execute(['scripts/convert_to_peft.py','--config',str(config),'--native',job['native'],'--cache',str(cache),
                '--train-tasks',str(Path(selection['training']['train_manifest']).resolve()),
                '--probe-contexts',str(args.probe_contexts),'--tokens-per-layer','256','--strength','1',
                '--device',args.device,'--memory-fraction','.50','--out',str(unit)],args.out/'logs'/(label+'-unit.log'))
            for strength in [1,.5,1.5]:
                destination=unit if strength==1 else args.out/(label+'-s'+str(strength).replace('.','p'))
                if strength!=1:
                    publish('deriving_cpu_dose',current_checkpoint=label,current_strength=strength)
                    execute(['scripts/derive_peft_strength.py','--source',str(unit),'--strength',str(strength),
                        '--out',str(destination)],args.out/'logs'/(destination.name+'.log'),cpu=True)
                report=destination.with_name(destination.name+'-conversion.json')
                identity=validate_conversion(destination,report,job,strength=strength,probe_sha256=declaration['probe_sha256'])
                spec=prepare_lora(read(config),adapter=destination,conversion_report=report)
                if spec['adapters']!={'particle_lora':identity}:raise ValueError('actual serving packet identity differs')
                exports[(job['preset'],job['completed_steps'],strength)]={
                    'native':job['native'],'cache':str(cache),'adapter':str(destination),'conversion_report':str(report),
                    'native_metadata_sha256':job['native_metadata_sha256'],'cache_index_sha256':job['cache_index_sha256'],
                    'conversion_report_sha256':sha256_file(report),'adapters':spec['adapters']}
            if staged:
                packets=[{'condition':condition,**exports[(*key,condition['strength'])]} for condition in learned
                         if (condition['preset'],condition['completed_steps'])==key]
                if len(packets)!=3:raise ValueError('all three declared strengths are required for a staged checkpoint')
                write(args.out/'staged-checkpoints'/f'{label}.json',{'schema':'gemma4-staged-checkpoint-packets-v1',
                    'selection_plan_sha256':selection_sha,'verified_job':job,'packets':packets})
            completed.add(key);publish('checkpoint_packets_ready',completed_unit_conversions=len(completed),candidate_packets=len(exports))
        if completed!=expected_jobs or len(exports)!=len(learned):
            raise ValueError('complete declared learned grid required')
        packets=[{'condition':condition,**exports[(condition['preset'],condition['completed_steps'],condition['strength'])]}
            for condition in learned]
        write(args.out/'candidate-packets.json',{'schema':'gemma4-measured-learned-packets-v1',
            'selection_plan_sha256':declaration['selection_plan_sha256'],'packets':packets,
            'scope':'measured conversion and immutable packet identities only; no model DEV or official score'})
        publish('prepared_complete_learned_grid',completed_unit_conversions=len(completed),candidate_packets=len(packets),
            candidate_packets_sha256=sha256_file(args.out/'candidate-packets.json'))
    except Exception as exc:
        publish('error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
