"""Wait for real full checkpoints, measure six conversions, stage18 DEV packets."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

from gemma_particle.conversion_handoff import validate_training_outputs,validate_conversion,read
from gemma_particle.dev_selection import load_selection_plan
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
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    controller=Path(__file__).resolve().parents[1]
    for name in ['training','pilot','executor','selection_plan','probe_contexts','probe_qualification','out']:
        setattr(args,name,getattr(args,name).resolve())
    if args.out.exists():raise FileExistsError('sweep preparation is immutable; reconcile incomplete outputs')
    selection=load_selection_plan(args.selection_plan);config=args.executor/'configs/31b-ct-shared-positional-pilot.json'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()!=args.executor_commit:
        raise ValueError('exact qualified native executor required')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=args.executor,check=True)
    training_plan=read(args.training/'plan.json');qualification=read(args.probe_qualification)
    if (training_plan['executor_commit']!=args.executor_commit or training_plan['device']!=args.device
            or training_plan['config_sha256']!=sha256_file(config)
            or qualification.get('status')!='passed' or qualification['probe_sha256']!=sha256_file(args.probe_contexts)
            or qualification['train_manifest_sha256']!=selection['training']['train_manifest_sha256']):
        raise ValueError('fixed native producer and pretraining TRAIN conversion probes required')
    sources={name:sha256_file(args.executor/name) for name in ['scripts/convert_to_peft.py','scripts/derive_peft_strength.py',
        'src/gemma_particle/model.py','src/gemma_particle/runtime.py','src/gemma_particle/export.py',
        'src/gemma_particle/conversion_probe.py','src/gemma_particle/lora_strengths.py']}
    controller_sources={name:sha256_file(controller/name) for name in ['scripts/prepare_learned_sweep.py',
        'src/gemma_particle/conversion_handoff.py','src/gemma_particle/dev_selection.py',
        'src/gemma_particle/lora_serving.py','src/gemma_particle/harness_identity.py','src/gemma_particle/adapters.py','src/gemma_particle/data.py']}
    birth=None;proc=None
    already_complete=read(args.training/'progress.json').get('state')=='completed_both_recipes'
    if not args.plan_only and not already_complete:
        if not args.training_pid:raise ValueError('owned live training supervisor required for this waiting handoff')
        proc=Path('/proc')/str(args.training_pid);argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
        if not any(v.endswith(b'/continue_full_training.py') for v in argv) or b'--out' not in argv:
            raise ValueError('exact full training supervisor required')
        output=Path(os.fsdecode(argv[argv.index(b'--out')+1]))
        if not output.is_absolute():output=proc.joinpath('cwd').resolve()/output
        if output.resolve()!=args.training:raise ValueError('training supervisor output differs')
        birth=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
    args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    declaration={'schema':'gemma4-learned-sweep-preparation-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'training':str(args.training),'training_plan_sha256':sha256_file(args.training/'plan.json'),
        'training_pid':args.training_pid,'training_birth_ticks':birth,'executor_commit':args.executor_commit,
        'selection_plan_sha256':sha256_file(args.selection_plan),'probe_sha256':sha256_file(args.probe_contexts),
        'probe_qualification_sha256':sha256_file(args.probe_qualification),'config_sha256':sha256_file(config),
        'sources':sources,'controller_sources':controller_sources,'source_sha256':sha256_file(Path(__file__)),'device':args.device,'memory_fraction':.5,
        'expected_unit_conversions':6,'expected_candidate_packets':18,
        'scope':'conversion/packet preparation only; no model DEV,holdout or official submission'}
    write(args.out/'plan.json',declaration);status={'state':'prepared' if args.plan_only else 'waiting_actual_full_training',
        'completed_unit_conversions':0,'candidate_packets':0,'model_dev_inference':False}
    def publish(state,**values):
        status.update(state=state,as_of=datetime.now(timezone.utc).isoformat(),**values)
        write(args.out/'progress.json',status);print(json.dumps(status),flush=True)
    publish(status['state'])
    if args.plan_only:return
    try:
        def same_live_producer():
            if proc is None:return False
            try:stat=proc.joinpath('stat').read_text().split(') ',1)[1].split()
            except FileNotFoundError:return False
            return stat[19]==birth and stat[0]!='Z'
        def check_declarations():
            if (sha256_file(args.training/'plan.json')!=declaration['training_plan_sha256']
                    or sha256_file(args.selection_plan)!=declaration['selection_plan_sha256']
                    or sha256_file(args.probe_qualification)!=declaration['probe_qualification_sha256']
                    or sha256_file(config)!=declaration['config_sha256']):
                raise ValueError('pinned producer/selection/probe/config declaration changed')
            if {k:sha256_file(controller/k) for k in controller_sources}!=controller_sources:
                raise ValueError('frozen conversion supervisor/auditor sources changed')
        while True:
            current=read(args.training/'progress.json')
            if current['state']=='completed_both_recipes':break
            if current['state']=='error':raise RuntimeError('actual native training failed; conversion cannot substitute checkpoints')
            if not same_live_producer():
                raise RuntimeError('native supervisor exited without qualified full completion')
            time.sleep(10)
        # Completion publication precedes process exit. Observe release before a
        # second native model can claim the same GPU/host memory allocation.
        while same_live_producer():time.sleep(1)
        check_declarations()
        cache=args.pilot/'cache'
        jobs=validate_training_outputs(args.training,cache,args.pilot,config,selection,args.probe_contexts,args.probe_qualification)
        write(args.out/'validated-native-jobs.json',jobs)
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
        exports={};completed=[]
        for job in jobs:
            label=f"{job['preset']}-step{job['completed_steps']}";unit=args.out/(label+'-s1')
            publish('waiting_shared_headroom',current_checkpoint=label)
            while not headroom():time.sleep(10)
            # Revalidate immutable checkpoints/cache after any resource wait.
            validate_training_outputs(args.training,cache,args.pilot,config,selection,args.probe_contexts,args.probe_qualification)
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
            completed.append(label);publish('checkpoint_packets_ready',completed_unit_conversions=len(completed),candidate_packets=len(exports))
        packets=[{'condition':condition,**exports[(condition['preset'],condition['completed_steps'],condition['strength'])]}
            for condition in selection['conditions'] if condition['role']=='learned']
        if len(packets)!=18:raise ValueError('complete declared learned grid required')
        write(args.out/'candidate-packets.json',{'schema':'gemma4-measured-learned-packets-v1',
            'selection_plan_sha256':declaration['selection_plan_sha256'],'packets':packets,
            'scope':'measured conversion and immutable packet identities only; no model DEV or official score'})
        publish('prepared_complete_learned_grid',completed_unit_conversions=6,candidate_packets=18,
            candidate_packets_sha256=sha256_file(args.out/'candidate-packets.json'))
    except Exception as exc:
        publish('error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
