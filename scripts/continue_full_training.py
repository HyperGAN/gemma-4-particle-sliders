"""Continue the qualified first update through the fixed two-recipe horizon.

This CPU supervisor waits for actual pilot completion. Its frozen executor
owns every model operation and exact-resume check. DEV outcomes are not inputs.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--pilot-pid',type=int)
    p.add_argument('--executor',type=Path,required=True)
    p.add_argument('--executor-commit',required=True)
    p.add_argument('--pilot-executor',type=Path,help='original pilot checkout when the sole training-script delta is CPU checkpoint loading')
    p.add_argument('--device',choices=['cuda:0','cuda:1'],required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.pilot,args.executor,args.out=[x.resolve() for x in (args.pilot,args.executor,args.out)]
    if args.out.exists():raise FileExistsError('training continuation is immutable; reconcile incomplete runs')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()!=args.executor_commit:
        raise ValueError('frozen native executor differs')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=args.executor,check=True)
    config=args.executor/'configs/31b-ct-shared-positional-pilot.json';settings=read(config)
    qualification=read(args.executor/'reports/long-history-gradient-qualification-v1.json')
    if (settings['updates']!=1024 or settings['batch_size']!=2 or settings['seed']!=0
            or settings['weights']!='raw' or settings['ema_decay'] is not None
            or qualification['status']!='passed' or qualification['memory_fraction']!=.5):
        raise ValueError('qualified unchanged two-recipe/1024-step/raw/batch2 configuration required')
    sources={name:sha(args.executor/name) for name in ['scripts/train.py','src/gemma_particle/model.py',
        'src/gemma_particle/runtime.py','src/gemma_particle/checkpoint_inputs.py',
        'src/gemma_particle/activation_storage.py','src/gemma_particle/ct_linear.py']}
    pilot=read(args.pilot/'progress.json')
    pilot_executor=(args.pilot_executor or args.executor).resolve()
    pilot_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot_executor,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=pilot_executor,check=True)
    if pilot['executor_commit']!=pilot_commit or pilot['declared_horizon']!=1024 or pilot['preset']!='empf':
        raise ValueError('original qualified EMPF pilot required')
    loader_delta=None
    if pilot_commit!=args.executor_commit:
        math_sources=[k for k in sources if k!='scripts/train.py']+[
            'src/gemma_particle/source_identity.py','src/gemma_particle/data.py','sources.lock.json',
            'configs/31b-ct-shared-positional-pilot.json']
        if any(sha(pilot_executor/k)!=sha(args.executor/k) for k in math_sources):
            raise ValueError('checkpoint-placement correction cannot change native math/config/source cohort')
        previous=ast.parse((pilot_executor/'scripts/train.py').read_text())
        current=ast.parse((args.executor/'scripts/train.py').read_text());changed=0
        for node in ast.walk(previous):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='load'):
                for keyword in node.keywords:
                    if keyword.arg=='map_location':keyword.value=ast.Constant(value='cpu');changed+=1
        if changed!=1 or ast.dump(previous)!=ast.dump(current):
            raise ValueError('only CPU checkpoint map_location may differ from the pilot training script')
        loader_delta={'original_executor_commit':pilot_commit,'original_train_source_sha256':sha(pilot_executor/'scripts/train.py'),
            'current_train_source_sha256':sources['scripts/train.py'],
            'semantic_delta':'only torch.load map_location=cpu; public game/optimizer loaders own placement',
            'native_math_source_comparison':'exact byte equality','training_script_comparison':'AST equality after this one allowed keyword change'}
    birth=None
    if not args.plan_only and pilot['state']!='completed_one_update':
        if not args.pilot_pid:raise ValueError('live pilot PID required while waiting')
        proc=Path('/proc')/str(args.pilot_pid)
        command=proc.joinpath('cmdline').read_bytes().split(b'\0')
        if not any(x.endswith(b'/wait_and_train_pilot.py') for x in command):raise ValueError('exact pilot supervisor required')
        source=Path(os.fsdecode(command[command.index(b'--out')+1]))
        if not source.is_absolute():source=proc.joinpath('cwd').resolve()/source
        if source.resolve()!=args.pilot:raise ValueError('pilot process output differs')
        birth=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
    args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    declaration={'schema':'gemma4-full-training-continuation-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'pilot':str(args.pilot),'pilot_pid':args.pilot_pid,'pilot_birth_ticks':birth,
        'executor_commit':args.executor_commit,'pilot_executor_commit':pilot_commit,
        'checkpoint_loading_delta':loader_delta,'sources':sources,'source_sha256':sha(__file__),
        'config_sha256':sha(config),'device':args.device,'memory_fraction':.5,'startup_headroom_GiB':8,
        'horizon':1024,'checkpoints':[256,512,1024],'presets':['empf','rhbi_optimized'],
        'optimizer_execution':{'empf':'unchanged recipe','rhbi_optimized':'explicit gpu_native_polar derivative'},
        'cache_rule':'same exact admitted TRAIN cache for both recipes; no DEV/holdout data',
        'pilot_preservation':'original pilot artifacts unchanged; its exact metric is copied into the EMPF continuation',
        'gpu_used':False,'scope':'prospective full native training; no model quality or official score'}
    write(args.out/'plan.json',declaration)
    status={'state':'prepared' if args.plan_only else 'waiting_for_actual_pilot','training_updates':0}
    write(args.out/'progress.json',status)
    if args.plan_only:print(json.dumps(status));return
    def publish(state,**values):
        status.update(state=state,as_of=datetime.now(timezone.utc).isoformat(),**values)
        write(args.out/'progress.json',status);print(json.dumps(status),flush=True)
    def headroom():
        output=subprocess.check_output(['nvidia-smi',f'--id={args.device[-1]}','--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'],text=True)
        free,total=(int(v.strip())*1024**2 for v in output.strip().split(','))
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        return free>=total*.5+8*1024**3 and available>=qualification['peak_host_rss_bytes']+8*1024**3
    def validate(directory,target):
        pin=read(directory/f'FULL-{target:04d}.json');native=read(directory/f'RAW-{target:04d}/native.json')
        metrics=[json.loads(s) for s in (directory/'updates.jsonl').read_text().splitlines() if s.strip()]
        if (pin['steps']!=target or pin['sha256']!=sha(directory/f'FULL-{target:04d}.pt')
                or native['completed_steps']!=target or native['weights']!='raw' or native['ema_decay'] is not None
                or native['sha256']!=sha(directory/f'RAW-{target:04d}/native.safetensors')
                or [m['step'] for m in metrics]!=list(range(1,target+1)) or not finite(metrics)):
            raise ValueError('actual contiguous finite update/checkpoint receipts required')
        return {'completed_steps':target,'full_checkpoint_sha256':pin['sha256'],
            'raw_metadata_sha256':sha(directory/f'RAW-{target:04d}/native.json'),
            'updates_sha256':sha(directory/'updates.jsonl')}
    try:
        while True:
            pilot=read(args.pilot/'progress.json')
            if pilot['state']=='completed_one_update':break
            if pilot['state']=='error':raise RuntimeError('pilot failed; no full training or synthetic substitute')
            if not proc.exists() or proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]!=birth:
                raise RuntimeError('pilot exited without verified completion')
            time.sleep(5)
        initial=validate(args.pilot/'empf',1);cache=args.pilot/'cache'
        if pilot['cache_index_sha256']!=sha(cache/'index.json'):raise ValueError('admitted cache changed after pilot')
        latest={'empf':1,'rhbi_optimized':0}
        publish('waiting_for_shared_headroom',cache_index_sha256=sha(cache/'index.json'),pilot_checkpoint=initial,
            completed_steps_by_preset=latest,training_updates=1)
        env=dict(os.environ,PYTHONPATH=str(args.executor/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        finished=[]
        for preset in ['empf','rhbi_optimized']:
            directory=args.out/preset;previous=1 if preset=='empf' else 0
            resume=args.pilot/'empf/FULL-0001.pt' if previous else None
            if previous:
                directory.mkdir();(directory/'updates.jsonl').write_bytes((args.pilot/'empf/updates.jsonl').read_bytes())
            for target in [256,512,1024]:
                while not headroom():time.sleep(10)
                if {k:sha(args.executor/k) for k in sources}!=sources or sha(config)!=declaration['config_sha256']:
                    raise ValueError('frozen native sources/config changed')
                if sha(cache/'index.json')!=pilot['cache_index_sha256']:raise ValueError('shared cache changed')
                command=[str(args.executor/'.train-venv/bin/python'),'scripts/train.py','--config',str(config),
                    '--cache',str(cache),'--preset',preset,'--device',args.device,'--memory-fraction','.50',
                    '--chunk-updates',str(target-previous),'--out',str(directory)]
                if resume:command.extend(['--resume',str(resume)])
                if preset=='rhbi_optimized':command.append('--gpu-native-polar')
                publish('training',preset=preset,target_step=target,finished=finished)
                log=args.out/'logs'/f'{preset}-to-{target:04d}.log'
                with log.open('x') as stream:
                    result=subprocess.run(command,cwd=args.executor,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'actual training exit{result.returncode}; retain {log}')
                receipt=validate(directory,target);finished.append({'preset':preset,**receipt})
                previous=target;resume=directory/f'FULL-{target:04d}.pt'
                latest[preset]=target
                publish('checkpoint_verified',finished=finished,completed_steps_by_preset=latest,
                    training_updates=sum(latest.values()))
        publish('completed_both_recipes',finished=finished,training_updates=2048)
    except Exception as exc:
        publish('error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
