"""One real EMPF update only after two independently revalidated initial targets.

This supervisor uses no model/GPU itself. The fixed executor owns the native
cache and public recipe. Missing data, failed provenance or occupied resources
cannot be replaced by synthetic targets or by stopping another worker's job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--pool-pid', type=int, required=True)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--executor-commit', required=True)
    p.add_argument('--device', choices=['cuda:0','cuda:1'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.pool, args.executor, args.out = args.pool.resolve(), args.executor.resolve(), args.out.resolve()
    if args.out.exists():
        raise FileExistsError('pilot handoff is immutable; incomplete runs require reconciliation')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.executor, text=True).strip()
    if commit != args.executor_commit:
        raise ValueError('frozen native executor differs')
    subprocess.run(['git','diff','--quiet','HEAD'], cwd=args.executor, check=True)
    qualification = read(args.executor/'reports/long-history-gradient-qualification-v1.json')
    if qualification['status']!='passed' or qualification['memory_fraction']!=.5 or qualification['training_updates']!=0:
        raise ValueError('full-history half-GPU gradient qualification required')
    for name, expected in qualification['execution_sources'].items():
        relative = 'scripts/profile_native.py' if Path(name).is_absolute() else name
        if sha(args.executor/relative)!=expected:
            raise ValueError('qualified native/config sources changed')
    pool_plan = read(args.pool/'plan.json')
    if (pool_plan.get('prompt_profile')!='repair-finalize-v2'
            or pool_plan.get('behavior_study')!='jev-evidence-loop-v1'
            or pool_plan.get('curator_commit')!=commit
            or pool_plan['server_identity']['device']!=args.device):
        raise ValueError('explicit finalizing initial-boundary cohort and matching GPU required')
    proc = Path(f'/proc/{args.pool_pid}')
    command = (proc/'cmdline').read_bytes().split(b'\0') if proc.exists() else []
    if not any(a.endswith(b'/collect_behavior_pool.py') for a in command):
        raise ValueError('exact live behavior-pool supervisor required')
    if b'--root' not in command:
        raise ValueError('pool process root argument required')
    process_root = Path(os.fsdecode(command[command.index(b'--root')+1]))
    if not process_root.is_absolute():
        process_root = (proc/'cwd').resolve()/process_root
    if process_root.resolve()!=args.pool:
        raise ValueError('pool PID belongs to another collection')
    birth = (proc/'stat').read_text().split(') ',1)[1].split()[19]
    args.out.mkdir(parents=True, mode=0o700); (args.out/'logs').mkdir()
    status = {'schema':'gemma4-admitted-pilot-v1', 'state':'waiting_for_targets',
        'created_at':datetime.now(timezone.utc).isoformat(), 'pool_plan_sha256':sha(args.pool/'plan.json'),
        'pool_pid':args.pool_pid, 'pool_birth_ticks':birth, 'executor_commit':commit,
        'source_sha256':sha(Path(__file__)), 'preset':'empf', 'chunk_updates':1,
        'declared_horizon':1024, 'memory_fraction':.5, 'training_updates':0,
        'scope':'first admitted-data engineering update; no DEV/holdout or official quality result'}
    write(args.out/'progress.json', status)
    env = dict(os.environ, PYTHONPATH=str(args.executor/'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(command, log, cpu=False):
        child_env = dict(env)
        if cpu:
            child_env['CUDA_VISIBLE_DEVICES']=''
        else:
            child_env.pop('CUDA_VISIBLE_DEVICES', None)
        python = args.executor/('.harness-venv/bin/python' if cpu else '.train-venv/bin/python')
        with log.open('x') as stream:
            result = subprocess.run([str(python),*command], cwd=args.executor, env=child_env,
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'pipeline exit {result.returncode}; preserve {log}')
    try:
        while True:
            progress = args.pool/'progress.json'
            result = read(progress) if progress.exists() else {}
            if result.get('state')=='target_ready':
                break
            if result.get('state')=='exhausted_without_targets':
                raise RuntimeError('TRAIN pool exhausted before two qualified initial targets')
            if not proc.exists() or (proc/'stat').read_text().split(') ',1)[1].split()[19]!=birth:
                raise RuntimeError('pool exited without targets; no restart or synthetic substitution')
            time.sleep(5)
        rows = []
        for record in result['finished']:
            if not record['admitted_contexts']:
                continue
            destination = args.out/'revalidated'/record['instance_id']
            execute(['scripts/curate_behavior_cohort.py','--cohort',record['cohort'],
                '--grades',record['grades'],'--out',str(destination)],
                args.out/'logs'/(record['instance_id']+'-revalidate.log'), cpu=True)
            if sha(destination/'jev-contexts.jsonl')!=record['contexts_sha256']:
                raise ValueError('admitted context differs from fresh offline evidence validation')
            rows.extend(json.loads(line) for line in (destination/'jev-contexts.jsonl').read_text().splitlines() if line.strip())
        contexts = Path(result['contexts']); transport = Path(result['prompt_transport'])
        if (len(rows)<2 or len({r['instance_id'] for r in rows})!=len(rows)
                or rows!=[json.loads(line) for line in contexts.read_text().splitlines() if line.strip()]
                or any(r['decision']!=0 or r.get('boundary')!='initial-assistant-decision' for r in rows)
                or sha(transport)!=result['prompt_transport_sha256']):
            raise ValueError('two distinct intact initial targets and exact prompt transport required')
        status.update(state='waiting_for_shared_headroom', contexts_sha256=sha(contexts),
            distinct_train_tasks=[r['instance_id'] for r in rows], prompt_transport_sha256=sha(transport))
        write(args.out/'progress.json', status)
        def headroom():
            output = subprocess.check_output(['nvidia-smi',f'--id={args.device[-1]}',
                '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'], text=True)
            free, total = (int(v.strip())*1024**2 for v in output.strip().split(','))
            available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            return free>=total*.5+8*1024**3 and available>=qualification['peak_host_rss_bytes']+8*1024**3
        while not headroom():
            time.sleep(10)
        config = args.executor/'configs/31b-ct-shared-positional-pilot.json'
        cache = args.out/'cache'
        status['state']='caching_actual_targets'; write(args.out/'progress.json', status)
        execute(['scripts/cache_contexts.py','--config',str(config),'--contexts',str(contexts),
            '--prompt-transport',str(transport),'--device',args.device,'--memory-fraction','.50','--out',str(cache)],
            args.out/'logs'/'cache.log')
        while not headroom():
            time.sleep(10)
        training = args.out/'empf'
        status['state']='one_public_recipe_update'; write(args.out/'progress.json', status)
        execute(['scripts/train.py','--config',str(config),'--cache',str(cache),'--preset','empf',
            '--device',args.device,'--memory-fraction','.50','--chunk-updates','1','--out',str(training)],
            args.out/'logs'/'train.log')
        metrics = [json.loads(line) for line in (training/'updates.jsonl').read_text().splitlines()]
        pin = read(training/'FULL-0001.json'); native = read(training/'RAW-0001/native.json')
        if (len(metrics)!=1 or not finite(metrics) or pin['steps']!=1
                or pin['sha256']!=sha(training/'FULL-0001.pt') or native['completed_steps']!=1
                or native['sha256']!=sha(training/'RAW-0001/native.safetensors')):
            raise ValueError('first-update metrics/checkpoint receipt failed')
        status.update(state='completed_one_update', training_updates=1, metrics=metrics,
            cache_index_sha256=sha(cache/'index.json'), full_checkpoint_sha256=pin['sha256'],
            raw_native_metadata_sha256=sha(training/'RAW-0001/native.json'))
        write(args.out/'progress.json', status); print(json.dumps(status), flush=True)
    except Exception as exc:
        status.update(state='error', error_type=type(exc).__name__, error=str(exc))
        write(args.out/'progress.json', status)
        raise


if __name__=='__main__':
    main()
