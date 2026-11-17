"""Continue the existing EMPF trajectory after its in-flight step-256 chunk."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from gemma_particle.conversion_handoff import finite, read
from gemma_particle.harness_identity import sha256_file


CHECKPOINTS = (256, 512, 1024)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_snapshot(binding):
    root = Path('/proc') / str(binding['pid'])
    fields = (root / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != binding['birth_ticks']:
        raise ValueError('process birth identity changed')
    return {'state': fields[0], 'parent': int(fields[1]), 'exit_status': int(fields[49])}


def check_handoff(hold):
    parent = process_snapshot(hold['controller'])
    child = process_snapshot(hold['empf_child'])
    if parent['state'] != 'T' or child['parent'] != hold['controller']['pid'] or child['state'] in ('T', 't'):
        raise ValueError('the stopped two-recipe supervisor must still own the exact EMPF worker')
    if child['state'] == 'Z' and child['exit_status'] != 0:
        raise RuntimeError(f"existing EMPF chunk failed with wait status {child['exit_status']}")
    return child['state'] == 'Z'


def checkpoint_receipt(directory, step):
    name = f'{step:04d}'
    full = directory / f'FULL-{name}.pt'
    pin = read(full.with_suffix('.json'))
    raw = directory / f'RAW-{name}'
    metadata = read(raw / 'native.json')
    journal = directory / 'updates.jsonl'
    metrics = [json.loads(line) for line in journal.read_text().splitlines()]
    recipe = metadata['game_signature']['recipe']
    if (pin['steps'] != step or pin['sha256'] != sha256_file(full)
            or metadata['schema'] != 'gemma4-native-slider-v1' or metadata['completed_steps'] != step
            or metadata['weights'] != 'raw' or metadata['ema_decay'] is not None
            or metadata['sha256'] != sha256_file(raw / 'native.safetensors')
            or (recipe['name'], recipe['total_steps'], recipe['batch_size']) != ('empf', 1024, 2)
            or [row['step'] for row in metrics] != list(range(1, step + 1)) or not finite(metrics)
            or any(type(row.get(key)) not in (int, float) for row in metrics for key in ('g_loss', 'd_loss'))):
        raise ValueError('verified exact EMPF checkpoint and contiguous metrics required')
    return {'preset': 'empf', 'completed_steps': step, 'full_checkpoint_sha256': pin['sha256'],
            'raw_metadata_sha256': sha256_file(raw / 'native.json'), 'updates_sha256': sha256_file(journal)}


def adopt_first_chunk(source, target):
    receipt = checkpoint_receipt(source, CHECKPOINTS[0])
    target.mkdir()
    for name in ('updates.jsonl', 'FULL-0256.pt', 'FULL-0256.json'):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / 'RAW-0256', target / 'RAW-0256')
    if checkpoint_receipt(target, CHECKPOINTS[0]) != receipt:
        raise ValueError('adopted checkpoint bytes differ')
    return receipt


def shared_headroom(device, qualification):
    command = ['nvidia-smi', f'--id={device[-1]}', '--query-gpu=memory.free,memory.total',
               '--format=csv,noheader,nounits']
    free, total = (int(value) * 1024**2 for value in subprocess.check_output(command, text=True).split(','))
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    return free >= .5 * total + 8 * 1024**3 and available >= qualification['peak_host_rss_bytes'] + 8 * 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.hold, args.out = args.hold.resolve(), args.out.resolve()
    if args.out.exists():
        raise FileExistsError('immutable EMPF run directory must be new')

    hold = read(args.hold)
    prior = Path(hold['controller']['out'])
    original = read(prior / 'plan.json')
    progress = read(prior / 'progress.json')
    executor = Path(hold['empf_child']['cwd'])
    device = original['device']
    if (hold['empf_child']['preset'] != 'empf' or progress['preset'] != 'empf'
            or progress['target_step'] != CHECKPOINTS[0] or hold['empf_child']['target_step'] != CHECKPOINTS[0]
            or device != hold['empf_child']['device'] or original['memory_fraction'] != .5
            or original['horizon'] != CHECKPOINTS[-1] or original['checkpoints'] != list(CHECKPOINTS)
            or (prior / 'rhbi_optimized').exists()):
        raise ValueError('expected the original admitted-data EMPF checkpoint chunk')
    check_handoff(hold)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=executor, text=True).strip()
    if commit != original['executor_commit']:
        raise ValueError('native executor commit changed')
    config = executor / 'configs/31b-ct-shared-positional-pilot.json'
    qualification_path = executor / 'reports/long-history-gradient-qualification-v1.json'
    qualification = read(qualification_path)
    if qualification['status'] != 'passed' or qualification['memory_fraction'] != .5:
        raise ValueError('expected qualified shared native configuration')
    pilot = Path(original['pilot'])
    cache = pilot / 'cache'
    cache_sha = progress['cache_index_sha256']
    pilot_status = read(pilot / 'progress.json')
    if pilot_status['state'] != 'completed_one_update' or pilot_status['cache_index_sha256'] != cache_sha:
        raise ValueError('original complete pilot and cache binding required')
    declaration = {**original, 'presets': ['empf'], 'optimizer_execution': {'empf': 'unchanged recipe'},
                   'scope': 'EMPF-only native continuation, original cache and exact checkpoint resume',
                   'source_sha256': sha256_file(__file__), 'source_hold': str(args.hold),
                   'source_hold_sha256': sha256_file(args.hold), 'source_training': str(prior),
                   'source_training_plan_sha256': sha256_file(prior / 'plan.json'),
                   'cache_index_sha256': cache_sha, 'qualification_sha256': sha256_file(qualification_path)}

    def check_inputs():
        if (sha256_file(args.hold) != declaration['source_hold_sha256']
                or sha256_file(prior / 'plan.json') != declaration['source_training_plan_sha256']
                or sha256_file(qualification_path) != declaration['qualification_sha256']
                or sha256_file(config) != original['config_sha256'] or sha256_file(cache / 'index.json') != cache_sha
                or {name: sha256_file(executor / name) for name in original['sources']} != original['sources']):
            raise ValueError('pinned EMPF handoff, native sources or cache changed')
    check_inputs()
    args.out.mkdir(parents=True)
    (args.out / 'logs').mkdir()
    write(args.out / 'plan.json', declaration)
    status = {'cache_index_sha256': cache_sha, 'pilot_checkpoint': progress['pilot_checkpoint'], 'finished': [],
              'training_updates': 1, 'completed_steps_by_preset': {'empf': 1}, 'preset': 'empf', 'target_step': CHECKPOINTS[0]}

    def publish(state, **values):
        status.update(state=state, as_of=datetime.now(timezone.utc).isoformat(), **values)
        write(args.out / 'progress.json', status)
        print(json.dumps(status, allow_nan=False), flush=True)

    publish('prepared' if args.plan_only else 'waiting_adopted_empf_checkpoint')
    if args.plan_only:
        return
    try:
        while not check_handoff(hold):
            time.sleep(5)
        check_inputs()
        if (prior / 'empf/updates.jsonl').read_bytes().splitlines()[0] != (pilot / 'empf/updates.jsonl').read_bytes().splitlines()[0]:
            raise ValueError('adopted trajectory differs from the original pilot update')
        own = args.out / 'empf'
        first = adopt_first_chunk(prior / 'empf', own)
        write(args.out / 'adopted-chunk.json', {'original_child': hold['empf_child'], 'observed_exit_status': 0,
                                               'original_output': str(prior / 'empf'), 'verified_checkpoint': first})
        finished = [first]
        previous = CHECKPOINTS[0]
        publish('checkpoint_verified', finished=finished, completed_steps_by_preset={'empf': previous}, training_updates=previous)
        env = dict(os.environ, PYTHONPATH=str(executor / 'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        for target in CHECKPOINTS[1:]:
            publish('waiting_for_shared_headroom', target_step=target)
            while not shared_headroom(device, qualification):
                time.sleep(10)
            check_inputs()
            if checkpoint_receipt(own, previous) != finished[-1]:
                raise ValueError('exact EMPF resume checkpoint changed')
            command = [str(executor / '.train-venv/bin/python'), 'scripts/train.py', '--config', str(config),
                       '--cache', str(cache), '--preset', 'empf', '--device', device, '--memory-fraction', '.50',
                       '--chunk-updates', str(target - previous), '--out', str(own), '--resume', str(own / f'FULL-{previous:04d}.pt')]
            publish('training')
            with (args.out / 'logs' / f'empf-to-{target:04d}.log').open('x') as logfile:
                subprocess.run(command, check=True, cwd=executor, env=env, stdout=logfile, stderr=subprocess.STDOUT)
            finished.append(checkpoint_receipt(own, target))
            previous = target
            publish('checkpoint_verified', finished=finished, completed_steps_by_preset={'empf': previous}, training_updates=previous)
        publish('completed_empf_only')
    except Exception as error:
        publish('error', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
