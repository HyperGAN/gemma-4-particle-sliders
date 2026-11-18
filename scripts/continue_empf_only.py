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


def continue_screen_winner(args):
    import math
    from gemma_particle.dev_dispatch import live
    from screen_empf_lr import DEVICE, RATES, TRIAL_UPDATES, load_plan, trial_dir, verified_trial

    executor = Path(__file__).resolve().parents[1]
    screen, out = args.rate_screen.resolve(), args.out.resolve()
    config = executor / 'configs/31b-ct-shared-positional-pilot.json'
    qualification_path = executor / 'reports/long-history-gradient-qualification-v1.json'
    screen_plan_path = screen / 'plan.json'
    screen_result_path = screen / 'result.json'
    if sha256_file(screen_plan_path) != args.screen_plan_sha256 or out.exists():
        raise ValueError('exact predeclared rate screen and a fresh native output root required')
    screen_plan = load_plan(screen)
    settings, qualification = read(config), read(qualification_path)
    if (sha256_file(config) != sha256_file(screen_plan['config'])
            or settings['updates'] != CHECKPOINTS[-1] or qualification['status'] != 'passed'
            or qualification['memory_fraction'] != .5):
        raise ValueError('the screen and production must use the same qualified native config')
    for name in ('scripts/train.py', 'src/gemma_particle/model.py', 'src/gemma_particle/runtime.py'):
        matches = [digest for path, digest in screen_plan['pins'].items() if path.endswith('/' + name)]
        if matches != [sha256_file(executor / name)]:
            raise ValueError('production executor changes screened training math: ' + name)
    if not args.plan_only and not args.screen_pid:
        raise ValueError('exact owned screen supervisor PID required')
    binding = None
    if args.screen_pid:
        proc = Path('/proc') / str(args.screen_pid)
        if not proc.exists() and not screen_result_path.exists():
            raise ValueError('expected live screen or its complete result')
        if proc.exists():
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if os.fsencode(screen / 'cutover.py') not in argv or args.screen_plan_sha256.encode() not in argv:
                raise ValueError('screen supervisor process differs from the declared output')
            binding = {'pid': args.screen_pid,
                       'birth_ticks': proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[19]}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=executor, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=executor, check=True)
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    write(out / 'plan.json', {
        'schema': 'gemma4-screen-selected-empf-continuation-v1', 'presets': ['empf'],
        'executor_commit': commit, 'device': DEVICE, 'memory_fraction': .5,
        'config_sha256': sha256_file(config), 'horizon': CHECKPOINTS[-1], 'checkpoints': list(CHECKPOINTS),
        'screen': str(screen), 'screen_plan_sha256': args.screen_plan_sha256, 'screen_supervisor': binding,
        'screen_tie_policy': 'keep default if tied; otherwise lowest tied rate',
    })
    cache_sha = sha256_file(Path(screen_plan['cache']) / 'index.json')
    state = {'cache_index_sha256': cache_sha, 'finished': [], 'training_updates': 0,
             'completed_steps_by_preset': {'empf': 0}, 'preset': 'empf'}

    def publish(label, **updates):
        state.update(state=label, as_of=datetime.now(timezone.utc).isoformat(), **updates)
        write(out / 'progress.json', state)
        print(json.dumps(state, allow_nan=False), flush=True)

    publish('prepared' if args.plan_only else 'waiting_adopted_empf_checkpoint')
    if args.plan_only:
        return
    try:
        while not screen_result_path.exists() or (binding and live(binding)):
            if not (binding and live(binding)):
                raise RuntimeError('owned screen supervisor exited before completing the rate screen')
            time.sleep(10)
        if (read(screen / 'progress.json')['state'] != 'complete'
                or sha256_file(screen_plan_path) != args.screen_plan_sha256):
            raise ValueError('actual complete screen with the original declaration is required')
        load_plan(screen)
        result = read(screen_result_path)
        if (result['plan_sha256'] != args.screen_plan_sha256
                or result['probe_responses_sha256'] != sha256_file(screen / 'probe-responses.safetensors')
                or result['trials'] != [{'rate': rate, **verified_trial(screen, settings, rate)} for rate in RATES]):
            raise ValueError('actual rate screen outputs differ from their checkpoint or metric receipts')
        scores = {item['rate']: item['distance']['mean'] for item in result['scores']}
        if set(scores) != {None, *RATES} or not all(math.isfinite(v) and v >= 0 for v in scores.values()):
            raise ValueError('one finite distribution score per candidate and a neutral reference required')
        best = min(scores[rate] for rate in RATES)
        winners = [rate for rate in RATES if scores[rate] == best]
        if winners != result['best_rates']:
            raise ValueError('screen winner differs from the measured minimum distance')
        selected = RATES[0] if RATES[0] in winners else min(winners)
        source = trial_dir(screen, selected)
        own = out / 'empf'
        own.mkdir()
        for name in ('updates.jsonl', 'FULL-0008.pt', 'FULL-0008.json'):
            shutil.copy2(source / name, own / name)
        shutil.copytree(source / 'RAW-0008', own / 'RAW-0008')
        receipt = checkpoint_receipt(own, TRIAL_UPDATES)
        if receipt != verified_trial(screen, settings, selected):
            raise ValueError('adopted winner checkpoint or journal differs from its measured source')
        write(out / 'adopted-rate.json', {
            'selected_lr': selected, 'best_rates': winners, 'score': best,
            'screen_result_sha256': sha256_file(screen_result_path), 'checkpoint': receipt,
        })
        previous = TRIAL_UPDATES
        env = dict(os.environ, PYTHONPATH=str(executor / 'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        finished = []
        for target in CHECKPOINTS:
            publish('waiting_for_shared_headroom', selected_lr=selected, target_step=target,
                    training_updates=previous, completed_steps_by_preset={'empf': previous})
            while not shared_headroom(DEVICE, qualification):
                time.sleep(10)
            if (load_plan(screen) != screen_plan or checkpoint_receipt(own, previous) != receipt
                    or sha256_file(screen_plan_path) != args.screen_plan_sha256):
                raise ValueError('screen declaration or exact production checkpoint changed')
            command = [str(executor / '.train-venv/bin/python'), 'scripts/train.py', '--config', str(config),
                       '--cache', screen_plan['cache'], '--train-tasks', screen_plan['train_tasks'],
                       '--preset', 'empf', '--device', DEVICE, '--memory-fraction', '.50', '--lr', str(selected),
                       '--chunk-updates', str(target - previous), '--out', str(own),
                       '--resume', str(own / f'FULL-{previous:04d}.pt')]
            publish('training')
            with (out / 'logs' / f'empf-to-{target:04d}.log').open('x') as logfile:
                subprocess.run(command, check=True, cwd=executor, env=env, stdout=logfile, stderr=subprocess.STDOUT)
            receipt = checkpoint_receipt(own, target)
            if read(own / f'RAW-{target:04d}' / 'native.json')['game_signature']['recipe']['lr'] != selected:
                raise ValueError('actual native checkpoint does not use the selected base rate')
            finished.append(receipt)
            previous = target
            publish('checkpoint_verified', finished=finished, completed_steps_by_preset={'empf': previous},
                    training_updates=previous)
        publish('completed_empf_only')
    except Exception as error:
        publish('error', error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--hold', type=Path)
    source.add_argument('--rate-screen', type=Path)
    parser.add_argument('--screen-plan-sha256')
    parser.add_argument('--screen-pid', type=int)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.rate_screen:
        if not args.screen_plan_sha256:
            parser.error('--screen-plan-sha256 is required with --rate-screen')
        continue_screen_winner(args)
        return
    if args.screen_plan_sha256 or args.screen_pid:
        parser.error('screen options require --rate-screen')
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
