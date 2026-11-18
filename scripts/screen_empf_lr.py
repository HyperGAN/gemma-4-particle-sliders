"""Screen EMPF base learning rates on a disjoint TRAIN hidden-state probe bank."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx
import torch
from particlegan import get_recipe
from safetensors.torch import save_file

from continue_empf_only import checkpoint_receipt, shared_headroom, write
from gemma_particle.behavior_curation import clean_capture, load_behavior_cohort
from gemma_particle.curation import pair_contexts
from gemma_particle.data import digest, read_jsonl
from gemma_particle.model import ContextRow
from gemma_particle.prompting import template_options
from gemma_particle.runtime import file_hash, load_cache, load_native, model_identity, restore_native
from gemma_particle.transport import weight_identity


SOURCE = Path(__file__).resolve()
REPO = SOURCE.parents[1]
RATES = (0.012, 0.006, 0.024)
TRIAL_UPDATES = 8
MEMORY_FRACTION = 0.5
DEVICE = 'cuda:1'
PROJECTION_COUNT = 256
PROJECTION_SEED = 20260918


def read(path):
    return json.loads(Path(path).read_text())


def recipe(config, rate):
    value = get_recipe('empf', lr=rate, batch_size=config['batch_size'],
                       total_steps=config['updates'], num_particles=config['particles'],
                       z_dim=config['particle_width'], ema_decay=0.0)
    if value.reg_every != 1:
        raise ValueError('EMPF penalty must run every update (lazy_k=1)')
    return value


def pin(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in paths}


def projections_for(width):
    rng = torch.Generator().manual_seed(PROJECTION_SEED)
    directions = torch.randn(PROJECTION_COUNT, width, generator=rng, dtype=torch.float64)
    return directions / directions.norm(dim=1, keepdim=True)


def sliced_wasserstein(left, right, projections):
    """Average W1 between the equal-size empirical distributions on fixed unit directions."""
    if left.ndim != 2 or left.shape != right.shape or len(left) == 0 or left.shape[1] != projections.shape[1]:
        raise ValueError('equal-size hidden edit distributions and aligned projection width required')
    delta = (left @ projections.T).sort(dim=0).values - (right @ projections.T).sort(dim=0).values
    distances = delta.abs().mean(dim=0)
    return {'mean': float(distances.mean()), 'median': float(distances.median()),
            'p95': float(torch.quantile(distances, 0.95))}


def prepare(args):
    if args.root.exists():
        raise FileExistsError('screen declaration already exists')
    if not args.archive:
        raise ValueError('completed TRAIN collection archive required')
    config = read(args.config)
    identity = model_identity(config)
    tasks = read_jsonl(args.train_tasks)
    if not tasks or any(task['split'] != 'train' for task in tasks):
        raise ValueError('explicit sanitized TRAIN-only manifest required')
    train_ids = {task['instance_id'] for task in tasks}
    fit, _ = load_cache(args.cache, identity=identity, train_ids=train_ids)
    fit_ids = {row.row.instance_id for row in fit}
    curated = read(args.archive)
    protocol = REPO / 'research/jev-evidence-loop-v1.json'
    contexts = read_jsonl(curated['contexts'])
    if ({ctx['instance_id'] for ctx in contexts} != fit_ids
            or curated['admitted_contexts'] != len(fit_ids)):
        raise ValueError('declared prior collection differs from the original fit cache')
    server = read(args.server_identity)
    if weight_identity(server['model_identity']) != weight_identity(identity) or not server['base_only']:
        raise ValueError('tokenization must use the bound frozen Gemma base server')
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('localhost', '127.0.0.1', '::1')
            or endpoint.path not in ('', '/v1', '/v1/') or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment):
        raise ValueError('explicit local tokenizer endpoint required')
    server_hash = digest(server)
    by_id = {ctx['instance_id']: ctx for ctx in contexts}

    def tokenize(client, ctx, field):
        response = client.post('/tokenize', json={
            'model': server['served_model'], 'messages': ctx[field + '_messages'],
            'tools': ctx['tools'], 'chat_template_kwargs': template_options(ctx['generation'])})
        response.raise_for_status()
        ids = response.json()['tokens']
        if not ids or len(ids) > 32768 or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError('probe requires full nonempty native input tokens')
        return ids

    probes = []
    with httpx.Client(base_url=args.endpoint.rstrip('/').removesuffix('/v1'), timeout=60,
                      headers={'X-Gemma-Identity': server_hash}) as client:
        def check_server():
            response = client.get('/health')
            response.raise_for_status()
            if response.json().get('identity_sha256') != server_hash:
                raise ValueError('live tokenizer identity differs')

        check_server()
        for row in fit:
            ctx = by_id[row.row.instance_id]
            if (tuple(tokenize(client, ctx, 'neutral')) != row.row.neutral_ids
                    or tuple(tokenize(client, ctx, 'positive')) != row.row.positive_ids):
                raise ValueError('tokenizer does not reproduce the exact native cache tokens')
        for old in curated['finished']:
            task_id = old['instance_id']
            if task_id in fit_ids:
                continue
            cohort = Path(old['cohort'])
            _, _, rows, outcomes = load_behavior_cohort(cohort, args.train_tasks, protocol)
            if [row['instance_id'] for row in rows] != [task_id]:
                raise ValueError('probe archive contains an unexpected task')
            arms = {r['arm']: r for r in outcomes}
            if not (clean_capture(arms['off']) and clean_capture(arms['verify'])):
                continue
            ctx, = pair_contexts(arms['verify'], arms['off'], initial_only=True)
            probes.append({'instance_id': task_id, 'decision': 0,
                           'neutral_ids': tokenize(client, ctx, 'neutral'),
                           'positive_ids': tokenize(client, ctx, 'positive'),
                           'archive_outcomes_sha256': file_hash(cohort / 'outcomes.jsonl')})
        check_server()
    if len({p['instance_id'] for p in probes}) != len(probes) or len(probes) < 2:
        raise ValueError('distinct disjoint paired TRAIN probes required')
    settings = []
    for rate in RATES:
        configured = recipe(config, rate)
        settings.append({'lr': rate, 'critic_lr': rate * configured.d_lr_mult,
                         'particle_bank_lr': rate * configured.prior_lr_mult})
    plan = {
        'schema': 'gemma4-empf-eight-update-lr-screen-v1',
        'config': str(args.config.resolve()), 'cache': str(args.cache.resolve()),
        'train_tasks': str(args.train_tasks.resolve()), 'fit_task_ids': sorted(fit_ids),
        'model_identity': identity, 'tokenizer_identity_sha256': server_hash,
        'candidates': settings, 'target_update': TRIAL_UPDATES,
        'schedule_horizon': config['updates'], 'lazy_k': 1, 'device': DEVICE,
        'memory_fraction': MEMORY_FRACTION, 'probes': probes,
        'metric': {'name': 'teacher_edit_sliced_wasserstein_1', 'direction': 'minimize',
                   'strength': 1.0, 'projections': PROJECTION_COUNT, 'projection_seed': PROJECTION_SEED,
                   'normalization': 'one fixed RMS magnitude from all coordinates of the two training teacher edits'},
        'pins': pin([args.config, args.train_tasks, args.cache / 'index.json',
                     args.cache / 'responses.safetensors', args.archive, args.server_identity, SOURCE,
                     REPO / 'scripts/train.py', REPO / 'src/gemma_particle/model.py',
                     REPO / 'src/gemma_particle/runtime.py']),
    }
    args.root.mkdir(parents=True)
    write(args.root / 'plan.json', plan)
    write(args.root / 'progress.json', {'state': 'prepared', 'gpu_used': False})
    print(json.dumps({'trials': settings, 'probes': len(probes), 'gpu_used': False}), flush=True)


def load_plan(root):
    plan = read(root / 'plan.json')
    if (plan['schema'] != 'gemma4-empf-eight-update-lr-screen-v1'
            or [row['lr'] for row in plan['candidates']] != list(RATES)
            or plan['target_update'] != TRIAL_UPDATES or plan['lazy_k'] != 1
            or plan['device'] != DEVICE or plan['memory_fraction'] != MEMORY_FRACTION
            or pin(plan['pins']) != plan['pins']):
        raise ValueError('locked learning-rate screen changed')
    return plan


def trial_dir(root, rate):
    return root / f'lr-{rate:g}'


def verified_trial(root, config, rate):
    directory = trial_dir(root, rate)
    receipt = checkpoint_receipt(directory, TRIAL_UPDATES)
    actual = read(directory / 'RAW-0008/native.json')['game_signature']['recipe']
    expected = json.loads(json.dumps(asdict(recipe(config, rate))))
    if actual != expected:
        raise ValueError('actual eight-update recipe differs from the declared default EMPF rate screen')
    return receipt


def wait_headroom():
    qualification = read(REPO / 'reports/long-history-gradient-qualification-v1.json')
    while not shared_headroom(DEVICE, qualification):
        time.sleep(15)


def score(root, plan, config):
    probes = [ContextRow(tuple(p['neutral_ids']), tuple(p['positive_ids']), p['instance_id'], p['decision'])
              for p in plan['probes']]
    fit, _ = load_cache(plan['cache'], identity=plan['model_identity'], train_ids=set(plan['fit_task_ids']))
    if set(plan['fit_task_ids']) & {p.instance_id for p in probes}:
        raise ValueError('probe overlaps the exact training cache')
    fit_edits = torch.cat([row.target.double() - row.neutral.double() for row in fit])
    scale = fit_edits.square().mean().sqrt().reshape(1)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('training teacher edit must have finite positive RMS')
    wait_headroom()
    native = load_native(config, device=DEVICE, memory_fraction=MEMORY_FRACTION, checkpointing=False)
    try:
        with torch.inference_mode():
            rows = [native.prepare(p) for p in probes]
            neutral = torch.cat([row.neutral for row in rows]).cpu().double()
            teacher = torch.cat([row.target for row in rows]).cpu().double()
            projected_teacher = (teacher - neutral) / scale
            projections = projections_for(teacher.shape[1])
            scores = [{'rate': None, 'label': 'neutral', 'distance': sliced_wasserstein(
                torch.zeros_like(projected_teacher), projected_teacher, projections)}]
            outputs = {'neutral': neutral, 'teacher': teacher, 'scale': scale, 'projections': projections}
            for rate in RATES:
                verified_trial(root, config, rate)
                restore_native(native, trial_dir(root, rate) / 'RAW-0008')
                prediction = torch.cat([native.generated(row).cpu().double() for row in rows])
                distance = sliced_wasserstein((prediction - neutral) / scale, projected_teacher, projections)
                if not all(math.isfinite(v) for v in distance.values()):
                    raise FloatingPointError('nonfinite distribution distance')
                outputs[f'lr-{rate:g}'] = prediction
                scores.append({'rate': rate, 'distance': distance})
    finally:
        native.close()
    minimum = min(row['distance']['mean'] for row in scores if row['rate'] is not None)
    winners = [row['rate'] for row in scores if row['rate'] is not None and row['distance']['mean'] == minimum]
    vectors = root / 'probe-responses.safetensors'
    save_file({name: value.contiguous() for name, value in outputs.items()}, vectors)
    return {'scores': scores, 'best_rates': winners, 'probe_responses_sha256': file_hash(vectors)}


def run(args):
    root = args.root.resolve()
    plan = load_plan(root)
    config = read(plan['config'])
    if (root / 'result.json').exists():
        raise FileExistsError('completed screen is immutable')
    receipts = []
    for rate in RATES:
        directory = trial_dir(root, rate)
        if not directory.exists():
            write(root / 'progress.json', {'state': 'waiting_for_shared_headroom', 'next_rate': rate})
            wait_headroom()
            load_plan(root)
            write(root / 'progress.json', {'state': 'training', 'rate': rate, 'completed_trials': receipts})
            cmd = [sys.executable, str(REPO / 'scripts/train.py'), '--config', plan['config'],
                   '--cache', plan['cache'], '--train-tasks', plan['train_tasks'], '--preset', 'empf',
                   '--device', DEVICE, '--memory-fraction', str(MEMORY_FRACTION),
                   '--chunk-updates', str(TRIAL_UPDATES), '--lr', str(rate), '--out', str(directory)]
            with (root / f'lr-{rate:g}.log').open('x') as log:
                subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
        receipts.append({'rate': rate, **verified_trial(root, config, rate)})
    write(root / 'progress.json', {'state': 'scoring', 'completed_trials': receipts})
    result = {**score(root, plan, config), 'trials': receipts, 'plan_sha256': file_hash(root / 'plan.json')}
    write(root / 'result.json', result)
    write(root / 'progress.json', {'state': 'complete', 'best_rates': result['best_rates']})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/31b-ct-shared-positional-pilot.json')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--train-tasks', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--server-identity', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not all((args.archive, args.cache, args.train_tasks, args.endpoint, args.server_identity)):
            parser.error('prepare requires --archive, --cache, --train-tasks, --endpoint, and --server-identity')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
