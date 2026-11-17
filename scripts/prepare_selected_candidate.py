"""Prepare the fixed DEV-selected/evaluated candidate; no upload or submission."""
import argparse
import json
from pathlib import Path
import subprocess

from gemma_particle.confirmation import load_confirmation_access
from gemma_particle.confirmation_environments import load_confirmation_environments
from gemma_particle.confirmation_results import compare_confirmation_pair
from gemma_particle.conversion_handoff import read
from gemma_particle.harness_identity import sha256_file
from gemma_particle.learned_effect import prepare_effect_probes, audit_learned_effect
from gemma_particle.selected_candidate import selected_evaluated_archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['access', 'environments', 'pair', 'comparison', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--owner', required=True)
    p.add_argument('--notebook-slug', required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root: raise ValueError('run inside the frozen packaging checkout')
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    if args.out.exists(): raise FileExistsError('selected candidate preparation is immutable')
    sources = {str(root / name): sha256_file(root / name) for name in [
        'scripts/prepare_selected_candidate.py', 'scripts/prepare_kaggle_candidate.py',
        'src/gemma_particle/selected_candidate.py', 'src/gemma_particle/learned_effect.py',
        'src/gemma_particle/confirmation.py', 'src/gemma_particle/confirmation_results.py',
        'src/gemma_particle/confirmation_environments.py']}
    if args.plan_only:
        args.out.mkdir(parents=True)
        (args.out / 'plan.json').write_text(json.dumps({'schema': 'gemma4-selected-candidate-plan-v1',
            'state': 'prepared', 'sources': sources, 'access_or_confirmation_contents_parsed': False,
            'model_inference': False, 'account_requests': False,
            'scope': 'require complete original choice,pair,effect and byte-identical evaluated archive;no new selection'}, indent=2) + '\n')
        print(json.dumps({'state': 'prepared', 'protected_inputs_opened': False, 'account_requests': False})); return
    # An unreleased access receipt fails before calibration, pair or outcome reads.
    scope = load_confirmation_access(args.access)
    environments = load_confirmation_environments(args.environments, scope)
    comparison = read(args.comparison)
    paths = [comparison['condition_results'][name]['path'] for name in scope.receipt['permitted_conditions']]
    if compare_confirmation_pair(paths, scope, args.pair, environments) != comparison:
        raise ValueError('unchanged complete paired confirmation audit required')
    access_plan = read(scope.receipt['access_plan']); grid = Path(access_plan['dev_grid'])
    declaration = read(grid / 'plan.json'); packet = scope.receipt['selected_packet']; name = packet['condition']['id']
    pins = declaration['declaration_pins']
    protocol_path = next(Path(path) for path in pins if Path(path).name == 'learned-effect-protocol-v1.json')
    contexts = next(Path(path) for path in pins if Path(path).name == 'contexts.json')
    qualification = next(Path(path) for path in pins if Path(path).name == 'conversion-probe-preparation-v1.json')
    if any(sha256_file(path) != value for path, value in pins.items()):
        raise ValueError('original learned dispatch declaration inputs changed')
    _, cases = prepare_effect_probes(protocol_path, contexts, qualification, scope.plan, scope.identity['selection_plan_sha256'])
    progress = read(grid / 'progress.json'); record = next(r for r in progress['finished'] if r['condition_id'] == name)
    effect_path = grid / (name + '-effect.json'); journal = grid / (name + '-preflight-http.jsonl')
    if sha256_file(effect_path) != record['learned_effect_sha256']:
        raise ValueError('original selected learned-effect observation changed')
    server_path = grid / (name + '-server/server-identity.json')
    audit_learned_effect(effect_path, journal, read(server_path), packet, cases, sha256_file(protocol_path))
    artifact, receipt = selected_evaluated_archive(scope, comparison)
    if any(sha256_file(path) != value for path, value in sources.items()):
        raise ValueError('fixed packaging source changed')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    args.out.mkdir(parents=True)
    command = [str(root / '.harness-venv/bin/python'), str(root / 'scripts/prepare_kaggle_candidate.py'),
        '--artifact', str(artifact), '--native', packet['native'], '--cache', packet['cache'],
        '--conversion-report', packet['conversion_report'], '--owner', args.owner,
        '--notebook-slug', args.notebook_slug, '--out', str(args.out / 'payload')]
    with (args.out / 'prepare.log').open('x') as log:
        child = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if child.returncode: raise RuntimeError('actual payload preparation failed;preserve log')
    payload = read(args.out / 'payload/preparation.json')
    if payload['artifact_sha256'] != receipt['sha256']:
        raise ValueError('payload differs from evaluated archive')
    result = {'schema': 'gemma4-selected-candidate-ready-v1', 'status': 'prepared',
        'confirmation': scope.identity, 'comparison_sha256': sha256_file(args.comparison),
        'learned_effect_sha256': sha256_file(effect_path), 'evaluated_artifact': str(artifact.resolve()),
        'artifact_sha256': receipt['sha256'], 'sources': sources, 'payload': str((args.out / 'payload').resolve()),
        'payload_preparation_sha256': sha256_file(args.out / 'payload/preparation.json'),
        'fixed_dev_choice_retained': True, 'new_candidate_selected': False, 'account_requests': False,
        'model_inference': False, 'official_score': None,
        'scope': 'local payload of exact evaluated DEV choice;confirmation outcomes reported without retuning;not an official score'}
    (args.out / 'readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'prepared', 'artifact_sha256': receipt['sha256'], 'account_requests': False}))


if __name__ == '__main__': main()
