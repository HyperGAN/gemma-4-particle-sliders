"""Reconstruct the selected candidate's completed experiment before publication."""
from pathlib import Path

from .confirmation import load_confirmation_access
from .confirmation_environments import load_confirmation_environments
from .confirmation_results import compare_confirmation_pair
from .conversion_handoff import read
from .harness_identity import sha256_file
from .kaggle_publication import audit_payload
from .learned_effect import prepare_effect_probes, audit_learned_effect
from .selected_candidate import selected_evaluated_archive


def audit_selected_publication(readiness_path, access, environments_path, pair, comparison_path, root, source_function):
    # Access fails before any calibration, confirmation, readiness or account work.
    scope = load_confirmation_access(access)
    environments = load_confirmation_environments(environments_path, scope)
    comparison = read(comparison_path)
    paths = [comparison['condition_results'][name]['path'] for name in scope.receipt['permitted_conditions']]
    if compare_confirmation_pair(paths, scope, pair, environments) != comparison:
        raise ValueError('actual complete unchanged paired confirmation audit required')
    artifact, build = selected_evaluated_archive(scope, comparison)
    readiness = read(readiness_path)
    if (readiness.get('schema') != 'gemma4-selected-candidate-ready-v1' or readiness.get('status') != 'prepared'
            or readiness.get('confirmation') != scope.identity
            or readiness.get('comparison_sha256') != sha256_file(comparison_path)
            or Path(readiness['evaluated_artifact']).resolve() != artifact.resolve()
            or readiness['artifact_sha256'] != build['sha256']
            or readiness.get('fixed_dev_choice_retained') is not True or readiness.get('new_candidate_selected') is not False
            or readiness.get('account_requests') is not False or readiness.get('model_inference') is not False
            or readiness.get('official_score') is not None):
        raise ValueError('original prepared DEV-selected and evaluated candidate readiness required')
    expected_sources = ['scripts/prepare_selected_candidate.py', 'scripts/prepare_kaggle_candidate.py',
        'src/gemma_particle/selected_candidate.py', 'src/gemma_particle/learned_effect.py',
        'src/gemma_particle/confirmation.py', 'src/gemma_particle/confirmation_results.py',
        'src/gemma_particle/confirmation_environments.py']
    sources = readiness['sources']
    if len(sources) != len(expected_sources): raise ValueError('complete original packaging source binding required')
    for name in expected_sources:
        matches = [(path, value) for path, value in sources.items() if path == name or path.endswith('/' + name)]
        if len(matches) != 1 or sha256_file(matches[0][0]) != matches[0][1] or sha256_file(Path(root) / name) != matches[0][1]:
            raise ValueError('unchanged original selected-candidate packaging implementation required')
    access_plan_path = Path(scope.receipt['access_plan']); access_plan = read(access_plan_path)
    grid = Path(access_plan['dev_grid']); declaration = read(grid / 'plan.json'); packet = scope.receipt['selected_packet']
    name = packet['condition']['id']; pins = declaration['declaration_pins']
    def one_pin(filename):
        matches = [Path(path) for path in pins if Path(path).name == filename]
        if len(matches) != 1 or sha256_file(matches[0]) != pins[str(matches[0])]:
            raise ValueError('original unique selected effect protocol input required')
        return matches[0]
    protocol = one_pin('learned-effect-protocol-v1.json'); contexts = one_pin('contexts.json')
    qualification = one_pin('conversion-probe-preparation-v1.json')
    _, cases = prepare_effect_probes(protocol, contexts, qualification, scope.plan, scope.identity['selection_plan_sha256'])
    finished = [r for r in read(grid / 'progress.json')['finished'] if r['condition_id'] == name]
    effect = grid / (name + '-effect.json'); journal = grid / (name + '-preflight-http.jsonl')
    if (len(finished) != 1 or sha256_file(effect) != finished[0]['learned_effect_sha256']
            or readiness['learned_effect_sha256'] != sha256_file(effect)):
        raise ValueError('original selected measured learned-effect observation required')
    audit_learned_effect(effect, journal, read(grid / (name + '-server/server-identity.json')), packet, cases, sha256_file(protocol))
    payload = Path(readiness['payload'])
    if sha256_file(payload / 'preparation.json') != readiness['payload_preparation_sha256']:
        raise ValueError('original prepared publication payload required')
    candidate = audit_payload(payload, build['sha256'], source_function)
    preparation = read(payload / 'preparation.json')
    if (preparation['adapter_identity'] != packet['adapters']['particle_lora']
            or any(preparation[k] != packet[k] for k in ['native_metadata_sha256', 'cache_index_sha256', 'conversion_report_sha256'])
            or preparation['export_strength'] != packet['condition']['strength']
            or preparation['source_sha256'] != sha256_file(Path(root) / 'scripts/prepare_kaggle_candidate.py')):
        raise ValueError('publication payload must retain the originally evaluated selected adapter and provenance')
    inputs = [readiness_path, access, access_plan_path, scope.receipt['dev_choice'], environments_path, comparison_path, pair, protocol, contexts, qualification,
        effect, journal, artifact / 'build.json', grid / 'plan.json', grid / 'progress.json', *map(Path, sources)]
    candidate['experiment_pins'] = {str(Path(path).resolve()): sha256_file(path) for path in inputs}
    for path, value in access_plan.get('declaration_pins', {}).items():
        if sha256_file(path) != value: raise ValueError('released access declaration changed during publication audit')
        candidate['experiment_pins'][path] = value
    return candidate
