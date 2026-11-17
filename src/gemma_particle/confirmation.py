"""Release the fixed holdout only after an unchanged, reaudited DEV choice.

These are local evidence bindings, not authentication of invented receipts.
Issuing access performs no model inference or confirmation evaluation.
"""
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .conversion_handoff import read
from .data import VISIBLE_FIELDS, read_jsonl, digest
from .harness_identity import sha256_file
from .lora_serving import prepare_lora, BASE_REPO, BASE_REVISION


def validate_dev_choice(directory, plan, plan_sha256):
    directory = Path(directory)
    status = read(directory / 'progress.json')
    if status.get('state') != 'complete_dev_choice':
        raise ValueError('complete immutable DEV choice required before confirmation access')
    path = directory / 'choice/selection.json'
    choice = read(path)
    conditions = {c['id']: c for c in plan['conditions']}
    learned = [c['id'] for c in plan['conditions'] if c['role'] == 'learned']
    if (choice.get('schema') != 'gemma4-locked-dev-selection-v1'
            or status.get('selection_sha256') != sha256_file(path)
            or choice.get('selection_plan_sha256') != plan_sha256
            or choice.get('model_dev_conditions') != 22 or choice.get('holdout_accessed') is not False
            or set(choice['condition_results']) != set(conditions)
            or status.get('planned_conditions') != 18
            or [r['condition_id'] for r in status['finished']] != learned):
        raise ValueError('complete unchanged twenty-two-condition DEV audit required')
    selected = choice.get('selected')
    if (selected is None or selected not in conditions or conditions[selected]['role'] != 'learned'
            or choice.get('advance_to_locked_confirmation') is not True
            or selected not in choice.get('eligible', []) or status.get('selected') != selected):
        raise ValueError('eligible learned DEV choice required; holdout remains locked')
    for name, record in choice['condition_results'].items():
        result_path = Path(record['path'])
        result = read(result_path)
        if (sha256_file(result_path) != record['sha256'] or result.get('condition') != conditions[name]
                or result.get('selection_plan_sha256') != plan_sha256):
            raise ValueError('actual selected DEV evidence changed')
    for record in status['finished']:
        if record['result_sha256'] != choice['condition_results'][record['condition_id']]['sha256']:
            raise ValueError('full learned dispatcher and locked DEV evidence differ')
    return choice, path


def build_confirmation_access(directory, plan, plan_sha256, reaudit_path, manifest, packets):
    # Every choice and packet check precedes reading the holdout manifest.
    choice, choice_path = validate_dev_choice(directory, plan, plan_sha256)
    if sha256_file(reaudit_path) != sha256_file(choice_path):
        raise ValueError('unchanged exact DEV choice must survive the independent offline reaudit')
    packet_root = Path(packets)
    packet_path = packet_root / 'candidate-packets.json'
    matrix = read(packet_path)
    matrix_status = read(packet_root / 'progress.json')
    expected = [c for c in plan['conditions'] if c['role'] == 'learned']
    if (matrix.get('schema') != 'gemma4-measured-learned-packets-v1'
            or matrix.get('selection_plan_sha256') != plan_sha256
            or matrix_status.get('state') != 'prepared_complete_learned_grid'
            or matrix_status.get('candidate_packets_sha256') != sha256_file(packet_path)
            or [p['condition'] for p in matrix['packets']] != expected):
        raise ValueError('unchanged complete measured packet grid required')
    packet = next(p for p in matrix['packets'] if p['condition']['id'] == choice['selected'])
    selected = read(choice['condition_results'][choice['selected']]['path'])
    bound = selected['packet']
    spec = prepare_lora({'model': BASE_REPO, 'model_revision': BASE_REVISION},
        adapter=packet['adapter'], conversion_report=packet['conversion_report'])
    native_path = Path(packet['native']) / 'native.json'
    cache_path = Path(packet['cache']) / 'index.json'
    native, cache = read(native_path), read(cache_path)
    expected_hashes = {
        'native_metadata_sha256': sha256_file(native_path),
        'cache_index_sha256': sha256_file(cache_path),
        'conversion_report_sha256': sha256_file(packet['conversion_report']),
    }
    if (selected.get('status') != 'complete' or selected['condition'] != packet['condition']
            or spec['adapters'] != packet['adapters'] or bound['adapters'] != packet['adapters']
            or any(packet[k] != value or bound[k] != value for k, value in expected_hashes.items())
            or spec['provenance']['export_strength'] != packet['condition']['strength']
            or any(spec['provenance']['conversion_provenance'][k] != expected_hashes[k]
                for k in ['native_metadata_sha256', 'cache_index_sha256'])
            or Path(bound['native_metadata']).resolve() != native_path.resolve()
            or Path(bound['cache_index']).resolve() != cache_path.resolve()
            or native['sha256'] != sha256_file(Path(packet['native']) / 'native.safetensors')
            or cache['responses_sha256'] != sha256_file(Path(packet['cache']) / 'responses.safetensors')):
        raise ValueError('selected model bytes differ from the actual locked DEV candidate')
    if sha256_file(manifest) != plan['confirmation']['manifest_sha256']:
        raise ValueError('original locked holdout manifest changed')
    rows = read_jsonl(manifest)
    if (len(rows) != plan['confirmation']['count'] or len(rows) != 49
            or any(not isinstance(r, dict) or set(r) != set(VISIBLE_FIELDS) | {'split'}
                or r['split'] != 'holdout' for r in rows)
            or len({r['instance_id'] for r in rows}) != 49
            or Counter(r['repo'] for r in rows) != {'Textualize/rich': 48, 'encode/httpx': 1}):
        raise ValueError('complete original sanitized forty-nine-task confirmation required')
    return {'schema': 'gemma4-locked-confirmation-access-v1', 'selection_plan_sha256': plan_sha256,
        'dev_choice': str(choice_path.resolve()), 'dev_choice_sha256': sha256_file(choice_path),
        'reaudited_dev_choice': str(Path(reaudit_path).resolve()), 'reaudited_dev_choice_sha256': sha256_file(reaudit_path),
        'selected_condition': packet['condition'], 'selected_packet': packet,
        'packet_manifest': str(packet_path.resolve()), 'packet_manifest_sha256': sha256_file(packet_path),
        'holdout_manifest': str(Path(manifest).resolve()), 'holdout_manifest_sha256': sha256_file(manifest),
        'count': 49, 'repositories': {'Textualize/rich': 48, 'encode/httpx': 1},
        'permitted_conditions': ['off', choice['selected']], 'forbidden_training_and_selection': True,
        'model_inference': False, 'confirmation_results': None,
        'scope': 'access after immutable DEV selection; exactly one paired confirmation remains to be executed'}


@dataclass(frozen=True)
class LockedConfirmation:
    receipt: dict
    plan: dict
    rows: list
    identity: dict


def load_confirmation_access(path):
    from .dev_selection import load_selection_plan
    path = Path(path)
    access = read(path)
    if (access.get('schema') != 'gemma4-locked-confirmation-access-v1'
            or access.get('forbidden_training_and_selection') is not True
            or access.get('model_inference') is not False or access.get('confirmation_results') is not None
            or access.get('offline_dev_auditor_returncode') != 0):
        raise ValueError('completed explicit confirmation access receipt required')
    access_plan_path = Path(access['access_plan']); declaration = read(access_plan_path)
    selection_path = Path(access['selection_plan'])
    if (sha256_file(access_plan_path) != access['access_plan_sha256']
            or declaration.get('schema') != 'gemma4-confirmation-access-plan-v1'
            or declaration['selection_plan_sha256'] != access['selection_plan_sha256']
            or sha256_file(selection_path) != access['selection_plan_sha256']
            or declaration['dev_executor_commit'] != access['offline_dev_auditor_executor_commit']
            or any(sha256_file(p) != value for p, value in declaration['declaration_pins'].items())
            or sha256_file(access_plan_path.parent / 'dev-reaudit.log') != access['offline_dev_audit_log_sha256']):
        raise ValueError('original access plan, source-bound offline reaudit and DEV declarations changed')
    plan = load_selection_plan(selection_path)
    expected = build_confirmation_access(declaration['dev_grid'], plan, access['selection_plan_sha256'],
        access['reaudited_dev_choice'], access['holdout_manifest'], declaration['packet_producer'])
    if any(access.get(key) != value for key, value in expected.items()):
        raise ValueError('confirmation access differs from current locked choice, packet or manifest')
    rows = read_jsonl(access['holdout_manifest'])
    if sha256_file(access['holdout_manifest']) != access['holdout_manifest_sha256']:
        raise ValueError('locked manifest changed during access validation')
    official_source = read(plan['secondary']['qualification'])['source_sha256']
    identity = {'access_receipt_sha256': sha256_file(path), 'selection_plan_sha256': access['selection_plan_sha256'],
        'dev_choice_sha256': access['dev_choice_sha256'], 'holdout_manifest_sha256': access['holdout_manifest_sha256'],
        'selected_condition_id': access['selected_condition']['id'], 'official_tasks_sha256': official_source,
        'forbidden_training_and_selection': True}
    return LockedConfirmation(access, plan, rows, identity)


def validate_confirmation_treatment(scope, *, adapter, arm, model, prompt_profile, instruction_protocol,
        budget_profile, history_profile, capture_protocol, source_root):
    from .adapters import adapter_identity
    from .capture import CONTEXT_CAPTURE_PROTOCOL
    from .lora_serving import BASE_ALIAS
    from .prompt_profiles import prompt_identity
    matching = scope.plan['matching']; source_root = Path(source_root)
    profiles = {'prompt_profile': prompt_profile, 'instruction_protocol': instruction_protocol,
        'budget_profile': budget_profile, 'history_profile': history_profile}
    if (arm != 'off' or model != BASE_ALIAS or capture_protocol != CONTEXT_CAPTURE_PROTOCOL
            or any(value != matching[key] for key, value in profiles.items())
            or sha256_file(source_root / 'submission/configs/sampling.yaml') != matching['sampling_file_sha256']
            or prompt_identity((source_root / 'submission/prompts/system.md').read_text(), prompt_profile)['sha256']
                != matching['common_system_sha256']):
        raise ValueError('confirmation requires the exact locked OFF prompt, model and execution profiles')
    if adapter is None: return 'off'
    if {'particle_lora': adapter_identity(adapter)} != scope.receipt['selected_packet']['adapters']:
        raise ValueError('only the exact selected learned adapter can enter confirmation')
    return scope.receipt['selected_condition']['id']


def validate_confirmation_server(scope, server, *, learned):
    from .dev_evidence import canonical_contract
    selected = scope.receipt['selected_packet']['adapters']
    if (server.get('kind') != 'gemma4' or server.get('qualification_only') is not False
            or server.get('pretrained_inference') is not True
            or server['model_identity']['repo'] != scope.plan['model']
            or server['model_identity']['revision'] != scope.plan['model_revision']
            or learned and (server.get('adapters') != selected or server.get('base_only') is not False)
            or not learned and not (server.get('adapters') == selected and server.get('base_only') is False
                or server.get('adapters') == {} and server.get('base_only') is True)):
        raise ValueError('actual pinned base or exact selected learned confirmation server required')
    if sha256_file(scope.receipt['dev_choice']) != scope.identity['dev_choice_sha256']:
        raise ValueError('locked DEV choice changed before confirmation server validation')
    choice = read(scope.receipt['dev_choice'])
    result_path = Path(choice['condition_results']['off']['path'])
    off_result = read(result_path)
    original_server = read(read(result_path.parent / 'plan.json')['server_identity'])
    if (sha256_file(result_path) != choice['condition_results']['off']['sha256']
            or digest(original_server) != off_result['server_identity_sha256']
            or canonical_contract({'model_identity': original_server}) != canonical_contract({'model_identity': server})):
        raise ValueError('confirmation serving contract differs from matched DEV baseline')


def confirmation_claim_path(scope):
    manifest = Path(scope.receipt['holdout_manifest']).resolve()
    key = scope.identity['holdout_manifest_sha256']
    return manifest.parent / '.confirmation-use' / (key + '.json')


def claim_confirmation_pair(scope, out):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError('paired confirmation root must be fresh')
    path = confirmation_claim_path(scope); path.parent.mkdir(parents=True, exist_ok=True)
    value = {'schema': 'gemma4-confirmation-pair-claim-v1', 'identity': scope.identity,
        'pair_root': str(out), 'permitted_conditions': scope.receipt['permitted_conditions'],
        'count_per_condition': 49, 'owner_pid': os.getpid()}
    with path.open('x') as stream: stream.write(json.dumps(value, indent=2) + '\n')
    out.mkdir(parents=True)
    return path


def validate_confirmation_claim(scope, path):
    path = Path(path)
    if path.resolve() != confirmation_claim_path(scope).resolve():
        raise ValueError('original choice/manifest anchored confirmation claim required')
    claim = read(path)
    if (claim.get('schema') != 'gemma4-confirmation-pair-claim-v1' or claim['identity'] != scope.identity
            or claim['permitted_conditions'] != scope.receipt['permitted_conditions'] or claim['count_per_condition'] != 49):
        raise ValueError('confirmation pair claim changed')
    return claim


def confirmation_pair_identity(scope, claim_path):
    validate_confirmation_claim(scope, claim_path)
    return {**scope.identity, 'pair_claim_sha256': sha256_file(claim_path)}


def reservation_path(scope, condition, task_id):
    key = digest({'condition': condition, 'instance_id': task_id})
    return confirmation_claim_path(scope).with_suffix('.tasks') / (key + '.json')


def reserve_confirmation_task(scope, claim_path, condition, task_id, cohort):
    claim = validate_confirmation_claim(scope, claim_path); cohort = Path(cohort).resolve()
    if (condition not in claim['permitted_conditions'] or task_id not in {r['instance_id'] for r in scope.rows}
            or not cohort.is_relative_to(Path(claim['pair_root']).resolve())):
        raise ValueError('only the fixed paired conditions/tasks/output root may run')
    path = reservation_path(scope, condition, task_id); path.parent.mkdir(parents=True, exist_ok=True)
    value = {'schema': 'gemma4-confirmation-task-reservation-v1', 'identity': scope.identity,
        'pair_claim_sha256': sha256_file(claim_path), 'condition': condition, 'instance_id': task_id,
        'cohort': str(cohort), 'policy': 'one model execution; preserve failures, never automatically reroll'}
    with path.open('x') as stream: stream.write(json.dumps(value, indent=2) + '\n')
    return sha256_file(path)


def validate_confirmation_reservation(scope, claim_path, condition, task_id, cohort):
    validate_confirmation_claim(scope, claim_path)
    path = reservation_path(scope, condition, task_id); value = read(path)
    if (value.get('schema') != 'gemma4-confirmation-task-reservation-v1' or value['identity'] != scope.identity
            or value['pair_claim_sha256'] != sha256_file(claim_path) or value['condition'] != condition
            or value['instance_id'] != task_id or value['cohort'] != str(Path(cohort).resolve())):
        raise ValueError('actual confirmation model execution reservation differs')
    return sha256_file(path)
