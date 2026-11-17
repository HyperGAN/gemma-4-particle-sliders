"""Release the fixed holdout only after an unchanged, reaudited DEV choice.

These are local evidence bindings, not authentication of invented receipts.
Issuing access performs no model inference or confirmation evaluation.
"""
from collections import Counter
from pathlib import Path

from .conversion_handoff import read
from .data import VISIBLE_FIELDS, read_jsonl
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
