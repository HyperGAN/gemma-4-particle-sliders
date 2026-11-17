"""Qualify an explicit DEV resource policy; preserve raw physical contracts.

These are local consistency checks, not authentication of invented experiments.
This module never launches a server or changes the ordinary DEV canonicalizer.
"""
from copy import deepcopy
from pathlib import Path

from .conversion_handoff import read
from .data import digest
from .dev_evidence import SERVER_SOURCES
from .dev_evidence import canonical_contract
from .hardware_evidence import audit_hardware_pair
from .harness_identity import sha256_file


def validate_resource_declaration(path, selection_path):
    declaration = read(path); selection = read(selection_path)
    learned = [c for c in selection['conditions'] if c['role'] == 'learned']
    expected = []
    for condition in selection['conditions']:
        index = next((i for i, c in enumerate(learned) if c['id'] == condition['id']), None)
        expected.append({'condition_id': condition['id'], 'physical_index': 0 if index is None else index % 2,
            'wave': None if index is None else index // 2 + 1})
    if (declaration.get('schema') != 'gemma4-dev-resource-declaration-v1'
            or declaration.get('state') != 'inactive_preparation'
            or declaration['core_selection_plan_sha256'] != sha256_file(selection_path)
            or len(selection['conditions']) != 22 or len(learned) != 18
            or declaration['assignments'] != expected
            or declaration['learned_waves'] != [[c['id'] for c in learned[i:i + 2]] for i in range(0, 18, 2)]
            or declaration['engine_memory_fraction'] != .6 or declaration['startup_headroom_GiB'] != 8
            or declaration['cpu_threads_per_slot'] != 4 or declaration['inherited_nice'] != 10
            or set(declaration['serving_sources_sha256']) != set(SERVER_SOURCES)
            or declaration['confirmation_device_policy'] != 'strict original GPU0 serving contract'):
        raise ValueError('original complete prospective resource assignment and shared serving limits required')
    return declaration


def prepare_resource_supplement(declaration_path, selection_path, qualification):
    declaration_path, selection_path, qualification = map(Path, (declaration_path, selection_path, qualification))
    declaration = validate_resource_declaration(declaration_path, selection_path)
    audit = audit_hardware_pair(qualification); plan = read(qualification / 'plan.json')
    inputs = plan['inputs']; pins = plan['declaration_pins']
    if (plan['selection_plan_sha256'] != declaration['core_selection_plan_sha256']
            or plan['executor_commit'] != declaration['serving_executor_commit']
            or sha256_file(inputs['proposal']) != declaration['qualification_proposal_sha256']
            or sha256_file(inputs['probe_inputs']) != declaration['qualification_probe_inputs_sha256']
            or sha256_file(Path(__file__).with_name('hardware_evidence.py')) != declaration['offline_hardware_auditor_sha256']):
        raise ValueError('exact declared source-bound hardware qualification required')
    for name, value in declaration['qualification_sources_sha256'].items():
        matches = [v for p, v in pins.items() if p == name or p.endswith('/' + name)]
        if matches != [value]: raise ValueError('original measured qualification source required')
    executor = Path(inputs['executor'])
    if any(sha256_file(executor / name) != value for name, value in declaration['serving_sources_sha256'].items()):
        raise ValueError('original frozen serving implementation required')
    baseline = read(Path(inputs['controls']) / 'plan.json')['server_identity']
    packet_manifest = Path(inputs['conversion']) / 'candidate-packets.json'
    return {'schema': 'gemma4-qualified-dev-resource-v1', 'status': 'qualified_for_explicit_dispatch',
        'declaration': str(declaration_path.resolve()), 'declaration_sha256': sha256_file(declaration_path),
        'selection_plan': str(selection_path.resolve()), 'selection_plan_sha256': sha256_file(selection_path),
        'qualification': str(qualification.resolve()), 'qualification_result_sha256': audit['result_sha256'],
        'qualification_plan_sha256': audit['plan_sha256'],
        'qualification_contract_sha256': audit['qualification_contract_sha256'], 'hardware': audit['hardware'],
        'baseline_server_identity': baseline, 'baseline_server_identity_sha256': digest(read(baseline)),
        'serving_executor': str(executor.resolve()), 'packet_manifest': str(packet_manifest.resolve()),
        'packet_manifest_sha256': sha256_file(packet_manifest),
        'assignments': declaration['assignments'], 'learned_waves': declaration['learned_waves'],
        'serving_launched': False, 'model_task_inference': False, 'ordinary_DEV_canonicalizer_changed': False,
        'scope': 'qualified explicit DEV resource policy only;raw physical evidence required per condition;confirmation remainsGPU0'}


def load_resource_supplement(path):
    record = read(path)
    if (record.get('schema') != 'gemma4-qualified-dev-resource-v1'
            or record.get('status') != 'qualified_for_explicit_dispatch'):
        raise ValueError('completed explicit qualified DEV resource record required')
    expected = prepare_resource_supplement(record['declaration'], record['selection_plan'], record['qualification'])
    if record != expected: raise ValueError('unchanged genuine resource declaration and numerical qualification required')
    return record


def normalize_qualified_contract(contract, condition_id, supplement):
    """Normalize only the assigned physical device after explicit qualification.

The caller must retain/audit the original physical fingerprint and slot evidence.
No prompt, environment, model, source, package or engine setting is normalized.
"""
    if supplement.get('status') != 'qualified_for_explicit_dispatch':
        raise ValueError('explicit qualified DEV resource policy required')
    assignment = next((a for a in supplement['assignments'] if a['condition_id'] == condition_id), None)
    if assignment is None or contract['model_identity']['device'] != f"cuda:{assignment['physical_index']}":
        raise ValueError('condition must retain its prospectively assigned physical device')
    server = deepcopy(contract['model_identity']); server.pop('device')
    if digest({'model_identity': server}) != supplement['qualification_contract_sha256']:
        raise ValueError('only the exact qualified serving contract may normalize its physical device')
    result = deepcopy(contract); result['model_identity']['device'] = 'cuda:0'
    return result


def audit_condition_slot(path, result, supplement_path, supplement):
    """Check a completed condition's recorded physical slot before normalization."""
    path = Path(path); condition = result['condition']; name = condition['id']
    plan = read(path.parent / 'plan.json'); server_path = Path(plan['server_identity']); server = read(server_path)
    if digest(server) != result['server_identity_sha256']:
        raise ValueError('original condition serving identity required')
    if (type(server.get('observed_free_bytes')) is not int or type(server.get('total_device_bytes')) is not int
            or server['observed_free_bytes'] < server['total_device_bytes'] * .6 + 8 * 1024**3):
        raise ValueError('actual per-condition shared GPU startup headroom required')
    normalize_qualified_contract(canonical_contract({'model_identity': server}), name, supplement)
    if condition['role'] != 'learned':
        if (digest(server) != supplement['baseline_server_identity_sha256']
                or server.get('base_only') is not True or server.get('adapters')):
            raise ValueError('original true-base GPU0 control server required')
        return {'server_identity_sha256': digest(server), 'baseline_control': True}
    evidence_path = path.parent / 'resource.json'; evidence = read(evidence_path)
    assignment = next(a for a in supplement['assignments'] if a['condition_id'] == name)
    index = assignment['physical_index']; fact = supplement['hardware'][index]
    packets = read(supplement['packet_manifest'])['packets']
    packet = next(p for p in packets if p['condition']['id'] == name)
    if (sha256_file(supplement['packet_manifest']) != supplement['packet_manifest_sha256']
            or evidence.get('schema') != 'gemma4-dev-serving-slot-v1' or evidence.get('status') != 'complete'
            or evidence['condition'] != condition or evidence['supplement_sha256'] != sha256_file(supplement_path)
            or evidence['server_identity_sha256'] != digest(server)
            or evidence['server_identity_file_sha256'] != sha256_file(server_path)
            or server.get('base_only') is not False or server['adapters'] != packet['adapters']
            or result['packet']['adapters'] != packet['adapters']
            or any(result['packet'][k] != packet[k] for k in ['native_metadata_sha256', 'cache_index_sha256', 'conversion_report_sha256'])):
        raise ValueError('exact completed learned condition,packet and physical slot evidence required')
    command = evidence['command']; argv = command['argv']
    from urllib.parse import urlsplit
    endpoint = urlsplit(plan['endpoint'])
    if (Path(command['cwd']).resolve() != Path(supplement['serving_executor']).resolve()
            or Path(command['identity_path']).resolve() != server_path.resolve()
            or argv[argv.index('--device') + 1] != f'cuda:{index}'
            or argv[argv.index('--port') + 1] != str(endpoint.port)
            or Path(argv[argv.index('--adapter') + 1]).resolve() != Path(packet['adapter']).resolve()
            or not any(a.endswith('/serve_vllm_lora.py') for a in argv)):
        raise ValueError('original assigned-device launcher command required')
    observations = evidence['observations']; binding = evidence['process_binding']
    if ([o['phase'] for o in observations] != ['before_first_task', 'after_final_task']
            or type(binding['pid']) is not int or binding['pid'] <= 0
            or not isinstance(binding['birth_ticks'], str) or not binding['birth_ticks'].isdigit()):
        raise ValueError('both original serving process observations required')
    for observed in observations:
        applications = observed['owned_gpu_applications']; descendants = observed['descendant_pids']
        if (observed['hardware'] != fact or observed['process_binding'] != binding
                or binding['pid'] not in descendants or not applications
                or type(observed.get('nice')) is not int or observed['nice'] < 10
                or observed.get('cpu_threads') != {'OMP_NUM_THREADS': 4, 'MKL_NUM_THREADS': 4}
                or any(a['gpu_uuid'] != fact['uuid'] or a['pid'] not in descendants for a in applications)):
            raise ValueError('original live owned GPU processes must match the qualified physical slot')
    return {'path': str(evidence_path.resolve()), 'sha256': sha256_file(evidence_path),
        'server_identity_sha256': digest(server), 'physical_index': index, 'gpu_uuid': fact['uuid']}
