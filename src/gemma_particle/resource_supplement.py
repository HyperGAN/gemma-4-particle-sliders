"""Qualify an explicit DEV resource policy; preserve raw physical contracts.

These are local consistency checks, not authentication of invented experiments.
This module never launches a server or changes the ordinary DEV canonicalizer.
"""
from copy import deepcopy
from pathlib import Path

from .conversion_handoff import read
from .data import digest
from .dev_evidence import SERVER_SOURCES
from .dev_evidence import canonical_contract, retained_case_paths
from .hardware_evidence import audit_hardware_pair
from .harness_identity import sha256_file


def resource_schedule(selection):
    learned = [c for c in selection['conditions'] if c['role'] == 'learned']
    indices = {condition['id']: i for i, condition in enumerate(learned)}
    expected = []
    for condition in selection['conditions']:
        index = indices.get(condition['id'])
        expected.append({'condition_id': condition['id'], 'physical_index': 0 if index is None else index % 2,
            'wave': None if index is None else index // 2 + 1})
    waves = [[c['id'] for c in learned[i:i + 2]] for i in range(0, len(learned), 2)]
    return expected, waves


def validate_resource_declaration(path, selection_path):
    declaration = read(path); selection = read(selection_path)
    learned = [c for c in selection['conditions'] if c['role'] == 'learned']
    scoped = selection.get('selection_scope', {}).get('kind') == 'empf_only'
    count = 9 if scoped else 18
    expected, waves = resource_schedule(selection)
    if (declaration.get('schema') != 'gemma4-dev-resource-declaration-v1'
            or declaration.get('state') != 'inactive_preparation'
            or declaration['core_selection_plan_sha256'] != sha256_file(selection_path)
            or len(selection['conditions']) != count + 4 or len(learned) != count
            or scoped and any(c['preset'] != 'empf' for c in learned)
            or declaration['assignments'] != expected
            or declaration['learned_waves'] != waves
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


def bind_resource_choice(choice, grid_plan, path, selection_sha256):
    """Bind an already-complete choice before any confirmation manifest read."""
    expected = {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}
    if choice.get('resource_supplement') != expected or grid_plan.get('resource_supplement') != expected:
        raise ValueError('locked choice and explicit dispatcher must bind the same original resource policy')
    resource = load_resource_supplement(path)
    if resource['selection_plan_sha256'] != selection_sha256:
        raise ValueError('qualified resources must preserve the original complete selection plan')
    return expected


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


def audit_task_slot(condition_folder, result, record, manifest, slot, supplement):
    """Bind each task to the serving identity of its original execution."""
    server = manifest['server_identity']
    reused_from = record.get('reused_from')
    if reused_from is None:
        if digest(server) != slot['server_identity_sha256']:
            raise ValueError('task server differs from actual recorded condition slot')
        return

    condition = result['condition']
    if not slot.get('baseline_control') or condition['role'] == 'learned':
        raise ValueError('only a recovered base control can reuse a prior serving identity')
    current_plan = read(Path(condition_folder) / 'plan.json')
    receipt = current_plan.get('reused_evidence')
    if (not receipt or receipt != result.get('reused_evidence')
            or receipt['prior_condition'] != reused_from
            or record['instance_id'] not in receipt['task_ids']):
        raise ValueError('task must belong to the declared recovered prefix')

    prior = Path(reused_from).resolve()
    before_path = prior / 'plan.json'
    before = read(before_path)
    if (sha256_file(before_path) != receipt['prior_plan_sha256']
            or any(before[k] != current_plan[k] for k in ['condition', 'selection_plan_sha256', 'task_ids', 'packet'])
            or before['condition'] != condition or before['packet'] is not None):
        raise ValueError('original matching base control recovery plan required')
    cohort, verification = retained_case_paths(prior, record['panel'], record['instance_id'])
    if (cohort != Path(record['cohort']).resolve() or verification != Path(record['verification']).resolve()):
        raise ValueError('recovered task must retain its original evidence paths')
    origin = read(cohort.parents[1] / 'plan.json')
    if (origin['condition'] != condition or origin['packet'] is not None
            or server != read(origin['server_identity']) or server != manifest['contract']['model_identity']
            or server.get('base_only') is not True or server.get('adapters')
            or type(server.get('observed_free_bytes')) is not int or type(server.get('total_device_bytes')) is not int
            or server['observed_free_bytes'] < server['total_device_bytes'] * .6 + 8 * 1024**3):
        raise ValueError('recovered task must match its original shared base server')
    normalize_qualified_contract(canonical_contract({'model_identity': server}), condition['id'], supplement)


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
