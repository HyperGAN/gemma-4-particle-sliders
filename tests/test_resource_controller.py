"""CPU scheduler integration with real children and mocked model/hardware.

Native/packet validation, hardware qualification and the final model comparison
are supplied fixtures. These controls establish scheduling and failure behavior,
not actual task execution, numerical parity, model scores or qualified hardware.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from gemma_particle.data import digest
from gemma_particle.harness_identity import sha256_file
from gemma_particle.hardware_parity import qualify_server_contracts
from gemma_particle.lora_serving import BASE_ALIAS
from gemma_particle.resource_dispatch import observe_slot
from test_conversion_handoff import put
from test_hardware_parity import servers
from test_learned_effect import protocol_fixture, response

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['complete', 'unqualified', 'no_effect', 'right_no_effect', 'child_failure'])
def test_full_controller_cpu_lifecycle_and_failure_boundaries(tmp_path, monkeypatch, mode):
    spec = importlib.util.spec_from_file_location('cpu_resource_controller', ROOT / 'scripts/run_resource_dev_grid.py')
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    selection_path = ROOT / 'research/dev-selection-v1.json'; selection = json.loads(selection_path.read_text())
    selection_sha = sha256_file(selection_path)
    declared = json.loads((ROOT / 'research/dev-resource-declaration-v1.json').read_text())
    executor, qualifier, auditor = [tmp_path / n for n in ['generation', 'qualifier', 'auditor']]
    for directory in [executor, qualifier, auditor]: directory.mkdir()
    for directory, names in [(executor, [*declared['serving_sources_sha256'], 'scripts/run_dev_controls.py',
            'configs/31b-ct-shared-positional-pilot.json', 'src/gemma_particle/dev_evidence.py', 'src/gemma_particle/dev_selection.py']),
            (qualifier, declared['qualification_sources_sha256']),
            (auditor, ['src/gemma_particle/dev_evidence.py', 'src/gemma_particle/dev_selection.py'])]:
        for name in names:
            path = directory / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(controller, 'frozen_sources', lambda directory, commit:
        {str(p): sha256_file(p) for p in directory.rglob('*') if p.is_file()})
    monkeypatch.setattr(controller, 'load_selection_plan', lambda path: deepcopy(selection))
    protocol, contexts, qualification, _, _ = protocol_fixture(tmp_path)
    for path in [contexts, qualification]:
        value = json.loads(path.read_text()); value['train_manifest_sha256'] = selection['training']['train_manifest_sha256']; put(path, value)
    value = json.loads(qualification.read_text()); value['probe_sha256'] = sha256_file(contexts); put(qualification, value)
    value = json.loads(protocol.read_text()); value.update(core_selection_plan_sha256=selection_sha,
        probe_contexts_sha256=sha256_file(contexts), probe_qualification_sha256=sha256_file(qualification)); put(protocol, value)
    conversion, controls, prior, training, pilot = [tmp_path / n for n in ['conversion', 'controls', 'prior', 'training', 'pilot']]
    for directory in [conversion, controls, prior, training, pilot]: directory.mkdir()
    put(training / 'plan.json', {'pilot': str(pilot)})
    put(conversion / 'plan.json', {'selection_plan_sha256': selection_sha, 'training': str(training),
        'training_plan_sha256': sha256_file(training / 'plan.json'),
        'config_sha256': sha256_file(executor / 'configs/31b-ct-shared-positional-pilot.json'),
        'probe_sha256': sha256_file(contexts), 'probe_qualification_sha256': sha256_file(qualification)})
    pair, baseline, adapters = servers()
    for identity in [*pair, baseline]:
        identity['sources'] = {n: sha256_file(executor / n) for n in declared['serving_sources_sha256']}
    baseline_path = tmp_path / 'base.json'; put(baseline_path, baseline)
    put(controls / 'plan.json', {'selection_plan_sha256': selection_sha, 'server_identity': str(baseline_path),
        'source_sha256': sha256_file(executor / 'scripts/run_dev_controls.py'), 'server_pid': 999999999, 'server_birth_ticks': '1'})
    producers = [{'directory': str(d), 'completed_state': state, 'binding': None} for d, state in [
        (conversion, 'prepared_complete_learned_grid'), (controls, 'complete')]]
    learned = [c for c in selection['conditions'] if c['role'] == 'learned']
    put(prior / 'plan.json', {'selection_plan_sha256': selection_sha, 'executor': str(executor),
        'executor_commit': declared['serving_executor_commit'], 'learned_effect_protocol_sha256': sha256_file(protocol),
        'producers': producers, 'condition_ids': [c['id'] for c in learned], 'tasks_per_condition': 55})
    put(prior / 'progress.json', {'state': 'waiting_actual_inputs', 'finished': [], 'model_dev_inference': False})
    packets = [{'condition': c, 'adapters': adapters, 'adapter': str(tmp_path / 'CPU-adapter'),
        'conversion_report': str(tmp_path / 'CPU-conversion.json'), 'native': str(tmp_path / 'CPU-native'),
        'cache': str(tmp_path / 'CPU-cache'), 'native_metadata_sha256': 'a' * 64,
        'cache_index_sha256': 'b' * 64, 'conversion_report_sha256': 'c' * 64} for c in learned]
    packet_path = conversion / 'candidate-packets.json'; put(packet_path, {'packets': packets})
    for producer in producers: put(Path(producer['directory']) / 'progress.json', {'state': producer['completed_state']})
    control_paths = []
    for condition in selection['conditions'][:4]:
        path = controls / condition['id'] / 'result.json'; put(path, {'condition': condition}); control_paths.append(path)
    monkeypatch.setattr(controller, 'validate_training_outputs', lambda *a: [])
    monkeypatch.setattr(controller, 'load_packets', lambda *a: deepcopy(packets))
    monkeypatch.setattr(controller, 'load_control_results', lambda *a: control_paths)
    policy = {'status': 'qualified_for_explicit_dispatch', 'assignments': declared['assignments'],
        'learned_waves': declared['learned_waves'], 'qualification_contract_sha256': qualify_server_contracts(pair, baseline, adapters),
        'hardware': [{'physical_index': i, 'name': 'CPU supplied fixture', 'uuid': f'CPU-uuid-{i}'} for i in range(2)],
        'packet_manifest': str(packet_path), 'packet_manifest_sha256': sha256_file(packet_path), 'serving_executor': str(executor)}
    def prepare(*args):
        actual = json.loads((Path(args[2]) / 'result.json').read_text())
        if actual['status'] != 'passed': raise ValueError('CPU fixture qualification refused')
        return deepcopy(policy)
    monkeypatch.setattr(controller, 'prepare_resource_supplement', prepare)
    monkeypatch.setattr(controller, 'load_resource_supplement', lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(controller, 'observe_slot', lambda binding, index, fact, phase:
        observe_slot(binding, index, fact, phase, sample=lambda:
            ({index: fact}, [{'pid': binding['pid'], 'gpu_uuid': fact['uuid']}])))
    worker = tmp_path / 'serve_vllm_lora.py'
    worker.write_text("""import json,sys,signal,time
from pathlib import Path
args=sys.argv
def get(name):return args[args.index(name)+1]
out=Path(get('--out'));out.mkdir()
signal.signal(signal.SIGINT,lambda *a:exit(0))
value=json.loads(Path(get('--CPU-template')).read_text())
(out/'server-identity.json').write_text('{')
time.sleep(.03)
(out/'server-identity.json').write_text(json.dumps(value))
while True:time.sleep(1)
""")
    child_worker = tmp_path / 'cpu_child.py'
    child_worker.write_text("""import json,sys,time
from pathlib import Path
payload=json.loads(sys.argv[1])
for path,value in payload['writes'].items():
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
time.sleep(.03)
raise SystemExit(payload['exit'])
""")
    spawned = []; live_children = []; real_popen = subprocess.Popen; real_sleep = time.sleep
    out = tmp_path / 'out'
    ports = []
    for _ in range(2):
        with socket.socket() as check: check.bind(('127.0.0.1', 0)); ports.append(check.getsockname()[1])
    while ports[0] == ports[1]:
        with socket.socket() as check: check.bind(('127.0.0.1', 0)); ports[1] = check.getsockname()[1]
    def option(argv, name): return argv[argv.index('--' + name) + 1]
    def spawn(argv, **kwargs):
        argv = list(argv); role = argv[1]; spawned.append(argv)
        if role.endswith('serve_vllm_lora.py'):
            index = int(option(argv, 'device')[-1]); template = tmp_path / f'template-{index}.json'; put(template, pair[index])
            command = [sys.executable, str(worker), *argv[2:], '--CPU-template', str(template)]
        else:
            writes = {}; exitcode = 0; destination = Path(option(argv, 'out'))
            if role.endswith('qualify_hardware_pair.py'):
                writes[str(destination / 'result.json')] = {'status': 'failed' if mode == 'unqualified' else 'passed'}
            elif role.endswith('run_dev_condition.py'):
                name = option(argv, 'condition'); packet = next(p for p in packets if p['condition']['id'] == name)
                server_path = Path(option(argv, 'server-identity')); identity = json.loads(server_path.read_text())
                writes[str(destination / 'plan.json')] = {'server_identity': str(server_path), 'endpoint': option(argv, 'endpoint')}
                writes[str(destination / 'result.json')] = {'status': 'complete', 'condition': packet['condition'], 'packet': packet,
                    'server_identity_sha256': digest(identity), 'primary': {'total': 42}, 'secondary': {'total': 13}}
                if mode == 'child_failure' and name == learned[0]['id']: exitcode = 9
            elif role.endswith('compare_dev_conditions.py'):
                assert len([a for a in argv if a == '--condition-result']) == 22
                assert json.loads((out / 'plan.json').read_text())['resource_supplement']['sha256'] == sha256_file(option(argv, 'resource-supplement'))
                writes[str(destination / 'selection.json')] = {'selected': learned[0]['id'], 'CPU-fixture': True}
            else: raise AssertionError('unexpected child ' + role)
            command = [sys.executable, str(child_worker), json.dumps({'writes': writes, 'exit': exitcode})]
        child = real_popen(command, **kwargs); live_children.append(child); return child
    monkeypatch.setattr(controller.subprocess, 'Popen', spawn)
    def checked(argv, **kwargs):
        if argv[0] == 'git': return 'scripts/run_resource_dev_grid.py\nsrc/gemma_particle/resource_dispatch.py\n'
        assert argv[0] == 'nvidia-smi' and '--query-gpu=memory.free,memory.total' in argv
        return '49140,49140\n'
    monkeypatch.setattr(controller.subprocess, 'check_output', checked)
    monkeypatch.setattr(controller.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(controller.time, 'sleep', lambda interval: real_sleep(.01))
    def respond(request):
        index = ports.index(request.url.port); identity = pair[index]
        if request.url.path == '/health': return httpx.Response(200, json={'identity_sha256': digest(identity)})
        if request.url.path == '/v1/models': return httpx.Response(200, json={'data': [{'id': BASE_ALIAS}, {'id': 'particle_lora'}]})
        if request.url.path == '/tokenize': return httpx.Response(200, json={'tokens': [2, 3]})
        payload = json.loads(request.content); value = response(payload)
        if mode != 'no_effect' and not (mode == 'right_no_effect' and index == 1) and payload['model'] == 'particle_lora':
            value['choices'][0]['logprobs']['top_logprobs'][0]['token_id:0'] += .001
        return httpx.Response(200, json=value)
    real_client = httpx.Client
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    argv = ['controller', '--prior-grid', str(prior), '--prior-pid', '999999999', '--prior-birth-ticks', '1',
        '--conversion', str(conversion), '--controls', str(controls), '--executor', str(executor),
        '--executor-commit', declared['serving_executor_commit'], '--qualifier', str(qualifier), '--qualifier-commit', 'CPU',
        '--auditor', str(auditor), '--auditor-commit', 'CPU', '--selection-plan', str(selection_path),
        '--declaration', str(ROOT / 'research/dev-resource-declaration-v1.json'),
        '--proposal', str(ROOT / 'research/dev-hardware-resource-proposal-v1.json'),
        '--probe-inputs', str(ROOT / 'research/dev-hardware-probe-inputs-v1.json'),
        '--probe-contexts', str(contexts), '--probe-qualification', str(qualification), '--effect-protocol', str(protocol),
        '--left-port', str(ports[0]), '--right-port', str(ports[1]), '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', argv)
    previous_handlers = {sig: signal.getsignal(sig) for sig in [signal.SIGTERM, signal.SIGINT]}
    try:
        if mode == 'complete': controller.main()
        else:
            with pytest.raises((ValueError, RuntimeError)): controller.main()
    finally:
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        for child in live_children:
            if child.poll() is None: child.terminate()
            child.wait()
    status = json.loads((out / 'progress.json').read_text())
    dev_calls = [call for call in spawned if call[1] == 'scripts/run_dev_condition.py']
    if mode == 'complete':
        assert status['state'] == 'complete_dev_choice'
        assert [r['condition_id'] for r in status['finished']] == [c['id'] for c in learned]
        assert len(dev_calls) == 18 and len(spawned) == 40  # 20 servers,qualifier,18 panels,comparator
        assert all(len((out / (c['id'] + '-preflight-http.jsonl')).read_text().splitlines()) == 24 for c in learned)
    elif mode in ['unqualified', 'no_effect', 'right_no_effect']:
        assert status['state'] == 'error' and not status['model_dev_inference'] and not dev_calls
        assert status['finished'] == [] and not (out / 'choice').exists()
    else:
        assert status['state'] == 'error' and len(dev_calls) == 2
        assert [r['condition_id'] for r in status['finished']] == [learned[1]['id']]
        assert (out / learned[0]['id'] / 'result.json').exists() and not (out / 'choice').exists()
    assert all(child.poll() is not None for child in live_children)
