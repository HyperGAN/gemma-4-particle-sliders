"""CPU dispatch fixtures, never real learned adapters or model DEV results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
import torch
from safetensors.torch import save_file

from gemma_particle.adapters import adapter_identity
from gemma_particle.conversion_handoff import validate_training_outputs
from gemma_particle.data import digest
from gemma_particle.dev_dispatch import (load_packets, load_control_results, candidate_preflight,
    process_binding, live)
from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import BASE_ALIAS, BASE_REPO
from test_conversion_handoff import fake_finished_run, put


def packet_fixture(root, presets=('empf', 'rhbi_optimized')):
    args = fake_finished_run(root / 'native-fixture', presets)
    selection = args[4]
    selection['conditions'] = [{'id': f'{p}-{step}-{s}', 'role': 'learned', 'arm': 'off',
        'preset': p, 'completed_steps': step, 'strength': s}
        for p in selection['training']['presets'] for step in [256, 512, 1024] for s in [.5, 1, 1.5]]
    jobs = validate_training_outputs(*args)
    directory = root / 'conversion'; packets = []
    for condition in selection['conditions']:
        job = next(j for j in jobs if j['preset'] == condition['preset'] and j['completed_steps'] == condition['completed_steps'])
        adapter = directory / condition['id']; adapter.mkdir(parents=True)
        put(adapter / 'adapter_config.json', {'peft_type': 'LORA', 'r': 8, 'base_model_name_or_path': BASE_REPO})
        save_file({'fixture.lora_A.weight': torch.ones(8, 16), 'fixture.lora_B.weight': torch.ones(16, 8)},
            adapter / 'adapter_model.safetensors')
        report = adapter.with_name(adapter.name + '-conversion.json')
        put(report, {'schema': 'gemma4-routed-to-peft-v1', 'exact_merge': False, 'strength': condition['strength'],
            'adapter_sha256': sha256_file(adapter / 'adapter_model.safetensors'),
            'provenance': {**{k: job[k] for k in ['native_metadata_sha256', 'cache_index_sha256', 'fit_task_ids', 'probe_task_ids']},
                'probe_contexts_sha256': sha256_file(args[5])},
            'layer_metrics': {name: {'fit_tokens': 256, 'probe_tokens': 256, 'ridge': .001,
                'delta_rmse': .1, 'delta_rms': 1., 'relative_delta_rmse': .1} for name in job['projection_names']}})
        packets.append({'condition': condition, 'native': job['native'], 'cache': str(args[1]), 'adapter': str(adapter),
            'conversion_report': str(report), 'native_metadata_sha256': job['native_metadata_sha256'],
            'cache_index_sha256': job['cache_index_sha256'], 'conversion_report_sha256': sha256_file(report),
            'adapters': {'particle_lora': adapter_identity(adapter)}})
    put(directory / 'candidate-packets.json', {'schema': 'gemma4-measured-learned-packets-v1',
        'selection_plan_sha256': 'a' * 64, 'packets': packets})
    put(directory / 'progress.json', {'state': 'prepared_complete_learned_grid', 'completed_unit_conversions': len(jobs),
        'candidate_packets': len(packets), 'candidate_packets_sha256': sha256_file(directory / 'candidate-packets.json')})
    return directory, selection, 'a' * 64, jobs, sha256_file(args[5])


@pytest.mark.parametrize('presets', [('empf',), ('empf', 'rhbi_optimized')])
def test_complete_measured_grid_and_all_packets_reaudited(tmp_path, presets):
    args = packet_fixture(tmp_path, presets)
    assert len(load_packets(*args)) == 9 * len(presets)
    adapter = args[0] / args[1]['conditions'][0]['id'] / 'adapter_model.safetensors'
    adapter.write_bytes(b'damaged CPU fixture')
    with pytest.raises(ValueError):
        load_packets(*args)


@pytest.mark.parametrize('damage', ['order', 'dose', 'native', 'alias'])
def test_internally_rebound_but_invalid_packet_manifest_rejected(tmp_path, damage):
    args = packet_fixture(tmp_path); directory = args[0]
    path = directory / 'candidate-packets.json'; manifest = json.loads(path.read_text())
    packet = manifest['packets'][0]
    if damage == 'order':
        manifest['packets'][0], manifest['packets'][1] = manifest['packets'][1], manifest['packets'][0]
    elif damage == 'dose':
        report_path = Path(packet['conversion_report']); report = json.loads(report_path.read_text())
        report['strength'] = 1; put(report_path, report)
        packet['conversion_report_sha256'] = sha256_file(report_path)
    elif damage == 'native':
        packet['native'] = '/CPU-fixture-wrong-checkpoint'
    else:
        packet['adapters'] = {'base-alias': packet['adapters']['particle_lora']}
    put(path, manifest)
    status_path = directory / 'progress.json'; status = json.loads(status_path.read_text())
    status['candidate_packets_sha256'] = sha256_file(path); put(status_path, status)
    with pytest.raises(ValueError):
        load_packets(*args)


def test_all_baseline_controls_and_denominators_required(tmp_path):
    controls = [{'id': name, 'role': 'off' if name == 'off' else 'prompt-control'}
        for name in ['off', 'prompt-reproduce', 'prompt-localize', 'prompt-verify']]
    records = []
    for condition in controls:
        path = tmp_path / condition['id'] / 'result.json'
        put(path, {'status': 'complete', 'condition': condition, 'selection_plan_sha256': 'a' * 64,
            'primary': {'total': 42}, 'secondary': {'total': 13}})
        records.append({'condition_id': condition['id'], 'result_sha256': sha256_file(path)})
    put(tmp_path / 'progress.json', {'state': 'complete', 'planned_conditions': 4, 'finished': records})
    assert len(load_control_results(tmp_path, {'conditions': controls}, 'a' * 64)) == 4
    revision = {'conditions': controls, 'selection_scope': {'original_plan_sha256': 'a' * 64}}
    assert len(load_control_results(tmp_path, revision, 'b' * 64)) == 4
    with pytest.raises(ValueError, match='denominators'):
        load_control_results(tmp_path, {'conditions': controls}, 'b' * 64)
    path = tmp_path / 'off/result.json'; value = json.loads(path.read_text()); value['secondary']['total'] = 5
    put(path, value); records[0]['result_sha256'] = sha256_file(path)
    put(tmp_path / 'progress.json', {'state': 'complete', 'planned_conditions': 4, 'finished': records})
    with pytest.raises(ValueError, match='denominators'):
        load_control_results(tmp_path, {'conditions': controls}, 'a' * 64)


def test_producer_binding_checks_real_pid_birth_command_and_output(tmp_path):
    script = tmp_path / 'fixture_producer.py'
    script.write_text("import time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    child = subprocess.Popen([sys.executable, str(script), '--out', str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        binding = process_binding(child.pid, script.name, tmp_path)
        assert live(binding)
        assert not live({**binding, 'birth_ticks': str(int(binding['birth_ticks']) + 1)})
        with pytest.raises(ValueError, match='output differs'):
            process_binding(child.pid, script.name, tmp_path / 'wrong')
        with pytest.raises(ValueError, match='exact owned'):
            process_binding(child.pid, 'different.py', tmp_path)
    finally:
        child.terminate(); child.wait()
    assert not live(binding)


@pytest.mark.parametrize('damage', [None, 'identity', 'missing_alias', 'response_alias', 'nonfinite', 'timeout'])
def test_synthetic_preflight_requires_actual_alias_and_finite_completion(damage):
    identity = {'fixture': 'not an actual model server'}; observed = []
    def respond(request):
        if damage == 'timeout':
            raise httpx.ReadTimeout('CPU fixture observation timeout', request=request)
        values = {'/health': {'identity_sha256': digest(identity)},
            '/v1/models': {'data': [{'id': BASE_ALIAS}, {'id': 'particle_lora'}]}, '/tokenize': {'tokens': [2, 3]},
            '/v1/completions': {'model': 'particle_lora', 'choices': [{'logprobs':
                {'tokens': ['token_id:7'], 'top_logprobs': [{'token_id:7': -.1}]}}]}}
        if damage == 'identity': values['/health']['identity_sha256'] = 'wrong'
        if damage == 'missing_alias': values['/v1/models']['data'] = [{'id': BASE_ALIAS}]
        if damage == 'response_alias': values['/v1/completions']['model'] = BASE_ALIAS
        if damage == 'nonfinite':
            # JSON's strict writer rejects NaN; use a deliberately invalid raw mock response.
            if request.url.path == '/v1/completions':
                return httpx.Response(200, content=json.dumps({**values['/v1/completions'], 'choices': [{'logprobs':
                    {'tokens': ['token_id:7'], 'top_logprobs': [{'token_id:7': float('nan')}]}}]}))
        return httpx.Response(200, json=values[request.url.path])
    def call(client, root, path, payload=None):
        observed.append((path, deepcopy(payload)))
        response = client.get(root + path) if payload is None else client.post(root + path, json=payload)
        response.raise_for_status(); return response.json()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if damage:
            with pytest.raises(httpx.ReadTimeout if damage == 'timeout' else ValueError):
                candidate_preflight(client, 'http://127.0.0.1:8031', identity, call)
        else:
            assert candidate_preflight(client, 'http://127.0.0.1:8031', identity, call)['status'] == 'passed'
            assert len(observed) == 4 and observed[-1][1]['prompt'] == [2, 3]
            assert observed[-1][1]['seed'] == 0
