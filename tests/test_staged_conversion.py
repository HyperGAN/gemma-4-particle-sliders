"""Exercise conversion orchestration with fixture processes and no CUDA calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from gemma_particle.harness_identity import sha256_file
from test_conversion_handoff import fake_finished_run, put


def test_staged_sweep_waits_for_release_and_publishes_each_checkpoint(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/prepare_learned_sweep.py'
    spec = importlib.util.spec_from_file_location('staged_conversion_controller', script)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    training, cache, pilot, config, selection, probe, qualification = fake_finished_run(tmp_path)
    (pilot / 'cache').symlink_to(cache, target_is_directory=True)
    completed = controller.read(training / 'progress.json')
    progress = {**completed, 'state': 'training', 'training_updates': 256, 'finished': completed['finished'][:1],
                'completed_steps_by_preset': {'empf': 256, 'rhbi_optimized': 0}}
    put(training / 'progress.json', progress)
    baseline_conditions = [{'id': name, 'role': 'control'} for name in ('off', 'reproduce', 'localize', 'verify')]
    learned = [{**row, 'id': f"{row['preset']}-{row['completed_steps']}-{strength}", 'strength': strength}
               for row in selection['conditions'] for strength in (1, .5, 1.5)]
    selection['conditions'] = baseline_conditions + learned
    selection_path = tmp_path / 'selection.json'
    put(selection_path, selection)
    selection_sha = sha256_file(selection_path)
    executor = tmp_path / 'executor'
    config_path = executor / 'configs/31b-ct-shared-positional-pilot.json'
    put(config_path, controller.read(config))
    put(executor / 'reports/long-history-gradient-qualification-v1.json', {'peak_host_rss_bytes': 1})
    put(training / 'plan.json', {'executor_commit': 'fixture', 'device': 'cuda:1', 'config_sha256': sha256_file(config_path)})
    inputs = ('scripts/convert_to_peft.py', 'scripts/derive_peft_strength.py', 'src/gemma_particle/model.py',
              'src/gemma_particle/runtime.py', 'src/gemma_particle/export.py', 'src/gemma_particle/conversion_probe.py',
              'src/gemma_particle/lora_strengths.py')
    for name in inputs:
        path = executor / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture source\n')
    controls = tmp_path / 'controls'
    identity = tmp_path / 'baseline.json'
    put(identity, {'device': 'cuda:0', 'base_only': True, 'adapters': {}})
    put(controls / 'plan.json', {'schema': 'gemma4-full-dev-controls-plan-v1', 'selection_plan_sha256': selection_sha,
                                 'server_identity': str(identity), 'server_pid': 30, 'server_birth_ticks': '300'})
    records = []
    for condition in baseline_conditions:
        path = controls / condition['id'] / 'result.json'
        put(path, {'status': 'complete', 'condition': condition, 'selection_plan_sha256': selection_sha,
                   'primary': {'total': 42}, 'secondary': {'total': 13}})
        records.append({'condition_id': condition['id'], 'result_sha256': sha256_file(path)})
    control_progress = controls / 'progress.json'
    put(control_progress, {'state': 'running'})

    out = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['prepare_learned_sweep.py', '--training', str(training), '--training-pid', '10',
        '--pilot', str(pilot), '--executor', str(executor), '--executor-commit', 'fixture', '--device', 'cuda:0',
        '--selection-plan', str(selection_path), '--probe-contexts', str(probe), '--probe-qualification', str(qualification),
        '--staged-controls', str(controls), '--staged-controls-pid', '20', '--out', str(out)])
    clock = {'baseline_server_live': True, 'unit_conversions': []}
    monkeypatch.setattr(controller, 'load_selection_plan', lambda _: selection)
    monkeypatch.setattr(controller, 'process_binding', lambda pid, *_: {'pid': pid, 'birth_ticks': str(pid * 10)})
    monkeypatch.setattr(controller, 'live', lambda binding: binding['pid'] != 30 or clock['baseline_server_live'])
    monkeypatch.setattr(controller.subprocess, 'check_output',
                        lambda command, **_: 'fixture\n' if command[0] == 'git' else '49140, 49140\n')
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: 'MemAvailable: 67108864 kB\n'
                        if path == Path('/proc/meminfo') else original_read(path, *a, **k))
    monkeypatch.setattr(controller, 'validate_conversion', lambda *a, **k: 'fixture-adapter')
    monkeypatch.setattr(controller, 'prepare_lora', lambda *a, **k: {'adapters': {'particle_lora': 'fixture-adapter'}})

    def run(command, **kwargs):
        if command[0] == 'git':
            return SimpleNamespace(returncode=0)
        if 'scripts/convert_to_peft.py' in command:
            assert not clock['baseline_server_live']
            assert command[command.index('--device') + 1] == 'cuda:0'
            assert command[command.index('--memory-fraction') + 1] == '.50'
            if not clock['unit_conversions']:
                assert controller.read(training / 'progress.json')['state'] == 'training'
            clock['unit_conversions'].append(command[command.index('--native') + 1])
        else:
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        destination = Path(command[command.index('--out') + 1])
        put(destination.with_name(destination.name + '-conversion.json'), {'fixture': str(destination)})
        return SimpleNamespace(returncode=0)

    def sleep(_):
        if controller.read(control_progress)['state'] == 'running':
            assert not clock['unit_conversions']
            put(control_progress, {'state': 'complete', 'planned_conditions': 4, 'finished': records})
        elif clock['baseline_server_live']:
            assert not clock['unit_conversions']
            clock['baseline_server_live'] = False
        else:
            assert len(clock['unit_conversions']) == 1
            packet = controller.read(out / 'staged-checkpoints/empf-step256.json')
            assert len(packet['packets']) == 3
            assert not (out / 'candidate-packets.json').exists()
            put(training / 'progress.json', completed)

    monkeypatch.setattr(controller.subprocess, 'run', run)
    monkeypatch.setattr(controller.time, 'sleep', sleep)
    controller.main()
    assert len(clock['unit_conversions']) == 6
    assert len(list((out / 'staged-checkpoints').glob('*.json'))) == 6
    assert controller.read(out / 'staged-baseline-release.json')['observed_exited']
    assert [packet['condition'] for packet in controller.read(out / 'candidate-packets.json')['packets']] == learned
    assert controller.read(out / 'progress.json')['state'] == 'prepared_complete_learned_grid'
