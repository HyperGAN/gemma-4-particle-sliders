"""Exercise the checkpoint handoff with fixture processes and native receipts."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from test_conversion_handoff import fake_finished_run, put


@pytest.fixture
def controller():
    script = Path(__file__).resolve().parents[1] / 'scripts/continue_empf_only.py'
    spec = importlib.util.spec_from_file_location('continue_empf_only', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_requires_exact_parent_holding_and_clean_worker_exit(controller, monkeypatch):
    hold = {'controller': {'pid': 5}, 'empf_child': {'pid': 8}}
    states = {5: {'state': 'T'}, 8: {'state': 'R', 'parent': 5, 'exit_status': 0}}
    monkeypatch.setattr(controller, 'process_snapshot', lambda binding: states[binding['pid']])
    assert not controller.check_handoff(hold)
    states[8].update(state='Z', exit_status=256)
    with pytest.raises(RuntimeError, match='failed with wait status'):
        controller.check_handoff(hold)
    states[8]['exit_status'] = 0
    assert controller.check_handoff(hold)
    states[5]['state'] = 'R'
    with pytest.raises(ValueError, match='stopped two-recipe supervisor'):
        controller.check_handoff(hold)


def test_takeover_uses_original_256_checkpoint_then_runs_only_empf(controller, tmp_path, monkeypatch):
    original, cache, pilot, config, _, _, _ = fake_finished_run(tmp_path)
    (pilot / 'cache').symlink_to(cache, target_is_directory=True)
    source = original / 'empf'
    archive = tmp_path / 'fixture-future'
    shutil.copytree(source, archive)
    metrics = (archive / 'updates.jsonl').read_bytes().splitlines(keepends=True)
    (source / 'updates.jsonl').write_bytes(b''.join(metrics[:256]))
    (pilot / 'empf').mkdir()
    (pilot / 'empf/updates.jsonl').write_bytes(metrics[0])
    # The original fixture builds both final recipes. This case models a run
    # still in its first EMPF chunk, before any second-recipe work exists.
    shutil.rmtree(original / 'rhbi_optimized')
    executor = tmp_path / 'executor'
    configuration = executor / 'configs/31b-ct-shared-positional-pilot.json'
    put(configuration, controller.read(config))
    put(executor / 'reports/long-history-gradient-qualification-v1.json', {'status': 'passed', 'memory_fraction': .5})
    source_script = executor / 'scripts/train.py'
    source_script.parent.mkdir()
    source_script.write_text('# fixture source\n')
    put(original / 'plan.json', {'executor_commit': 'fixture', 'pilot': str(pilot), 'device': 'cuda:1',
                                 'memory_fraction': .5, 'horizon': 1024, 'checkpoints': [256, 512, 1024],
                                 'config_sha256': controller.sha256_file(configuration),
                                 'sources': {'scripts/train.py': controller.sha256_file(source_script)}})
    put(original / 'progress.json', {'preset': 'empf', 'target_step': 256, 'pilot_checkpoint': {'completed_steps': 1},
                                     'cache_index_sha256': controller.sha256_file(cache / 'index.json')})
    hold = tmp_path / 'hold.json'
    put(hold, {'controller': {'out': str(original)}, 'empf_child': {'preset': 'empf', 'target_step': 256,
                                                                 'device': 'cuda:1', 'cwd': str(executor)}})
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['continue_empf_only.py', '--hold', str(hold), '--out', str(output)])
    states = iter([False, False, True])
    monkeypatch.setattr(controller, 'check_handoff', lambda _: next(states))
    monkeypatch.setattr(controller, 'shared_headroom', lambda *_: True)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **k: 'fixture\n')
    commands = []

    def wait(seconds):
        assert seconds == 5 and not commands
        assert not (output / 'empf').exists()

    def run(command, **kwargs):
        assert kwargs['check'] and kwargs['cwd'] == executor
        assert command[command.index('--preset') + 1] == 'empf'
        assert command[command.index('--memory-fraction') + 1] == '.50'
        own = output / 'empf'
        previous, target = [(256, 512), (512, 1024)][len(commands)]
        assert command[command.index('--resume') + 1] == str(own / f'FULL-{previous:04d}.pt')
        assert command[command.index('--chunk-updates') + 1] == str(target - previous)
        for suffix in ('.pt', '.json'):
            name = f'FULL-{target:04d}' + suffix
            shutil.copyfile(archive / name, own / name)
        shutil.copytree(archive / f'RAW-{target:04d}', own / f'RAW-{target:04d}')
        (own / 'updates.jsonl').write_bytes(b''.join(metrics[:target]))
        commands.append(command)

    monkeypatch.setattr(controller.time, 'sleep', wait)
    monkeypatch.setattr(controller.subprocess, 'run', run)
    controller.main()
    assert len(commands) == 2
    assert not (output / 'rhbi_optimized').exists()
    assert controller.read(output / 'adopted-chunk.json')['verified_checkpoint'] == controller.checkpoint_receipt(source, 256)
    progress = controller.read(output / 'progress.json')
    assert (progress['state'], progress['training_updates']) == ('completed_empf_only', 1024)
    assert [record['completed_steps'] for record in progress['finished']] == [256, 512, 1024]
    assert json.loads(metrics[0])['step'] == 1
