"""Real CPU-process boundaries; supplied hardware facts are fixtures, not parity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gemma_particle.dev_dispatch import live, process_binding
from gemma_particle.harness_identity import sha256_file
from gemma_particle.resource_dispatch import (pending_producers, stopped_prior_grid, observe_slot,
    slot_record, stop_owned, wait_wave)
from gemma_particle.resource_supplement import audit_condition_slot
from test_conversion_handoff import put
from test_resource_supplement import slot_fixture


@pytest.fixture
def cpu_owner(tmp_path):
    script = tmp_path / 'run_learned_dev_grid.py'
    script.write_text('import signal,time\nsignal.signal(signal.SIGINT,lambda *a:exit(0))\ntime.sleep(60)\n')
    out = tmp_path / 'grid'; out.mkdir()
    put(out / 'progress.json', {'state': 'waiting_actual_inputs', 'model_dev_inference': False, 'finished': []})
    def lower_priority():
        os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    child = subprocess.Popen([sys.executable, str(script), '--out', str(out)],
        env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'), preexec_fn=lower_priority)
    binding = process_binding(child.pid, script.name, out)
    # Do not rely on the Python signal handler being installed immediately after exec.
    time.sleep(.05)
    try:
        yield child, binding, out
    finally:
        if child.poll() is None: child.terminate()
        child.wait()


def test_prior_grid_needs_actual_owner_exit_and_competing_owner_is_detected(cpu_owner):
    child, binding, out = cpu_owner
    with pytest.raises(ValueError, match='observed stopped'): stopped_prior_grid(out, binding)
    wrong_birth = dict(binding, birth_ticks=str(int(binding['birth_ticks']) + 1))
    with pytest.raises(ValueError, match='live prior dispatcher'): stopped_prior_grid(out, wrong_birth)
    child.terminate(); child.wait()
    assert stopped_prior_grid(out, binding) == sha256_file(out / 'progress.json')


@pytest.mark.parametrize('damage', ['started', 'finished', 'state'])
def test_stopped_process_does_not_erase_prior_task_attempts(cpu_owner, damage):
    child, binding, out = cpu_owner; child.terminate(); child.wait()
    status = json.loads((out / 'progress.json').read_text())
    if damage == 'started': status['model_dev_inference'] = True
    elif damage == 'finished': status['finished'] = [{'condition_id': 'attempted'}]
    else: status['state'] = 'error'
    put(out / 'progress.json', status)
    with pytest.raises(ValueError): stopped_prior_grid(out, binding)


def test_producer_completion_and_process_release_are_both_needed(cpu_owner):
    child, binding, out = cpu_owner
    producer = {'directory': str(out), 'binding': binding, 'completed_state': 'complete'}
    assert pending_producers([producer]) == [str(out)]
    put(out / 'progress.json', {'state': 'complete'})
    assert pending_producers([producer]) == [str(out) + ':awaiting_process_release']
    child.terminate(); child.wait()
    assert pending_producers([producer]) == []


@pytest.mark.parametrize('state', ['waiting_actual_inputs', 'error', None])
def test_missing_producer_or_failed_producer_never_becomes_completion(cpu_owner, state):
    child, binding, out = cpu_owner; child.terminate(); child.wait()
    if state is None: (out / 'progress.json').unlink()
    else: put(out / 'progress.json', {'state': state})
    with pytest.raises(RuntimeError):
        pending_producers([{'directory': str(out), 'binding': binding, 'completed_state': 'complete'}])


def test_cleanup_refuses_different_output_then_stops_only_retained_owner(cpu_owner):
    child, binding, out = cpu_owner
    with pytest.raises(ValueError): stop_owned(child, binding, 'run_learned_dev_grid.py', out / 'other')
    assert child.poll() is None and live(binding)
    assert stop_owned(child, binding, 'run_learned_dev_grid.py', out) == 0
    assert not live(binding)


def test_failed_wave_member_does_not_cancel_or_retry_its_running_peer(tmp_path):
    marker = tmp_path / 'peer-finished'
    failed = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'])
    peer = subprocess.Popen([sys.executable, '-c',
        'import time;from pathlib import Path;import sys;time.sleep(.2);Path(sys.argv[1]).write_text("once")', str(marker)])
    observations = []
    assert wait_wave([failed, peer], observe=lambda: observations.append(True), interval=.01) == [7, 0]
    assert observations and marker.read_text() == 'once'


@pytest.mark.parametrize('damage', [None, 'uuid', 'no-owned-app', 'hardware', 'dead'])
def test_slot_observation_binds_real_cpu_owner_and_selected_settings(cpu_owner, damage):
    child, binding, out = cpu_owner
    fact = {'physical_index': 1, 'name': 'CPU supplied fixture', 'uuid': 'fixture-uuid'}
    facts = {1: dict(fact)}; apps = [{'pid': child.pid, 'gpu_uuid': fact['uuid']}]; calls = []
    if damage == 'uuid': apps[0]['gpu_uuid'] = 'other'
    elif damage == 'no-owned-app': apps[0]['pid'] = 0
    elif damage == 'hardware': facts[1]['name'] = 'changed'
    elif damage == 'dead': child.terminate(); child.wait()
    def sample():
        calls.append(True); return facts, apps
    if damage:
        with pytest.raises(ValueError): observe_slot(binding, 1, fact, 'before_first_task', sample=sample)
        if damage == 'dead': assert not calls
    else:
        observed = observe_slot(binding, 1, fact, 'before_first_task', sample=sample)
        assert observed['nice'] >= 10 and observed['process_binding'] == binding
        assert observed['cpu_threads'] == {'OMP_NUM_THREADS': 4, 'MKL_NUM_THREADS': 4}
        assert observed['owned_gpu_applications'] == apps


def test_dispatcher_slot_record_is_accepted_by_independent_resource_reader(tmp_path):
    result_path, result, policy_path, policy = slot_fixture(tmp_path)
    original = json.loads((result_path.parent / 'resource.json').read_text())
    identity = Path(original['command']['identity_path'])
    args = (result['packet'], policy_path, policy, identity, original['process_binding'], original['command'])
    before = slot_record(*args, original['observations'][:1])
    assert before['status'] == 'incomplete'
    put(result_path.parent / 'resource.json', before)
    with pytest.raises(ValueError): audit_condition_slot(result_path, result, policy_path, policy)
    complete = slot_record(*args, original['observations'])
    put(result_path.parent / 'resource.json', complete)
    assert audit_condition_slot(result_path, result, policy_path, policy)['physical_index'] == 1
