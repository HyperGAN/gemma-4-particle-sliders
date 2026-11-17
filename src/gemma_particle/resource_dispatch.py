"""Process and physical-slot evidence for explicit two-slot DEV dispatch.

No task is retried here. Process exit, producer completion, and a successful
observation are separate facts. Hardware is sampled only by explicit calls.
"""
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time

from .conversion_handoff import read
from .data import digest
from .dev_dispatch import live, process_binding
from .dev_evidence import canonical_contract
from .harness_identity import sha256_file
from .resource_supplement import normalize_qualified_contract


def pending_producers(producers):
    pending = []
    for producer in producers:
        path = Path(producer['directory']) / 'progress.json'
        current = read(path) if path.exists() else {}
        binding = producer['binding']
        if current.get('state') == 'error':
            raise RuntimeError('actual input producer failed: ' + producer['directory'])
        if current.get('state') != producer['completed_state']:
            if binding is None or not live(binding):
                raise RuntimeError('producer exited without full completion: ' + producer['directory'])
            pending.append(producer['directory'])
        elif binding is not None and live(binding):
            pending.append(producer['directory'] + ':awaiting_process_release')
    return pending


def stopped_prior_grid(directory, binding):
    """Check both the exact prior owner and any competing owner of its output."""
    directory = Path(directory).resolve()
    status = read(directory / 'progress.json')
    if (status.get('state') != 'waiting_actual_inputs' or status.get('model_dev_inference') is not False
            or status.get('finished') != [] or live(binding)):
        raise ValueError('prior dispatcher must be observed stopped with zero learned task attempts')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if not any(a.endswith(b'/run_learned_dev_grid.py') for a in argv) or b'--out' not in argv:
                continue
            target = Path(os.fsdecode(argv[argv.index(b'--out') + 1]))
            if not target.is_absolute():
                target = proc.joinpath('cwd').resolve() / target
            fields = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
            if target.resolve() == directory and fields[0] != 'Z':
                raise ValueError('a live prior dispatcher still owns this output')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sha256_file(directory / 'progress.json')


def descendants(pid):
    parents = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parents[int(proc.name)] = int(proc.joinpath('stat').read_text().split(') ', 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {pid}
    while True:
        expanded = owned | {p for p, parent in parents.items() if parent in owned}
        if expanded == owned:
            return sorted(owned)
        owned = expanded


def hardware_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,compute_cap,memory.total,driver_version,uuid',
        '--format=csv,noheader,nounits'], text=True)
    facts = {}
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        index, name, capability, total, driver, uuid = row
        facts[int(index)] = {'physical_index': int(index), 'name': name, 'compute_capability': capability,
            'total_MiB': int(total), 'driver': driver, 'uuid': uuid}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    applications = [{'pid': int(row[0]), 'gpu_uuid': row[1]}
        for row in csv.reader(raw.splitlines(), skipinitialspace=True)]
    return facts, applications


def observe_slot(binding, index, expected_fact, phase, sample=hardware_snapshot):
    if not live(binding):
        raise ValueError('original owned serving process is no longer live')
    proc = Path('/proc', str(binding['pid']))
    fields = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
    nice = int(fields[16])
    # Select only these two non-secret settings; never copy the process environment.
    selected = {}
    for value in proc.joinpath('environ').read_bytes().split(b'\0'):
        name, _, contents = value.partition(b'=')
        if name in (b'OMP_NUM_THREADS', b'MKL_NUM_THREADS'):
            selected[name.decode()] = int(contents)
    owned = descendants(binding['pid']); facts, applications = sample()
    apps = [a for a in applications if a['pid'] in owned]
    if (facts.get(index) != expected_fact or not apps or nice < 10
            or selected != {'OMP_NUM_THREADS': 4, 'MKL_NUM_THREADS': 4}
            or any(a['gpu_uuid'] != expected_fact['uuid'] for a in apps) or not live(binding)):
        raise ValueError('actual owned slot,priority,thread settings or qualified hardware differs')
    return {'phase': phase, 'observed_at': datetime.now(timezone.utc).isoformat(), 'hardware': facts[index],
        'process_binding': binding, 'descendant_pids': owned, 'owned_gpu_applications': apps,
        'nice': nice, 'cpu_threads': selected}


def slot_record(packet, supplement_path, supplement, identity_path, binding, command, observations):
    identity = read(identity_path); condition = packet['condition']
    normalize_qualified_contract(canonical_contract({'model_identity': identity}), condition['id'], supplement)
    if identity.get('base_only') is not False or identity['adapters'] != packet['adapters']:
        raise ValueError('exact loaded candidate packet required')
    return {'schema': 'gemma4-dev-serving-slot-v1',
        'status': 'complete' if [o['phase'] for o in observations] == ['before_first_task', 'after_final_task'] else 'incomplete',
        'condition': condition, 'supplement_sha256': sha256_file(supplement_path),
        'server_identity_sha256': digest(identity), 'server_identity_file_sha256': sha256_file(identity_path),
        'process_binding': binding, 'command': command, 'observations': list(observations)}


def stop_owned(process, binding, script, output):
    """Signal only the retained Popen child with the original exact ownership."""
    if process.poll() is None:
        if not live(binding) or process_binding(process.pid, script, Path(output).resolve()) != binding:
            raise ValueError('refusing to signal a changed process owner')
        process.send_signal(signal.SIGINT)
    return process.wait()


def wait_wave(jobs, observe=lambda: None, interval=1):
    """Let an already-running peer finish even if its sibling fails; no rerolls."""
    remaining = list(jobs)
    while remaining:
        observe()
        remaining = [job for job in remaining if job.poll() is None]
        if remaining:
            time.sleep(interval)
    return [job.returncode for job in jobs]
