"""Bind the complete measured packet grid before dispatching real DEV jobs."""
import math
import os
from pathlib import Path

from .conversion_handoff import read, validate_conversion
from .data import digest
from .dev_selection import baseline_selection_sha256
from .harness_identity import sha256_file
from .lora_serving import BASE_ALIAS


def process_binding(pid, script, output):
    proc = Path('/proc') / str(pid)
    argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if not any(a.endswith(('/' + script).encode()) for a in argv) or b'--out' not in argv:
        raise ValueError('exact owned producer required: ' + script)
    actual = Path(os.fsdecode(argv[argv.index(b'--out') + 1]))
    if not actual.is_absolute():
        actual = proc.joinpath('cwd').resolve() / actual
    if actual.resolve() != output:
        raise ValueError('producer output differs: ' + script)
    return {'pid': pid, 'birth_ticks': proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]}


def live(binding):
    try:
        fields = Path('/proc', str(binding['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and fields[19] == binding['birth_ticks']


def load_packets(directory, selection, plan_sha256, jobs, probe_sha256):
    directory = Path(directory)
    status = read(directory / 'progress.json')
    path = directory / 'candidate-packets.json'
    manifest = read(path)
    expected = [c for c in selection['conditions'] if c['role'] == 'learned']
    checkpoints = {(c['preset'], c['completed_steps']) for c in expected}
    by_checkpoint = {(j['preset'], j['completed_steps']): j for j in jobs}
    packets = manifest['packets']
    if (status.get('state') != 'prepared_complete_learned_grid'
            or status.get('completed_unit_conversions') != len(checkpoints) or status.get('candidate_packets') != len(expected)
            or len(jobs) != len(checkpoints) or by_checkpoint.keys() != checkpoints
            or status.get('candidate_packets_sha256') != sha256_file(path)
            or manifest.get('schema') != 'gemma4-measured-learned-packets-v1'
            or manifest.get('selection_plan_sha256') != plan_sha256
            or [p['condition'] for p in packets] != expected):
        raise ValueError('complete measured packet grid in fixed condition order required')
    for packet in packets:
        condition = packet['condition']
        job = by_checkpoint[(condition['preset'], condition['completed_steps'])]
        identity = validate_conversion(packet['adapter'], packet['conversion_report'], job,
            strength=condition['strength'], probe_sha256=probe_sha256)
        if (packet['native'] != job['native']
                or packet['native_metadata_sha256'] != job['native_metadata_sha256']
                or packet['cache_index_sha256'] != job['cache_index_sha256']
                or sha256_file(Path(packet['cache']) / 'index.json') != job['cache_index_sha256']
                or packet['conversion_report_sha256'] != sha256_file(packet['conversion_report'])
                or packet['adapters'] != {'particle_lora': identity}):
            raise ValueError('actual measured native/cache/conversion/serving packet differs')
    return packets


def load_control_results(directory, selection, plan_sha256):
    directory = Path(directory)
    status = read(directory / 'progress.json')
    controls = [c for c in selection['conditions'] if c['role'] != 'learned']
    control_sha = baseline_selection_sha256(selection, plan_sha256)
    if (status.get('state') != 'complete' or status.get('planned_conditions') != 4
            or [r['condition_id'] for r in status['finished']] != [c['id'] for c in controls]):
        raise ValueError('all four complete baseline controls required')
    paths = []
    for condition, record in zip(controls, status['finished']):
        path = directory / condition['id'] / 'result.json'
        result = read(path)
        if (record['result_sha256'] != sha256_file(path)
                or result.get('status') != 'complete' or result['condition'] != condition
                or result['selection_plan_sha256'] != control_sha
                or result['primary']['total'] != 42 or result['secondary']['total'] != 13):
            raise ValueError('unchanged complete baseline result and fixed denominators required')
        paths.append(path.resolve())
    return paths


def candidate_preflight(client, root, identity, call):
    """One fixed synthetic completion; no DEV or holdout content or quality score.

    The caller journals every attempt before issuing HTTP and its actual response.
    Network observation failures propagate and never become an inferred rejection.
    """
    if call(client, root, '/health').get('identity_sha256') != digest(identity):
        raise ValueError('actual live server identity differs')
    names = {r['id'] for r in call(client, root, '/v1/models')['data']}
    if not {BASE_ALIAS, 'particle_lora'} <= names:
        raise ValueError('actual server does not advertise both base and learned aliases')
    tokens = call(client, root, '/tokenize', {'model': 'particle_lora',
        'prompt': 'Repository repair protocol control.\nReturn the word ready.\nAnswer:'})['tokens']
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('actual nonempty synthetic prompt token IDs required')
    response = call(client, root, '/v1/completions', {'model': 'particle_lora', 'prompt': tokens,
        'temperature': 0, 'seed': 0, 'max_tokens': 1, 'logprobs': 5,
        'return_tokens_as_token_ids': True, 'ignore_eos': True})
    if response.get('model') != 'particle_lora':
        raise ValueError('actual completion did not use the learned alias')
    probs = response['choices'][0]['logprobs']
    top = probs['top_logprobs'][0]
    if (len(probs['tokens']) != 1 or not top or probs['tokens'][0] not in top
            or not all(k.startswith('token_id:') and k[9:].isdigit() for k in top)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in top.values())):
        raise ValueError('one real finite token-ID completion required')
    return {'status': 'passed', 'prompt_token_ids': tokens, 'response_sha256': digest(response),
        'scope': 'synthetic alias loading only; no task quality or native conversion fidelity claim'}
