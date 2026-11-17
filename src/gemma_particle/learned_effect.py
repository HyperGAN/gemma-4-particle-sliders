"""Observe learned-versus-base effects on fixed disjoint TRAIN prefixes.

Local artifacts bind an observation; they cannot authenticate fabricated data.
An effect is neither a task-quality score nor native-conversion fidelity.
"""
import math
import re
import json
from pathlib import Path

from .conversion_handoff import read
from .data import digest
from .dev_dispatch import candidate_preflight
from .harness_identity import sha256_file
from .lora_serving import BASE_ALIAS


GENERATION = {'temperature': 0, 'seed': 0, 'max_tokens': 1, 'logprobs': 20,
    'ignore_eos': True, 'return_tokens_as_token_ids': True}
TOLERANCE = 1e-5


def prepare_effect_probes(protocol_path, contexts_path, qualification_path, selection, selection_sha256):
    protocol, contexts, qualification = map(read, (protocol_path, contexts_path, qualification_path))
    entries = contexts['entries']; ids = [e['instance_id'] for e in entries]
    lengths = [len(e['neutral_ids']) for e in entries]
    if (protocol.get('schema') != 'gemma4-learned-effect-protocol-v1'
            or protocol['core_selection_plan_sha256'] != selection_sha256
            or protocol['probe_contexts_sha256'] != sha256_file(contexts_path)
            or protocol['probe_qualification_sha256'] != sha256_file(qualification_path)
            or protocol['aliases'] != [BASE_ALIAS, 'particle_lora']
            or protocol['generation'] != GENERATION or protocol['tolerance'] != TOLERANCE
            or protocol['repeats_per_alias'] != 2 or protocol['required_effect_probes'] != 1
            or len(ids) != 2 or len(set(ids)) != 2 or ids != protocol['probe_task_ids']
            or lengths != protocol['probe_token_counts']
            or contexts.get('schema') != 'gemma4-conversion-probe-v1'
            or contexts.get('purpose') != 'conversion diagnostic only; forbidden as a training target'
            or contexts['train_manifest_sha256'] != selection['training']['train_manifest_sha256']
            or qualification.get('status') != 'passed' or qualification['probe_sha256'] != sha256_file(contexts_path)
            or qualification['train_manifest_sha256'] != contexts['train_manifest_sha256']
            or qualification['task_ids'] != ids or qualification['token_counts'] != lengths
            or qualification['request_sha256'] != [e['request_sha256'] for e in entries]):
        raise ValueError('fixed qualified TRAIN effect protocol and original probe bytes required')
    cases = []
    for entry in entries:
        tokens = entry['neutral_ids']
        if (not tokens or len(tokens) > 32768 or any(type(t) is not int or t < 0 for t in tokens)
                or digest(tokens) != entry['neutral_ids_sha256']
                or digest(entry['request']) != entry['request_sha256']
                or [m.get('role') for m in entry['request'].get('messages', [])] != ['system', 'user']):
            raise ValueError('unchanged initial TRAIN prompt and exact token IDs required')
        cases.append({'instance_id': entry['instance_id'], 'prompt_token_ids': tokens,
            'prompt_token_ids_sha256': digest(tokens), 'prompt_tokens': len(tokens)})
    return protocol, cases


def _distribution(response, alias, length):
    usage = response.get('usage', {}); choices = response.get('choices', [])
    if (response.get('model') != alias or len(choices) != 1
            or usage.get('prompt_tokens') != length or usage.get('completion_tokens') != 1):
        raise ValueError('actual requested alias, full prefix and one completion token required')
    probabilities = choices[0]['logprobs']; tokens = probabilities['tokens']; top = probabilities['top_logprobs']
    if (len(tokens) != 1 or len(top) != 1 or len(top[0]) != 20 or tokens[0] not in top[0]
            or any(not isinstance(k, str) or re.fullmatch(r'token_id:\d+', k) is None for k in top[0])
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in top[0].values())):
        raise ValueError('one generated token ID and twenty finite top logprobs required')
    return tokens[0], top[0]


def _difference(left, right):
    token_a, probs_a = left; token_b, probs_b = right
    common = probs_a.keys() & probs_b.keys()
    delta = max((abs(probs_a[k] - probs_b[k]) for k in common), default=None)
    same_set = probs_a.keys() == probs_b.keys()
    changed = token_a != token_b or not same_set or delta is not None and delta > TOLERANCE
    return {'generated_token_ids': [token_a, token_b], 'generated_token_equal': token_a == token_b,
        'top_token_id_sets_equal': same_set, 'common_top_tokens': len(common),
        'max_abs_common_logprob_delta': delta, 'differs_beyond_tolerance': bool(changed)}


def learned_effect_preflight(client, root, identity, packet, cases, call):
    """Journaled actual calls only; observation errors propagate without retries."""
    if (identity.get('base_only') is not False or identity.get('adapters') != packet['adapters']
            or set(packet['adapters']) != {'particle_lora'}):
        raise ValueError('actual exact learned packet on a dual-alias server required')
    records = []
    for case in cases:
        if (len(case['prompt_token_ids']) != case['prompt_tokens']
                or digest(case['prompt_token_ids']) != case['prompt_token_ids_sha256']):
            raise ValueError('fixed complete TRAIN token prefix changed')
        distributions = {}; repeats = {}; responses = {}; requests = {}
        for alias in [BASE_ALIAS, 'particle_lora']:
            payload = {'model': alias, 'prompt': case['prompt_token_ids'], **GENERATION}
            values = []; parsed = []
            for _ in range(2):
                value = call(client, root, '/v1/completions', payload)
                parsed.append(_distribution(value, alias, case['prompt_tokens'])); values.append(value)
            repeats[alias] = _difference(*parsed)
            distributions[alias] = parsed[0]
            responses[alias] = [digest(value) for value in values]; requests[alias] = digest(payload)
        records.append({'instance_id': case['instance_id'], 'prompt_tokens': case['prompt_tokens'],
            'prompt_token_ids_sha256': case['prompt_token_ids_sha256'], 'repeat_comparisons': repeats,
            'base_vs_learned': _difference(distributions[BASE_ALIAS], distributions['particle_lora']),
            'request_sha256': requests, 'response_sha256': responses})
    stable = all(not r['differs_beyond_tolerance'] for c in records for r in c['repeat_comparisons'].values())
    effect_count = sum(c['base_vs_learned']['differs_beyond_tolerance'] for c in records)
    return {'schema': 'gemma4-learned-effect-observation-v1',
        'status': 'unstable_observation' if not stable else 'passed' if effect_count else 'no_observed_effect',
        'stable_repeats': stable, 'effect_probes': effect_count, 'probe_count': len(records), 'cases': records,
        'tolerance': TOLERANCE, 'generation': GENERATION, 'server_identity_sha256': digest(identity),
        'condition': packet['condition'], 'adapters': packet['adapters'],
        'native_metadata_sha256': packet['native_metadata_sha256'], 'cache_index_sha256': packet['cache_index_sha256'],
        'conversion_report_sha256': packet['conversion_report_sha256'],
        'scope': 'fixed disjoint TRAIN prefix effect only; no DEV score,native fidelity,confirmation or official result'}


def audit_learned_effect(path, journal, identity, packet, cases, protocol_sha256):
    """Recompute alias and effect observations from every original HTTP exchange."""
    entries = [json.loads(line) for line in Path(journal).read_text().splitlines()]
    cursor = 0
    def replay(client, root, endpoint, payload=None):
        nonlocal cursor
        if cursor + 2 > len(entries):
            raise ValueError('complete original learned preflight HTTP exchanges required')
        attempt, response = entries[cursor:cursor + 2]; cursor += 2
        if (attempt != {'phase': 'attempt', 'path': endpoint, 'request': payload}
                or response.get('phase') != 'response' or response.get('path') != endpoint
                or response.get('status_code') != 200):
            raise ValueError('exact original successful learned preflight request/response order required')
        return response['response']
    candidate_preflight(None, 'offline-journal', identity, replay)
    expected = learned_effect_preflight(None, 'offline-journal', identity, packet, cases, replay)
    expected.update(protocol_sha256=protocol_sha256, http_journal_sha256=sha256_file(journal))
    if cursor != len(entries) or expected != read(path) or expected['status'] != 'passed':
        raise ValueError('actual complete stable learned-effect observation and unchanged journal required')
    return expected
