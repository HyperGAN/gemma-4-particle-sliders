"""Numerical resource qualification; never relax the active DEV contract."""
import math
import re

from .conversion_handoff import read
from .data import digest
from .dev_evidence import canonical_contract
from .harness_identity import sha256_file
from .lora_serving import BASE_ALIAS, BASE_REPO, BASE_REVISION

LENGTHS = [32, 4096, 28672]
ALIASES = [BASE_ALIAS, 'particle_lora']
GENERATION = {'temperature': 0, 'seed': 0, 'max_tokens': 1, 'logprobs': 20,
    'ignore_eos': True, 'return_tokens_as_token_ids': True}
TOLERANCE = 1e-5


def prepare_probes(inputs, proposal, selection_sha256):
    packet, resource = read(inputs), read(proposal)
    seed = packet['seed_token_ids']
    if (packet.get('schema') != 'gemma4-hardware-probe-inputs-v1'
            or resource.get('schema') != 'gemma4-dev-hardware-resource-proposal-v1'
            or packet['proposal_sha256'] != sha256_file(proposal)
            or resource['core_selection_plan_sha256'] != selection_sha256
            or packet['model'] != BASE_REPO or packet['model_revision'] != BASE_REVISION
            or packet['lengths'] != LENGTHS or resource['probe_lengths'] != LENGTHS
            or packet['aliases'] != ALIASES or resource['aliases'] != ALIASES
            or packet['generation'] != GENERATION or resource['generation'] != GENERATION
            or {k: resource['probe_adapter'][k] for k in ['preset', 'completed_steps', 'strength']}
                != {'preset': 'empf', 'completed_steps': 256, 'strength': 1}
            or not seed or any(type(t) is not int or t < 0 for t in seed)):
        raise ValueError('original prospective fixed model/adapter/token/generation protocol required')
    cases = []
    for length in LENGTHS:
        tokens = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
        for alias in ALIASES:
            payload = {'model': alias, 'prompt': tokens, **GENERATION}
            cases.append({'length': length, 'alias': alias, 'request': payload, 'request_sha256': digest(payload)})
    return packet, resource, cases


def qualify_server_contracts(servers, baseline, adapters):
    """Compare serving facts for this numerical control only; keep devices in DEV.

    Actual physical bindings are separately checked against GPU process facts.
    No source object is mutated, and canonical_contract itself is unchanged.
    """
    normalized = []
    for i, server in enumerate([*servers, baseline]):
        if i < 2 and (server.get('device') != f'cuda:{i}' or server.get('base_only') is not False
                or server.get('adapters') != adapters):
            raise ValueError('exact same measured candidate on physical GPU0 and GPU1 required')
        if (server['memory_fraction'] != .6 or server['observed_free_bytes'] < server['total_device_bytes'] * .6 + 8 * 1024**3):
            raise ValueError('recorded shared .60 target and eight GiB startup headroom required')
        contract = canonical_contract({'model_identity': server})
        contract['model_identity'].pop('device')
        normalized.append(contract)
    if normalized[0] != normalized[1] or normalized[0] != normalized[2]:
        raise ValueError('actual serving implementations, packages, weights or engine settings differ')
    return digest(normalized[0])


def validate_device_facts(facts, owned_applications, resource):
    if len(facts) != 2 or len(owned_applications) != 2 or len(resource['observed_slots']) != 2:
        raise ValueError('both actual physical GPU observations required')
    if len({f['uuid'] for f in facts}) != 2:
        raise ValueError('two distinct physical GPU UUIDs required')
    for index, (fact, expected, applications) in enumerate(zip(facts, resource['observed_slots'], owned_applications)):
        if ({k: fact[k] for k in expected} != expected or fact['physical_index'] != index
                or not applications or any(a['gpu_uuid'] != fact['uuid'] for a in applications)):
            raise ValueError('actual owned GPU processes or original hardware descriptors differ')
    return True


def compare_probe_responses(case, left, right):
    if (len(case['request']['prompt']) != case['length'] or case['request']['model'] != case['alias']
            or digest(case['request']) != case['request_sha256']):
        raise ValueError('exact original requested token IDs and alias required')
    parsed = []
    for response in [left, right]:
        usage = response.get('usage', {})
        choices = response.get('choices', [])
        if (response.get('model') != case['alias'] or len(choices) != 1
                or usage.get('prompt_tokens') != case['length'] or usage.get('completion_tokens') != 1):
            raise ValueError('actual requested alias/input length and one generated token required')
        probs = choices[0]['logprobs']; tokens = probs['tokens']; top = probs['top_logprobs']
        if (len(tokens) != 1 or len(top) != 1 or len(top[0]) != 20 or tokens[0] not in top[0]
                or any(not isinstance(k, str) or re.fullmatch(r'token_id:\d+', k) is None for k in top[0])
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in top[0].values())):
            raise ValueError('one exact generated token ID and twenty finite top logprobs required')
        parsed.append((tokens[0], top[0]))
    (a, pa), (b, pb) = parsed
    same_ids = pa.keys() == pb.keys()
    delta = max(abs(pa[k] - pb[k]) for k in pa) if same_ids else None
    return {'length': case['length'], 'alias': case['alias'], 'request_sha256': case['request_sha256'],
        'generated_token_ids': [a, b], 'generated_token_equal': a == b, 'top_token_id_sets_equal': same_ids,
        'max_abs_logprob_delta': delta, 'tolerance': TOLERANCE,
        'passed': a == b and same_ids and delta is not None and delta <= TOLERANCE,
        'response_sha256': [digest(left), digest(right)]}
