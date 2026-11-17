"""TRAIN-only issue prompts for disjoint conversion diagnostics, never targets."""
from .data import digest
from .model import ContextRow
from .transport import weight_identity


def initial_off_request(outcome, train_ids):
    task=outcome.get('instance_id')
    request=outcome.get('first_request',{})
    verification=outcome.get('verification',{})
    if (outcome.get('schema')!='gemma4-verified-trajectory-v4'
            or task not in train_ids or outcome.get('split')!='train'
            or outcome.get('arm')!='off' or outcome.get('generator_kind')!='gemma4'
            or outcome.get('frozen_base_generator') is not True
            or outcome.get('resolved') is not True or outcome.get('capture_complete') is not True
            or verification.get('stage')!='fresh-container-tests'
            or verification.get('instance_id')!=task or verification.get('resolved') is not True
            or verification.get('first_request_sha256')!=digest(request)
            or verification.get('requests_sha256')!=digest(outcome.get('requests'))
            or not outcome.get('requests') or outcome['requests'][0].get('request')!=request
            or [m.get('role') for m in request.get('messages',[])]!=['system','user']):
        raise ValueError('conversion probe requires a bound, complete passing TRAIN OFF initial issue prompt')
    return request


def validate_probe(packet, *, response_identity, train_ids, fit_ids):
    server=packet.get('server_identity',{})
    generator=server.get('model_identity',{})
    if (packet.get('schema')!='gemma4-conversion-probe-v1'
            or packet.get('purpose')!='conversion diagnostic only; forbidden as a training target'
            or packet.get('response_model_identity')!=response_identity
            or packet.get('server_identity_sha256')!=digest(server)
            or server.get('kind')!='gemma4' or server.get('pretrained_inference') is not True
            or server.get('strength')!=0 or server.get('native_checkpoint') is not None
            or generator.get('attn_implementation')!='vllm:auto'
            or weight_identity(generator)!=weight_identity(response_identity)):
        raise ValueError('conversion probe model/transport identity differs')
    rows=[];seen=set()
    for entry in packet.get('entries',[]):
        task=entry.get('instance_id');ids=entry.get('neutral_ids',[])
        if (task not in train_ids or task in fit_ids or task in seen
                or not ids or len(ids)>32768 or any(type(t) is not int or t<0 for t in ids)
                or entry.get('neutral_ids_sha256')!=digest(ids)
                or entry.get('request_sha256')!=digest(entry.get('request'))
                or [m.get('role') for m in entry.get('request',{}).get('messages',[])]!=['system','user']):
            raise ValueError('conversion probe changed, overlaps fit tasks, or contains non-TRAIN history')
        seen.add(task);rows.append(ContextRow(tuple(ids),(),task,0))
    if not rows:raise ValueError('nonempty conversion probe required')
    return rows
