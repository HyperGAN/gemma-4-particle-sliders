"""Bind official vLLM prompt IDs to a separate native response computation.

Only input tokens transfer. No vLLM activations or numerical equivalence claim
enters a target: native HF computes both neutral and positive hidden states.
"""
from .data import digest


def weight_identity(identity):
    return {k:identity.get(k) for k in ('repo','revision','dtype','weight_format')}


def validate_transport(transport,contexts,*,contexts_sha256,response_identity):
    generator=transport.get('generator_model_identity',{})
    if (transport.get('schema')!='gemma4-prompt-transport-v1'
            or transport.get('contexts_sha256')!=contexts_sha256
            or transport.get('contexts_digest')!=digest(contexts)
            or transport.get('response_model_identity')!=response_identity):
        raise ValueError('prompt transport differs from contexts or native response identity')
    if (generator.get('attn_implementation')!='vllm:auto'
            or generator.get('repo')!='google/gemma-4-31B-it-qat-w4a16-ct'
            or generator.get('revision')!='52f3f65bc7a02d555763bc923bd1d9094898219d'
            or generator.get('weight_format')!='ct-w4g32'
            or generator.get('dtype')!='bfloat16'
            or response_identity.get('attn_implementation')!='sdpa'
            or weight_identity(generator)!=weight_identity(response_identity)):
        raise ValueError('only matching pinned 31B CT weights permit this explicit prompt transport')
    if not contexts or any(r.get('model_identity')!=generator for r in contexts):
        raise ValueError('transport generator differs from trajectory provenance')
    server=transport.get('server_identity',{})
    if (server.get('model_identity')!=generator or server.get('strength')!=0
            or server.get('native_checkpoint') is not None
            or transport.get('server_identity_sha256')!=digest(server)):
        raise ValueError('source-bound frozen generator identity required')
    entries=transport.get('entries',[])
    if len(entries)!=len(contexts):raise ValueError('one tokenized pair per context required')
    for row,entry in zip(contexts,entries):
        if (entry.get('instance_id')!=row['instance_id'] or entry.get('decision')!=row['decision']
                or entry.get('context_sha256')!=digest(row)):
            raise ValueError('transport context order or bytes changed')
        for side in ('neutral','positive'):
            ids=entry.get(side+'_ids',[])
            if (not ids or len(ids)>32768 or any(type(i) is not int or i<0 for i in ids)
                    or entry.get(side+'_ids_sha256')!=digest(ids)):
                raise ValueError('invalid or changed transported token IDs')
    return entries
