"""Bind a packaged PEFT adapter to its exact discovered and served alias."""
import hashlib
import json
from pathlib import Path


def adapter_identity(directory):
    directory=Path(directory)
    names=('adapter_config.json','adapter_model.safetensors')
    if any((directory/name).is_symlink() or not (directory/name).is_file() for name in names):
        raise ValueError('regular PEFT config and safetensors files required')
    config=json.loads((directory/names[0]).read_text())
    rank=config.get('r')
    if config.get('peft_type')!='LORA' or type(rank) is not int or not 1<=rank<=128:
        raise ValueError('PEFT LoRA rank one to 128 required')
    if not config.get('base_model_name_or_path'):
        raise ValueError('PEFT base model identity required')
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    return {'base_model_name_or_path':config['base_model_name_or_path'],'rank':rank,
            'files':{name:sha(directory/name) for name in names}}


def register_packaged_adapter(models,manifest,base_llm,*,server_adapters=None,fixture=False):
    """Use official adapter discovery/registration with no base-alias fallback.

    This first candidate cohort deliberately has one packaged particle_lora.
    The live server must bind its alias to the same config/weight bytes.
    """
    if set(manifest.adapters)!={'particle_lora'}:
        raise ValueError('exactly one discovered particle_lora candidate required')
    identities={}
    for name,info in manifest.adapters.items():
        if info.path.name!='adapter_model.safetensors' or info.name!=name:
            raise ValueError('canonical PEFT adapter discovery required')
        identities[name]=adapter_identity(info.path.parent)
        if info.rank!=identities[name]['rank']:
            raise ValueError('discovered adapter rank differs from its config')
    if not fixture and server_adapters!=identities:
        raise ValueError('live server adapter aliases or weight/config bytes differ')
    manifest.register_all(models,lambda info:base_llm.model_copy(
        update={'model':'openai/'+info.name},deep=False))
    return identities
