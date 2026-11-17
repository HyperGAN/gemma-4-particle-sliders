"""CPU packaging fixtures only: no pretrained model, real update or upload."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
from safetensors.torch import save_file

from gemma_particle.harness_identity import sha256_file

ROOT=Path(__file__).resolve().parents[1]


def module(name, file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fixture(tmp_path):
    # Pinned identity labels exercise file bindings; these tiny/random tensors
    # are deliberately not loadable as a full31B candidate or usable targets.
    config=json.loads((ROOT/'configs/31b-ct-shared-positional-pilot.json').read_text())
    identity=dict(repo=config['model'],revision=config['model_revision'],dtype='bfloat16',
        attn_implementation='sdpa',weight_format='ct-w4g32')
    adapter=tmp_path/'fixture-adapter';adapter.mkdir()
    generator=torch.Generator().manual_seed(0)
    save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':torch.randn(8,20000,generator=generator),
        'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':torch.randn(200,8,generator=generator)},
        adapter/'adapter_model.safetensors')
    (adapter/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA',task_type='CAUSAL_LM',r=8,lora_alpha=8,
        base_model_name_or_path=config['model'],target_modules=['q_proj'])))
    native=tmp_path/'fixture-native';native.mkdir()
    (native/'native.safetensors').write_bytes(b'CPU packaging fixture, not a pretrained checkpoint')
    (native/'native.json').write_text(json.dumps(dict(schema='gemma4-native-slider-v1',model_identity=identity,
        completed_steps=1,weights='raw',ema_decay=None,sha256=sha256_file(native/'native.safetensors'))))
    cache=tmp_path/'fixture-cache';cache.mkdir()
    (cache/'responses.safetensors').write_bytes(b'CPU packaging fixture, not real targets')
    (cache/'index.json').write_text(json.dumps(dict(metadata={'model_identity':identity},
        responses_sha256=sha256_file(cache/'responses.safetensors'))))
    conversion=tmp_path/'fixture-conversion.json'
    conversion.write_text(json.dumps(dict(schema='gemma4-routed-to-peft-v1',exact_merge=False,strength=1,
        adapter_sha256=sha256_file(adapter/'adapter_model.safetensors'),
        provenance=dict(native_metadata_sha256=sha256_file(native/'native.json'),cache_index_sha256=sha256_file(cache/'index.json')))))
    builder=module('candidate_packaging_builder','build_submission.py')
    artifact=tmp_path/'artifact'
    builder.build(artifact,builder.MODELS[0],adapter=adapter,prompt_profile='repair-finalize-v2')
    assert (artifact/'submission.zip').stat().st_size>512*1024
    return artifact,native,cache,conversion


def test_large_candidate_dataset_notebook_preserves_exact_zip_and_rejects_corruption(tmp_path,monkeypatch):
    artifact,native,cache,conversion=fixture(tmp_path)
    preparer=module('candidate_dataset_preparer','prepare_kaggle_candidate.py')
    out=tmp_path/'prepared'
    argv=['prepare','--artifact',str(artifact),'--native',str(native),'--cache',str(cache),
        '--conversion-report',str(conversion),'--owner','fixture-owner',
        '--notebook-slug','fixture-candidate','--out',str(out)]
    monkeypatch.setattr(sys,'argv',argv);preparer.main()
    summary=json.loads((out/'preparation.json').read_text())
    assert summary['dataset'].endswith(summary['artifact_sha256'][:12])
    kernel=json.loads((out/'notebook/kernel-metadata.json').read_text())
    assert kernel['is_private'] and not kernel['enable_gpu'] and not kernel['enable_internet']
    assert kernel['dataset_sources']==[summary['dataset']]
    inputs=tmp_path/'input';owner,slug=summary['dataset'].split('/')
    dataset=inputs/'datasets'/owner/slug
    shutil.copytree(out/'dataset',dataset)
    working=tmp_path/'working'
    monkeypatch.setenv('KAGGLE_INPUT_DIR',str(inputs));monkeypatch.setenv('KAGGLE_WORKING_DIR',str(working))
    document=json.loads((out/'notebook/candidate.ipynb').read_text())
    exec(document['cells'][1]['source'],{})
    assert (working/'submission.zip').read_bytes()==(artifact/'submission.zip').read_bytes()
    payload=dataset/'agent_bundle.bin';payload.write_bytes(payload.read_bytes()+b'changed')
    monkeypatch.setenv('KAGGLE_WORKING_DIR',str(tmp_path/'corrupted-output'))
    with pytest.raises(AssertionError,match='Dataset bytes differ'):
        exec(document['cells'][1]['source'],{})
    assert not (tmp_path/'corrupted-output/submission.zip').exists()


def test_changed_native_provenance_cannot_prepare_a_candidate(tmp_path,monkeypatch):
    artifact,native,cache,conversion=fixture(tmp_path)
    (native/'native.safetensors').write_bytes(b'changed')
    preparer=module('candidate_bad_provenance','prepare_kaggle_candidate.py')
    out=tmp_path/'rejected'
    monkeypatch.setattr(sys,'argv',['prepare','--artifact',str(artifact),'--native',str(native),'--cache',str(cache),
        '--conversion-report',str(conversion),'--owner','fixture-owner','--notebook-slug','fixture-candidate','--out',str(out)])
    with pytest.raises(ValueError,match='native/cache bytes differ'):preparer.main()
    assert not out.exists()
