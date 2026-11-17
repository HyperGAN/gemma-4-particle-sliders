"""Compare CPU strength derivation to the independent native export path."""
import json

import pytest
from safetensors.torch import load_file
import torch

from gemma_particle.export import collect_activations,export_peft
from gemma_particle.lora_strengths import derive_peft_strength
from gemma_particle.model import ContextRow,Native,make_game
from test_gemma_binding import tiny_model


def test_derived_doses_equal_fresh_native_exports_and_reject_tampering(tmp_path):
    native=Native(tiny_model(),identity={'fixture':'CPU random Gemma4'})
    try:
        rows=[ContextRow((2,3,4+i),(2,8,9,3,4+i),f'fixture-{i}',0) for i in range(4)]
        game=make_game(native,[native.prepare(r) for r in rows],preset='empf',steps=1,batch=2)
        game.step()
        fit=collect_activations(native,rows[:2],tokens_per_layer=5)
        probe=collect_activations(native,rows[2:],tokens_per_layer=5)
        provenance={'fit_task_ids':['fixture-0','fixture-1'],'probe_task_ids':['fixture-2','fixture-3'],
            'native_metadata_sha256':'a'*64,'cache_index_sha256':'b'*64}
        source=tmp_path/'source'
        unit=export_peft(native,fit,probe,source,base_model_name='CPU-fixture',strength=1,provenance=provenance)
        original=(source/'adapter_model.safetensors').read_bytes()
        source_weights=load_file(source/'adapter_model.safetensors')
        assert any(torch.count_nonzero(v) for k,v in source_weights.items() if k.endswith('.lora_B.weight'))
        for strength in [.5,1.5]:
            derived=tmp_path/f'derived-{strength}';direct=tmp_path/f'direct-{strength}'
            receipt=derive_peft_strength(source,derived,strength=strength)
            export_peft(native,fit,probe,direct,base_model_name='CPU-fixture',strength=strength,provenance=provenance)
            observed,expected=load_file(derived/'adapter_model.safetensors'),load_file(direct/'adapter_model.safetensors')
            assert observed.keys()==expected.keys()
            assert all(torch.equal(observed[k],expected[k]) for k in observed)
            assert any(not torch.equal(observed[k],source_weights[k]) for k in observed if k.endswith('.lora_B.weight'))
            assert (derived/'adapter_config.json').read_bytes()==(direct/'adapter_config.json').read_bytes()
            assert receipt['provenance']==unit['provenance'] and receipt['layer_metrics']==unit['layer_metrics']
            assert receipt['strength']==strength and receipt['strength_derivation']['source_strength']==1
            with pytest.raises(FileExistsError):derive_peft_strength(source,derived,strength=strength)
        assert (source/'adapter_model.safetensors').read_bytes()==original
        with pytest.raises(ValueError,match='unit-strength'):
            derive_peft_strength(derived,tmp_path/'double-dose',strength=.5)
        report=source.with_name(source.name+'-conversion.json');damaged=json.loads(report.read_text())
        damaged['adapter_sha256']='c'*64;report.write_text(json.dumps(damaged))
        with pytest.raises(ValueError,match='unit-strength'):derive_peft_strength(source,tmp_path/'damaged',strength=.5)
    finally:native.close()
