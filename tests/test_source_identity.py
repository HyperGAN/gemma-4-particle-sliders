import pytest

from gemma_particle import source_identity
from gemma_particle.model import Native
from gemma_particle.runtime import load_native
from test_gemma_binding import tiny_model


def test_wrong_environment_fails_before_freezing_or_wrapping_the_model(monkeypatch):
    model=tiny_model();flags=[p.requires_grad for p in model.parameters()]
    forwards=[layer.self_attn.q_proj.forward for layer in model.model.layers]
    monkeypatch.setattr(source_identity,'version',lambda name:'0.10.0')
    with pytest.raises(RuntimeError,match='environment does not match'):
        Native(model,identity={'fixture':'wrong-source'})
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [layer.self_attn.q_proj.forward for layer in model.model.layers]==forwards


def test_wrong_environment_fails_before_loading_pretrained_weights(monkeypatch):
    from transformers import AutoModelForImageTextToText
    def forbidden(*args,**kwargs):raise AssertionError('pretrained load reached before source admission')
    monkeypatch.setattr(AutoModelForImageTextToText,'from_pretrained',forbidden)
    monkeypatch.setattr(source_identity,'version',lambda name:'0.10.0')
    with pytest.raises(RuntimeError,match='environment does not match'):
        load_native({},device='cuda:1')


def test_changed_shared_engine_is_rejected_even_with_matching_version(monkeypatch,tmp_path):
    from gemma_particle import response_engine
    changed=tmp_path/'response_engine.py';changed.write_text('# changed engine\n')
    monkeypatch.setattr(response_engine,'__file__',str(changed))
    with pytest.raises(RuntimeError,match='locked bridge'):source_identity.require_core_source()
