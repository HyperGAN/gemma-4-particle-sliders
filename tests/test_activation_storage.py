from copy import deepcopy
import gc
import weakref

import pytest
import torch

from gemma_particle.activation_storage import CPUActivationStorage
from gemma_particle.model import Native, ContextRow, make_game
from test_gemma_binding import tiny_model, assert_equal


def test_storage_deduplicates_same_live_version_and_releases_source():
    value=torch.arange(12.,requires_grad=True).reshape(3,4).t()
    source=weakref.ref(value)
    store=CPUActivationStorage()
    a=store.pack(value);b=store.pack(value)
    assert a is b and a.value.device.type=='cpu' and a.value.data_ptr()!=value.data_ptr()
    assert torch.equal(store.unpack(a),value)
    assert a.value.dtype==value.dtype and a.value.shape==value.shape
    assert store.stats=={'saves':2,'copies':1,'deduplicated_saves':1,'copied_bytes':48}
    del value
    gc.collect()
    assert source() is None
    assert torch.equal(store.unpack(a),torch.arange(12.).reshape(3,4).t())


def test_live_inplace_mutation_is_rejected_and_hooks_restore_after_error():
    value=torch.arange(4.,requires_grad=True)
    with CPUActivationStorage():
        loss=value.square().sum()
    with torch.no_grad():value.add_(1)
    with pytest.raises(RuntimeError,match='modified in place'):loss.backward()
    with pytest.raises(ValueError,match='forward failure'):
        with CPUActivationStorage():raise ValueError('forward failure')
    # A later ordinary autograd save retains PyTorch's own version check.
    other=torch.ones(2,requires_grad=True);loss=other.square().sum()
    with torch.no_grad():other.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):loss.backward()


@pytest.mark.parametrize('storage',['device','cpu'])
@pytest.mark.parametrize('checkpoint_profile',['block','nested_attention_mlp','nested_positional_attention_mlp'])
def test_checkpointed_native_matches_direct_full_gradient_and_deferred_scale(storage,checkpoint_profile):
    direct=Native(tiny_model(),identity={'fixture':'activation-storage'})
    wrapped=Native(tiny_model(),identity=direct.identity,checkpointing=True,activation_storage=storage,
                   checkpoint_profile=checkpoint_profile)
    try:
        with torch.no_grad():
            for _,_,adapter in direct.slider.hosts:adapter.lora_up.weight.fill_(.01)
            wrapped.network.load_state_dict(direct.network.state_dict())
        before=deepcopy(wrapped.model.state_dict())
        a=direct.hidden((2,3,4,5),strength=.75)
        b=wrapped.hidden((2,3,4,5),strength=.75)
        assert torch.equal(a,b)
        cotangent=torch.linspace(-1,1,direct.width).reshape_as(a)
        expected=torch.autograd.grad(a,tuple(direct.network.parameters()),cotangent,allow_unused=True)
        # Recompute occurs under a different ambient strength after forward.
        with wrapped.network.scaled(.125):
            actual=torch.autograd.grad(b,tuple(wrapped.network.parameters()),cotangent,allow_unused=True)
            assert wrapped.network.scale==.125
        for left,right in zip(expected,actual):
            assert (left is None)==(right is None)
            if left is not None:torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-8)
        assert_equal(before,wrapped.model.state_dict())
        if storage=='cpu':assert wrapped.last_activation_storage['copied_bytes']>0
    finally:direct.close();wrapped.close()
    assert all(not getattr(layer,'_adapter_scale_checkpoint',False) for layer in wrapped.trunk.layers)
    assert all(not getattr(m,'_adapter_scale_checkpoint',False) for layer in wrapped.trunk.layers for m in (layer.self_attn,layer.mlp))


@pytest.mark.parametrize('preset',['rhbi_optimized','empf'])
@pytest.mark.parametrize('checkpoint_profile',['block','nested_attention_mlp','nested_positional_attention_mlp'])
def test_cpu_storage_recipe_update_and_exact_continuation(preset,checkpoint_profile):
    natives=[]
    try:
        games=[]
        for storage,profile in [('device','block'),('cpu',checkpoint_profile),('cpu',checkpoint_profile)]:
            native=Native(tiny_model(),identity={'fixture':'activation-game'},checkpointing=True,activation_storage=storage,
                          checkpoint_profile=profile)
            natives.append(native)
            rows=[native.prepare(ContextRow((2,3,4+i),(2,8,9,3,4+i),str(i),0)) for i in range(2)]
            games.append(make_game(native,rows,preset=preset,steps=3,batch=2))
        assert_equal(games[0].step(),games[1].step())
        assert_equal(natives[0].network.state_dict(),natives[1].network.state_dict())
        packet=games[1].state_dict();games[2].load_state_dict(packet)
        assert_equal(games[1].step(),games[2].step())
        assert_equal(games[1].state_dict(),games[2].state_dict())
        with pytest.raises(ValueError):games[0].load_state_dict(packet)
    finally:
        for native in natives:native.close()


@pytest.mark.parametrize('nested_profile',['nested_attention_mlp','nested_positional_attention_mlp'])
def test_nested_profile_full_resume_is_rejected_by_block_profile_before_mutation(nested_profile):
    natives=[]
    try:
        games=[]
        for profile in ('block',nested_profile):
            native=Native(tiny_model(),identity={'fixture':'checkpoint-profile'},checkpointing=True,
                          activation_storage='cpu',checkpoint_profile=profile)
            natives.append(native)
            rows=[native.prepare(ContextRow((2,3,4+i),(2,8,9,3,4+i),str(i),0)) for i in range(2)]
            games.append(make_game(native,rows,preset='empf',steps=3,batch=2))
        assert_equal(games[0].step(),games[1].step())
        assert_equal(natives[0].network.state_dict(),natives[1].network.state_dict())
        before=games[0].state_dict()
        with pytest.raises(ValueError):games[0].load_state_dict(games[1].state_dict())
        assert_equal(before,games[0].state_dict())
    finally:
        for native in natives:native.close()


def test_keyword_input_lifetime_is_released_by_binding_without_changing_gradient():
    from contextlib import contextmanager
    from torch import nn
    from particle_sliders.backends import checkpoint_with_adapter_scale
    from gemma_particle.checkpoint_inputs import checkpoint_with_positional_inputs
    class Network:
        scale=1.
        @contextmanager
        def scaled(self,scale):yield
    class Block(nn.Module):
        def forward(self,hidden_states):return hidden_states.square().sum()
    gradients=[]
    for canonicalized in (False,True):
        block=Block();original=block.forward
        install=checkpoint_with_positional_inputs if canonicalized else checkpoint_with_adapter_scale
        handle=install(block,Network())
        try:
            parameter=torch.tensor(2.,requires_grad=True)
            value=parameter*torch.arange(1.,5.);source=weakref.ref(value)
            with CPUActivationStorage():result=block(hidden_states=value)
            del value;gc.collect()
            assert (source() is None)==canonicalized
            gradients.append(torch.autograd.grad(result,parameter)[0])
        finally:handle.close()
        assert block.forward==original and not getattr(block,'_adapter_scale_checkpoint',False)
    assert torch.equal(gradients[0],gradients[1])
