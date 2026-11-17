import importlib.util
import os

import pytest
import torch
from torch.nn import functional as F

pytestmark=pytest.mark.skipif(importlib.util.find_spec('compressed_tensors') is None,
                            reason='official CT wheel is a separate optional input')


def fixture():
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationScheme
    from gemma_particle.ct_linear import FrozenCTLinear
    scheme=QuantizationScheme.model_validate({'targets':['Linear'],'format':'pack-quantized',
        'weights':{'num_bits':4,'type':'int','strategy':'group','group_size':32,'symmetric':True}})
    g=torch.Generator().manual_seed(0)
    dense=torch.randn(16,64,generator=g)
    scale=dense.reshape(16,2,32).abs().amax(-1)/7
    state=PackedQuantizationCompressor.compress({'weight':dense,'weight_scale':scale},scheme)
    expected=PackedQuantizationCompressor.decompress(state,scheme)['weight']
    return FrozenCTLinear(state,scheme),expected


def test_packed_forward_and_input_gradient_equal_public_dense_matrix():
    module,weight=fixture()
    frozen={k:v.clone() for k,v in module.state_dict().items()}
    x=torch.linspace(-1,1,2*3*64).reshape(2,3,64).requires_grad_()
    a=module(x);b=F.linear(x,weight)
    assert torch.equal(a,b)
    cotangent=torch.linspace(-.5,.5,a.numel()).reshape_as(a)
    ga=torch.autograd.grad(a,x,cotangent)[0]
    gb=torch.autograd.grad(b,x,cotangent)[0]
    assert torch.equal(ga,gb)
    assert not list(module.parameters())
    assert all(torch.equal(v,module.state_dict()[k]) for k,v in frozen.items())
    assert module.weight.numel()==0 and module.weight_packed.numel()*8==weight.numel()


def test_exact_derivative_and_no_saved_dense_matrix():
    module,_=fixture()
    x=torch.linspace(-1,1,64,dtype=torch.float64).reshape(1,64).requires_grad_()
    assert torch.autograd.gradcheck(module,(x,),eps=1e-6,atol=1e-6,rtol=1e-6)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append((tuple(t.shape),t.dtype)) or t),lambda t:t):
        module(x).sum().backward()
    assert ((16,64),torch.float64) not in saved
    assert ((16,8),torch.int32) in saved


@pytest.mark.parametrize('checkpoint_profile',['none','nested_attention_mlp'])
@pytest.mark.parametrize('packed_storage',['device','cpu'])
def test_native_quantized_gemma_forward_and_adapter_gradients_match_dense_decompression(tmp_path,checkpoint_profile,packed_storage):
    from copy import deepcopy
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from gemma_particle.ct_linear import replace_compressed_linears
    from compressed_tensors.quantization import QuantizationConfig
    from transformers import AutoModelForCausalLM,CompressedTensorsConfig
    from gemma_particle.model import Native
    from test_gemma_binding import tiny_model
    template,_=fixture();scheme=template.quantization_scheme
    model=tiny_model();reference=deepcopy(model)
    for name,linear in list(model.named_modules()):
        if type(linear) is not torch.nn.Linear or not name.startswith('model.layers.'):continue
        weight=linear.weight.detach();groups=weight.reshape(weight.shape[0],-1,32)
        scale=groups.abs().amax(-1).clamp_min(1e-8)/7
        state=PackedQuantizationCompressor.compress({'weight':weight,'weight_scale':scale},scheme)
        dense=PackedQuantizationCompressor.decompress(state,scheme)['weight']
        with torch.no_grad():reference.get_submodule(name).weight.copy_(dense)
        linear.quantization_scheme=scheme
        linear.register_parameter('weight_scale',torch.nn.Parameter(scale,requires_grad=False))
        PackedQuantizationCompressor.compress_module(linear)
    qconfig=QuantizationConfig(config_groups={'group_0':scheme},ignore=['lm_head'],quantization_status='compressed')
    model.config.quantization_config={**qconfig.model_dump(mode='json'),'quant_method':'compressed-tensors'}
    checkpoint=tmp_path/'ct-checkpoint';model.save_pretrained(checkpoint)
    device=torch.device(os.environ.get('GEMMA_CT_TEST_DEVICE','cpu'))
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.03,device)
    reference=reference.to(device)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,device_map=str(device),
        quantization_config=CompressedTensorsConfig(run_compressed=True))
    profile=replace_compressed_linears(model,packed_storage=packed_storage)
    assert len(profile['modules'])==19 and not hasattr(model,'ct_decompress_hook')
    options={} if checkpoint_profile=='none' else dict(checkpointing=True,activation_storage='cpu',checkpoint_profile=checkpoint_profile)
    a=Native(model,identity={'fixture':'packed-gemma4'},host_profile='frozen-ct-w4g32',**options)
    b=Native(reference,identity={'fixture':'decompressed-gemma4'})
    try:
        assert a.metadata().get('packed_weight_storage','device')==packed_storage
        if packed_storage=='cpu':
            assert all(getattr(m,n).device.type=='cpu' for m in model.modules() if hasattr(m,'_packed_names') for n in m._packed_names)
        with torch.no_grad():
            for _,_,adapter in a.slider.hosts:adapter.lora_up.weight.fill_(.01)
            b.network.load_state_dict(a.network.state_dict())
        va=a.hidden((2,3,4,5),strength=1);vb=b.hidden((2,3,4,5),strength=1)
        assert torch.equal(va,vb)
        ga=torch.autograd.grad(va.square().sum(),tuple(a.network.parameters()),allow_unused=True)
        gb=torch.autograd.grad(vb.square().sum(),tuple(b.network.parameters()),allow_unused=True)
        for left,right in zip(ga,gb):
            assert (left is None)==(right is None)
            if left is not None:assert torch.equal(left,right)
    finally:a.close();b.close()
