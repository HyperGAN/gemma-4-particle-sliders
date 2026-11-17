"""Frozen CT W4 linear with the exact dense input derivative and packed storage.

The public CT compressor owns unpacking and dequantization. A dense weight is
materialized for each forward/backward call, then released; autograd saves only
the original packed constants. No base-weight gradient or approximation is used.
This host is separate from the qualified BF16 E2B collection cohort.
"""
import torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path
import zipfile

CT_WHEEL=Path('data/kaggle-official/wheelhouse/compressed_tensors-0.15.0.1-py3-none-any.whl')
CT_WHEEL_SHA256='e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9'


@lru_cache(maxsize=2)
def verify_ct_sources(wheel=CT_WHEEL):
    wheel=Path(wheel)
    with wheel.open('rb') as stream:observed=hashlib.file_digest(stream,'sha256').hexdigest()
    if observed!=CT_WHEEL_SHA256:raise ValueError('official CT wheel hash differs')
    dist=importlib.metadata.distribution('compressed-tensors')
    with zipfile.ZipFile(wheel) as archive:
        files=[n for n in archive.namelist() if n.endswith('.py')]
        if any(dist.locate_file(n).read_bytes()!=archive.read(n) for n in files):
            raise ValueError('installed CT sources differ from the pinned official wheel')
    return {'version':dist.version,'wheel_sha256':observed,'verified_python_files':len(files)}


def _decompress(state,scheme,dtype):
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    return PackedQuantizationCompressor.decompress(state,scheme)['weight'].to(dtype=dtype)


def _compute_state(state,device):
    # Storage transfer only: keep packed integers/scales and decompress with the
    # same public implementation on the activation device. No dense CPU cache.
    return {name:value.to(device=device) for name,value in state.items()}


class _FrozenPackedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bias,module):
        ctx.keys=module._packed_names
        ctx.shape=(module.out_features,module.in_features)
        ctx.scheme=module.quantization_scheme
        ctx.input_dtype=x.dtype
        ctx.input_device=x.device
        ctx.save_for_backward(*(getattr(module,name) for name in ctx.keys))
        state=_compute_state({name:getattr(module,name) for name in ctx.keys},x.device)
        state['weight_shape']=ctx.shape
        weight=_decompress(state,ctx.scheme,x.dtype)
        return F.linear(x,weight,bias.to(dtype=x.dtype) if bias is not None else None)

    @staticmethod
    def backward(ctx,grad_output):
        state=_compute_state(dict(zip(ctx.keys,ctx.saved_tensors)),ctx.input_device)
        state['weight_shape']=ctx.shape
        weight=_decompress(state,ctx.scheme,ctx.input_dtype)
        # This is the derivative of y = x W^T + b for constant dequantized W.
        return grad_output.matmul(weight),None,None


class FrozenCTLinear(nn.Module):
    """Explicitly supported symmetric group-32 INT4 CT linear task host."""
    def __init__(self,state,scheme,*,compute_dtype=None,packed_storage='device'):
        super().__init__()
        if packed_storage not in ('device','cpu'):
            raise ValueError('packed storage must be device or cpu')
        self.packed_storage=packed_storage
        weights=scheme.weights
        value=lambda x:getattr(x,'value',x)
        if (weights is None or weights.num_bits!=4 or value(weights.type)!='int'
                or value(weights.strategy)!='group' or weights.group_size!=32 or not weights.symmetric
                or scheme.input_activations is not None or scheme.output_activations is not None):
            raise ValueError('only frozen symmetric group-32 W4 CT weights are qualified')
        if set(state)-{'weight_packed','weight_scale','weight_shape','weight_zero_point','weight_g_idx','bias'}:
            raise ValueError('unsupported packed state fields')
        shape=tuple(int(x) for x in state['weight_shape'])
        if len(shape)!=2 or any(n<=0 for n in shape):raise ValueError('linear matrix shape required')
        self.out_features,self.in_features=shape
        self.quantization_scheme=scheme.model_copy(deep=True)
        self._packed_names=tuple(sorted(k for k in state if k not in ('weight_shape','bias')))
        if 'weight_packed' not in self._packed_names or 'weight_scale' not in self._packed_names:
            raise ValueError('packed weights and scales required')
        if state['weight_packed'].dtype!=torch.int32:raise ValueError('CT packed INT32 storage required')
        compute_device=state['weight_packed'].device
        storage_device='cpu' if packed_storage=='cpu' else compute_device
        for name in self._packed_names:self.register_buffer(name,state[name].detach().to(device=storage_device))
        self.register_buffer('bias',state['bias'].detach().to(device=storage_device) if state.get('bias') is not None else None)
        dtype=compute_dtype or state['weight_scale'].dtype
        if not dtype.is_floating_point:raise ValueError('floating activation dtype required')
        # TinyParticleProjection reads only host weight device/dtype at creation.
        # This empty descriptor never substitutes for the actual packed matrix.
        self.register_buffer('weight',torch.empty(0,device=compute_device,dtype=dtype))

    @classmethod
    def from_compressed_module(cls,module,*,packed_storage='device'):
        names=('weight_packed','weight_scale','weight_shape','weight_zero_point','weight_g_idx','bias')
        state={n:getattr(module,n) for n in names if getattr(module,n,None) is not None}
        return cls(state,module.quantization_scheme,packed_storage=packed_storage)

    def forward(self,x):
        if self.packed_storage=='cpu' and any(getattr(self,name).device.type!='cpu' for name in self._packed_names):
            raise ValueError('CPU packed storage moved away from its declared device')
        bias=self.bias.to(device=x.device) if self.bias is not None else None
        return _FrozenPackedFunction.apply(x,bias,self)


def replace_compressed_linears(model,*,packed_storage='device'):
    """Replace only loaded packed Linear modules; never patch shared classes."""
    quantizer=getattr(model,'hf_quantizer',None)
    if quantizer is None:raise ValueError('a loaded CT model is required')
    selected=[(name,module) for name,module in model.named_modules() if hasattr(module,'weight_packed')]
    if not selected:raise ValueError('no packed weights found')
    replacements=[]
    for name,module in selected:
        if type(module) is not nn.Linear:raise ValueError('unsupported compressed module type')
        replacements.append((name,FrozenCTLinear.from_compressed_module(module,packed_storage=packed_storage)))
    quantizer.compressor.remove_decompression_hook(model)
    for name,replacement in replacements:model.set_submodule(name,replacement)
    return {'profile':'frozen-ct-w4g32-public-decompression-exact-input-gradient-v1',
            'modules':{name:[m.out_features,m.in_features] for name,m in replacements},
            'saved_backward_state':'original packed constants only; no dense base weights',
            **({'packed_storage':packed_storage} if packed_storage!='device' else {})}


def place_cpu_packed_compute(model,device):
    """Place ordinary model tensors without staging the full packed GPU model.

    Run once after CPU loading/replacement and before adapters, graphs or an
    optimizer exist. PyTorch's own module conversion handles direct registered
    parameters/buffers; recursion is disabled so CPU-packed children stay put.
    """
    device=torch.device(device)
    packed=[m for m in model.modules() if isinstance(m,FrozenCTLinear)]
    if (not packed or any(m.packed_storage!='cpu' for m in packed)
            or any(getattr(m,n).device.type!='cpu' for m in packed for n in m._packed_names)):
        raise ValueError('uniform CPU packed storage required before compute placement')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('compute placement must precede all model gradients')
    for module in model.modules():
        if isinstance(module,FrozenCTLinear):
            module.weight=module.weight.to(device=device)
        else:
            module._apply(lambda tensor:tensor.to(device=device),recurse=False)
