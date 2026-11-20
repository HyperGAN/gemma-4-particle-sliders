"""Expose declared positional-or-keyword tensors to public checkpoint saves.

PyTorch's non-reentrant checkpoint saves positional tensor arguments through
saved-tensor hooks; its recomputation closure retains keyword arguments. Bind
the original signature before entering the unchanged core wrapper. Container
objects, notably Gemma's shared-KV UserDict, retain their identity and mutations.
"""
import inspect

from particle_sliders.backends import checkpoint_with_adapter_scale


class _PositionalCheckpoint:
    def __init__(self,module,forward,checkpointed,handle):
        self.module,self.forward,self.checkpointed,self.handle=module,forward,checkpointed,handle
        self.closed=False

    def close(self):
        if self.closed:return
        if self.module.forward is not self.forward:
            raise RuntimeError('positional checkpoint ownership changed')
        self.module.forward=self.checkpointed
        self.handle.close()
        self.closed=True


def checkpoint_with_positional_inputs(module,network):
    signature=inspect.signature(module.forward)
    handle=checkpoint_with_adapter_scale(module,network)
    checkpointed=module.forward
    def forward(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        bound.apply_defaults()
        return checkpointed(*bound.args,**bound.kwargs)
    try:
        module.forward=forward
        return _PositionalCheckpoint(module,forward,checkpointed,handle)
    except BaseException:
        handle.close()
        raise
