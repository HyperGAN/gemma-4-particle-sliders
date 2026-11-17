"""Lossless CPU storage for tensors saved outside checkpointed blocks.

The shared non-reentrant block checkpoint owns recomputation and adapter scale.
These scoped hooks only change storage of its inputs and uncheckpointed saves.
Copies are synchronous, preserve dtype, and never retain a strong device reference.
"""
from dataclasses import dataclass
import weakref

import torch


@dataclass(frozen=True)
class SavedCPU:
    value: torch.Tensor
    device: torch.device
    source: weakref.ReferenceType
    version: int


class CPUActivationStorage:
    def __init__(self):
        self._cache = {}
        self.stats = {'saves': 0, 'copies': 0, 'deduplicated_saves': 0, 'copied_bytes': 0}
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def pack(self, tensor):
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError('activation storage requires ordinary strided tensors')
        self.stats['saves'] += 1
        prior = self._cache.get(id(tensor))
        if prior is not None and prior.source() is tensor and prior.version == tensor._version:
            self.stats['deduplicated_saves'] += 1
            return prior
        saved = SavedCPU(tensor.detach().to(device='cpu', copy=True), tensor.device,
                         weakref.ref(tensor), tensor._version)
        self._cache[id(tensor)] = saved
        self.stats['copies'] += 1
        self.stats['copied_bytes'] += tensor.numel() * tensor.element_size()
        return saved

    @staticmethod
    def unpack(saved):
        source = saved.source()
        if source is not None and source._version != saved.version:
            raise RuntimeError('saved activation was modified in place after CPU storage')
        return saved.value.to(device=saved.device)

    def __enter__(self):
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._hooks.__exit__(*exc)
        finally:
            # Individual graph saves own their packets after the forward. The
            # deduplication table must not extend any packet's lifetime.
            self._cache.clear()
