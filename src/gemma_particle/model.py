"""Native Gemma 4 final-state binding using the shared routed projection.

No loss or optimizer mathematics live here. Gemma 4 omits K/V projections in
KV-sharing layers; targets follow the actual native module tree.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from particlegan import init, get_recipe
from particle_sliders.routed_projection import TinyParticleProjection
from particle_sliders.pg09_backends import AdapterParameters, checkpoint_with_adapter_scale
from particle_sliders.pg0100 import ConditionalPG0100, validate_pg09_domain
from .source_identity import require_core_source


def text_trunk(model):
    matches = [(name, module) for name, module in model.named_modules()
               if getattr(getattr(module, "config", None), "model_type", None) in ("gemma4_text", "gemma4_unified_text")
               and hasattr(module, "layers") and hasattr(module, "embed_tokens")]
    if len(matches) != 1:
        raise ValueError("exactly one native Gemma 4 text trunk required")
    return matches[0]


class Slider(nn.Module):
    def __init__(self, model, *, rank=8, alpha=8, particles=128, particle_width=4, seed=0, host_profile='float-linear'):
        super().__init__()
        if particle_width != 4:
            raise ValueError("shared TinyParticleProjection requires four-dimensional particles")
        if host_profile=='float-linear':
            host_type=nn.Linear
        elif host_profile=='frozen-ct-w4g32':
            from .ct_linear import FrozenCTLinear
            host_type=FrozenCTLinear
        else:
            raise ValueError('unqualified projection host profile')
        self.host_profile=host_profile
        prefix, trunk = text_trunk(model)
        self.particles = nn.Parameter(torch.randn(particles, particle_width, device=next(trunk.parameters()).device))
        # Public table initializer supplies deterministic R2 particle locations.
        holder = get_recipe('rhbi_optimized', num_particles=particles, z_dim=particle_width).make_prior()
        init.deterministic_orthogonal_(holder, seed=seed)
        with torch.no_grad():
            self.particles.copy_(holder.z.to(self.particles))
        self.adapters = nn.ModuleDict()
        self.hosts = []
        self.rank, self.alpha = rank, alpha
        try:
            for i, layer in enumerate(trunk.layers):
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    host = getattr(layer.self_attn, projection, None)
                    if host is None:
                        continue
                    if type(host) is not host_type or isinstance(getattr(host.forward, "__self__", None), TinyParticleProjection):
                        raise ValueError("unwrapped projection matching the explicit host profile required")
                    name = f"{prefix}.layers.{i}.self_attn.{projection}"
                    adapter = TinyParticleProjection(name, host, self.particles, rank=rank, alpha=alpha)
                    self.hosts.append((name, host, adapter))
                    key = name.replace(".", "-")
                    self.adapters[key] = adapter
                    adapter.to(device=host.weight.device, dtype=torch.float32)
                    branch_seed = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:15], 16)
                    init.deterministic_orthogonal_(adapter, seed=branch_seed)
            if not self.hosts:
                raise ValueError("no native attention targets")
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("finite slider strength required")
        old = [adapter.multiplier for _, _, adapter in self.hosts]
        try:
            for _, _, adapter in self.hosts:
                adapter.multiplier = strength
            yield self
        finally:
            for (_, _, adapter), value in zip(self.hosts, old):
                adapter.multiplier = value

    def close(self):
        for _, host, adapter in reversed(self.hosts):
            if host.forward == adapter.forward:
                host.forward = adapter.org_forward
            elif host.forward != adapter.org_forward:
                raise RuntimeError("foreign forward mutation; cannot detach slider")
        self.hosts.clear()


@dataclass(frozen=True)
class ContextRow:
    neutral_ids: tuple[int, ...]
    positive_ids: tuple[int, ...]
    instance_id: str
    decision: int


@dataclass(frozen=True)
class Prepared:
    row: ContextRow
    target: torch.Tensor
    neutral: torch.Tensor


class Native:
    def __init__(self, model, *, identity, checkpointing=False, **slider_options):
        if not identity:
            raise ValueError("pinned pretrained identity or explicit CPU fixture identity required")
        self.source_cohort=require_core_source()
        self.model, self.identity = model, dict(identity)
        self._flags = [(p, p.requires_grad) for p in model.parameters()]
        self._modes = [(m, m.training) for m in model.modules()]
        self._handles = []
        self.slider = None
        _, self.trunk = text_trunk(model)
        self.width = self.trunk.config.hidden_size
        validate_pg09_domain((1, self.width), response_domain="continuous_vector", penalty_input_domain="continuous")
        model.eval().requires_grad_(False)
        try:
            self.slider = Slider(model, **slider_options)
            self.network = AdapterParameters(self.slider.named_parameters(), self.slider.scaled)
            if checkpointing:
                for layer in self.trunk.layers:
                    self._handles.append(checkpoint_with_adapter_scale(layer, self.network))
        except BaseException:
            self.close()
            raise

    @property
    def particles(self):
        return self.network.particles

    def hidden(self, ids, *, strength=1):
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("nonempty nonnegative token IDs required")
        tokens = torch.tensor([ids], device=self.trunk.embed_tokens.weight.device)
        with self.network.scaled(strength):
            result = self.trunk(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True)
        response = result.last_hidden_state[:, -1, :].float()
        if response.shape != (1, self.width) or not torch.isfinite(response).all():
            raise ValueError("invalid native response")
        return response

    def prepare(self, row):
        if row.neutral_ids == row.positive_ids:
            raise ValueError("teacher instruction must change the context")
        with torch.no_grad():
            return Prepared(row, self.hidden(row.positive_ids, strength=0).detach().clone(),
                            self.hidden(row.neutral_ids, strength=0).detach().clone())

    def generated(self, prepared):
        return self.hidden(prepared.row.neutral_ids, strength=1)

    def metadata(self):
        return {"protocol": "gemma4-matched-assistant-decision-pg0100-v1", "model_identity": self.identity,
                "source_cohort":self.source_cohort,
                "response": "full native final hidden state", "adapter_names": list(self.network.names),
                "host_profile":self.slider.host_profile,
                "target_convention": "same question and observed history; teacher has selected behavioral system instruction",
                "ema_decay": None}

    def close(self):
        for handle in reversed(self._handles):
            handle.close()
        self._handles.clear()
        if self.slider is not None:
            self.slider.close()
        for parameter, flag in self._flags:
            parameter.requires_grad_(flag)
        for module, mode in self._modes:
            module.training = mode


class Critic(nn.Module):
    def __init__(self, width, *, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        init.deterministic_orthogonal_(self, seed=1)

    def forward(self, response, condition):
        return self.net(torch.cat((response, condition), dim=1)).squeeze(-1)


def make_game(native, rows, *, preset="rhbi_optimized", steps=1024, batch=8, **options):
    critic = Critic(native.width).to(device=native.particles.device, dtype=torch.float32)
    return ConditionalPG0100(native, rows, critic, particles=native.particles,
                            conditions=torch.cat([p.neutral for p in rows]), preset=preset,
                            total_steps=steps, batch_size=batch, seed=0, ema_decay=None,
                            response_domain="continuous_vector", penalty_input_domain="continuous", **options)
