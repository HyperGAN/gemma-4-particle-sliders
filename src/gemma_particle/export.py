"""Measure a routed bottleneck's linear approximation and write PEFT weights.

This is a calibrated approximation. Routing is nonlinear, so a native particle
adapter has no exact PEFT merge. Fit rows and diagnostic rows must be disjoint.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file
from particle_sliders.distillation import fit_routed_down


def peft_target(name):
    # HF multimodal checkpoints nest the text trunk; the competition serves
    # Gemma4ForCausalLM. This mapping still requires scorer/vLLM parity testing.
    return name.replace("model.language_model.layers.", "model.layers.")


@torch.no_grad()
def collect_activations(native, rows, *, tokens_per_layer=256):
    if tokens_per_layer < 2:
        raise ValueError("at least two calibration tokens required")
    rows=list(rows)
    if not rows:raise ValueError('nonempty activation contexts required')
    # Every row receives a fixed share. Select positions across each full
    # history, avoiding a calibration set made only of the common system prefix.
    if len(rows)>tokens_per_layer:
        indices=torch.linspace(0,len(rows)-1,tokens_per_layer,dtype=torch.float64).round().long().tolist()
        rows=[rows[i] for i in indices]
    budgets=[tokens_per_layer//len(rows)+(i<tokens_per_layer%len(rows)) for i in range(len(rows))]
    current_budget=0
    captured = {name: [] for name, _, _ in native.slider.hosts}
    counts = dict.fromkeys(captured, 0)
    handles = []
    try:
        for name, host, _ in native.slider.hosts:
            def capture(module, inputs, *, name=name):
                left = min(current_budget,tokens_per_layer - counts[name])
                if left > 0:
                    flat=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
                    positions=torch.linspace(0,len(flat)-1,min(left,len(flat)),device=flat.device,dtype=torch.float64).round().long()
                    x = flat.index_select(0,positions).float().cpu().clone()
                    captured[name].append(x)
                    counts[name] += len(x)
            handles.append(host.register_forward_pre_hook(capture))
        for row,current_budget in zip(rows,budgets):
            native.hidden(row.neutral_ids, strength=1)
            if all(count >= tokens_per_layer for count in counts.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
    if any(not chunks for chunks in captured.values()):
        raise ValueError("calibration did not execute every adapted projection")
    return {name: torch.cat(chunks) for name, chunks in captured.items()}


@torch.no_grad()
def export_peft(native, fit, probe, destination, *, base_model_name, strength=1, provenance):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("export directory is immutable")
    if not math.isfinite(strength) or not base_model_name:
        raise ValueError("finite strength and base model identity required")
    if not provenance.get("fit_task_ids") or not provenance.get("probe_task_ids"):
        raise ValueError("record calibration and probe task identities")
    if set(provenance["fit_task_ids"]) & set(provenance["probe_task_ids"]):
        raise ValueError("fit and probe tasks overlap")
    weights, metrics = {}, {}
    for name, _, adapter in native.slider.hosts:
        x = fit[name].to(device=adapter.lora_down.weight.device, dtype=torch.float32)
        test = probe[name].to(x)
        targets = adapter.bridge(adapter.lora_down(x), native.particles)
        down, ridge = fit_routed_down(x, targets)
        truth = adapter.lora_up(adapter.bridge(adapter.lora_down(test), native.particles))
        approximation = adapter.lora_up(test @ down.T)
        error = (truth - approximation).square().mean().sqrt()
        norm = truth.square().mean().sqrt()
        metrics[name] = {"fit_tokens": len(x), "probe_tokens": len(test), "ridge": float(ridge),
                         "delta_rmse": float(error), "delta_rms": float(norm),
                         "relative_delta_rmse": float(error / norm) if norm > 0 else None}
        target = "base_model.model." + peft_target(name)
        weights[target + ".lora_A.weight"] = down.cpu().contiguous()
        weights[target + ".lora_B.weight"] = (adapter.lora_up.weight * strength).cpu().contiguous()
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": native.slider.rank,
              "lora_alpha": native.slider.alpha, "lora_dropout": 0.0, "bias": "none",
              "inference_mode": True, "fan_in_fan_out": False, "use_rslora": False,
              "use_dora": False, "base_model_name_or_path": base_model_name,
              "target_modules": sorted({name.rsplit(".", 1)[-1] for name, _, _ in native.slider.hosts})}
    destination.mkdir(parents=True)
    save_file(weights, destination / "adapter_model.safetensors", metadata={"format": "pt", "method": "routed-bottleneck-linear-fit"})
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = {"schema": "gemma4-routed-to-peft-v1", "exact_merge": False, "strength": strength,
               "adapter_sha256": hashlib.sha256((destination / "adapter_model.safetensors").read_bytes()).hexdigest(),
               "provenance": provenance, "layer_metrics": metrics,
               "vllm_parity": "unmeasured", "repository_resolution_rate": "unmeasured"}
    # Receipt stays outside the submission's adapter directory.
    destination.with_name(destination.name + "-conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
