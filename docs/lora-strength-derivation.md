# Reuse a measured conversion across strengths

The routed bottleneck fit and its probe diagnostics do not depend on export strength. The existing native exporter applies strength only when writing the FP32 LoRA B matrix. `derive_peft_strength.py` therefore creates the declared .5 and 1.5 doses on CPU from the measured unit-strength conversion, preserving A and exact config bytes. It saves separate weights and a receipt binding the original conversion, adapter identity, native checkpoint, cache, and probe provenance.

The [numerical control](../reports/lora-dose-derivation-control-v1.json) uses a nonzero CPU random Gemma adapter after a real public-recipe fixture update. Both derived doses exactly match separate native exports at those strengths. It also checks source preservation and rejects duplicate outputs, a derived source, and a damaged conversion receipt. This is CPU fixture evidence; learned 31B fidelity, serving effects, and task outcomes still require their actual runs.

Original layer diagnostics describe the unscaled native delta and are explicitly labeled as such in derived receipts. Scaling performs no new fitting and introduces no new quality observation. This reduces six checkpoints from eighteen repeated model conversions to six measured conversions plus twelve CPU dose derivations.

```bash
CUDA_VISIBLE_DEVICES='' .train-venv/bin/python scripts/derive_peft_strength.py \
  --source runs/converted-unit --strength .5 --out runs/converted-half
```
