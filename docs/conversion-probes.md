# Disjoint TRAIN conversion probes

The first successful teacher may supply many decisions from one TRAIN task. That is sufficient for a native engineering chunk, but it cannot provide task-disjoint fit and probe sets by itself. Conversion now accepts an explicit `--probe-contexts` packet from other TRAIN tasks. Those prompts provide approximation diagnostics only and never become hidden-state training targets or DEV results.

Before any pretrained optimizer update, the initial OFF requests from `fastapi_13207` and `fastapi_13537` were reserved as probes. Both baseline attempts already passed fresh tests with complete capture, so this conditional pool excluded them from targets. Only the initial system and user messages enter the probe; later model answers, tool output and patches are excluded. The exact server-bound tokenizer produced 2,884 and 3,199 input tokens. This preparatory call performed tokenization and no model inference.

The private packet `data/conversion-probe-v1/contexts.json` binds the TRAIN manifest, source outcome hashes, initial requests, exact token IDs and frozen server/native weight identities. Conversion rejects changed token hashes, fit/probe task overlap, a different manifest/model, or non-TRAIN tasks. These are internal provenance checks, not authentication of arbitrary external JSON. The actual source records come from the local fresh-container verifier.

Activation sampling now budgets tokens evenly across the supplied rows and samples positions evenly across each full history. The previous implementation filled the budget with the beginning of the first row, which could measure only the common system prefix. This declared conversion-only change is tested by changing a late issue token and checking that both input rows contribute. It changes no native training objective or recipe. Historical conversion receipts retain their original sampling law.

```bash
# After native training, using admitted contexts for fitting:
.train-venv/bin/python scripts/convert_to_peft.py \
  --config configs/31b-ct-offload-pilot.json --cache runs/cache --native runs/empf/RAW-1024 \
  --probe-contexts data/conversion-probe-v1/contexts.json --tokens-per-layer 256 \
  --out runs/empf/peft --device cuda:1 --memory-fraction .60
```

The [preparation receipt](../reports/conversion-probe-preparation-v1.json) records CPU controls and tokenization. Actual 31B approximation error, vLLM LoRA transport and task resolution remain unmeasured. DEV selection and holdout confirmation still require their own complete evaluations.
