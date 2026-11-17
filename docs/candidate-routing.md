# PEFT candidate evaluation routing

`run_trajectories.py --adapter PATH --arm off` packages a PEFT candidate with the OFF system prompt and uses the official `discover_adapters` and `AdapterManifest.register_all` APIs. The discovered `particle_lora` entry receives its own registered model, `openai/particle_lora`, sharing the task's recording client. The official compiler selects that registered adapter model. No candidate alias is rewritten to the frozen base alias.

Before inference, the runner requires `server_identity["adapters"]` to equal the discovered identities: alias, base model, rank and SHA-256 of both PEFT files. Missing aliases or changed config/weight bytes fail. The artifact, cohort contract, captured result and subsequent verified outcome retain the candidate identity. Candidate trajectories explicitly have `frozen_base_generator=false`, so they cannot supply frozen-base teacher targets. Combining this candidate with a teacher prompt is rejected before execution; this initial treatment evaluates the learned adapter with the OFF prompt.

The [CPU qualification](../reports/candidate-alias-cpu-v1.json) runs a real official compiler and ADK loop with clearly labeled synthetic adapter files. Both agent decisions and public compaction summaries use the selected adapter model at the captured ADK boundary, with seed 0 and thinking budget 1024 preserved. Two CPU checks also cover server-identity mismatches and the CLI prompt restriction. These fixtures do not claim loadable 31B weights, LoRA effects, task success or training targets.

This control captures calls before LiteLLM provider dispatch; the `openai/` prefix is its provider selector. Actual HTTP alias transport, loading the trained 31B adapter through the official vLLM wheel, zero/nonzero effect controls and repository-test evaluation remain required. The current frozen collection server has no adapters enabled and is unchanged. A future candidate server must advertise the exact `adapters` mapping and serve its entries under their discovered names.

```bash
# After a trained conversion and its actual LoRA-enabled serving qualification:
.harness-venv/bin/python scripts/run_trajectories.py \
  --tasks data/external-dev-v1/dev.jsonl --task-id DECLARED_DEV_TASK --arm off \
  --adapter runs/selected/peft --model gemma-4-31b-it-qat-w4a16-ct \
  --prompt-profile repair-finalize-v2 \
  --endpoint http://127.0.0.1:8027/v1 --server-identity runs/candidate-server/server-identity.json \
  --external-dev-manifest data/external-dev-v1/manifest.json \
  --external-dev-qualification reports/supplementary-dev-qualification-v1.json \
  --wheels PATH_TO_QUALIFIED_DEV_WHEELS --image QUALIFIED_DEV_IMAGE \
  --inputs PATH_TO_QUALIFIED_DEV_INPUTS --out runs/own-candidate-dev
```

The example requires actual task, wheel, image and input values from the qualified DEV manifest. It is not an executed model result.

The current prospective cohort uses `repair-finalize-v2` for both candidate and matched OFF controls. Keep that common profile and its contract hash through DEV, confirmation and final packaging. Historical runs retain `repair-v1`; the explicit flag prevents inheriting that historical default for a new candidate comparison.
