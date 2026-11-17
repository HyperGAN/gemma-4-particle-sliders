# Native pretrained collection

Native collection uses the pinned Gemma E2B BF16 model and the shared Particle Sliders wrapper at strength zero. The eligible Kaggle model name is a routing alias; `server-identity.json` records the actual BF16 weights and runtime. These results are separate from production W4A16/vLLM results.

Install `requirements-model.txt` and `requirements-serving.txt` in the training environment. Download the pinned model's JSON, safetensors and Jinja files with `huggingface_hub.snapshot_download` before starting the server. The thinking-budget implementation also requires the exact official wheel:

```bash
.kaggle-venv/bin/kaggle datasets download metric/gemma-4-developer-agent-wheelhouse \
  -f vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl -p data/kaggle-official/wheelhouse --quiet
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 .venv/bin/python scripts/serve_native.py \
  --device cuda:1 --memory-fraction .35 --port 8018 --out runs/native-server
```

`official_thinking.py` verifies the wheel and source SHA, then compiles only the three unchanged state methods of the official `ThinkingTokenBudgetLogitsProcessor`. Its countdown and forcing policy are retained. The surrounding sampler is HF generation, so this does not establish vLLM output parity. Unrequested HF top-k filtering is explicitly disabled, and unsupported generation controls fail rather than disappearing.

Captured OpenAI tool arguments remain strings in the journal. A copy is decoded into mappings for the pinned native Gemma template. Captured thinking controls and tool schemas enter the same template used for state caching. Native responses use the tokenizer's own response parser; tool arguments are serialized back to OpenAI JSON strings for ADK.

The live endpoint exposes its identity digest. The runner verifies it before collection and sends the digest on every request; a different or missing digest returns HTTP 412. Use a separate port for each live cohort. The [initial memory-failure receipt](../reports/native-memory-failure-v1.json) records eager attention exceeding the shared cap and an excluded mixed episode caused by endpoint reuse. The SDPA cohort restarts the comparison with explicit attention identity in both trajectories and state caches.

Create a private `plan.json` before collecting, declaring the task IDs, all four arms, server identity, budget and selection rule. `collect_arms.py` finishes complete existing arms, runs missing arms, verifies every patch in a separate process, and writes the combined outcomes and admitted contexts. It refuses to restart incomplete directories. Its optional server PID cleanup checks the exact command and output directory before stopping that owned process.

Verified schema v3 binds the actual first request separately from complete model captures. A genuine task-budget timeout counts as a failed OFF outcome even if its final request was canceled. The real first request supplies the control system text, tools and sampling controls. Other incomplete OFF captures cannot supply a pair. Every selected successful teacher trajectory still requires a complete, hash-checked capture from the frozen base model. This correction was declared after observing the first OFF timeout and before running any positive arm.

The first SDPA OFF run on `fastapi_11194` timed out without a patch. The remaining instruction arms must finish before admitting any target. No adapter training or official Kaggle score has been measured.
