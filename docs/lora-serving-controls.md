# LoRA serving controls

The separate `serve_vllm_lora.py` entry point prepares and launches the pinned official vLLM wheel with explicit adapter aliases. It leaves the active frozen collection server unchanged. A learned candidate must match its measured conversion receipt and native/cache bindings. Synthetic zero/nonzero controls have a distinct server kind, `gemma4_lora_transport_fixture`, so the ordinary trajectory runner rejects them as benchmark/teacher servers.

The CPU fixture generator reads only the cached, pinned safetensors header. It creates valid-shape rank-eight matrices for all 230 actual attention targets, using the same PEFT name mapping as native conversion. Both packets share fixed seed-zero down matrices. The zero packet has zero up matrices; the probe packet has fixed nonzero up matrices. They are transport fixtures, not learned adapters or coding results.

CPU checks cover isolated fixture identities, exact alias/config/weight bindings, receipt mismatch rejection and parsing the prepared JSON aliases through the actual official vLLM `LoRAParserAction`. The full-weight fixture check independently confirms 460 tensor keys, matching shapes/down matrices, all-zero zero up matrices and nonzero probe up matrices. A prepared plan accesses no GPU and does not claim a verified serving environment.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .train-venv/bin/python scripts/prepare_lora_transport_fixtures.py \
  --out runs/own-lora-fixtures
PYTHONPATH=src .harness-venv/bin/python scripts/serve_vllm_lora.py \
  --synthetic-fixtures runs/own-lora-fixtures/fixtures.json --device cuda:1 \
  --out runs/own-lora-plan --plan-only
```

Actual launch separately verifies installed official wheel bytes, observed GPU headroom and fixed model revision. Runtime adapter updates are disabled. The server uses the same 32,768-token window and Gemma 4 tool/reasoning parsers as the existing frozen cohort. Adapter dtype follows the official vLLM auto/BF16 condition; native FP32 adapter fidelity remains a separate measurement.

`qualify_vllm_lora.py` requires the real fixture server identity, checks its health/model list and compares exact tokenization under the base, zero and probe aliases. Each alias then receives the same fixed prompt token IDs and one greedy generated token. Exact token IDs and finite top logprobs must preserve zero parity within the predeclared 1e-5 tolerance and show a nonzero effect. Inert adapters, changed zero behavior and nonfinite outputs cannot pass. Request attempts and responses are journaled, including incomplete attempts.

```bash
# Only after the selected GPU has actual shared headroom:
PYTHONPATH=src .vllm-venv/bin/python scripts/serve_vllm_lora.py \
  --synthetic-fixtures runs/own-lora-fixtures/fixtures.json --device cuda:1 \
  --memory-fraction .60 --out runs/own-lora-server
PYTHONPATH=src .harness-venv/bin/python scripts/qualify_vllm_lora.py \
  --endpoint http://127.0.0.1:8027 --server-identity runs/own-lora-server/server-identity.json \
  --out runs/own-lora-effect
```

The [actual official-vLLM control](../reports/vllm-lora-effect-control-v1.json) passed all eight requests: identical prompt token IDs, exactly zero logprob difference for the zero adapter, and a .5296175 maximum common-token logprob change for the nonzero probe. The owned test server then exited and GPU 1 returned to 20 MiB used. This establishes short-prompt loading and effect for synthetic matrices; native nonlinear-to-linear conversion fidelity, real candidate task success and an official learned score remain unmeasured.
