# Official vLLM collection

The 31B cohort uses the authenticated competition wheelhouse's vLLM 0.19.1, compressed-tensors 0.15.0.1 and Transformers 5.13.1 wheels. It runs in a separate Python 3.12 environment with Torch 2.10.0+cu128. Training retains its own Torch environment. [The environment receipt](../reports/vllm-environment-v1.json) records installed versions; server startup verifies every installed Python source against those three official wheels.

```bash
uv venv --python 3.12 .vllm-venv
uv pip install --python .vllm-venv/bin/python \
  data/kaggle-official/wheelhouse/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl \
  data/kaggle-official/wheelhouse/compressed_tensors-0.15.0.1-py3-none-any.whl \
  data/kaggle-official/wheelhouse/transformers-5.13.1-py3-none-any.whl
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
  .vllm-venv/bin/python scripts/serve_vllm.py \
  --device cuda:0 --memory-fraction .60 --port 8026 --out runs/own-vllm-server
```

The launcher fixes the Hub revision, one sequence, tensor parallelism 1, eager execution, text-only input, official Gemma4 tool/reasoning parsers and official thinking delimiters. vLLM selects Marlin W4A16 linears and Triton attention for this model. Its pinned model generation configuration supplies top-k 64 when the request omits top-k; this differs from the earlier native E2B cohort. These are separate conditions.

The .60 setting is vLLM's engine memory target, not a hard CUDA allocator cap. Startup checks for at least 8 GiB of observed headroom beyond the requested budget. The local single-A6000 configuration differs from Kaggle's four L4 GPUs. The server advertises a 32,768-token context; the initial engine reported 10,368 KV-cache slots. Full-window requests and production hardware equivalence remain unqualified.

`server-identity.json` is immutable and records exact model, wheel sources, host settings and launch arguments. The official server's middleware returns its identity digest only after health succeeds, and rejects `/v1/` requests without the exact `X-Gemma-Identity` header. Each collection plan names that identity. Use a new run directory and port for a new engine.

## Exact input tokens and independent hidden targets

The Gemma template treats string and OpenAI-list system content differently: the list branch appends a space before the tool declarations. On the first TRAIN task, explicit OpenAI-list formatting matched all 25 captured requests' official `/tokenize` output. The original native string formatting differed by one token. These observations concern input serialization only.

After a complete four-arm cohort supplies verified successful TRAIN contexts, capture the official server's positive and neutral token IDs:

```bash
.venv/bin/python scripts/tokenize_contexts.py \
  --contexts runs/own-cohort/contexts.jsonl \
  --server-identity runs/own-vllm-server/server-identity.json \
  --endpoint http://127.0.0.1:8026/v1 --out runs/own-cohort/prompt-transport.json
.venv/bin/python scripts/cache_contexts.py --config configs/31b-ct-pilot.json \
  --contexts runs/own-cohort/contexts.jsonl \
  --prompt-transport runs/own-cohort/prompt-transport.json \
  --device cuda:1 --memory-fraction .70 --out runs/own-cache
```

The transport checks identical pinned weights, context bytes, order, token digests, frozen generator identity and the separate native response identity. It transfers only inputs. Native HF computes both positive and neutral hidden states itself; no vLLM activations or numerical-equivalence claim becomes a target. Ordinary native contexts retain their exact-identity requirement. Transport controls do not establish learned quality, native-generation parity, nonlinear-to-PEFT conversion or production adapter parity.

[History compaction](history-compaction.md) is now an explicit prospective profile matching the official starter. Earlier local collection omitted it; those runs remain a separate no-compaction cohort. Exact input-token controls alone did not establish the complete inference setup.
