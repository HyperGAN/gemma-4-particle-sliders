# Frozen quantized Gemma host

The [official 31B CT checkpoint](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/tree/52f3f65bc7a02d555763bc923bd1d9094898219d) uses symmetric group-32 INT4 weights. The stock CT loader decompresses the model at first forward. This project has an explicit alternative host that keeps the original packed constants and materializes one dense matrix per linear forward/backward call. It releases that temporary matrix afterward.

`FrozenCTLinear` delegates all unpacking and dequantization to the exact official `compressed-tensors==0.15.0.1` wheel. Its backward computes `dX = dY @ W` for the same fixed dequantized matrix used by the forward. Base weights and scales are buffers, and autograd saves their packed representations. The empty `weight` descriptor supplies only device/dtype information to the existing shared projection factory. It never replaces the mathematical matrix.

This changes storage and execution, not the ParticleGAN loss, penalty, prior, optimizer or update schedule. Imports do not modify shared classes. The loader replaces only explicitly supported packed Linear instances and removes the stock whole-model decompression hook. Unsupported quantization formats fail explicitly.

Install the exact downloaded wheel into the training environment:

```bash
uv pip install --python .venv/bin/python --no-deps \
  data/kaggle-official/wheelhouse/compressed_tensors-0.15.0.1-py3-none-any.whl
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_ct_linear.py
```

The CPU controls compare outputs and input gradients with the public decompressor, run a double-precision derivative check, verify that no dense weight is saved for backward, and load an actual tiny Gemma CT checkpoint. Its adapter gradients match the corresponding dense decompression exactly.

The [resource receipt](../reports/frozen-ct-qualification-v1.json) records real 31B checks at 128 and 4,096 tokens. Both preserve exact initial ON/OFF, finite connected adapter gradients, a frozen base and unchanged adapter parameters. The 4,096-token check peaked at about 25.4 GiB allocated under a .60 process cap. These synthetic probes perform zero optimizer updates and cannot supply training targets.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 .venv/bin/python scripts/profile_native.py \
  --config configs/31b-ct-pilot.json --device cuda:0 --memory-fraction .60 \
  --max-tokens 4096 --out runs/ct-resource-check
```

Floating hosts retain their .50 maximum process fraction; the explicit CT host permits up to .75 and checks for at least 8 GiB of observed headroom before allocation. The measured profile uses .60, leaving 40% of device capacity for other processes. Existing jobs remain running.

Longer contexts, actual recipe-update memory, coding outcomes, linear conversion and production serving parity are still required. Native CT generation is expensive, so a separate official vLLM/Torch 2.10 environment is being prepared for faster collection. Its results must be identified separately from native HF collection; no compatible target-transfer claim has been established yet.
