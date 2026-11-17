# CPU storage of frozen packed weights

The [shared storage config](../configs/31b-ct-shared-storage-pilot.json) combines CPU checkpoint-input storage, nested attention/MLP checkpoints and CPU storage of the frozen CT matrices. It preserves both full token sequences, batch size, hidden-vector width, adapter initialization and public ParticleGAN recipes.

`FrozenCTLinear` retains the original packed INT4 integers, scales and optional auxiliary constants as CPU buffers. Each forward or input-gradient call copies those constants to the activation device, invokes the same public CT decompressor there, uses the dense matrix for that operation and releases it. There is no dense CPU weight cache, altered quantization, approximate gradient or asynchronously reused staging buffer. The derivative remains `dX = dY @ W` for the original dequantized constant matrix. Base gradients are absent.

The empty host weight descriptor stays on the compute device, so the core creates FP32 adapters on that device. The frozen embedding and normalizations also remain on the compute device. Native metadata infers the storage profile from the actual CT modules and binds it into the full game signature. Moving a declared CPU-packed module's buffers onto a GPU later is rejected rather than silently changing its storage contract.

The first .40-cap attempt [failed before forward](../reports/cpu-packed-initial-loader-error-v1.json): GPU-first loading requested a single 19.04-GiB allocation against an 18.96-GiB cap. The corrected loader loads the original compressed model on CPU, replaces its packed linears there, and places only ordinary registered parameters/buffers and empty compute descriptors on the GPU. PyTorch's own module conversion applies to each module's direct tensors, with recursion disabled at this placement step so packed children remain on CPU. This happens before adapters, graphs or optimizers exist. It does not alter any tensor value. Its [CPU-first CUDA control](../reports/cpu-first-packed-cuda-v1.json) checks exact outputs and all adapter gradients after that placement.

The [81-check CPU preflight](../reports/cpu-packed-weight-preflight-v1.json) and official compiler pass. The [tiny CUDA control](../reports/cpu-packed-weight-cuda-v1.json) also passes all four combinations of device/CPU packed storage and direct/nested checkpointing. Every output and adapter-gradient tensor exactly matches a separately decompressed dense reference. Peak CUDA allocation was 10,129,408 bytes on this small random model; this does not estimate 31B memory or quality. The separate nested profile's [recipe and continuation controls](../reports/nested-checkpoint-cpu-v1.json) remain available.

The current pretrained resource probe uses two full 28,672-token sequences and a .40 CUDA allocator cap, about 19 GiB on an A6000. This leaves concurrent capacity for another worker's declared 19-GiB cap and more than eight GiB of physical headroom. Its full-history result is pending. The probe is synthetic, runs no optimizer and cannot supply training targets.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  nice -n 10 .train-venv/bin/python scripts/profile_native.py \
  --config configs/31b-ct-shared-storage-pilot.json --device cuda:1 \
  --memory-fraction .40 --max-tokens 28672 --batch-size 2 \
  --out runs/own-shared-storage-resource
```

Only actual resource results can establish whether this fits. Transfer latency, host RAM and a public-recipe step on genuine admitted contexts still need measurement. The original .70/.75 resource failures and block-storage receipts are preserved.

The CPU-first .40 and .50 attempts subsequently failed in their first gradient-enabled forward. A [keyword-input lifetime correction](checkpoint-input-binding.md) now passes 91 CPU checks and six tiny CUDA controls as a distinct profile, preserving the full model computation while exposing declared tensor inputs to public checkpoint saves. Its full-history .40-cap result is pending.
