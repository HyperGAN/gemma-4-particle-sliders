# Full-history activation storage

The existing 31B resource checks cover two 4,096-token sequences. They do not qualify the observed histories that can approach 28,000 tokens. Keeping every layer's checkpoint input on the device can exceed the shared GPU cap even though the cached final hidden target has only 5,376 values per decision.

The opt-in [CPU storage config](../configs/31b-ct-offload-pilot.json) preserves all input tokens, the native full-width response, batch size, initialization and recipe. It uses the public core's existing non-reentrant layer checkpoint and adapter-scale recomputation. An outer, forward-scoped `saved_tensors_hooks` context copies saved inputs and other saves outside those checkpointed blocks synchronously to CPU. Backward restores their original device and dtype. The inner checkpoint hooks continue to own block intermediates, so this does not copy every packed model constant to CPU at every layer.

Packets hold a CPU tensor and a weak source reference. Repeated saves of the same live tensor/version share a packet. The per-forward lookup table is cleared on exit; graph saves own their packets afterward. A retained source that changes version causes an error before unpacking. As with saved-tensor hook APIs generally, callers must not mutate saved storage through aliases whose original tensor object has already died. Native training does not mutate model activations in this way. There is no asynchronous copy, compression or dtype conversion.

The storage choice enters native metadata and the full game signature, so full checkpoints cannot resume under a different storage profile. Inference and cache-only `no_grad` forwards do not need offloading. Restoring a raw inference adapter remains separate from full optimizer continuation.

The [CPU qualification](../reports/activation-storage-cpu-v1.json) covers direct-versus-checkpointed gradients at nonzero adapter weights, deferred backward under a changed ambient adapter strength, both recipes' exact updates and continuation, mutation detection, weak-reference lifetime and hook cleanup. The complete 63-test CPU suite and official compiler pass. The earlier 57-test source-upgrade receipt is archived byte for byte as [pg0101-cpu-preflight-v1.json](../reports/pg0101-cpu-preflight-v1.json).

GPU long-history memory, wall time and actual recipe updates remain unmeasured for this profile. Run a zero-update resource probe after an owned collection server has released its GPU, inspect the observed headroom, and then qualify an actual recipe step on admitted data before committing to a longer chunk:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 .train-venv/bin/python scripts/profile_native.py \
  --config configs/31b-ct-offload-pilot.json --device cuda:1 --memory-fraction .70 \
  --max-tokens 28672 --batch-size 2 --out runs/ct-offload-long-resource-v1
```

This command is a synthetic derivative/resource probe. It performs zero optimizer updates and cannot supply a training target or a coding score. The current collectors remain in their original frozen checkout and environment.
