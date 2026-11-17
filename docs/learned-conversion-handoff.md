# Prepare the complete learned conversion grid

`scripts/prepare_learned_sweep.py` waits for the actual full native training producer before preparing the eighteen learned conditions in the frozen [DEV plan](../research/dev-selection-v1.json). It does not run DEV tasks, open the holdout, select a winner, or submit to Kaggle.

Before loading a model, it requires the completed original one-update pilot, both completed 1,024-update trajectories, six checkpoint receipts, and one exact admitted TRAIN response cache. It checks checkpoint and raw tensor hashes, contiguous finite metrics, the exact journal prefix saved for each checkpoint, native objective/source signatures, and all 230 attention projection hosts. The qualified issue-only conversion probes must remain disjoint from the fitting tasks. These local consistency checks cannot authenticate fabricated receipts.

For a live producer, the supervisor records its exact PID, birth time, and output directory, then observes completion and process release. Each native conversion waits for half the GPU memory plus 8 GiB free and the qualified host-memory bound plus 8 GiB free. Native model children use the frozen executor, a .50 GPU allocation cap, four CPU threads, and offline model files. Launch the supervisor with `nice -n 10` to share CPU resources.

The six measured unit-strength conversions use 256 fitting and 256 disjoint probe activation rows per host. The .5 and 1.5 doses are then [derived on CPU](lora-strength-derivation.md), preserving the original unscaled fitting diagnostics. Each packet binds the actual native checkpoint, cache, probes, conversion report, adapter weights, and serving identity. `candidate-packets.json` follows the eighteen conditions in their original plan order. Failure preserves partial outputs and stops without automatic retries.

Run CPU preparation first, using absolute paths to the existing producer and original pilot:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/prepare_learned_sweep.py \
  --training /path/to/full-training \
  --pilot /path/to/original-pilot \
  --executor /path/to/frozen-native-executor \
  --executor-commit EXACT_COMMIT --device cuda:1 \
  --plan-only --out runs/conversion-plan
```

To arm the handoff, omit `--plan-only`, supply `--training-pid` for the exact live producer, and choose a fresh output directory. An already completed producer does not require a live PID. Freeze the supervisor checkout before launching; source or declaration changes cause failure before the next child operation.

The [CPU preparation receipt](../reports/learned-conversion-preparation-v1.json) records receipt-tampering controls, an actual random CPU Native signature check, and the public-source suite without private data. It demonstrates preparation only. No learned 31B conversion, pretrained training update, or model-quality result is claimed by that receipt.
