# EMPF learning-rate screen

The screening budget is eight native updates per rate on the same admitted TRAIN cache. Each trial starts from the same seed and keeps the production horizon of 1,024 updates so its first eight scheduled steps match a full run. The study varies the base rate and leaves EMPF's critic and particle-bank multipliers at 1.5 and 2.5. The penalty runs on every update with `lazy_k=1`.

| Base rate | Critic rate | Particle-bank rate |
|---:|---:|---:|
| 0.006 | 0.009 | 0.015 |
| 0.012 | 0.018 | 0.030 |
| 0.024 | 0.036 | 0.060 |

The evaluation uses seven other TRAIN questions with clean OFF and Verify captures. It checks that the live tokenizer reproduces the exact two fitting pairs before it tokenizes the scoring prompts. Each scoring pair differs only in the Verify instruction. No scoring prompt enters the training cache.

At update eight, measure the sliced Wasserstein distance between the adapted and teacher hidden-state edits from the frozen neutral baseline. Use unit slider strength, 256 frozen directions and one RMS normalization from the fitting pairs. A lower mean distance ranks higher. The result retains exact ties, all measured distances and the neutral reference. If the default rate ties for first, carry it forward. Otherwise carry forward the lowest tied rate. This metric measures hidden-state distributions on these seven prompts. Repository-test DEV and Kaggle scores answer the task-quality question separately.

## Measured result

All three eight-update trials completed on November 18, 2026. An independent CPU replay of the scoring tensors reproduced the ranking and verified that all 4,141 adapter tensors in each raw export matched its full checkpoint bit for bit.

| State | Mean sliced Wasserstein distance ↓ |
|---|---:|
| Untrained reference | 0.606359 |
| `0.006` at update 8 | 3.139985 |
| `0.012` at update 8 | 2.227117 |
| `0.024` at update 8 | 1.380404 |

`0.024` ranked first among the three learning rates and its full checkpoint was adopted for training. The untrained reference had the lowest distance overall. The local `result.json` for this run has SHA-256 `bae80c0b9cc6c26785784d0413b7293ac2583bd13204240845dca52c3dfb1948`.

## Running the screen

Use the completed Jev TRAIN pool's progress file, its verified native response cache and the identity JSON from a live frozen base server. Prepare before allocating the native GPU. The tokenizer requests reuse the server's local tokenization endpoint.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .train-venv/bin/python scripts/screen_empf_lr.py prepare \
  --root runs/empf-lr-screen \
  --archive runs/jev-train-pool/progress.json \
  --cache runs/empf-pilot/cache \
  --train-tasks data/splits-v1/train.jsonl \
  --endpoint http://127.0.0.1:8030/v1 \
  --server-identity runs/base-server/server-identity.json

PYTHONPATH=src HF_HUB_OFFLINE=1 .train-venv/bin/python scripts/screen_empf_lr.py run \
  --root runs/empf-lr-screen
```

The runner executes the trials in series on `cuda:1` with the 50% memory cap and the shared GPU and host-memory checks. It writes the exact full and raw update-eight checkpoints, the measured probe vectors and `result.json`. After the score is available, the following command verifies the inputs and adopts the winning full checkpoint into a fresh production directory.

```bash
PYTHONPATH=src .train-venv/bin/python scripts/continue_empf_only.py \
  --rate-screen runs/empf-lr-screen \
  --screen-plan-sha256 "$(sha256sum runs/empf-lr-screen/plan.json | cut -d ' ' -f 1)" \
  --out runs/empf-production
```

The production runner resumes the exact winning optimizer and sampler state with the same rate, then emits the declared checkpoints at 256, 512 and 1,024 updates. Apply the existing staged conversion and matched DEV plan to those checkpoints. [Conversion handoff](learned-conversion-handoff.md)
