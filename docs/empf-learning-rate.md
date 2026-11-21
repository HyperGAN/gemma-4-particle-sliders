# EMPF learning-rate screen

The screening budget is eight native updates per rate on the same admitted TRAIN cache. Each trial starts from the same seed and keeps the production horizon of 1,024 updates so its first eight scheduled steps match a full run. The study varies the base rate and leaves EMPF's critic and particle-bank multipliers at 1.5 and 2.5. The penalty runs on every update with `lazy_k=1`.

| Base rate | Critic rate | Particle-bank rate |
|---:|---:|---:|
| 0.006 | 0.009 | 0.015 |
| 0.012 | 0.018 | 0.030 |
| 0.024 | 0.036 | 0.060 |

The original screen used seven other TRAIN questions with clean OFF and Verify captures. It checked that the live tokenizer reproduced the two fitting pairs before tokenizing the scoring prompts. Each scoring pair differs only in the Verify instruction. The fitting and scoring questions are disjoint.

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

## Expanded training pairs

The November 19 follow-up repeated the rate screen with four fitting pairs, then tested seven fitting pairs at the winning rate. Each trial ran eight EMPF updates with batch size 2 and `lazy_k=1`. Scoring used the same seven held-out TRAIN questions, 256 projection directions, unit slider strength and the four-pair RMS normalization. The neutral and teacher probe responses matched bit for bit across the four- and seven-pair runs.

| Fitting pairs | Base rate | Mean sliced Wasserstein distance ↓ |
|---:|---:|---:|
| Slider off | | 0.631899 |
| 4 | 0.006 | 2.872362 |
| 4 | 0.012 | 2.583717 |
| 4 | 0.024 | 1.513819 |
| 7 | 0.024 | 1.819301 |

`0.024` had the lowest distance of the three four-pair trials. At that rate the four-pair run scored ahead of the seven-pair run, and the slider-off reference scored ahead of both. That ordering held after omitting each probe in turn. These figures measure held-out TRAIN hidden-state distributions. The main native run continued from the original two-pair checkpoint. Task repair quality will be measured in the separate DEV evaluation.

The archived four-pair and seven-pair results have SHA-256 hashes `464d5d0e7e5e4e11f71ef4e953185468efc1ee83605a3c5a6451719acc7ee317` and `058085e6a710b98774a7ce4d7b564f95af2bc9d826ec39ae9b4565fc436ee627`, respectively. An independent CPU replay recomputed all distances from the recorded probe tensors.

## Step-384 checkpoint and new sweep

The original two-question run stopped at a sealed recovery checkpoint on November 19, 2026. An independent CPU replay reproduced the following means from its archived probe tensors, using the same seven scoring questions, 256 projection directions, fitting-set normalization and unit slider strength.

| State | Mean sliced Wasserstein distance ↓ |
|---|---:|
| Slider off | 0.606359 |
| Step 384 at `0.024` | 2.474720 |

The probe tensor archive has SHA-256 `de3ea69c16e3e50aa67ea4e9ad9cc732b17a5c55a0f89101cc0ce7ebc0df3b5b`. These distances describe hidden-state edits on TRAIN questions. Repair quality is measured separately with fresh tests.

The November 20 sweep completed all three rates on seven fixed, freshly verified teacher contexts from FastAPI. Two are failure-to-fix pairs. In those pairs the baseline fails and the teacher passes. The other five come from questions where the baseline passes and the teacher earns a higher behavior grade. All three trials used batch size 2, `lazy_k=1`, identical sampled rows and eight EMPF updates. Scoring used seven disjoint TRAIN questions, unit slider strength, 256 fixed directions and the RMS normalization from the seven fitting pairs.

| State | Mean sliced Wasserstein distance ↓ |
|---|---:|
| Slider off | 0.587806 |
| `0.006` at update 8 | 2.993953 |
| `0.012` at update 8 | 3.396773 |
| `0.024` at update 8 | 2.108837 |

An independent CPU replay reproduced the ranking. After omitting each probe in turn, `0.024` remained the best trained rate and the slider-off reference had the lowest distance. The exact `0.024` optimizer checkpoint was adopted for the update-64 continuation. Fresh repository tests will measure repair quality after distillation.

The scored result has SHA-256 `2e783fc6cb6e90700b8f860cb1cb2d1078c38b7215d1f8e50a2c53fde72c06a6`. The probe vector archive has SHA-256 `5110ee951a15e254d8decf7d5aff29ab3a1953b505dcd950e605340e45bc5723`.

## Update-12 distillation check

The seven-context `0.024` run continued to update 12. The conversion fitted rank-eight PEFT corrections for 230 projection modules on the seven fitting contexts, with 256 sampled tokens per module. On two disjoint TRAIN calibration contexts, the aggregate relative RMS error of the converted layer corrections was 9.82%.

The measurements below use Gemma hidden states from actual forward passes on the same seven TRAIN scoring questions, with the frozen directions and normalization from the rate sweep. The PEFT rows apply the exported matrices as linear hooks on the text model.

| State | Mean sliced Wasserstein distance ↓ |
|---:|---:|
| Slider off | 0.587806 |
| Native EMPF at update 12, unit strength | 1.556428 |
| Update-12 PEFT, unit strength | 1.493268 |
| Update-12 PEFT, half strength | 0.931554 |
| Update-12 PEFT, 0.15 strength | 0.427866 |

The 0.15 strength was selected from linear approximations to the half- and unit-strength responses, then measured with a separate direct forward pass. Its mean distance was 27.2% lower than slider off. The reduction ranged from 20.8% to 27.6% when each probe was omitted in turn. Fresh repository tests will measure repair quality in the paired DEV evaluation.

The full- and half-strength forward tensor archives have SHA-256 hashes `03d35c15524a1e871a2654768815117e58cd975329d528a5d1da30fbd00db91d` and `f91a2884a21ef6953988d735f34fe08e49e8124cea944848cf8cc71f4765c2fa`. The 0.15-strength forward tensor archive has SHA-256 hash `51d93cc5d69fc12aa8c0874df1e2e46434eb724762131d6c67f9ee4a3f44d682`. An independent CPU replay reproduced the five distances. [PEFT conversion workflow](learned-conversion-handoff.md#distill-one-31b-empf-checkpoint)

## Update-32 strength check

The same seven-context `0.024` run continued to update 32. Two direct native forward passes per probe measured unit and 0.15 strengths. They use the same seven held-out TRAIN questions, projection directions and normalization as the update-12 check.

| State | Mean sliced Wasserstein distance ↓ |
|---|---:|
| Slider off | 0.587806 |
| Update-12 PEFT, 0.15 strength | 0.427866 |
| Native EMPF at update 32, unit strength | 2.679911 |
| Native EMPF at update 32, 0.15 strength | 0.677768 |

At 0.15 strength, the native update-32 distance was 15.3% above slider off. The increase ranged from 11.7% to 27.3% when each probe was omitted in turn. The update-12 PEFT adapter at 0.15 strength is the first candidate for a paired DEV repair test. This choice uses the hidden-state probe result, and fresh repository tests will measure repair quality.

An independent CPU replay reproduced both update-32 distances from the saved forward tensors. The local comparison result has SHA-256 `36bd382305a9bd1dba7fc2ad53dc5b46f91d7c043f70508ee94f240d1978150e`.

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
  --resource-qualification /path/to/passing-resource-qualification.json \
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
