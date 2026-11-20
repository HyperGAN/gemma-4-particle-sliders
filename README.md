# Gemma 4 Particle Sliders

An application of [paired-context behavior learning](https://huggingface.co/blog/ntc-ai/particle-sliders-learn-the-behavior-keep-the-quest) to the [Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent), using source-pinned ParticleGAN **0.10.1** and an experimental Particle Sliders core.

**Work in progress.** Both Kaggle OFF controls exceeded the notebook runtime limit and ended without scores. Learned repair quality and leaderboard performance remain unmeasured. [Official OFF controls](docs/official-submission.md)

The original two-question EMPF run resumed the winning `0.024` trial and stopped at a verified step-384 checkpoint. Its mean sliced Wasserstein distance was 2.474720 with the slider on and 0.606359 with it off. As of November 20, 2026, a separate eight-update sweep is running on seven fixed TRAIN contexts, including two where a failed baseline became a passing teacher repair. [EMPF calibration results](docs/empf-learning-rate.md)

The intended behavior is systematic repository repair: reproduce a failure, locate its cause, make a focused patch, and verify it within budget. The frozen model tries several behavioral instructions on TRAIN tasks. Only complete trajectories that pass fresh-container tests and improve on the matched OFF run can supply targets. At each selected assistant decision, teacher and student receive the same question and observed history; the teacher additionally receives the successful instruction. The adapter learns the full native final hidden state through the shared paired-error adversarial game. There is no answer-token cross-entropy or output MSE in native training.

The optional [NanoGPT decision client](docs/decision-model.md) uses `typesafe/jev-latest` and passes a real typed-API control with synthetic repair traces. It is available for behavior analysis; fresh repository tests still determine task success.

A [prospective initial-decision cohort](docs/jev-initial-cohort.md) applies the article's common behavior-grade selection to successful teacher patches verified in fresh containers. Its [first real quartet](reports/first-real-behavior-admission-v1.json) supplies an initial target from Verify, with a Jev score of .99 against OFF at .40. Subsequent TRAIN collection admitted a second distinct question from Verify before native training started.

The [full EMPF execution plan](research/dev-selection-empf-v1.json) compares four controls with nine learned conditions. It tests checkpoints at 256, 512 and 1,024 updates at adapter strengths 0.5, 1 and 1.5. Each condition uses the full 42-task and 13-task panels from the [original DEV design](docs/dev-selection.md). Controls and learned candidates use matched LoRA-enabled engine settings, actual returned model aliases, and fresh patch verification. The selected condition advances to a locked 49-task comparison with OFF.

An explicit [common finalization profile](docs/submission-finalization.md) addresses observed repeated successful `submit_patch` calls. Its four-case synthetic control supports a prospective task test, and its 104-check CPU qualification passes. The original prompt and ZIP bytes remain reproducible.

To run the learned adapter once its weights are available, load its two PEFT files in vLLM with the pinned Gemma 4 31B base. [Serving quickstart](docs/learned-conversion-handoff.md#run-the-distilled-adapter)

To reproduce the adapter, prepare the paired TRAIN cache, run the 31B EMPF learning-rate screen, and continue training from the selected checkpoint. Distill a resulting raw slider export to rank-eight PEFT LoRA for the same base model. [Train the 31B EMPF slider](docs/empf-learning-rate.md#running-the-screen) · [Distill the slider](docs/learned-conversion-handoff.md#distill-one-31b-empf-checkpoint)

[CPU dose derivation](docs/lora-strength-derivation.md) reuses each measured unit-strength conversion for .5 and 1.5, avoiding repeated GPU fitting. Nonzero CPU fixture weights match separate native exports exactly; learned 31B quality remains unmeasured.

The [conversion handoff](docs/learned-conversion-handoff.md) validates completed native checkpoint receipts and prepares three adapter strengths per checkpoint. In the EMPF workflow, staged conversion can start as each checkpoint becomes available after baseline serving releases its GPU.

The [learned DEV dispatcher](docs/learned-dev-dispatch.md) waits for the measured packets and baseline controls. It checks the adapter's effect on fixed, disjoint TRAIN prefixes before it runs full DEV conditions through the frozen task executor and invokes offline selection.

The [confirmation access gate](docs/confirmation-access.md) independently reaudits that immutable DEV choice and binds the selected packet before parsing the original holdout. CPU preparation passes; current incomplete DEV state keeps the holdout locked, and no confirmation has run.

The [protected confirmation entry points](docs/confirmation-execution.md) require that access receipt, exact original execution profiles and exclusive pair/task claims. CPU guards cover all ninety-eight synthetic reservations and reject incomplete access before task/reference reads; actual confirmation remains unexecuted.

The [confirmation asset and environment runner](docs/confirmation-environments.md) fetches the missing snapshot/graph/embedding files only after release, then binds all forty-nine CPU reference controls to explicit immutable inputs. Failures stay in the panel; no actual confirmation download or calibration has run.

The [paired confirmation driver and offline audit](docs/confirmation-pair.md) preserve all forty-nine OFF and forty-nine selected-candidate attempts, exclusive reservations and fresh-test evidence. Their CPU preparation passes; no real confirmation pair has run.

The [two-GPU numerical checker and offline reader](docs/hardware-parity.md) require measured same-adapter parity before DEV uses both cards. The reader reconstructs the recorded HTTP exchanges and retains the physical device identities.

The [EMPF resource declaration](research/dev-resource-declaration-empf-v1.json) assigns the nine learned conditions across five waves. The [resource access runner](docs/resource-confirmation-access.md) independently reaudits the complete choice through a separate frozen auditor and checks the serving evidence for each condition.

The [prepared learned publication runner](docs/learned-publication.md) continues from an actually evaluated selected payload to a private dataset, an exact offline CPU notebook version and one competition request. It verifies downloaded archive bytes and actual quota, and stops after ambiguous publishing results. Preparation has made no learned account mutation or official score claim.

The [official-result observer](docs/official-result-verification.md) supports a local ZIP handoff followed by submission from the Kaggle account owner. It downloads the submitted archive by numeric ID and checks its SHA-256 against the evaluated ZIP before comparing public scores. A verified public record requires the exact numeric submission ID in the ranked entries.

The [candidate evaluation runner](docs/candidate-routing.md) now selects the official discovered adapter alias and checks its weight/config hashes against the declared server. Its compiler/ADK routing controls pass on CPU; [actual 31B LoRA loading and effect controls](reports/vllm-lora-effect-control-v1.json) also pass with synthetic zero/nonzero adapters. Learned conversion fidelity and task performance remain unqualified.

The [dataset-backed candidate notebook](docs/kaggle-candidate-packaging.md) preserves exact learned-bundle ZIP bytes beyond the small inline limit and binds native/cache/conversion provenance. Its selected-candidate gate requires complete confirmation, replays the selected effect journal and reuses the exact evaluated archive. CPU packaging and official compiler controls pass; no learned candidate has yet been uploaded or scored.

[Disjoint TRAIN issue probes](docs/conversion-probes.md) are fixed before pretrained training, and conversion samples across full histories. [Checkpoint input binding](docs/checkpoint-input-binding.md), CPU packed weights and saved-input storage now pass a full two-sequence, 28,672-token gradient probe at a 50% shared GPU cap. All 4,141 gradient tensors are connected and finite, with about 21.7 GiB peak allocation. This probe performs zero optimizer updates and supplies no targets.

[Tiny CUDA disk-resume controls](reports/cuda-cpu-loaded-resume-qualification-v1.json) verify that both recipes can resume after preserving serialized RNG bytes on CPU. The EMPF continuation verifies the previous checkpoint and available shared memory before resuming.

| Component | Current evidence |
|---|---|
| ParticleGAN | `develop` commit `53f5ff3757334f02341df66d107d27db900f4a5d`, package 0.10.1 |
| Particle Sliders | `119d7e05…` plus the hash-pinned [0.10.1 bridge patch](patches/particle-sliders-pg0101.patch); [source qualification](docs/source-upgrade-0101.md) |
| Gemma binding | Actual small random Gemma 4 architecture on CPU; updates, frozen base, exact OFF and resume checked |
| PEFT conversion | Approximation metrics and loading through the real PEFT loader checked on the CPU fixture |
| Submission | Prompt baseline compiled by official `adk-submission==0.2.13`; no inference during this check |
| Real-task scorer controls | [58 TRAIN environments](reports/train-environment-qualification-v2.json) plus [2 isolated JSON-dependency repairs](reports/train-json-environment-qualification-v1.json) qualify60 distinct original tasks across recorded images; all42 supplementary DEV environments pass |
| Behavior admission | Two distinct initial TRAIN contexts from Verify. The [first admission receipt](reports/first-real-behavior-admission-v1.json) records the actual Jev judgments and fresh test outcomes. |
| Trajectory capture | Actual ADK requests, tool schemas, generation controls and completions journaled; a separate verifier reads reference data |
| Native serving | Pinned E2B BF16 tool calling on shared GPU 1; SDPA keeps long-context memory within the declared cap; [serving evidence](reports/native-serving-calibration.json) |
| Quantized 31B host | [Frozen CT storage and exact input derivative](docs/frozen-ct-host.md); real 128/4096-token forward/backward checks pass; no updates or coding score from these probes |
| Official vLLM collection | [Pinned 31B serving](docs/vllm-collection.md) passes exact multiline tool and input-format controls; first full-budget TRAIN baseline patch passes fresh tests |
| History management | [Official compaction settings](docs/history-compaction.md) qualified through an actual ADK CPU loop; old runs retain their separate no-compaction scope |
| Competition | Both official OFF controls are terminal and unscored after exceeding the notebook runtime limit. [Official OFF controls](docs/official-submission.md) |

[sources.lock.json](sources.lock.json) records source identities. The source patch reuses an existing experimental conditional engine rather than replacing its loss or optimizer formulas. It preserves the released 0.9.5 engine and adds a separate 0.10 cohort. A Gemma-specific fix avoids forwarding the BISPF/EMPF domain keyword to RHBI's intrinsically continuous penalty. Historical checkpoints are not migrated.

Install PyTorch appropriate to the machine first. The tested training environment uses Python 3.11, Torch 2.14.1, and Transformers 5.13.1. The competition harness requires a separate Python 3.12 environment.

```bash
python3.11 -m venv --system-site-packages .train-venv
.train-venv/bin/python -m pip install 'setuptools>=84.2.1' wheel backports.tarfile
.train-venv/bin/python scripts/bootstrap.py
.train-venv/bin/python -m pip install -r requirements-model.txt pytest
CUDA_VISIBLE_DEVICES='' .train-venv/bin/python -m pytest -q \
  tests/test_gemma_binding.py tests/test_runtime.py tests/test_export.py
```

The complete CPU suite also uses the separate `.harness-venv` and `.vllm-venv` software environments for real compiler/parser checks. A [public-source portability run](reports/public-source-cpu-portability-v1.json) passes without competition task downloads or prebuilt bundles; its two GPU checks and the downloaded-wheel byte check are explicitly skipped.

The bootstrap checks exact commits and the source patch hash. Native loading also checks the installed engine and ParticleGAN source bytes before loading weights. It never downloads model weights or starts GPU work. The ordinary `pip install -e .` does not select a released ParticleGAN as a substitute for these source pins. Existing 0.10.0 collections retain their original environment and frozen checkout.

Official inputs can now be downloaded with Kaggle's CLI authentication. Use an isolated environment with `kaggle==2.2.5` and `six==1.17.0` installed; authentication remains in Kaggle's standard private location. The task file and sandbox setup match the preparation mirror. The current official `swegemma` wheel has five changed Python files despite the same package version, so source hashes are recorded and the official wheel is installed. Raw reference patches stay in ignored `data/`; agent-visible splits remove both `patch` and `test_patch`.

```bash
.kaggle-venv/bin/python scripts/fetch_official_inputs.py --wheelhouse
# After creating the sanitized split below, fetch a selected TRAIN snapshot and code indexes:
.kaggle-venv/bin/python scripts/fetch_official_inputs.py \
  --visible-tasks data/splits-v1/train.jsonl --task-id fastapi_11194
```

The pinned [public mirror](https://huggingface.co/datasets/HanG/gemma-4-developer-agent) remains a reproducible preparation source. ZIP byte ranges are checked exactly and capped at a transfer budget. [Input receipts](reports/input-provenance.json) bind the extracted members.

```bash
.train-venv/bin/python scripts/fetch_public_inputs.py  # list ZIP members, without downloading snapshots
.train-venv/bin/python scripts/fetch_public_inputs.py \
  --member root/.cache/kagglehub/competitions/gemma-4-developer-agent/tasks.jsonl
.train-venv/bin/python scripts/prepare_tasks.py \
  --source data/public-mirror/root/.cache/kagglehub/competitions/gemma-4-developer-agent/tasks.jsonl
.train-venv/bin/python scripts/build_submission.py --out dist/baseline-31b
```

The frozen pilot split is TRAIN: FastAPI (67), DEV: Requests (13), HOLDOUT: Rich and HTTPX (49). This repository-disjoint split is a small engineering pilot, not an estimate of hidden competition performance. No holdout supplies targets, conversion activations, or selection feedback. Larger training-data coverage and DEV adequacy remain research work.

[The experiment protocol](docs/protocol.md) describes the remaining work. `run_trajectories.py` runs sanitized TRAIN/DEV tasks and captures actual model requests. `verify_trajectories.py` is a separate process that alone reads reference-bearing tasks, validates matching environment identities and tests each patch in a fresh container. Curation requires all four declared arms, a failed OFF control and a complete successful instruction run from the exact pinned BF16 frozen base. Genuine OFF task timeouts retain their failure status and hash-bound initial request. It preserves observed history, tools, ADK's appended system text and captured thinking mode. Scripted CPU probes are explicitly rejected as targets.

Cache, training and conversion commands require an explicit device after CPU preparation. The 31B vLLM serving plan targets 60% GPU memory and the native EMPF training plan caps PyTorch at 50%. [Native serving instructions](docs/native-serving.md) cover the earlier SDPA cohort and request identity checks.

The first four E2B TRAIN tasks completed all four arms without a verified fix and supplied zero contexts. This is a result under the project's former five-minute pilot budget. [Collection receipts](reports/e2b-native-collection-v1.json) preserve failures and timeouts. New artifacts and trajectory runs default to the documented 60-minute budget; see [operational budgets](docs/operational-budgets.md).

Three official-vLLM 31B short-pilot cohorts completed with zero verified fixes and zero admitted targets ([receipts](reports/31b-vllm-short-pilots-v1.json)). The first baseline in a fresh 60-minute trial produced a patch that passed fresh tests after an 11-minute attempt ([receipt](reports/31b-first-full-budget-result-v1.json)); its context-limit error and incomplete capture are retained. The `localize` arm also completed a [verified solve with full capture](reports/31b-first-complete-instructed-result-v1.json) in about nine minutes; OFF already passes, so this task supplies no targets. The [full cohort is complete](reports/31b-first-full-budget-cohort-v1.json). [Its protocol](docs/vllm-collection.md) preserves the exact official prompt tokens while computing native hidden targets independently. A [supplementary 42-task DEV cohort](docs/supplementary-dev.md) was fixed across 24 repositories before inference; all 42 tasks passed independent negative/reference environment controls with workspace imports confirmed ([receipt](reports/supplementary-dev-qualification-v1.json)). The [CPU handoff](reports/supplementary-dev-handoff-v1.json) also passes. No model score has been measured on this cohort. The original Requests DEV results and Rich/HTTPX holdout remain intact.

The commands below use the Gemma 4 E2B engineering pilot.

```bash
# After complete TRAIN runs have genuine fresh-container verification receipts:
.train-venv/bin/python scripts/curate.py --outcomes data/train-outcomes.jsonl --out data/contexts.jsonl
# These next commands load pretrained weights and execute model work:
.train-venv/bin/python scripts/cache_contexts.py --contexts data/contexts.jsonl --out runs/cache --device cuda:0
.train-venv/bin/python scripts/train.py --cache runs/cache --preset empf --out runs/empf --device cuda:0 --chunk-updates 64
# Continue with the original config, cache and checkpoint; never relabel a new cohort as resume.
.train-venv/bin/python scripts/train.py --cache runs/cache --preset empf --out runs/empf --device cuda:0 \
  --resume runs/empf/FULL-0064.pt --chunk-updates 64
.train-venv/bin/python scripts/convert_to_peft.py --cache runs/cache --native runs/empf/RAW-1024 \
  --out runs/empf/peft --device cuda:0
.train-venv/bin/python scripts/build_submission.py --model gemma-4-e2b-it-qat-w4a16-ct \
  --adapter runs/empf/peft --out dist/empf-e2b
```

RHBI's public recipe uses a CPU polar execution profile. CUDA training requires the explicitly recorded `--gpu-native-polar` derivative; it is not silently selected. Native research uses BF16 pretrained weights with FP32 adapter/critic parameters. Production W4A16 quantization is a separate evaluation condition. Current source recipes and default rates are preserved; a learning-rate search must be declared before execution.

Successful CPU tests do not establish better coding behavior. Genuine task outcomes, conversion loss, wall time, error rates, model/quantization/harness identity, and the official scored submission remain the evidence needed to meet the goal.

The [original blocked verifier receipt](reports/verifier-control-v1.json) remains unchanged. Later controls exposed missing test dependencies, imports of an installed package instead of edited workspace code, and an unkeyed shared wheel cache. The declared local images add dependencies and prioritize workspace imports; each run gets a private cache. Tests, air gap, clock, TLS validation and scoring remain unchanged. The calibrated FastAPI and Requests utility tasks now distinguish an empty patch from their reference fixes. This qualifies those exact local task/environments; it does not establish hidden-scorer parity or agent accuracy.

[Context-budget controls](docs/context-budget-controls.md) describe the new prospective OFF failure proof and conditional TRAIN teacher policy. [Broader TRAIN environment qualification](docs/train-environments.md) records the fixed 67-task sweep and separate missing-dependency repairs.

[Fixed-order TRAIN collection](docs/train-pool-collection.md) preserves failures and captures exact target tokens before releasing its server. The collector can also run a fixed 12-task pilot with OFF and Verify on each task and two additional teacher arms on four comparison tasks. The isolated [0.10.1 source upgrade](docs/source-upgrade-0101.md) passes 57 CPU checks and exact one-update fixture comparisons for both explicit recipes. All 87 Python files in the downloaded 0.10.1 wheel match the pinned source ([source check](reports/pg0101-wheel-source-check-v1.json)).
