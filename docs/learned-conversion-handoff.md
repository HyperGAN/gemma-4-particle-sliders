# Distill and serve a trained Particle Slider

Train the native EMPF slider first. `FULL-*.pt` saves the optimizer and sampler state for resuming training. `RAW-*/` contains the trained slider tensors used for distillation. To produce a standard PEFT LoRA, the exporter fits each nonlinear routed bottleneck with a linear projection and measures the approximation on separate TRAIN probes. The report records the source checkpoint, response cache, sample counts and error per layer.

## Run the distilled adapter

Once the learned export is available, put its `adapter_config.json` and `adapter_model.safetensors` together in an `empf-peft/` directory. The PEFT files are enough to serve the adapter with vLLM. The conversion receipt records how they were produced. Training checkpoints and Kaggle task files are only needed to reproduce the experiment.

The local reference setup uses one 48 GiB NVIDIA RTX A6000, vLLM 0.19.1, compressed-tensors 0.15.0.1, and Transformers 5.13.1. With that software installed, start the server from the directory containing `empf-peft/`. vLLM downloads the pinned [Gemma 4 31B base](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) if it is missing from the Hugging Face cache.

```bash
vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
  --revision 52f3f65bc7a02d555763bc923bd1d9094898219d \
  --tokenizer-revision 52f3f65bc7a02d555763bc923bd1d9094898219d \
  --host 127.0.0.1 --port 8027 --dtype bfloat16 \
  --gpu-memory-utilization 0.60 --max-model-len 32768 \
  --max-num-seqs 1 --max-num-batched-tokens 2048 --enforce-eager \
  --language-model-only --no-scheduler-reserve-full-isl \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}' \
  --enable-lora --max-lora-rank 8 --lora-modules particle_lora=./empf-peft
```

Once the server is ready, request the `particle_lora` alias to activate the adapter.

```bash
curl -fsS http://127.0.0.1:8027/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"particle_lora","messages":[{"role":"user","content":"How do I reverse a Python list without modifying it?"}],"max_tokens":128}'
```

## Distill one 31B EMPF checkpoint

Use a sealed raw export, its original TRAIN response cache, and the disjoint issue-only packet from `prepare_conversion_probe.py`. The example uses update 256. Run it from the repository root with the pinned model and wheelhouse available and sufficient headroom on the chosen device.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .train-venv/bin/python scripts/convert_to_peft.py \
  --config configs/31b-ct-shared-positional-pilot.json \
  --native runs/empf-production/empf/RAW-0256 \
  --cache runs/empf-pilot/cache \
  --probe-contexts data/conversion-probe-v1/contexts.json \
  --device cuda:0 --memory-fraction .50 --out runs/empf-peft
```

The output directory contains `adapter_config.json` and `adapter_model.safetensors`. The adjacent `runs/empf-peft-conversion.json` records the approximation. The repository's evaluation launcher also checks that receipt before serving the adapter.

```bash
PYTHONPATH=src .vllm-venv/bin/python scripts/serve_vllm_lora.py \
  --adapter runs/empf-peft --conversion-report runs/empf-peft-conversion.json \
  --device cuda:0 --port 8027 --out runs/empf-peft-server
```

Kaggle packages the same PEFT files under `adapters/<alias>/`, with the agent YAML selecting that alias. The official starter's four-L4 command accepts a fixture matching the rank and all 230 attention projection shapes. The CPU check verified adapter discovery and command construction, including rank 8, tensor parallelism 4, and the CUDA graph memory profiler setting. Hosted adapter behavior requires a real inference result. [Official starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent) · [Kaggle staff clarification](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191)

## Prepare all nine EMPF evaluation conditions

Run `scripts/prepare_learned_sweep.py` with `--selection-plan research/dev-selection-empf-v1.json`. It verifies the three native checkpoints at 256, 512 and 1,024, then fits one adapter per checkpoint and derives strengths 0.5, 1 and 1.5. To convert completed checkpoints while native training continues on the other GPU, supply the staged controls root and the matching process IDs with `--staged-controls`, `--staged-controls-pid` and `--training-pid`. The dispatcher waits for the controls to finish and release the conversion GPU. Use the resulting candidate packets with the matched DEV plan and [prepare a selected Kaggle submission](kaggle-candidate-packaging.md) after evaluation.

## Original two-preset grid

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
