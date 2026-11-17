# Run the complete learned DEV grid

`scripts/run_learned_dev_grid.py` connects the [measured conversion handoff](learned-conversion-handoff.md) to the existing frozen DEV executor. It waits for all eighteen measured packets, all four complete baseline controls, both producer processes to exit, and the owned baseline server to release its allocation. Live producer identities bind their exact command, output directory, PID, and birth time.

The supervisor revalidates both actual full native trajectories and every packet's native/cache/probe/strength/adapter bindings before spending GPU time. Every learned condition uses the same frozen `run_dev_condition.py`, generation, verification, serving code, prompts, budgets, package checks, and immutable images as the baseline. The current comparison remains on physical GPU 0. The separate hardware resource proposal is inactive; this dispatcher does not normalize device indices or activate it.

Each candidate server waits for its .60 memory target plus 8 GiB of observed free GPU memory. Four CPU threads and the launcher's inherited `nice -n 10` share the host. A fixed synthetic prompt checks the actual learned alias and one finite token-ID completion before any DEV task runs. The separate [learned-effect protocol](../research/learned-effect-protocol-v1.json) then compares the base and learned aliases on both original disjoint TRAIN conversion prefixes. These prefixes were fixed before any pretrained optimizer update and retain their old `repair-v1` prompt condition. No DEV or confirmation question is used.

For each prefix, the fixed order is base twice, then learned twice: eight one-token completions per candidate, at temperature zero and seed zero. Both repeats of each alias must agree in generated token, top-twenty token-ID set and log probabilities within `1e-5`. At least one prefix must show a base-versus-learned token or top-set difference, or a common-token log-probability difference greater than `1e-5`. The observation binds the exact packet, strength, native checkpoint, cache, conversion and probe hashes. A missing effect or unstable repeats stop the dispatcher with the actual observation retained. No threshold is relaxed, task repeated, or condition automatically labeled unqualified. These numerical controls supply no task-quality or native-conversion fidelity evidence. HTTP attempts and responses are journaled; observation errors propagate without retries or an inferred result.

The dispatcher runs every condition in the original plan order and all 42+13 tasks per condition. It stops on failures, preserving every completed task and partial log; it never rerolls a task or automatically labels a condition unqualified. Only its owned candidate server receives a stop signal, and the next candidate waits for actual process release and GPU headroom. After all eighteen conditions complete, the original offline comparison reaudits all twenty-two results and writes the immutable DEV choice. A null choice remains a real possible result.

Prepare without CUDA or model access first:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/run_learned_dev_grid.py \
  --conversion /path/to/measured-conversion-root \
  --controls /path/to/full-baseline-controls \
  --executor /path/to/frozen-baseline-dev-executor \
  --executor-commit EXACT_BASELINE_COMMIT \
  --plan-only --out runs/learned-dev-plan
```

Freeze the supervisor checkout before arming it. Omit `--plan-only`, provide `--conversion-pid` and `--controls-pid` for producers that are still running, choose a fresh output directory and a free loopback `--port`, and launch with `nice -n 10`. Completed producers do not require live PIDs. Source and declaration pins are rechecked before each candidate starts.

The [original preparation receipt](../reports/learned-dev-dispatch-preparation-v1.json) covers synthetic packet tampering, baseline denominators, an actual CPU child process identity, mock HTTP alias/error/timeout boundaries, and a plan against the live producer declarations. The [effect preparation receipt](../reports/learned-effect-preparation-v1.json) adds fixed-protocol, numerical effect, repeatability and observation-failure controls. These checks establish CPU preparation. Actual learned effects, learned DEV results, the locked 49-task confirmation, and an official learned score remain required. This supervisor never opens the holdout or submits to Kaggle.
