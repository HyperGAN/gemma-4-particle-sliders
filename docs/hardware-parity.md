# Prepare the proposed two-GPU numerical control

The [resource proposal](../research/dev-hardware-resource-proposal-v1.json) preserves the full twenty-two-condition study while considering both shared A6000 slots after native training and the four baseline controls complete. It remains inactive. `scripts/qualify_hardware_pair.py` prepares and measures its fixed numerical control; it does not activate a resource supplement or change DEV contract comparison.

The [frozen inputs](../research/dev-hardware-probe-inputs-v1.json) repeat fourteen token IDs from the earlier actual synthetic LoRA transport control. The exact synthetic string must still tokenize to those IDs under both aliases on both servers. Cyclic repetition produces exactly 32, 4,096 and 28,672 input tokens. Each length uses the base and learned aliases, one greedy output token, seed zero and twenty top log probabilities. Both slots receive identical request bytes. There is no task content in the probes.

The fixed adapter is the measured EMPF step-256 unit-strength packet, chosen prospectively rather than by a DEV outcome. Before HTTP inference, the checker requires actual completed native/conversion and baseline producers, their process release, no prior learned DEV inference, and observed exit of the owned waiting learned dispatcher. The current waiter remains running; the actual-mode boundary control confirms that it blocks qualification before any HTTP or driver query. The checker never starts, stops or replaces a server.

For each actual server, the checker binds the frozen serving source, exact adapter, command, endpoint, PID, birth time and physical device. It compares the original model, official wheel/package identities and engine settings with the baseline, including a .60 memory target and recorded 8 GiB startup headroom. Actual driver/hardware descriptors and descendant GPU-process UUIDs must match the original two slots. Other applications are excluded from the owned-process receipt and remain untouched.

All six paired cases must preserve the generated token ID and the complete top-twenty token-ID set, with maximum absolute log-probability difference at most `1e-5`. Actual HTTP attempts and responses are journaled. Transport errors produce an incomplete observation without a numerical verdict, restart or tolerance change. A passed numerical control still requires an explicit versioned resource supplement and comparison-audit support. The ordinary DEV canonical contract continues to distinguish physical GPU indices.

The checker also records each actual identity file, launcher command, PID/birth binding, observed descendant process IDs, original prerequisite progress hashes and named input roots. Identity files must stay unchanged throughout measurement. These fields let the later offline reader reconstruct the numerical observation after the owned servers have exited.

`scripts/audit_hardware_pair.py` replays all twenty HTTP exchanges, including twelve completions, recomputes all six numerical comparisons, and rechecks the source, prerequisite, packet matrix, identity, command and GPU-process records. It preserves both physical device identities and does not activate a resource supplement:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/audit_hardware_pair.py \
  --qualification /path/to/completed/hardware-control \
  --out runs/hardware-control-offline-audit
```

This reader verifies consistency of recorded artifacts. It cannot authenticate fabricated records, and six synthetic probes cannot establish equivalence on all possible inputs. The current [offline preparation report](../reports/hardware-evidence-preparation-v1.json) uses explicitly synthetic CPU artifacts. No real two-slot numerical control has passed.

Prepare on CPU without any HTTP request, driver query, model or CUDA access:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/qualify_hardware_pair.py \
  --dev-grid /path/to/full-learned-dev-grid \
  --conversion /path/to/measured-conversions \
  --controls /path/to/full-baseline-controls \
  --executor /path/to/frozen-baseline-dev-executor \
  --executor-commit EXACT_BASELINE_COMMIT \
  --plan-only --out runs/hardware-parity-plan
```

For the future actual measurement, use a frozen checker checkout and a fresh output directory. Supply `--left-identity`, `--left-pid`, `--right-identity`, `--right-pid`, the two local root endpoints, and the exact stopped dispatcher's `--dev-dispatch-pid` and `--dev-dispatch-birth-ticks`. Both already-running servers must use the same measured packet and original frozen serving executor, on physical GPUs 0 and 1 respectively. These prerequisites are not currently satisfied.

The [CPU preparation receipt](../reports/hardware-parity-preparation-v1.json) records numerical mismatch controls, actual process barriers, and read-only hardware/process metadata checks. No real two-GPU numerical parity, activated resource protocol, learned DEV score or holdout result is established by that receipt.
