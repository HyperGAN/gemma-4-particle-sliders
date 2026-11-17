# Fixed-order TRAIN collection

The original pool and separately declared dependency repairs qualify 58 of the 67 TRAIN tasks with explicit per-task wheel profiles ([environment view](../reports/train-environment-qualification-v2.json)). Nine failures remain visible. The environment selection uses no model outcomes. Original control grades remain immutable.

`collect_train_pool.py` declares its task order, exclusions, frozen executor, exact model identity, source hashes and stopping rule before launching a model attempt. It uses the original TRAIN order, skips unqualified environments and explicit reservations for other collectors, and checks each snapshot, wheel profile and container image before execution.

Each task uses the documented 60-minute budget, `tools-v2` instructions, `context-budget-off-v1` capture, [compiled canonical-model history compaction](history-compaction.md) (`official-compaction-v2`) and `failed-off-only` teacher policy. A failed usable OFF control triggers all three teachers. Curation still requires the matched four-arm evidence and a completely captured verified teacher. An already passing baseline or unusable control remains in the results with a reason for skipping teachers. These selective TRAIN results are not a four-arm benchmark score.

The runner retains all admitted contexts, including any single-row result while it continues collecting. After a completed task brings the total to at least two, it captures the exact positive/neutral token IDs from the bound official server, records the transport and stops. This is an initial data-collection stage; a ready dataset does not establish learned quality or complete the competition objective.

```bash
.harness-venv/bin/python scripts/collect_train_pool.py \
  --root runs/own-train-pool \
  --executor /path/to/frozen-executor \
  --executor-commit FULL_COMMIT_OF_FROZEN_EXECUTOR \
  --history-profile official-compaction-v2 \
  --qualification reports/train-environment-qualification-v2.json \
  --server-identity runs/own-server/server-identity.json \
  --endpoint http://127.0.0.1:8028/v1
```

The executor must contain the current compaction source and have the harness environment, `.train-venv`, and downloaded inputs available. Ten source files bind the default compacted collector. An optional `--stop-server-pid` signals only a process whose serving command and output directory match this server identity. The GPU's actual release must subsequently be observed. Pipeline errors preserve the incomplete directory and require explicit reconciliation. `--plan-only` prints a CPU preparation preview without starting inference.

The original [control receipt](../reports/train-pool-controls-v1.json) records 54 CPU checks, the frozen-executor tokenizer CLI and the original 55-task no-compaction plan. That pool closed after four tasks with zero targets; 51 unattempted tasks were withdrawn after discovering the history-profile discrepancy. Its results retain their original source and scope. Serving uses shared A6000 hardware with a .60 vLLM memory target. No pretrained optimizer updates have run.
