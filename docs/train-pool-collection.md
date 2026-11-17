# Fixed-order TRAIN collection

The original pool and separately declared dependency repairs qualify 58 of the 67 TRAIN tasks with explicit per-task wheel profiles ([environment view](../reports/train-environment-qualification-v2.json)). Nine failures remain visible. The environment selection uses no model outcomes. Original control grades remain immutable.

`collect_train_pool.py` declares its task order, exclusions, frozen executor, exact model identity, source hashes and stopping rule before launching a model attempt. It uses the original TRAIN order, skips unqualified environments and explicit reservations for other collectors, and checks each snapshot, wheel profile and container image before execution.

Each task uses the documented 60-minute budget, `tools-v2` instructions, `context-budget-off-v1` capture and `failed-off-only` teacher policy. A failed usable OFF control triggers all three teachers. Curation still requires the matched four-arm evidence and a completely captured verified teacher. An already passing baseline or unusable control remains in the results with a reason for skipping teachers. These selective TRAIN results are not a four-arm benchmark score.

The runner retains all admitted contexts, including any single-row result while it continues collecting. After a completed task brings the total to at least two, it captures the exact positive/neutral token IDs from the bound official server, records the transport and stops. This is an initial data-collection stage; a ready dataset does not establish learned quality or complete the competition objective.

```bash
.harness-venv/bin/python scripts/collect_train_pool.py \
  --root runs/own-train-pool \
  --executor /path/to/frozen-executor \
  --executor-commit 1d439443297ab3672d5e3305fd121af67bb1be9e \
  --qualification reports/train-environment-qualification-v2.json \
  --server-identity runs/own-server/server-identity.json \
  --endpoint http://127.0.0.1:8028/v1
```

The executor must have the existing harness/training environments and downloaded inputs available. An optional `--stop-server-pid` signals only a process whose serving command and output directory match this server identity. The GPU's actual release must subsequently be observed. Pipeline errors preserve the incomplete directory and require explicit reconciliation. `--plan-only` prints a CPU preparation preview without starting inference.

The [control receipt](../reports/train-pool-controls-v1.json) records 54 CPU checks, the actual frozen-executor tokenizer CLI, and a live 55-task pool plan excluding the three separately declared TRAIN tasks. Both collection engines use one shared A6000 each with a .60 vLLM memory target. No additional native training updates have run.
