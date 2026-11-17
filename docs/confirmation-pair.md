# Run the locked confirmation pair

`run_confirmation_pair.py` runs the original forty-nine confirmation tasks once with OFF and once with the candidate already selected by the complete DEV grid. It requires the released [access receipt](confirmation-access.md) and the complete [environment qualification](confirmation-environments.md). The original task order and all failed environment controls remain in the report.

Prepare a declaration without opening the access receipt, calibration results or held-out tasks, querying the GPU driver, or launching a model:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/run_confirmation_pair.py \
  --access /path/to/future/access.json \
  --environments /path/to/future/qualification.json \
  --executor /path/to/frozen-confirmation-executor \
  --executor-commit EXACT_COMMIT --plan-only \
  --out runs/confirmation-pair-plan
```

Actual execution uses the same arguments without `--plan-only`, a new output directory, and inherited `nice -n 10`. The runner claims the single permitted pair before model work. It waits for GPU 0 to have the declared 60% engine allocation plus 8 GiB startup headroom, then starts one owned server with the selected packet. OFF calls the true base alias; the selected condition calls the adapter alias on that same server. Both conditions use the original common finalization, tool, sixty-minute budget, sampling and compaction profiles. The fixed protected executor performs generation and fresh-container verification.

The preflight checks registration and a finite model response. It does not establish a numerical learned-adapter effect. That qualification remains a separate requirement before learned DEV execution. The runner records every attempted confirmation case and stops on a process or audit failure. Partial attempts retain their exclusive reservations and need explicit reconciliation; restarting with a new output directory cannot silently repeat the panel. Shutdown targets only the server started by this runner.

Each complete condition binds all forty-nine model journals, patches, fresh test receipts, reservations, environment qualification and source hashes. `compare_confirmation.py` independently reaudits those files:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/compare_confirmation.py \
  --access /path/to/released/access.json \
  --environments /path/to/qualification.json --pair /path/to/pair-claim.json \
  --condition-result /path/to/pair/off/result.json \
  --condition-result /path/to/pair/SELECTED_ID/result.json \
  --out runs/confirmation-comparison
```

The comparison reports the full denominator, environment-qualified subset, repository counts, Wilson intervals, errors, tool proposals, elapsed time and paired wins/losses/ties with an exact discordance test. An agent error with an empty returned patch remains unresolved under the official rule, while the raw fresh-test result is preserved. Confirmation never chooses another candidate. Local file consistency checks cannot authenticate fabricated artifacts, and a confirmation gain is not an official leaderboard score.

The [preparation report](../reports/confirmation-pair-preparation-v1.json) records synthetic CPU controls and an actual declaration-only run. No real confirmation access, model run or score was produced during preparation.
