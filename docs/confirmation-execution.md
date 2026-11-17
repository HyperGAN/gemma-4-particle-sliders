# Execute only the released confirmation pair

The ordinary TRAIN/DEV entry points still reject holdout tasks. Their explicit `--confirmation-access` path loads the [post-selection access receipt](confirmation-access.md), rechecks the original choice and selected model bytes, and preserves the task label `holdout`. Supplementary DEV relabeling and CPU protocol fixtures cannot enter this path.

`run_trajectories.py` and `verify_trajectories.py` also require `--confirmation-pair`. Create that claim only for the actual released choice:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/claim_confirmation_pair.py \
  --access /path/to/released/access.json --out runs/locked-confirmation-pair
```

The printed claim is anchored beside the original manifest, outside the output directory. A second output directory or another choice cannot obtain a new claim for that panel. Each permitted condition/task pair has an exclusive reservation before its model run. The verifier checks the same reservation and cohort path. Completed failures remain attempts; there is no automatic reroll. An interrupted process requires evidence-preserving reconciliation.

Both treatments use the OFF prompt: true base alias `gemma-4-31b-it-qat-w4a16-ct`, and the exact selected adapter under `particle_lora`. Other adapters, teacher instructions and changed execution profiles are rejected. Generation requires `tools-v2`, `repair-finalize-v2`, `official-default-60m-v1`, `official-compaction-v2` and `context-budget-off-v1`. The system-prompt and sampling bytes must match the locked plan. A candidate server may advertise both aliases; OFF uses the actual base alias without an active adapter. The matched DEV serving contract, including physical GPU 0, remains enforced.

For each of the forty-nine original tasks, use the original visible manifest, the released receipt and pair claim, and the qualified immutable image/input/wheel profile. Run `run_trajectories.py` once under each permitted condition, passing the selected `--adapter` only for the learned treatment. Every output cohort must be under the claimed pair root. Then use `verify_trajectories.py` with the same `--confirmation-access` and `--confirmation-pair`, the original official reference dataset and the captured environment. Only the verifier reads references.

`verify_controls.py --split holdout --confirmation-access ACCESS` provides the separate negative/reference environment controls after release. It binds the original official dataset and selected task to the receipt and checks Rich/HTTPX imports from the edited workspace. It performs no model inference and does not consume a model-task reservation. Reference results remain calibration evidence, never an agent score.

The [CPU preparation receipt](../reports/confirmation-execution-preparation-v1.json) covers default split rejection, exact released profiles/model bytes, provenance tampering, exclusive claims and all ninety-eight synthetic task reservations. Three actual CLI controls reject incomplete access before touching their sentinel task/reference paths. The syntax and arguments of the named official agent/compiler/model/sandbox/verifier calls match the frozen DEV executor; this is a bounded source check, not a complete semantic proof.

No real access, pair claim, calibration or confirmation task has run. Full-panel environment calibration, a complete paired driver and its audited comparison report remain required. The confirmation path retains holdout labels, which TRAIN curation rejects; its results cannot supply another DEV choice. Local confirmation cannot establish official leaderboard superiority.
