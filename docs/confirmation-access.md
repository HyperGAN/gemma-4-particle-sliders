# Release the locked confirmation panel

`scripts/unlock_confirmation.py` prepares access to the original forty-nine-task confirmation panel after the actual complete DEV grid selects a learned candidate. It performs no model inference, environment calibration, confirmation evaluation, or Kaggle submission.

The gate requires the learned dispatcher's completed state, its exact immutable choice hash, all twenty-two declared condition results, and an eligible learned selection. A null choice keeps confirmation locked. It runs the original offline comparison again from the exact clean baseline DEV executor. Every recorded task, request journal, patch, verification receipt, comparison contract, and aggregate therefore faces that existing audit again without repeating model inference. The new choice must be byte-identical to the original.

Before parsing any holdout task, the gate also binds the selected adapter's actual files, conversion strength/report, raw native checkpoint, and exact response cache to the selected DEV result. It then checks the original manifest hash, all forty-nine unique IDs, the original Rich/HTTPX repository membership, and the exact visible-field allowlist. Reference fields are forbidden. The access receipt records the original choice, independent reaudit, selected packet and manifest identities. These local consistency checks cannot authenticate invented JSON as real execution.

CPU preparation can run while training and DEV are incomplete. It hashes the locked manifest as metadata and never parses its task contents:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/unlock_confirmation.py \
  --dev-grid /path/to/full-learned-dev-grid \
  --dev-executor /path/to/frozen-baseline-dev-executor \
  --dev-executor-commit EXACT_BASELINE_COMMIT \
  --packets /path/to/measured-conversions \
  --plan-only --out runs/confirmation-access-plan
```

After the actual immutable DEV choice exists, use a frozen access-gate checkout, omit `--plan-only`, and choose a fresh output directory. An incomplete or changed choice stops before the manifest parser, preserves the failure receipt, and creates no access receipt. There is no automatic retry or fallback candidate.

The [CPU preparation receipt](../reports/confirmation-access-preparation-v1.json) includes synthetic boundary controls and an actual check against the current unfinished DEV dispatcher. That check observed zero calls to the holdout parser and no access file. No real confirmation access has been issued.

Actual paired confirmation still requires a dedicated execution path with this receipt, qualified environments and one-run bookkeeping. The current TRAIN/DEV trajectory and verification entry points continue to exclude the holdout. Confirmation results cannot supply training targets or another DEV choice. A local confirmation result also cannot establish official leaderboard superiority.
