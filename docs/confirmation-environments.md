# Prepare the full confirmation environment panel

The current official input cache contains eighty TRAIN/DEV snapshots and no Rich/HTTPX assets. Confirmation therefore needs an explicit download step after the [locked access receipt](confirmation-access.md) exists. No confirmation download or reference control has run during this preparation.

`fetch_official_inputs.py --confirmation-access ACCESS --all-confirmation` derives all forty-nine snapshot paths and their graph/embedding paths from the released manifest. It rejects other task IDs and unsafe task/commit names. The ordinary TRAIN/DEV path still rejects holdout tasks. Kaggle is imported only after access validation; `--plan-only` needs no Kaggle import, account access or network request. Its requested paths do not prove provider availability or download size.

For the actual download, use the frozen source and an isolated Kaggle environment with the existing private authentication:

```bash
PYTHONPATH="$PWD/src" /path/to/kaggle-env/bin/python scripts/fetch_official_inputs.py \
  --confirmation-access /path/to/released/access.json --all-confirmation \
  --out data/kaggle-official --max-download-gib 4
```

The provider file listing must fit the explicit transfer budget before downloads begin. A fresh official task source is checked before the larger assets; existing files retain their recorded hashes. The download receipt binds each released batch to its access identity and records file provenance. Repeated calls verify and reuse existing files.

`verify_confirmation_environments.py` then runs the unchanged official negative/reference verifier in separate CPU children for all forty-nine tasks. Its default environment uses the original immutable secondary-DEV image and original wheel profile. Explicit alternatives are recorded as their own immutable calibration cohort. One copied wheel profile, image, setup and official snapshot/graph/embedding files apply to the whole panel. Two workers, hidden CUDA devices and inherited `nice -n 10` share the host.

CPU declaration preparation is available before release without parsing any confirmation task:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/verify_confirmation_environments.py \
  --access /path/to/future/access.json \
  --executor /path/to/frozen-confirmation-verifier \
  --executor-commit EXACT_COMMIT --plan-only \
  --out runs/confirmation-environment-plan
```

Actual mode requires the real receipt first and every asset's official post-access download provenance. Each control binds the original reference dataset, snapshot, image, wheels, execution source, access identity, terminal exit and log. Qualification requires an unresolved clean negative, a resolved clean reference, and both workspace-origin checks. Failed references and evidenced failures before a receipt remain in the forty-nine-record report. The CLI returns 2 when some environments fail qualification; these are calibration outcomes, not inferred agent results.

The qualification reader rechecks the complete ordered panel, input and wheel bytes, original download receipt, sources and all control receipts. The [CPU preparation report](../reports/confirmation-environments-preparation-v1.json) uses synthetic fixtures and a fake Kaggle service, plus a declaration plan against actual local software/image/wheel metadata. It establishes no real confirmation download, control result or model score. The complete paired driver and its audited comparison remain required.
