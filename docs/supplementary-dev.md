# Supplementary DEV protocol

The original 13 Requests DEV tasks remain recorded, including the blocked network/TLS reference controls. Rich and HTTPX remain untouched holdouts. A separate supplementary cohort uses public [FreshSWE-Py](https://www.kaggle.com/datasets/moshegolderson/freshswe-py) tasks to seek a broader test environment that can support model selection.

Before model inference, `prepare_external_dev.py` selects 42 issue-sourced tasks across 24 repositories. It excludes every original split repository and the FastAPI/Textualize organizations. It takes at most two tasks per repository by fixed SHA256 ordering. Neither model outcomes nor reference-patch difficulty enter selection. Only the visible allowlist reaches an agent. These tasks are forbidden as training or conversion inputs.

```bash
.kaggle-venv/bin/kaggle datasets download moshegolderson/freshswe-py \
  -p data/freshswe/bundle --quiet
.kaggle-venv/bin/kaggle datasets download moshegolderson/freshswe-py \
  -f tasks.jsonl -p data/freshswe/raw --unzip --quiet
.venv/bin/python scripts/prepare_external_dev.py
.venv/bin/python scripts/extract_external_dev.py
docker build -t gemma-particle-sandbox:freshswe-v1 \
  -f data/freshswe/selected/docker/Dockerfile data/freshswe/selected
.harness-venv/bin/python scripts/verify_external_dev.py \
  --out runs/own-external-dev-controls
```

Extraction takes only the fixed snapshots/graphs/embeddings/locks, the provider setup/Dockerfile and offline wheelhouse. Baseline patches/results are excluded. Every extracted byte and the downloaded bundle are hash-bound. The provider setup was inspected before execution; it installs per-task locked dependencies and the workspace in the container.

The provider's validation claims do not qualify our environment. Each task must independently fail with an empty patch and pass with its reference patch through unchanged official `verify_task`, in fresh offline 4 GiB containers. An import probe must resolve the tested library under `/workspace/`. These CPU oracle controls never supply agent answers or learned targets. The runner retains all failures and does not silently remove them from the fixed cohort.

The first image is `sha256:13e0e36764a6e1b2c052daeb204f8051fe7c9be8bda6716278cfac775de6bfa9`, built from the downloaded Dockerfile and pinned input bytes. Full-cohort qualification is still running. A passing subset alone is not a DEV benchmark score.
