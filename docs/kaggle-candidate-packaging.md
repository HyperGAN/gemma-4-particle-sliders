# Dataset-backed learned submissions

`prepare_kaggle_candidate.py` prepares a local dataset and an offline CPU notebook for a measured PEFT candidate. The existing small OFF notebook remains separate. Before packaging, it checks exact archive members and hashes, the LoRA identity, native weight bytes and completed updates, cached response bytes, and conversion provenance binding both native metadata and cache index. These consistency checks rely on genuine local training/conversion execution; they cannot authenticate fabricated external receipts.

The pinned [Google model card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/52f3f65bc7a02d555763bc923bd1d9094898219d/README.md) identifies Apache 2.0. Learned bundles include its weights license and source notice, plus the MIT license for this repository's code. The dataset uses `other` with both licenses described, as supported by [Kaggle's dataset metadata specification](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

The prepared dataset slug contains twelve characters of the exact archive SHA-256. A binary envelope carries the ZIP without depending on archive-extension handling during upload. The notebook binds both envelope and ZIP hashes, then emits the original `submission.zip` bytes. It enables neither GPU nor internet. Its [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) names the exact dataset. Payloads contain the agent and provenance hashes; full training episodes and native weights remain local.

```bash
# After actual learned conversion, matched DEV selection and holdout confirmation:
PYTHONPATH=src .harness-venv/bin/python scripts/prepare_kaggle_candidate.py \
  --artifact runs/selected/artifact --native runs/selected/native \
  --cache runs/selected/cache --conversion-report runs/selected/peft-conversion.json \
  --owner mikkel255 --notebook-slug gemma4-particle-selected \
  --out runs/selected/kaggle-prepared
```

The command prepares files only. Publish the prepared `dataset` directory privately with the authenticated Kaggle API, push the `notebook` directory, wait for that exact notebook version to complete, and download its output. Verify the downloaded ZIP against `preparation.json` before submitting that notebook version through `competition_submit_code`. Preserve dataset, notebook/version and competition submission identifiers. An existing pending submission is polled by its ID rather than resubmitted.

CPU fixtures exercise a larger-than-512-KiB bundle, execute the emitted notebook locally, verify exact output bytes and reject corrupted payloads or changed native provenance. Licensed bundles also pass the real official compiler/ADK alias control. These fixtures are explicitly not loadable 31B candidates or real trained results. Actual dataset upload, Kaggle notebook execution and a learned official score remain unmeasured.
