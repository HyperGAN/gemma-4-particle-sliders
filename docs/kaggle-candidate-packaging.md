# Dataset-backed learned submissions

`prepare_selected_candidate.py` connects the immutable DEV choice and the complete paired confirmation audit to a local dataset and offline CPU notebook. It reaudits the selected candidate's original HTTP effect observations and uses the exact archive already exercised during DEV and confirmation. Every selected DEV/confirmation case must record that same build receipt. The common finalization prompt, sampling, tools, sixty-minute budget and adapter bytes must match the original comparison. Confirmation outcomes are reported without changing the DEV choice or introducing a confirmation promotion threshold.

The lower-level `prepare_kaggle_candidate.py` writes the payload after these experiment gates. It checks exact archive members and hashes, the LoRA identity, native weight bytes and completed updates, cached response bytes, and conversion provenance binding both native metadata and cache index. These consistency checks rely on genuine local execution; they cannot authenticate fabricated external receipts.

The pinned [Google model card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/52f3f65bc7a02d555763bc923bd1d9094898219d/README.md) identifies Apache 2.0. Learned bundles include its weights license and source notice, plus the MIT license for this repository's code. The dataset uses `other` with both licenses described, as supported by [Kaggle's dataset metadata specification](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

The prepared dataset slug contains twelve characters of the exact archive SHA-256. A binary envelope carries the ZIP without depending on archive-extension handling during upload. The notebook binds both envelope and ZIP hashes, then emits the original `submission.zip` bytes. It enables neither GPU nor internet. Its [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) names the exact dataset. Payloads contain the agent and provenance hashes; full training episodes and native weights remain local.

```bash
# Inside the frozen packaging checkout, after the complete confirmation pair:
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" .harness-venv/bin/python \
  scripts/prepare_selected_candidate.py \
  --access /path/to/released/access.json \
  --environments /path/to/qualification.json --pair /path/to/pair-claim.json \
  --comparison /path/to/pair/comparison.json \
  --owner mikkel255 --notebook-slug gemma4-particle-selected \
  --out runs/selected/kaggle-prepared
```

`--plan-only` prepares a declaration without opening access, calibration, confirmation or effect records and without account requests. Actual mode requires released access before reading protected results, reconstructs the complete pair and selected effect observations, and preserves an immutable readiness receipt. It performs no model inference, upload or submission.

After actual preparation, publish `payload/dataset` privately with the authenticated Kaggle API, push `payload/notebook`, wait for that exact notebook version to complete, and download its output. Verify the downloaded ZIP against `payload/preparation.json` before submitting that notebook version through `competition_submit_code`. Preserve dataset, notebook/version and competition submission identifiers. An existing pending submission is polled by its ID rather than resubmitted.

CPU fixtures exercise a larger-than-512-KiB bundle, execute the emitted notebook locally, verify exact output bytes and reject corrupted payloads or changed native provenance. Licensed bundles also pass the real official compiler/ADK alias control. These fixtures are explicitly not loadable 31B candidates or real trained results. Actual dataset upload, Kaggle notebook execution and a learned official score remain unmeasured.

The [selected-candidate preparation report](../reports/selected-candidate-preparation-v1.json) adds evaluated-archive, protocol and complete HTTP-journal controls. This is CPU preparation; no real candidate has passed the experiment gates.
