# Publish the exact evaluated learned candidate

`scripts/publish_selected_candidate.py` continues from [selected-candidate packaging](kaggle-candidate-packaging.md). Actual mode reconstructs the original released access, qualified environments, complete paired confirmation comparison, selected evaluated archive, original twenty-four HTTP effect-journal entries and prepared payload before importing the authenticated Kaggle client. The DEV choice remains fixed even if confirmation is disappointing. This runner neither selects a new candidate nor runs a model locally.

The payload audit allows only the two declared dataset resources and their metadata, plus the canonical offline CPU emitter notebook and its metadata. It checks the full binary envelope and embedded archive hashes, adapter and native/cache/conversion provenance, original preparation source and private CPU settings. Extra files, executable notebook changes and altered dataset bindings fail before publication.

Publication creates a private dataset once and observes its processing state, privacy and version. A separate notebook staging directory preserves the prepared notebook bytes. Its only metadata changes make the title match the requested slug and pin the dataset to version 1. The notebook retains the [Kaggle metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) settings for private execution with GPU, TPU and internet disabled. Dataset resources follow [Kaggle's dataset metadata interface](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

The new notebook must return the expected owned URL and version 1. Both status observation and output download use that explicit version. The downloaded `submission.zip` must match the exact archive evaluated in DEV and confirmation. An unversioned latest notebook is never substituted. Competition submission supplies that same kernel and version through the [code competition interface](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md).

Before each publishing mutation, the runner checks unchanged experiment and source bindings and writes an exclusive, durable intent. Each operation gets one SDK invocation; an exception or unverifiable response preserves the intent and stops for reconciliation. The dataset adapter disables the SDK's create-request retry wrapper. Resumable byte transfers and repeated read observations do not create another dataset, notebook version or competition attempt. An existing remote dataset or notebook is not overwritten. Re-entering a used publication output directory is rejected.

The competition gate checks actual pending submission handles and Kaggle's current server allowance. Unknown states remain pending. A transient status-read failure observes the same handle again; it does not imply terminal failure or restart notebook work. An already matching submission description stops for explicit reconciliation. A created competition ID means queued evaluation, with no score or SOTA claim. Poll that exact ID and compare a completed learned score against a freshly observed leaderboard.

The publisher also records the public best immediately before its competition request and hashes that observation into the intent. The [prepared official-result observer](official-result-verification.md) retains this reference, reads the exact created learned submission and verifies any displayed own entry through its numeric public submission ID. A control, tie, pending evaluation or stronger preceding record cannot become a strict learned-record verdict.

The [runtime declaration](../research/kaggle-publication-runtime-v1.json) pins the tested Kaggle 2.2.5 and kagglesdk 0.1.38 interfaces, including six source files for the API and typed requests, responses and enums. Actual mode rejects an altered interface before publication. This is a bounded compatibility check, not authentication of an entire environment or of invented experiment records.

```sh
# Run from a clean frozen publication checkout after genuine readiness exists.
env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .kaggle-venv/bin/python \
  scripts/publish_selected_candidate.py \
  --readiness /path/to/selected/readiness.json \
  --access /path/to/released/access.json \
  --environments /path/to/confirmation/qualification.json \
  --pair /path/to/pair-claim.json \
  --comparison /path/to/confirmation/comparison.json \
  --out /path/to/new-immutable-publication
```

`--plan-only` writes source and operation declarations without opening protected inputs, importing an account client or making a request. The [preparation report](../reports/learned-publication-preparation-v1.json) separates real local CPU controls and installed-SDK interface checks from fake service responses and supplied experiment boundaries. No real learned dataset, notebook, competition request or official learned score has been produced by these preparation controls.
