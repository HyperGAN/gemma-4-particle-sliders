# Observe an official learned result

The publisher saves a first-page public leaderboard observation immediately before the competition request and binds its hash in the submission intent. This preserves the preceding public best, including any stronger result from our own team. Neither the reference nor an eventual score changes the already locked DEV choice.

`scripts/monitor_selected_submission.py` first reconstructs the selected experiment and exact created publication. It requires the genuinely selected archive, original readiness, downloaded output bytes, one publishing attempt per stage, matching notebook/version, submission intent and returned numeric submission ID. An OFF-control receipt cannot supply this binding. Only then does it import the tested account client.

Observation reads that exact submission ID. Pending and unknown statuses remain pending, and missing or invalid scores are not converted to zero. A completed error is not a win. On clean completion, the monitor reads the current first public leaderboard page and compares the returned score strings as exact finite decimals. A tie is insufficient. The learned score must strictly exceed both the preceding all-team public best and the freshly observed comparison.

If our completed candidate already appears as the sole current best, the monitor can exclude that displayed entry only after Kaggle's public team-submission interface identifies the exact numeric candidate submission ID and matching score. Matching a team name alone is insufficient. A tied second team or stronger earlier record prevents a strict-record verdict. These are comparisons of reported public scores; private or unreported precision is not inferred.

Every read attempt and selected response is retained without credentials. A transient observation failure preserves the same ID. `--watch` waits without an episode time limit and never resubmits or reruns a notebook or model. A fresh output directory can hold a later read-only observation. The monitor reports evidence; it never marks the thread goal complete. Completion still requires the full experiment, repository and contemporaneous official-result audit.

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .kaggle-venv/bin/python \
  scripts/monitor_selected_submission.py \
  --publication /path/to/created-publication \
  --readiness /path/to/selected/readiness.json \
  --access /path/to/released/access.json \
  --environments /path/to/confirmation/qualification.json \
  --pair /path/to/pair-claim.json \
  --comparison /path/to/confirmation/comparison.json \
  --out /path/to/new-official-observation --watch
```

Run inside the clean frozen observer checkout. `--plan-only` opens no protected inputs and makes no account or model request. The [preparation report](../reports/official-result-preparation-v1.json) distinguishes CPU/fake-service controls from actual read-only public-interface observations. No real learned submission or winning learned score exists in these preparation results.
