# Original TRAIN environment qualification

`verify_train_environments.py` checks the full fixed 67-task FastAPI TRAIN manifest with two CPU workers. Each new task gets a wheel profile selected from its snapshot's declared core constraints, then an empty/reference control through the unchanged official verifier in separate offline containers. Every failure stays in the progress and final summary. These controls do not constitute model accuracy or learned targets.

Matching prior controls can be explicitly reused. Reuse checks their dataset, snapshot, image, wheel bytes and workspace imports. An older verifier source requires an explicit audited Git commit and keeps its original source hash. The four accepted controls from `cf4486b` differ from the current verifier only in the separately added supplementary DEV path; their FastAPI behavior is unchanged. The other two prior TRAIN controls use the current verifier source.

```bash
.harness-venv/bin/python scripts/verify_train_environments.py \
  --out runs/own-train-environments \
  --reuse-source-commit cf4486b \
  --reuse-control runs/verifier-fastapi-official-v1/result.json
```

Omit reuse flags for a new local qualification. The cohort directory is immutable and a terminal failure returns a nonzero exit code. A live process is not restarted after an observation timeout.

The initial sweep completed with 45 of 67 environments qualified under its original wheel pool ([receipt](../reports/train-environment-qualification-v1.json)). It retains all 22 failures: one empty control already passes, two reference controls fail, and 19 lack a declared dependency. Observed failures include an empty patch that already passes, a required test that skips, snapshot-output incompatibility under the installed Pydantic version, and a missing declared dependency. None has been repaired by changing tests or reference patches.

An explicit second wheel pool adds the unchanged PyPI `typing-inspection==0.4.2` wheel, the minimum required by newer snapshots. Its SHA256 is `4ed1cacbdc298c220f1bd249ed5287caa16f34d44ef4e9c3d0cbad5b521545e7`. The initial `fastapi_14258` control passes with this dependency present. Its earlier failure remains recorded. This is a separate task/environment condition; the running original sweep and previously qualified wheel profiles are unchanged.

All 19 tasks blocked solely by the missing wheel now form a separate, fixed dependency-repair cohort with the added wheel. The already passing `fastapi_14258` control was reused explicitly; the other 18 controls completed on two CPU workers. Thirteen of the 19 repair environments qualify, with all six failures retained ([repair receipt](../reports/train-dependency-repair-v1.json)). The [combined environment view](../reports/train-environment-qualification-v2.json) records 58 qualified tasks and nine remaining failures, with each original grade preserved. Previously passing environments are not rerun.

Two of those remaining tasks report missing declared JSON test dependencies: `fastapi_14962` requires `ujson`, and `fastapi_14964` requires both `ujson` and `orjson`. Their snapshot extras declare `ujson>=5.8.0` and `orjson>=3.9.3`. A [separate plan](../research/json-test-dependency-repair-v1.json) fixes these two tasks before new controls. It adds only the highest stable compatible wheels selected from PyPI before testing: ujson6.1.0 and orjson3.13.0, with exact URLs/hashes retained. Compatibility uses the actual base image's Python3.13 pip tags.

The [isolated Dockerfile](../environments/fastapi-json-deps-v3.Dockerfile) extends the unchanged qualified base, installs those two wheels offline without dependency resolution, and produces `gemma-particle-sandbox:fastapi-json-deps-v3`. Both empty/reference control pairs pass with workspace imports confirmed, using the original snapshot-core wheel profiles. The [two-task qualification](../reports/train-json-environment-qualification-v1.json) binds the new image ID and preserved original controls. There are now60 distinct original TRAIN tasks with qualified recorded environments, across the separate images. The seven remaining failures stay recorded.

The fixed private subset is `data/train-json-repair-v1/train.jsonl`. A future behavior pool can use that subset and its qualification with `--visible-tasks`, `--qualification` and `--image gemma-particle-sandbox:fastapi-json-deps-v3`. The preparer and collector propagate the explicit image and reject a mismatch with its qualified receipt. This does not change the running original58-task environment view or the existing collection plans.
