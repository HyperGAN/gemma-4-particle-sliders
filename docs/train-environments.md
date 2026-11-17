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

The initial sweep remains running under its original wheel pool. Observed failures include an empty patch that already passes, a required test that skips, snapshot-output incompatibility under the installed Pydantic version, and a missing declared dependency. None has been repaired by changing tests or reference patches.

An explicit second wheel pool adds the unchanged PyPI `typing-inspection==0.4.2` wheel, the minimum required by newer snapshots. Its SHA256 is `4ed1cacbdc298c220f1bd249ed5287caa16f34d44ef4e9c3d0cbad5b521545e7`. The initial `fastapi_14258` control passes with this dependency present. Its earlier failure remains recorded. This is a separate task/environment condition; the running original sweep and previously qualified wheel profiles are unchanged.
