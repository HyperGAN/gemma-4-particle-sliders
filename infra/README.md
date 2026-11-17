# Local public-task environments

These images are explicit research derivatives. They retain the task tests, official scoring function, network isolation, wall clock and TLS validation. Image IDs and input hashes in the control receipts identify the environments actually tested. Rebuilds must repeat the controls before their results are interpreted.

Build `gemma-particle-sandbox:mirror-v30` from the downloaded competition `docker/Dockerfile.sandbox`, including its companion `imp.py` and `telnetlib.py`. Then build the dependency and source-import layers:

```bash
docker build -f infra/requests-dependencies.Dockerfile -t gemma-particle-sandbox:requests-deps-v1 infra
docker build -f infra/workspace-source.Dockerfile -t gemma-particle-sandbox:source-v1 infra
docker build -f infra/fastapi-dependencies.Dockerfile -t gemma-particle-sandbox:fastapi-deps-v2 infra
```

The dependency images retain `pip freeze` output in `/requests-control-packages.txt` and `/fastapi-control-packages.txt`. The source layer corrects a measured import problem: the official setup's plain `.pth` appended the workspace after preinstalled packages. `SourceBoundManager` refuses to count a passing test run unless the imported subject package belongs to `/workspace/`.

The harness wheel-tar cache uses one filename independently of wheel identity. Each run therefore sets a private temporary root before official container setup, preserving other sessions' caches. FastAPI wheels are filtered against the snapshot's declared core requirements with `prepare_task_wheels.py`; their bytes remain unchanged. This filter is not a full dependency resolver.

`reports/verifier-controls-v2.json` records two calibrated task/environment combinations and the remaining blocked Requests full-test control. These are verifier checks, not coding-agent scores.
