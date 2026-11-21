# Run the full DEV grid with qualified physical slots

Restore the [frozen selection inputs](dev-selection.md) before preparing a new dispatch.

The [active EMPF selection plan](https://github.com/HyperGAN/gemma-4-particle-sliders/blob/f97a3e9a0711a8b81f87e396a88ff125090f0c26/research/dev-selection-empf-v1.json) uses nine learned conditions and four controls. Pass it as `--selection-plan` with the matching [EMPF resource declaration](../research/dev-resource-declaration-empf-v1.json) as `--declaration`. The original task runner still uses the [original DEV plan](https://github.com/HyperGAN/gemma-4-particle-sliders/blob/f97a3e9a0711a8b81f87e396a88ff125090f0c26/research/dev-selection-v1.json) for the fixed task panels and matching settings.

`scripts/run_resource_dev_grid.py` is a separate controller for the original eighteen learned conditions. It preserves the original twenty-two-condition comparison, complete forty-two plus thirteen task panels, recipes, checkpoint order, strengths, seed, raw weights, selection law and secondary floor. The four controls remain on their original GPU 0 server. The learned conditions follow the nine paired waves in the [prospective resource declaration](../research/dev-resource-declaration-v1.json). Confirmation retains its original GPU 0 contract.

Actual qualification needs complete native training and conversions, all four complete controls, observed release of their producer processes and the original control server, and observed shared GPU headroom. Each serving process inherits nice 10 and four CPU threads, uses the original frozen generation source, and targets 60% GPU memory with at least 8 GiB of observed startup headroom.

Before arming the replacement, verify the old supervisor's exact PID, birth, source, command and output, its zero learned-task attempts and lack of children. Stop that owned CPU waiter, observe its exit, and retain its checkout and output. The replacement refuses a live old owner or any previous learned task. It never stops the collection or baseline servers. Its plan-only mode accepts the existing live waiter for preparation and makes no HTTP, GPU-driver, model or account request.

The controller waits on actual producer handles. A missing producer with incomplete outputs is an error; a completed producer still running must release. It then validates all native receipts, all eighteen packets and all four complete controls. It launches the fixed EMPf step-256 strength-1 packet on both slots and invokes the separately frozen hardware qualifier. The offline reader must reconstruct all twenty HTTP exchanges and pass all six predeclared numerical pairs before a qualified resource record can be written. There is no automatic tolerance change, retry or reassignment after failure.

The plan binds the actual qualified policy before the first learned task. Both conditions in a wave must pass their alias readiness and fixed TRAIN learned-effect checks before either panel starts. Each condition records its original serving PID and birth, command, identity, packet and physical GPU UUID before the first task and after the last. Raw physical task contracts remain intact. The original generation executor runs all fifty-five tasks serially within each condition; two complete conditions can run concurrently on the qualified slots.

An actual process failure is retained. Its already-running peer may finish, after which the controller stops before another wave. It does not infer missing task scores or rerun attempted tasks. On normal wave completion, it stops only its own exact serving owners. A separate frozen resource-aware comparator audits all twenty-two results and produces the original prospective choice. The [independent confirmation-access runner](resource-confirmation-access.md) subsequently reaudits that choice before any protected task is opened.

Example preparation, using clean frozen checkouts and actual retained owner identifiers:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$CONTROLLER/src" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
  "$CONTROLLER/.harness-venv/bin/python" scripts/run_resource_dev_grid.py \
  --prior-grid /path/to/old-learned-grid \
  --prior-pid "$PRIOR_PID" --prior-birth-ticks "$PRIOR_BIRTH" \
  --conversion /path/to/measured-learned-sweep \
  --controls /path/to/full-dev-controls \
  --executor /path/to/original-dev-executor --executor-commit "$GENERATION_COMMIT" \
  --qualifier /path/to/hardware-qualifier --qualifier-commit "$QUALIFIER_COMMIT" \
  --auditor /path/to/resource-auditor --auditor-commit "$AUDITOR_COMMIT" \
  --out /path/to/new-immutable-plan --plan-only
```

Run from the frozen controller checkout. Actual dispatch uses a fresh output directory without `--plan-only` after the exact old waiter has been observed stopped. A partial run requires evidence reconciliation, not an automatic restart.

The [preparation report](../reports/resource-dispatch-preparation-v1.json) distinguishes real CPU lifecycle tests and the current live-waiter rejection from supplied model and hardware fixtures. Neither CPU fixtures nor internally consistent files authenticate an experiment, establish GPU parity or demonstrate model quality. Real qualification, learned evaluation, confirmation and an official score above the current leader remain outstanding.
