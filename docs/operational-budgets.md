# Operational budget cohorts

The five-minute limit was chosen by this project for short pilots. It is not a Kaggle maximum. A pilot timeout means the agent exhausted that local wall-clock allowance; it does not establish how the model performs with the documented longer budget. All such failures remain recorded.

The downloaded official `HARNESS_README.md`, section 7.1, documents 60 minutes, 100 tool calls in `inference.py`, 500 turns and a 300-second command timeout as defaults. `official-default-60m-v1` is now the default in the submission template, artifact builder and trajectory runner. The older `pilot-5m-v1` profile remains an explicit option for reproducing historical artifacts. No further short-budget experiments are planned. Verification remains the official fixed 300-second pytest timeout.

These are distinct local cohorts. Their OFF and teacher arms use matching budgets within each trial. The longer profile changes all four operational limits, so a difference cannot be attributed to wall-clock time alone. Actual Kaggle host overrides and four-L4 throughput remain separate from these single-A6000 local experiments.

`build_submission.py` writes the documented limits into the packaged `eval_config.yaml`. `run_trajectories.py` uses the same default for its actual `EvalConfig` and persists the contract before starting inference. Every new collection plan must explicitly declare its profile; an omitted profile is rejected. A full-budget plan also declares the literal budget dictionary, teacher instructions and execution-source hashes before inference. Outcomes and selected contexts preserve the budget identity. Explicit historical pilot builds reproduce byte-for-byte.

The first longer trial repeats TRAIN `fastapi_11194` with the original `behavior-v1` prompts. It uses a new immutable directory and a complete new OFF control. No earlier pilot outcome is relabeled. It is running from an isolated checkout with frozen source hashes. Only verified successful teachers against the matched failed OFF control may supply training contexts.

The [default-budget control](../reports/default-operational-budget-v1.json) records a real CPU agent-loop invocation with no budget flag, its persisted 60-minute runtime contract and packaged limits, official compilation, and 31 passing CPU checks. The probe is scripted and cannot supply a learned target or model score.
