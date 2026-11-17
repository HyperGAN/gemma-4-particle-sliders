# Operational budget cohorts

The five-minute limit was chosen by this project for short pilots. It is not a Kaggle maximum. A pilot timeout means the agent exhausted that local wall-clock allowance; it does not establish how the model performs with the documented longer budget. All such failures remain recorded.

The downloaded official `HARNESS_README.md`, section 7.1, documents 60 minutes, 100 tool calls in `inference.py`, 500 turns and a 300-second command timeout as defaults. The new `official-default-60m-v1` profile uses those four settings together. The older `pilot-5m-v1` profile retains 5 minutes, 40 tool calls, 70 turns and a 60-second command timeout. Verification remains the official fixed 300-second pytest timeout.

These are distinct local cohorts. Their OFF and teacher arms use matching budgets within each trial. The longer profile changes all four operational limits, so a difference cannot be attributed to wall-clock time alone. Actual Kaggle host overrides and four-L4 throughput remain separate from these single-A6000 local experiments.

`build_submission.py --budget-profile official-default-60m-v1` writes the limits into the packaged `eval_config.yaml`. `run_trajectories.py` uses the same profile for its actual `EvalConfig`. A full-budget collection plan declares the profile, literal budget dictionary, teacher instructions and execution-source hashes before inference. Its outcomes and selected contexts preserve the budget identity. Prior pilot artifacts reproduce byte-for-byte.

The first longer trial repeats TRAIN `fastapi_11194` with the original `behavior-v1` prompts. It uses a new immutable directory and a complete new OFF control. No earlier pilot outcome is relabeled. It is prepared in an isolated checkout so the active five-minute cohorts retain their original code. Only verified successful teachers against the matched failed OFF control may supply training contexts.
