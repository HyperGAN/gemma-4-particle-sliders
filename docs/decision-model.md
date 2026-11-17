# NanoGPT decision model

The optional decision client uses **`typesafe/jev-1.14`**, pinned rather than a moving alias, through NanoGPT's native `https://api.nano-gpt.com/api/v1/decisions` endpoint. The live host catalog includes this version. [NanoGPT's API contract](https://docs.nano-gpt.com/api-reference/endpoint/decisions) and [TypeSafe's Score guide](https://docs.typesafe.ai/primitives/score) describe its typed judgments. A Score is a position on ordered descriptive levels; normalize by the maximum level when a zero-to-one reporting scale is needed.

The [qualification receipt](../reports/decision-model-qualification-v1.json) records a real authenticated request using the user's new NanoGPT key. One request asks independent Score, Noul and Choice questions over two synthetic traces. Jev ranks the trace with observed execution evidence above the unsupported repair claim, identifies the claim and recognizes an observed test. This qualifies transport and these small controls, not real repair judgment quality.

The first client validation was too strict about independently rounded probabilities and scores. The retained response has a score of 1.44 while the weighted mean of its displayed probabilities is 1.45. Validation now allows the rounding bound of two-decimal displayed components, preserves all original values and rejects materially inconsistent distributions. The archived response passes offline revalidation without another paid call. Both original error receipts remain unchanged.

Credentials are never written into requests, reports or this repository. By default the client reads `NANOGPT_API_KEY` from its process environment. An explicit `--key-file ~/.zshrc` reads exactly one literal assignment and takes precedence over stale inherited values; it does not source or execute the startup file. HTTP redirects are rejected, and HTTP failure messages omit bodies and headers. No automatic paid retries run.

```bash
PYTHONPATH=src .harness-venv/bin/python scripts/qualify_decision_model.py \
  --key-file ~/.zshrc --out runs/own-decision-api-control
```

The command makes one API request and creates an immutable output directory. It does not load Gemma or use a local GPU. `gemma_particle.decisions.decide` exposes the same validated typed client for subsequent behavior analysis. Refused, missing, nonfinite or mismatched-model answers fail rather than becoming inferred scores.

This integration does not change the already running collection or target-admission rule. Fresh repository tests remain the task-success criterion. The decision model is separate from the ParticleGAN critic and its numerical training feedback. No Jev-scored real task, new curation threshold or learned adapter is claimed by this control.
