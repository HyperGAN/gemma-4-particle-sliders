# Official OFF controls

The first official submission is a frozen 31B prompt-only OFF control. It uses the current 60-minute/100-tool/500-turn/300-second-command artifact. It contains no learned adapter and does not establish Particle Sliders quality.

`scripts/prepare_kaggle_notebook.py` embeds the small validated ZIP into a private CPU-only, offline Kaggle notebook. The notebook performs no inference; it emits the exact archive for competition evaluation. Its local execution and downloaded Kaggle output both match the compiled archive byte for byte. The notebook completed as version 1, then `competition_submit_code` created submission **58457991**. Its first authenticated status is `PENDING`, with no score ([receipt](../reports/official-off-baseline-submission-v1.json)). Poll this exact submission handle; do not create another submission because an observation takes time.

The notebook is [Gemma 4 Particle Sliders OFF Control V1](https://www.kaggle.com/code/mikkel255/gemma-4-particle-sliders-off-control-v1). Official competition evaluation supplies the model-serving/scoring environment separately. The local shared A6000 runs remain separate evidence. The archive SHA256 is `05a2bee7bc26199e3b260c20548ecb6a545951d59bce85e77627ada46daa0a02`.

Future learned submissions require measured conversion, actual LoRA transport, DEV selection and untouched holdout confirmation. An official OFF score is a control, not completion of the learned-behavior objective.

The matched finalization-v2 OFF control uses a ZIP byte-identical to the actual DEV OFF artifact. A separate private CPU notebook completed and its downloaded output matched that ZIP. Kaggle initially rejected the competition request while the first control was pending and the daily quota was exhausted. The deferred request created the matched submission on November 18, 2026 UTC. [Matched archive receipt](../reports/official-finalizing-off-control-v2.json)

The November 19 authenticated observation found both submissions terminal with empty scores. Kaggle reported that each submission notebook exceeded the allowed runtime. The archived terminal status receipt has SHA-256 `96fd1845927798f5c4097fa23c32e7ac76e4746ba027b4699ae74cfeacbbf6af`.

`wait_and_submit_off_control.py` provides a one-shot handoff for that exact completed notebook version. It waits for the observed UTC window and pending gate, detects an existing matching submission, and preserves an intent before making one request. A failed or ambiguous request stops for reconciliation. The [CPU boundary controls](../reports/deferred-off-submit-preparation-v1.json) check these cases with a fake service; the actual plan-only check sends no submission request.
