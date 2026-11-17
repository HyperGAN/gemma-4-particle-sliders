# Explicit submission finalization

The original `repair-v1` prompt tells the agent to call `submit_patch` after verification. Official swegemma 0.2.11 permits repeated submissions and declares this tool with `count_tool_call=False`. Its inner ADK invocation ends cleanly after a final text reply. The first prospective Jev OFF episode made a one-file patch and received successful submission responses, then repeatedly submitted again while thinking it would stop. These free submission calls explain why emitted proposals can exceed the declared 100 counted tools.

The [fixed synthetic control](../reports/submission-finalization-control-v1.json) made four one-shot calls with the same seed, sampling and synthetic tool history. Both prompts ended after one prior submission. After twelve repeated submissions, the current prompt emitted another tool call; the explicit cue returned final text. No actual repository was repaired or tool executed in this control. This supports testing the cue on real tasks, without establishing a task-performance gain.

The separately declared [repair-finalize-v2 profile](../research/repair-finalize-v2.json) adds the same instruction to OFF and all three teachers:

> After submit_patch returns status ok, finish with one short final text response. Do not call submit_patch again unless you made another code change after the previous submission.

`--prompt-profile repair-finalize-v2` records the profile and common prompt hash in the collection plan and evaluation contract. Source pins include the profile module and base template. Captured outcomes and selected pairs retain the profile. Teacher and student still differ only by the declared behavioral cue; the finalization instruction is common to both. The three teacher instructions, behavior rubric, public harness, sampling, full operational budget, compaction and initial-state training boundary remain unchanged.

The historical default remains `repair-v1`, with exact original ZIP bytes verified. Existing frozen runs retain their original prompt and source. Stopping the looping Jev cohort is an explicit intervention with incomplete arms preserved; it supplies no fitting data. The new fixed-order cohort starts independently, and actual clean completion plus fresh repository tests must establish whether finalization improves.
