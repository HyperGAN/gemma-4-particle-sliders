"""Admit successful paired contexts selected by real repository-test outcomes."""
from copy import deepcopy
from .data import digest

RUBRIC = "Resolve the repository issue with a focused patch, demonstrated by the issue validation tests, while using the available tools within budget."
TARGET_INSTRUCTIONS = {
    "reproduce": "Before editing, reproduce the reported failure with a focused executable check. Trace the failing values into the implementation, make the smallest justified fix, then rerun that check and related tests. Submit the patch within the tool and time budget.",
    "localize": "Locate the implementation and its closest tests. State the concrete cause supported by those files, check boundary cases and public API compatibility, make a focused fix, and verify it with targeted tests before submitting the patch.",
    "verify": "Use an iterative repair loop: inspect the issue and repository, form one falsifiable hypothesis, test it with a short command, edit the relevant code, and check both the original failure and adjacent behavior. Preserve time to submit a patch.",
}


def select_contexts(tasks, outcomes):
    """Outcomes contain matched complete runs, not judge-generated target answers.

    A verifier receipt binds the task, patch SHA, test result, and exact
    trajectory. The caller must obtain it from the two-container harness.
    This function validates internal binding; it does not authenticate arbitrary
    external JSON as a real test execution.
    """
    allowed = {r["instance_id"]: r for r in tasks}
    if any(row["split"] != "train" for row in tasks):
        raise ValueError("only TRAIN tasks may supply target contexts")
    by_task = {}
    for row in outcomes:
        task_id = row["instance_id"]
        if task_id not in allowed:
            raise ValueError("non-TRAIN outcome")
        if type(row.get("resolved")) is not bool:
            raise ValueError("resolved must be a boolean, not a string or score")
        if row["arm"] != "off" and row["arm"] not in TARGET_INSTRUCTIONS:
            raise ValueError("unknown target-instruction arm")
        receipt = row["verification"]
        if (receipt["instance_id"] != task_id or receipt["patch_sha256"] != row["patch_sha256"]
                or receipt["trajectory_sha256"] != digest(row["messages"])
                or receipt["resolved"] is not row["resolved"] or receipt["stage"] != "fresh-container-tests"):
            raise ValueError("outcome is not bound to the verified patch/trajectory")
        arms = by_task.setdefault(task_id, {})
        if row["arm"] in arms:
            raise ValueError("duplicate arm; define a new declared trial instead")
        arms[row["arm"]] = row
    selected = []
    for task_id, arms in sorted(by_task.items()):
        if "off" not in arms or arms["off"]["resolved"]:
            continue
        candidates = [row for arm, row in arms.items() if arm != "off" and row["resolved"]]
        if not candidates:
            continue
        # Fixed tie break, declared before observing results.
        winner = min(candidates, key=lambda r: (r["tool_calls"], r["elapsed_seconds"], r["arm"]))
        for index, message in enumerate(winner["messages"]):
            if message["role"] != "assistant":
                continue
            history = deepcopy(winner["messages"][:index])
            if not history or history[0]["role"] != "system":
                raise ValueError("a paired context requires the base system prompt")
            suffix = "\n\n" + TARGET_INSTRUCTIONS[winner["arm"]]
            if not history[0]["content"].endswith(suffix):
                raise ValueError("teacher context does not contain its declared target instruction")
            neutral = deepcopy(history)
            neutral[0]["content"] = neutral[0]["content"][:-len(suffix)]
            selected.append({"instance_id": task_id, "split": "train", "decision": index,
                             "neutral_messages": neutral, "positive_messages": history,
                             "target_arm": winner["arm"], "verification": winner["verification"],
                             "quality_rubric": RUBRIC})
    return selected
