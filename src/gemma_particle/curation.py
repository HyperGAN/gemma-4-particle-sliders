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
        if row.get('schema') != 'gemma4-verified-trajectory-v3' or row.get('generator_kind') != 'gemma4':
            raise ValueError('only verified Gemma model runs may supply targets; CPU protocol probes are forbidden')
        if not row.get('frozen_base_generator'):
            raise ValueError('curation targets require the frozen base generator')
        if type(row.get("resolved")) is not bool:
            raise ValueError("resolved must be a boolean, not a string or score")
        if row["arm"] != "off" and row["arm"] not in TARGET_INSTRUCTIONS:
            raise ValueError("unknown target-instruction arm")
        identity=row.get('model_identity',{})
        if not identity or identity.get('dtype')!='bfloat16' or len(identity.get('revision',''))!=40:
            raise ValueError('target collection requires a pinned BF16 native Gemma model')
        for decision,packet in enumerate(row['requests']):
            if (packet.get('instance_id')!=task_id or packet.get('arm')!=row['arm']
                    or packet['decision']!=decision or packet['request_sha256']!=digest(packet['request'])):
                raise ValueError('captured request task, arm, order or bytes changed')
        receipt = row["verification"]
        if (receipt["instance_id"] != task_id or receipt["patch_sha256"] != row["patch_sha256"]
                or receipt["requests_sha256"] != digest(row["requests"])
                or receipt['first_request_sha256'] != digest(row['first_request'])
                or receipt["resolved"] is not row["resolved"] or receipt["stage"] != "fresh-container-tests"):
            raise ValueError("outcome is not bound to the verified patch/trajectory")
        arms = by_task.setdefault(task_id, {})
        if row["arm"] in arms:
            raise ValueError("duplicate arm; define a new declared trial instead")
        arms[row["arm"]] = row
    selected = []
    for task_id, arms in sorted(by_task.items()):
        if set(arms) != {'off',*TARGET_INSTRUCTIONS}:
            raise ValueError('all four declared curation arms must finish before selection')
        baseline=arms['off']
        if any(r['evaluation_contract_sha256'] != baseline['evaluation_contract_sha256'] for r in arms.values()):
            raise ValueError('curation arms have unmatched model/environment/budget contracts')
        if any(r['model_identity']!=baseline['model_identity'] for r in arms.values()):
            raise ValueError('curation arms have unmatched model identities')
        if baseline["resolved"]:
            continue
        if not baseline.get('first_request'):
            continue
        if not baseline.get('capture_complete') and not baseline.get('task_budget_exhausted'):
            continue
        candidates = [row for arm, row in arms.items() if arm != "off" and row["resolved"] and row.get('capture_complete')]
        if not candidates:
            continue
        # Fixed tie break, declared before observing results.
        winner = min(candidates, key=lambda r: (r["tool_calls"], r["elapsed_seconds"], r["arm"]))
        off_messages=baseline['first_request']['messages']
        if not off_messages or off_messages[0]['role']!='system':
            raise ValueError('captured OFF system prompt required')
        base_system=off_messages[0]['content']
        off_request=baseline['first_request']
        insertion='\n\n'+TARGET_INSTRUCTIONS[winner['arm']]
        for packet in winner["requests"]:
            request=packet['request']
            if any(request[k]!=off_request[k] for k in ('model','tools','generation')):
                raise ValueError('teacher tools or generation differ from the OFF control')
            if packet['request_sha256'] != digest(request):
                raise ValueError('captured request changed')
            history = deepcopy(request['messages'])
            if not history or history[0]["role"] != "system":
                raise ValueError("a paired context requires the base system prompt")
            positive_system=history[0]['content']
            if positive_system.count(insertion)!=1 or positive_system.replace(insertion,'',1)!=base_system:
                raise ValueError('teacher must differ from actual OFF system bytes by the declared instruction only')
            neutral = deepcopy(history)
            neutral[0]["content"] = base_system
            selected.append({"schema":"gemma4-paired-context-v2", "instance_id": task_id, "split": "train", "decision": packet['decision'],
                             "neutral_messages": neutral, "positive_messages": history,
                             "tools":deepcopy(request['tools']), "generation":deepcopy(request['generation']),
                             "teacher_request_sha256":packet['request_sha256'],
                             "evaluation_contract_sha256":winner['evaluation_contract_sha256'],
                             "model_identity":winner['model_identity'],
                             "target_arm": winner["arm"], "verification": winner["verification"],
                             "quality_rubric": RUBRIC})
    return selected
