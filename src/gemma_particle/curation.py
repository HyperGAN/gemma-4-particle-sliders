"""Admit successful paired contexts selected by real repository-test outcomes."""
from copy import deepcopy
from .data import digest
from .capture import LEGACY_CAPTURE_PROTOCOL,CONTEXT_CAPTURE_PROTOCOL

RUBRIC = "Resolve the repository issue with a focused patch, demonstrated by the issue validation tests, while using the available tools within budget."
TARGET_INSTRUCTIONS = {
    "reproduce": "Before editing, reproduce the reported failure with a focused executable check. Trace the failing values into the implementation, make the smallest justified fix, then rerun that check and related tests. Submit the patch within the tool and time budget.",
    "localize": "Locate the implementation and its closest tests. State the concrete cause supported by those files, check boundary cases and public API compatibility, make a focused fix, and verify it with targeted tests before submitting the patch.",
    "verify": "Use an iterative repair loop: inspect the issue and repository, form one falsifiable hypothesis, test it with a short command, edit the relevant code, and check both the original failure and adjacent behavior. Preserve time to submit a patch.",
}
DEFAULT_INSTRUCTION_PROTOCOL = 'behavior-v1'
TOOL_PRECISION = ("Use each tool's declared schema literally. Keep filepath a plain path and pass line ranges as separate integer fields. "
    "If a range read fails or repeats the file header, use run_command with sed to read the exact range. "
    "Avoid repeating unchanged calls. Batch narrow searches when useful and preserve time for a code edit, a focused check, and submit_patch.")
INSTRUCTION_PROTOCOLS = {DEFAULT_INSTRUCTION_PROTOCOL:TARGET_INSTRUCTIONS,
    'tools-v2':{key:value+'\n\n'+TOOL_PRECISION for key,value in TARGET_INSTRUCTIONS.items()}}


def instruction_set(protocol=DEFAULT_INSTRUCTION_PROTOCOL):
    if protocol not in INSTRUCTION_PROTOCOLS:raise ValueError('unknown instruction protocol')
    return INSTRUCTION_PROTOCOLS[protocol]


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
        instructions=instruction_set(row.get('instruction_protocol',DEFAULT_INSTRUCTION_PROTOCOL))
        task_id = row["instance_id"]
        if task_id not in allowed:
            raise ValueError("non-TRAIN outcome")
        if row.get('schema') not in ('gemma4-verified-trajectory-v3','gemma4-verified-trajectory-v4') or row.get('generator_kind') != 'gemma4':
            raise ValueError('only verified Gemma model runs may supply targets; CPU protocol probes are forbidden')
        if not row.get('frozen_base_generator'):
            raise ValueError('curation targets require the frozen base generator')
        if type(row.get("resolved")) is not bool:
            raise ValueError("resolved must be a boolean, not a string or score")
        if row["arm"] != "off" and row["arm"] not in instructions:
            raise ValueError("unknown target-instruction arm")
        identity=row.get('model_identity',{})
        if not identity or identity.get('dtype')!='bfloat16' or len(identity.get('revision',''))!=40:
            raise ValueError('target collection requires a pinned BF16 native Gemma model')
        for decision,packet in enumerate(row['requests']):
            if (packet.get('instance_id')!=task_id or packet.get('arm')!=row['arm']
                    or packet['decision']!=decision or packet['request_sha256']!=digest(packet['request'])):
                raise ValueError('captured request task, arm, order or bytes changed')
        receipt = row["verification"]
        capture_protocol=row.get('capture_protocol',LEGACY_CAPTURE_PROTOCOL)
        if capture_protocol==CONTEXT_CAPTURE_PROTOCOL:
            proof=row.get('context_budget_proof')
            if (row['schema']!='gemma4-verified-trajectory-v4'
                    or receipt.get('capture_protocol')!=capture_protocol
                    or receipt.get('context_budget_proof_sha256')!=digest(proof)):
                raise ValueError('context-budget proof is not bound to the declared v4 outcome')
            if proof is not None:
                if (row['arm']!='off' or row.get('capture_complete')
                        or proof.get('schema')!='gemma4-context-budget-proof-v1'
                        or proof.get('context_limit')!=32768
                        or type(proof.get('prompt_tokens')) is not int
                        or type(proof.get('requested_output_tokens')) is not int
                        or proof['prompt_tokens']<=0 or proof['requested_output_tokens']<=0
                        or proof['prompt_tokens']+proof['requested_output_tokens']<=32768):
                    raise ValueError('invalid fixed-context budget proof')
        elif capture_protocol!=LEGACY_CAPTURE_PROTOCOL or row['schema']!='gemma4-verified-trajectory-v3':
            raise ValueError('unknown captured request protocol')
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
        protocols={r.get('instruction_protocol',DEFAULT_INSTRUCTION_PROTOCOL) for r in arms.values()}
        if len(protocols)!=1:raise ValueError('curation arms have unmatched instruction protocols')
        if len({r.get('capture_protocol',LEGACY_CAPTURE_PROTOCOL) for r in arms.values()})!=1:
            raise ValueError('curation arms have unmatched capture protocols')
        protocol=next(iter(protocols));instructions=instruction_set(protocol)
        if set(arms) != {'off',*instructions}:
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
        context_exhausted=(baseline.get('capture_protocol')==CONTEXT_CAPTURE_PROTOCOL
            and baseline.get('context_budget_proof') is not None)
        if not baseline.get('capture_complete') and not baseline.get('task_budget_exhausted') and not context_exhausted:
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
        insertion='\n\n'+instructions[winner['arm']]
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
            if protocol!=DEFAULT_INSTRUCTION_PROTOCOL:selected[-1]['instruction_protocol']=protocol
            if 'budget_profile' in winner:selected[-1]['budget_profile']=winner['budget_profile']
            if 'capture_protocol' in winner:selected[-1]['capture_protocol']=winner['capture_protocol']
    return selected
