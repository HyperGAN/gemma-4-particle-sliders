"""Load the prospective initial-decision behavior study without budget drift."""
import json
from pathlib import Path

from .budgets import OFFICIAL_BUDGET,budget_values
from .decisions import ENDPOINT


def load_behavior_protocol(path):
    protocol=json.loads(Path(path).read_text())
    if protocol.get('schema')!='gemma4-behavior-admission-protocol-v1' or protocol.get('name')!='jev-evidence-loop-v1':
        raise ValueError('explicit evidence-loop study identity required')
    cohort=protocol['cohort_policy'];target=protocol['target_selection'];admission=protocol['admission']
    if cohort.get('budget_profile')!=OFFICIAL_BUDGET or cohort.get('operational_budget')!=budget_values(OFFICIAL_BUDGET):
        raise ValueError('full official operational budget must match its declaration')
    if (target.get('max_contexts_per_task')!=1 or target.get('full_initial_request') is not True
            or target.get('prior_assistant_or_tool_history_allowed') is not False):
        raise ValueError('one complete initial context without prior assistant/tool history required')
    if (admission.get('minimum_teacher_normalized_score')!=.75 or admission.get('teacher_score_strictly_above_off') is not True
            or admission.get('teacher_fresh_verification_resolved') is not True
            or admission.get('teacher_complete_capture_and_clean_terminal') is not True
            or admission.get('off_complete_capture_and_clean_terminal') is not True):
        raise ValueError('declared complete successful teacher and common-quality gates required')
    if (protocol['question'].get('type')!='score' or len(protocol['question'].get('criteria',[]))!=5
            or protocol['judge'].get('endpoint')!=ENDPOINT or protocol['judge'].get('model')!='typesafe/jev-latest'):
        raise ValueError('fixed five-level Jev-latest question and NanoGPT endpoint required')
    return protocol
