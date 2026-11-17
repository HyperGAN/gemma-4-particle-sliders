from copy import deepcopy
import json
from pathlib import Path

import pytest

from gemma_particle.behavior_protocol import load_behavior_protocol


def test_prospective_common_judge_cannot_silently_change_budget_or_admit_failed_teachers(tmp_path):
    root=Path(__file__).resolve().parents[1]
    protocol=load_behavior_protocol(root/'research/jev-evidence-loop-v1.json')
    variants=[]
    row=deepcopy(protocol);row['cohort_policy']['operational_budget']['max_time_minutes']=5;variants.append(row)
    row=deepcopy(protocol);row['admission']['teacher_fresh_verification_resolved']=False;variants.append(row)
    row=deepcopy(protocol);row['admission']['teacher_score_strictly_above_off']=False;variants.append(row)
    row=deepcopy(protocol);row['target_selection']['prior_assistant_or_tool_history_allowed']=True;variants.append(row)
    for index,row in enumerate(variants):
        path=tmp_path/f'{index}.json';path.write_text(json.dumps(row))
        with pytest.raises(ValueError):load_behavior_protocol(path)
