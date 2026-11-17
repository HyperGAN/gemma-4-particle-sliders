import importlib.util
from pathlib import Path

import pytest


def test_conditional_teacher_policy_retains_all_arms_needed_for_any_admission():
    path=Path(__file__).resolve().parents[1]/'scripts/collect_arms.py'
    spec=importlib.util.spec_from_file_location('collection_policy',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=dict(resolved=False,first_request={'model':'gemma'},capture_complete=True)
    assert module.teacher_skip_reason('failed-off-only',row) is None
    assert module.teacher_skip_reason('all-four',dict(row,resolved=True)) is None
    assert 'already passes' in module.teacher_skip_reason('failed-off-only',dict(row,resolved=True))
    assert 'cannot supply' in module.teacher_skip_reason('failed-off-only',dict(row,capture_complete=False))
    assert module.teacher_skip_reason('failed-off-only',dict(row,capture_complete=False,task_budget_exhausted=True)) is None
    assert module.teacher_skip_reason('failed-off-only',dict(row,capture_complete=False,
        capture_protocol='context-budget-off-v1',context_budget_proof={'verified':'fixture'})) is None
    with pytest.raises(ValueError):module.teacher_skip_reason('undeclared',row)
