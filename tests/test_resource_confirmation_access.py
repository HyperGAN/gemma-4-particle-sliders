"""CPU binding/compatibility controls; no real release or holdout/model run."""
import json
from pathlib import Path

import pytest

from gemma_particle.confirmation import load_confirmation_access
from gemma_particle.harness_identity import sha256_file
import gemma_particle.resource_supplement as resources
from test_confirmation_execution import execution_fixture
from test_conversion_handoff import put


@pytest.mark.parametrize('damage', [None, 'choice', 'grid', 'path', 'hash', 'plan'])
def test_original_choice_and_dispatch_must_bind_the_same_qualified_policy(tmp_path, monkeypatch, damage):
    path = tmp_path / 'policy.json'; put(path, {'fixture': 'binding-only CPU policy;not qualified hardware'})
    binding = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    choice = {'resource_supplement': dict(binding)}; grid = {'resource_supplement': dict(binding)}
    observed = []
    def qualified(p):
        observed.append(p)
        return {'selection_plan_sha256': 'b' * 64 if damage == 'plan' else 'a' * 64}
    # The hardware loader has its own complete artifact controls. This test
    # isolates the new choice/dispatch/plan references before manifest access.
    monkeypatch.setattr(resources, 'load_resource_supplement', qualified)
    if damage == 'choice': choice.pop('resource_supplement')
    elif damage == 'grid': grid.pop('resource_supplement')
    elif damage == 'path': choice['resource_supplement']['path'] = str(tmp_path / 'wrong')
    elif damage == 'hash': grid['resource_supplement']['sha256'] = '0' * 64
    if damage:
        with pytest.raises(ValueError): resources.bind_resource_choice(choice, grid, path, 'a' * 64)
        assert len(observed) == (1 if damage == 'plan' else 0)
    else:
        assert resources.bind_resource_choice(choice, grid, path, 'a' * 64) == binding
        assert observed == [path]


@pytest.mark.parametrize('wrong_auditor', [False, True])
def test_unchanged_protected_loader_binds_actual_auditor_and_separate_generation_commit(tmp_path, wrong_auditor):
    path, _ = execution_fixture(tmp_path)
    access = json.loads(path.read_text()); plan_path = Path(access['access_plan']); plan = json.loads(plan_path.read_text())
    plan.update(dev_executor_commit='CPU-offline-resource-auditor', generation_executor_commit='CPU-original-generation',
        executor_field_semantics='actual offline auditor;generation source recorded separately')
    put(plan_path, plan)
    access['access_plan_sha256'] = sha256_file(plan_path)
    access['offline_dev_auditor_executor_commit'] = 'wrong' if wrong_auditor else plan['dev_executor_commit']
    put(path, access)
    if wrong_auditor:
        with pytest.raises(ValueError, match='source-bound offline reaudit'): load_confirmation_access(path)
    else:
        scope = load_confirmation_access(path)
        assert scope.receipt['offline_dev_auditor_executor_commit'] == 'CPU-offline-resource-auditor'
        assert plan['generation_executor_commit'] != scope.receipt['offline_dev_auditor_executor_commit']
