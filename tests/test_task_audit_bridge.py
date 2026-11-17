"""CPU provenance controls for changing audit readers without changing inference."""
import pytest

from gemma_particle.resource_dispatch import task_audit_bridge


def test_audit_corrections_preserve_all_generation_and_verification_sources():
    names={'src/gemma_particle/dev_evidence.py':'old','scripts/run_dev_condition.py':'old',
        'scripts/run_trajectories.py':'same','scripts/verify_trajectories.py':'same'}
    before={'/serving/'+k:v for k,v in names.items()}
    names['src/gemma_particle/dev_evidence.py']='fixed'
    after={'/audit/'+k:v for k,v in names.items()}
    result=task_audit_bridge('/serving','/audit',before,after)
    assert result['audit_only_changes']==['src/gemma_particle/dev_evidence.py']
    assert result['unchanged_generation_and_verification'] is True
    after['/audit/scripts/run_trajectories.py']='changed'
    with pytest.raises(ValueError,match='generation and verification'):
        task_audit_bridge('/serving','/audit',before,after)


def test_added_or_removed_generation_sources_are_rejected():
    before={'/serving/scripts/run_trajectories.py':'same'}
    for after in [{},{'/audit/scripts/run_trajectories.py':'same','/audit/new.py':'new'}]:
        with pytest.raises(ValueError):task_audit_bridge('/serving','/audit',before,after)
