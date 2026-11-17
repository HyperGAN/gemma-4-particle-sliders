from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/collect_train_pool.py'
    spec=importlib.util.spec_from_file_location('train_pool',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def inputs():
    rows=[dict(instance_id=f'fastapi_{i}',repo='fastapi/fastapi',base_commit='a'*40,
        problem_statement='issue',hints_text='',created_at='2026-01-01',split='train') for i in (3,1,2)]
    controls=[{'instance_id':r['instance_id'],'status':'passed' if r['instance_id']!='fastapi_2' else 'blocked'} for r in reversed(rows)]
    qualification=dict(schema='gemma4-train-environment-qualification-v1',qualified=2,controls=controls)
    return rows,qualification


def test_qualified_selection_preserves_declared_order_and_exclusions():
    m=module();rows,q=inputs()
    assert [r['instance_id'] for r in m.select_tasks(rows,q,[])]==['fastapi_3','fastapi_1']
    assert [r['instance_id'] for r in m.select_tasks(rows,q,['fastapi_3'])]==['fastapi_1']
    with pytest.raises(ValueError,match='unknown excluded'):m.select_tasks(rows,q,['missing'])
    damaged=deepcopy(q);damaged['controls'].pop()
    with pytest.raises(ValueError,match='whole fixed'):m.select_tasks(rows,damaged,[])
    damaged=deepcopy(q);damaged['qualified']=3
    with pytest.raises(ValueError,match='count differs'):m.select_tasks(rows,damaged,[])
    rows[0].update(repo='psf/requests',split='dev')
    with pytest.raises(ValueError,match='TRAIN pool'):m.select_tasks(rows,q,[])


def test_server_release_refuses_an_unrelated_process(tmp_path):
    with pytest.raises(ValueError,match='refusing to stop'):
        module().stop_owned_server(os.getpid(),tmp_path/'server-identity.json')
