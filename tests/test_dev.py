import json

import pytest

from gemma_particle.data import digest,write_jsonl
from gemma_particle.dev import load_external_dev,qualified_dev_identity,validate_agent_tasks
from gemma_particle.harness_identity import sha256_file


@pytest.fixture
def cohort(tmp_path):
    rows=[dict(instance_id=f'loguru_{i}',repo='Delgan/loguru',base_commit='a'*40,
        problem_statement='Repair the issue',hints_text='',created_at='2026-06-01',split='dev') for i in (1,2)]
    write_jsonl(tmp_path/'dev.jsonl',rows)
    manifest=dict(schema='gemma4-supplementary-dev-v1',dataset='moshegolderson/freshswe-py',
        forbidden_training=True,count=2,task_ids=[r['instance_id'] for r in rows],
        repositories=['Delgan/loguru'],source_sha256='b'*64,visible_sha256=digest(rows))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    inputs=tmp_path/'inputs';(inputs/'snapshots').mkdir(parents=True)
    (inputs/'sandbox').mkdir();(inputs/'sandbox/setup.py').write_text('# fixture only\n')
    files=[{'path':'sandbox/setup.py','sha256':sha256_file(inputs/'sandbox/setup.py')}]
    controls=[]
    for row in rows:
        snapshot=inputs/'snapshots'/f"{row['instance_id']}.tgz";snapshot.write_bytes(b'fixture snapshot')
        controls.append(dict(task=row['instance_id'],status='passed',snapshot_sha256=sha256_file(snapshot),
            arms={'empty':dict(resolved=False,test_exit_code=1,error=None),
                  'reference':dict(resolved=True,test_exit_code=0,error=None)},
            origin_checks=[dict(workspace=True,origin='/workspace/loguru/__init__.py') for _ in range(2)]))
    (inputs/'receipt.json').write_text(json.dumps({'files':files}))
    wheels=tmp_path/'wheels';wheels.mkdir()
    (wheels/'receipt.json').write_text(json.dumps(dict(injected_wheels=[],inputs_receipt_sha256=sha256_file(inputs/'receipt.json'))))
    receipt=dict(schema='gemma4-supplementary-dev-qualification-v1',status='passed',
        no_model_inference=True,forbidden_training=True,selection_manifest_sha256=sha256_file(path),
        source_sha256=manifest['source_sha256'],visible_sha256=manifest['visible_sha256'],
        input_receipt_sha256=sha256_file(inputs/'receipt.json'),image_id='fixture-image',planned=2,qualified=2,controls=controls)
    qualified=tmp_path/'qualification.json';qualified.write_text(json.dumps(receipt))
    return path,qualified,inputs,wheels


def test_matching_external_dev_qualification_and_original_split_guards(cohort):
    path,qualified,inputs,wheels=cohort
    selection=load_external_dev(path,source_sha256='b'*64)
    identity=qualified_dev_identity(selection,qualified,inputs=inputs,image_id='fixture-image',wheels=wheels)
    assert identity['forbidden_training'] is True and identity['manifest_sha256']==sha256_file(path)
    validate_agent_tasks(selection.rows,selection)
    with pytest.raises(ValueError,match='explicit manifest'):validate_agent_tasks(selection.rows)
    with pytest.raises(ValueError,match='source/selection'):load_external_dev(path,source_sha256='c'*64)
    rich={**selection.rows[0],'repo':'Textualize/rich'}
    with pytest.raises(ValueError):validate_agent_tasks([rich])
    reference={**selection.rows[0],'patch':'reference answer'}
    with pytest.raises(ValueError,match='allowlist'):validate_agent_tasks([reference],selection)
    with pytest.raises(ValueError,match='frozen'):validate_agent_tasks(selection.rows[:1],selection)


@pytest.mark.parametrize('repo',['Textualize/rich','encode/httpx','fastapi/fastapi','psf/requests'])
def test_external_manifest_cannot_relabel_an_original_split(cohort,repo):
    path,_,_,_=cohort
    rows=[dict(instance_id='relabeled',repo=repo,base_commit='a'*40,problem_statement='issue',
        hints_text='',created_at='2026-06-01',split='dev')]
    manifest=json.loads(path.read_text());manifest.update(count=1,task_ids=['relabeled'],repositories=[repo],visible_sha256=digest(rows))
    path.write_text(json.dumps(manifest));write_jsonl(path.parent/'dev.jsonl',rows)
    with pytest.raises(ValueError,match='exclude original'):load_external_dev(path)


@pytest.mark.parametrize('change',['drop','failed_reference','passing_empty','bad_origin','image','snapshot','setup','wheel'])
def test_incomplete_or_changed_dev_environment_is_rejected(cohort,change):
    path,qualified,inputs,wheels=cohort;selection=load_external_dev(path)
    receipt=json.loads(qualified.read_text());image='fixture-image'
    if change=='drop':receipt['controls'].pop()
    elif change=='failed_reference':receipt['controls'][0]['arms']['reference']['resolved']=False
    elif change=='passing_empty':receipt['controls'][0]['arms']['empty']['resolved']=True
    elif change=='bad_origin':receipt['controls'][0]['origin_checks'][0]['origin']='/site-packages/loguru.py'
    elif change=='image':image='different-image'
    elif change=='snapshot':(inputs/'snapshots/loguru_1.tgz').write_bytes(b'changed snapshot')
    elif change=='setup':(inputs/'sandbox/setup.py').write_text('# changed setup')
    elif change=='wheel':(wheels/'extra.whl').write_bytes(b'different wheel')
    qualified.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):qualified_dev_identity(selection,qualified,inputs=inputs,image_id=image,wheels=wheels)
