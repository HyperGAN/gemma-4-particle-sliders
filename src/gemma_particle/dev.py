"""Bind supplementary DEV execution to its frozen, qualified environment.

This module reads visible tasks and control receipts. Only the separate
verifier receives a reference-bearing dataset or its source hash.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from .data import VISIBLE_FIELDS, REPO_SPLITS, digest, read_jsonl
from .harness_identity import sha256_file

WORKSPACE_PACKAGES = {
    'psf/requests':'requests', 'fastapi/fastapi':'fastapi',
    'pypa/packaging':'packaging', 'psf/black':'black', 'scrapy/w3lib':'w3lib',
    'pydantic/pydantic-settings':'pydantic_settings', 'tox-dev/platformdirs':'platformdirs',
    'PyCQA/isort':'isort', 'strawberry-graphql/strawberry':'strawberry',
    'litestar-org/litestar':'litestar', 'pallets/click':'click',
    'simonw/sqlite-utils':'sqlite_utils', 'encode/uvicorn':'uvicorn',
    'pallets/werkzeug':'werkzeug', 'more-itertools/more-itertools':'more_itertools',
    'pytest-dev/pluggy':'pluggy', 'scrapy/parsel':'parsel', 'urllib3/urllib3':'urllib3',
    'jd/tenacity':'tenacity', 'python-humanize/humanize':'humanize',
    'python-attrs/attrs':'attrs', 'Python-Markdown/markdown':'markdown',
    'marshmallow-code/marshmallow':'marshmallow', 'pallets/flask':'flask',
    'executablebooks/markdown-it-py':'markdown_it', 'Delgan/loguru':'loguru',
}


@dataclass(frozen=True)
class ExternalDevSelection:
    manifest: dict
    rows: list
    manifest_sha256: str
    visible_file_sha256: str


def validate_agent_tasks(rows, external=None):
    fields=set(VISIBLE_FIELDS)|{'split'}
    if not rows or any(set(r)!=fields or r['split'] not in ('train','dev') for r in rows):
        raise ValueError('agent tasks require the exact visible TRAIN/DEV field allowlist')
    if len({r['instance_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate visible task IDs')
    if external is not None:
        if rows!=external.rows:
            raise ValueError('agent tasks differ from the frozen supplementary DEV cohort')
    elif any(REPO_SPLITS.get(r['repo'])!=r['split'] for r in rows):
        raise ValueError('original repository split required; supplementary DEV needs an explicit manifest')


def load_external_dev(path, *, source_sha256=None):
    path=Path(path)
    manifest=json.loads(path.read_text())
    visible_path=path.parent/'dev.jsonl'
    rows=read_jsonl(visible_path)
    if (manifest.get('schema')!='gemma4-supplementary-dev-v1'
            or manifest.get('dataset')!='moshegolderson/freshswe-py'
            or manifest.get('forbidden_training') is not True
            or manifest.get('count')!=len(rows)
            or manifest.get('task_ids')!=[r.get('instance_id') for r in rows]
            or manifest.get('visible_sha256')!=digest(rows)
            or (source_sha256 is not None and source_sha256!=manifest.get('source_sha256'))):
        raise ValueError('supplementary DEV source/selection differs')
    for row in rows:
        repo=row.get('repo','')
        if (row.get('split')!='dev' or repo in REPO_SPLITS
                or repo.split('/')[0] in ('fastapi','Textualize') or repo not in WORKSPACE_PACKAGES):
            raise ValueError('supplementary DEV must exclude original split repositories')
    if manifest.get('repositories')!=sorted({r['repo'] for r in rows}):
        raise ValueError('supplementary DEV repository membership differs')
    selection=ExternalDevSelection(manifest,rows,sha256_file(path),sha256_file(visible_path))
    validate_agent_tasks(rows,selection)
    return selection


def qualified_dev_identity(selection, qualification, *, inputs, image_id, wheels):
    """Validate every fixed control, without opening reference patches."""
    qualification=Path(qualification)
    receipt=json.loads(qualification.read_text())
    inputs=Path(inputs)
    inputs_sha=sha256_file(inputs/'receipt.json')
    manifest=selection.manifest
    controls=receipt.get('controls',[])
    if (receipt.get('schema')!='gemma4-supplementary-dev-qualification-v1'
            or receipt.get('status')!='passed' or receipt.get('no_model_inference') is not True
            or receipt.get('forbidden_training') is not True
            or receipt.get('selection_manifest_sha256')!=selection.manifest_sha256
            or receipt.get('source_sha256')!=manifest['source_sha256']
            or receipt.get('visible_sha256')!=manifest['visible_sha256']
            or receipt.get('input_receipt_sha256')!=inputs_sha or receipt.get('image_id')!=image_id
            or receipt.get('planned')!=len(selection.rows) or receipt.get('qualified')!=len(selection.rows)
            or len(controls)!=len(selection.rows)
            or {c['task'] for c in controls}!=set(manifest['task_ids'])):
        raise ValueError('complete matching supplementary DEV qualification required')
    for control in controls:
        empty,reference=control['arms']['empty'],control['arms']['reference']
        origins=control['origin_checks']
        if (control['status']!='passed' or empty['resolved'] is not False
                or empty['test_exit_code'] in (None,0) or empty['error'] is not None
                or reference['resolved'] is not True or reference['test_exit_code']!=0
                or reference['error'] is not None or len(origins)!=2
                or not all(o['workspace'] and o['origin'].startswith('/workspace/') for o in origins)):
            raise ValueError('every supplementary DEV negative/reference control must pass')
        if sha256_file(inputs/'snapshots'/f"{control['task']}.tgz")!=control['snapshot_sha256']:
            raise ValueError('supplementary DEV snapshot differs from its qualified control')
    input_receipt=json.loads((inputs/'receipt.json').read_text())
    for entry in input_receipt['files']:
        if entry['path'].startswith(('sandbox/','graphs/','embeddings/')):
            if sha256_file(inputs/entry['path'])!=entry['sha256']:
                raise ValueError('supplementary DEV active sandbox input differs')
    # This provider image already contains the exact task locks and wheelhouse.
    # Injecting a different wheel profile would change the qualified environment.
    wheels=Path(wheels)
    wheel_receipt=json.loads((wheels/'receipt.json').read_text())
    if (list(wheels.glob('*.whl')) or wheel_receipt.get('injected_wheels')!=[]
            or wheel_receipt.get('inputs_receipt_sha256')!=inputs_sha):
        raise ValueError('supplementary DEV requires the qualified image wheelhouse')
    return {'manifest_sha256':selection.manifest_sha256,
        'qualification_sha256':sha256_file(qualification),'source_sha256':manifest['source_sha256'],
        'visible_sha256':manifest['visible_sha256'],'visible_file_sha256':selection.visible_file_sha256,
        'inputs_receipt_sha256':inputs_sha,'forbidden_training':True}
