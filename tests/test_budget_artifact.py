import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import yaml

from gemma_particle.budgets import OFFICIAL_BUDGET,PILOT_BUDGET,DEFAULT_BUDGET,budget_values


def test_official_budget_changes_only_packaged_operational_limits(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/build_submission.py'
    spec=importlib.util.spec_from_file_location('budget_artifact_builder',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pilot=module.build(tmp_path/'pilot','gemma-4-31b-it-qat-w4a16-ct',instruction='verify',budget_profile=PILOT_BUDGET)
    full=module.build(tmp_path/'full','gemma-4-31b-it-qat-w4a16-ct',instruction='verify')
    assert DEFAULT_BUDGET==OFFICIAL_BUDGET
    assert budget_values()==budget_values(OFFICIAL_BUDGET)
    with ZipFile(tmp_path/'pilot/submission.zip') as a,ZipFile(tmp_path/'full/submission.zip') as b:
        assert a.namelist()==b.namelist()
        changed=[name for name in a.namelist() if a.read(name)!=b.read(name)]
        assert changed==['eval_config.yaml']
        assert yaml.safe_load(b.read('eval_config.yaml'))=={'evaluation':budget_values(OFFICIAL_BUDGET)}
    assert pilot['sha256']!=full['sha256']
    assert full['sha256']==hashlib.sha256((tmp_path/'full/submission.zip').read_bytes()).hexdigest()
    assert full['budget_profile']==OFFICIAL_BUDGET
    historical=json.loads((source.parents[1]/'reports/operational-budget-controls-v1.json').read_text())
    prior=next(row for row in historical['checks'] if row['arm']=='verify')
    assert pilot['sha256']==prior['pilot_sha256']
    assert full['sha256']==prior['full_sha256']


def test_collection_plan_cannot_silently_select_a_short_budget(tmp_path):
    (tmp_path/'plan.json').write_text('{}\n')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/collect_arms.py'),
        '--root',str(tmp_path),'--server-identity',str(tmp_path/'not-read.json'),
        '--endpoint','http://127.0.0.1:8000/v1','--wheels',str(tmp_path/'not-read')],
        cwd=root,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'must explicitly declare their operational budget' in result.stderr
