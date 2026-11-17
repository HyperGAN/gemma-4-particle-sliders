import hashlib
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import yaml

from gemma_particle.budgets import OFFICIAL_BUDGET,budget_values


def test_official_budget_changes_only_packaged_operational_limits(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/build_submission.py'
    spec=importlib.util.spec_from_file_location('budget_artifact_builder',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pilot=module.build(tmp_path/'pilot','gemma-4-31b-it-qat-w4a16-ct',instruction='verify')
    full=module.build(tmp_path/'full','gemma-4-31b-it-qat-w4a16-ct',instruction='verify',budget_profile=OFFICIAL_BUDGET)
    with ZipFile(tmp_path/'pilot/submission.zip') as a,ZipFile(tmp_path/'full/submission.zip') as b:
        assert a.namelist()==b.namelist()
        changed=[name for name in a.namelist() if a.read(name)!=b.read(name)]
        assert changed==['eval_config.yaml']
        assert yaml.safe_load(b.read('eval_config.yaml'))=={'evaluation':budget_values(OFFICIAL_BUDGET)}
    assert pilot['sha256']!=full['sha256']
    assert full['sha256']==hashlib.sha256((tmp_path/'full/submission.zip').read_bytes()).hexdigest()
    assert full['budget_profile']==OFFICIAL_BUDGET
