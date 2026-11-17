"""CPU fixtures for asset and full-panel calibration binding; no real holdout."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gemma_particle.confirmation import confirmation_asset_names
from gemma_particle.confirmation_environments import (confirmation_input_hashes,
    control_qualified, load_confirmation_environments)
from gemma_particle.harness_identity import sha256_file, wheel_identity
from test_confirmation_execution import execution_fixture
from test_conversion_handoff import put

ROOT=Path(__file__).resolve().parents[1]


def test_assets_derive_only_from_explicit_fixed_scope(tmp_path):
    _,scope=execution_fixture(tmp_path)
    names=confirmation_asset_names(scope)
    assert len([n for n in names if n.startswith('snapshots/')])==49
    assert len([n for n in names if n.startswith('graphs/')])==2
    assert len([n for n in names if n.startswith('embeddings/')])==2
    assert len(confirmation_asset_names(scope,[scope.rows[0]['instance_id']]))==3
    with pytest.raises(ValueError):confirmation_asset_names(None)
    with pytest.raises(ValueError):confirmation_asset_names(scope,['not-in-locked-panel'])


def environment_fixture(root, *, failed_reference=False, before_receipt=False):
    access,scope=execution_fixture(root)
    inputs=root/'inputs'
    for name in [*confirmation_asset_names(scope),'sandbox/setup.py']:
        path=inputs/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU file hash fixture only')
    names=confirmation_asset_names(scope)
    put(inputs/'download-receipt.json',{'schema':'gemma4-official-download-v1','competition':'gemma-4-developer-agent',
        'files':{**{name:{'source':'competition','sha256':sha256_file(inputs/name)} for name in [*names,'sandbox/setup.py']},
            'tasks.jsonl':{'source':'competition','sha256':scope.identity['official_tasks_sha256']}},
        'confirmation_batches':{scope.identity['access_receipt_sha256']:{'access':scope.identity,'requested_assets':names}},
        'fixture':'CPU only; not an actual Kaggle download'})
    out=root/'environments';out.mkdir();wheels=out/'wheels';wheels.mkdir()
    (wheels/'CPU-fixture.whl').write_bytes(b'CPU bytes only, not an installable wheel')
    put(wheels/'receipt.json',{'wheels':wheel_identity(wheels)})
    verifier=root/'executor/scripts/verify_controls.py';verifier.parent.mkdir(parents=True);verifier.write_text('# CPU source-binding fixture\n')
    controller=root/'CPU-controller.py';controller.write_text('# CPU metadata fixture\n')
    raw=root/'CPU-official-reference-fixture.jsonl'
    plan={'schema':'gemma4-confirmation-environment-plan-v1','planned':49,'workers':2,'gpu_used':False,'model_inference':False,
        'access_identity':scope.identity,'access_receipt':str(access),'task_ids':[r['instance_id'] for r in scope.rows],
        'raw_tasks':str(raw),'raw_tasks_sha256':sha256_file(raw),'inputs':str(inputs),
        'image_id':'sha256:CPU-fixture','input_files':confirmation_input_hashes(scope,inputs),
        'download_receipt_sha256':sha256_file(inputs/'download-receipt.json'),
        'wheel_profile':str(wheels),'wheels':wheel_identity(wheels),'wheel_profile_sha256':sha256_file(wheels/'receipt.json'),
        'execution_sources':{str(verifier):sha256_file(verifier)},'controller_sources':{str(controller):sha256_file(controller)}}
    plan_path=out/'plan.json';put(plan_path,plan);controls=[]
    for index,row in enumerate(scope.rows):
        task=row['instance_id'];path=out/task/'result.json';log=out/(task+'.log');log.write_text('CPU verifier log fixture\n')
        record={'instance_id':task,'returncode':0,'log':str(log),'log_sha256':sha256_file(log)}
        if before_receipt and index==0:
            record.update(returncode=3,receipt=None,receipt_sha256=None,status='failed before verifier receipt',qualified=False)
        else:
            receipt={'schema':'gemma4-real-task-verifier-controls-v1','task':task,'split':'holdout','gpu_used':False,
                'confirmation':scope.identity,'image_id':plan['image_id'],'tasks_sha256':plan['raw_tasks_sha256'],
                'snapshot_sha256':plan['input_files'][f'snapshots/{task}.tgz'],
                'wheel_selection_sha256':plan['wheel_profile_sha256'],'execution_script_sha256':sha256_file(verifier),
                'status':'passed','arms':{'empty':{'resolved':False,'test_exit_code':1,'error':None},
                    'reference':{'resolved':True,'test_exit_code':0,'error':None}},
                'origin_checks':[{'workspace':True,'origin':'/workspace/fixture/__init__.py'}]*2}
            if failed_reference and index==0:
                receipt['status']='blocked: reference control did not pass'
                receipt['arms']['reference'].update(resolved=False,test_exit_code=1);record['returncode']=2
            put(path,receipt);record.update(receipt=str(path),receipt_sha256=sha256_file(path),status=receipt['status'],
                qualified=bool(control_qualified(receipt,record['returncode'])))
        controls.append(record)
    result={'schema':'gemma4-confirmation-environments-v1','state':'complete','gpu_used':False,'model_inference':False,
        'access_identity':scope.identity,'planned':49,'qualified':sum(r['qualified'] for r in controls),'controls':controls,
        'image_id':plan['image_id'],'wheels':str(wheels),'plan':str(plan_path),'plan_sha256':sha256_file(plan_path)}
    path=out/'qualification.json';put(path,result)
    return path,scope


@pytest.mark.parametrize('failure',[None,'reference','pre_receipt'])
def test_all49_controls_retained_with_actual_failure_annotations(tmp_path,failure):
    path,scope=environment_fixture(tmp_path,failed_reference=failure=='reference',before_receipt=failure=='pre_receipt')
    result=load_confirmation_environments(path,scope)
    assert len(result['qualification']['controls'])==49
    assert len(result['qualified_task_ids'])==(48 if failure else 49)


@pytest.mark.parametrize('damage',['drop','snapshot','graph','wheel','raw','source','log','count','reference_hash','download_provenance'])
def test_changed_or_incomplete_environment_evidence_is_rejected(tmp_path,damage):
    path,scope=environment_fixture(tmp_path);result=json.loads(path.read_text());plan=json.loads(Path(result['plan']).read_text())
    if damage=='drop':result['controls'].pop()
    elif damage=='snapshot':(Path(plan['inputs'])/f"snapshots/{scope.rows[0]['instance_id']}.tgz").write_bytes(b'changed')
    elif damage=='graph':next((Path(plan['inputs'])/'graphs').iterdir()).write_bytes(b'changed')
    elif damage=='wheel':(Path(plan['wheel_profile'])/'CPU-fixture.whl').write_bytes(b'changed')
    elif damage=='raw':Path(plan['raw_tasks']).write_text('changed')
    elif damage=='source':Path(next(iter(plan['execution_sources']))).write_text('changed')
    elif damage=='log':Path(result['controls'][0]['log']).write_text('changed')
    elif damage=='count':result['qualified']=48
    elif damage=='download_provenance':
        receipt=Path(plan['inputs'])/'download-receipt.json';value=json.loads(receipt.read_text());value['confirmation_batches']={};put(receipt,value)
    else:
        receipt=Path(result['controls'][0]['receipt']);value=json.loads(receipt.read_text());value['tasks_sha256']='a'*64
        put(receipt,value);result['controls'][0]['receipt_sha256']=sha256_file(receipt)
    put(path,result)
    with pytest.raises(ValueError):load_confirmation_environments(path,scope)


def cli(script,*arguments):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1')
    return subprocess.run([str(ROOT/'.harness-venv/bin/python'),str(ROOT/'scripts'/script),*map(str,arguments)],
        cwd=ROOT,env=env,capture_output=True,text=True)


def test_fetch_plan_all_fixed_assets_needs_no_kaggle_import_or_network(tmp_path):
    access,_=execution_fixture(tmp_path);out=tmp_path/'asset-plan'
    result=cli('fetch_official_inputs.py','--confirmation-access',access,'--all-confirmation','--plan-only','--out',out)
    assert result.returncode==0,result.stderr
    plan=json.loads((out/'fetch-plan.json').read_text())
    assert len(plan['requested_files'])==59 and plan['network_access'] is False and plan['downloaded_files']==0
    assert plan['provider_sizes_verified'] is False


@pytest.mark.parametrize('script',['fetch_official_inputs.py','verify_confirmation_environments.py'])
def test_real_cli_denies_unreleased_access_before_assets_or_reference_metadata(tmp_path,script):
    access=tmp_path/'incomplete.json';put(access,{'schema':'CPU incomplete access fixture'})
    out=tmp_path/'out';arguments=['--out',out]
    if script=='fetch_official_inputs.py':
        arguments+=['--confirmation-access',access,'--all-confirmation','--visible-tasks',tmp_path/'MUST_NOT_READ.jsonl']
    else:
        arguments+=['--access',access,'--executor',tmp_path/'MUST_NOT_INSPECT','--executor-commit','CPU-fixture',
            '--raw-tasks',tmp_path/'MUST_NOT_READ.jsonl']
    result=cli(script,*arguments)
    assert result.returncode!=0 and 'completed explicit confirmation access receipt required' in result.stderr
    assert 'FileNotFoundError' not in result.stderr and not out.exists()


@pytest.mark.parametrize('mode',['download_and_reuse','budget','changed_official_source'])
def test_fake_kaggle_handoff_binds_access_and_checks_source_before_assets(tmp_path,monkeypatch,mode):
    access,scope=execution_fixture(tmp_path);out=tmp_path/'fake-download'
    base=['HARNESS_README.md','tasks.jsonl','sandbox/setup.py','docker/Dockerfile.sandbox','docker/imp.py','docker/telnetlib.py']
    files={name:b'CPU fake provider payload only' for name in [*base,*confirmation_asset_names(scope)]}
    files['tasks.jsonl']=(tmp_path/'CPU-official-reference-fixture.jsonl').read_bytes()
    if mode=='changed_official_source':files['tasks.jsonl']=b'different CPU reference source'
    class FakeAPI:
        def __init__(self):self.downloads=[]
        def competition_list_files(self,*args,**kwargs):
            return SimpleNamespace(files=[SimpleNamespace(name=name,total_bytes=len(value)) for name,value in files.items()],next_page_token=None)
        def competition_download_file(self,competition,name,path,**kwargs):
            self.downloads.append(name);destination=Path(path)/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(files[name])
        def competition_leaderboard_view(self,*args,**kwargs):
            return [SimpleNamespace(team_name='CPU fixture, not real leaderboard',score=0,submission_date='CPU fixture')]
    api=FakeAPI();monkeypatch.setitem(sys.modules,'kaggle',SimpleNamespace(api=api))
    argv=['fetch_official_inputs.py','--confirmation-access',str(access),'--all-confirmation','--out',str(out)]
    if mode=='budget':argv+=['--max-download-gib','0.000000001']
    monkeypatch.setattr(sys,'argv',argv)
    if mode in ['budget','changed_official_source']:
        with pytest.raises(ValueError):runpy.run_path(str(ROOT/'scripts/fetch_official_inputs.py'),run_name='__main__')
        assert api.downloads==(['tasks.jsonl'] if mode=='changed_official_source' else [])
    else:
        runpy.run_path(str(ROOT/'scripts/fetch_official_inputs.py'),run_name='__main__')
        assert api.downloads[0]=='tasks.jsonl' and len(api.downloads)==59
        receipt=json.loads((out/'download-receipt.json').read_text())
        assert scope.identity['access_receipt_sha256'] in receipt['confirmation_batches']
        runpy.run_path(str(ROOT/'scripts/fetch_official_inputs.py'),run_name='__main__')
        assert len(api.downloads)==59
