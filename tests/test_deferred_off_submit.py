"""CPU fake-service controls for duplicate prevention and ambiguous outcomes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('scenario',['new','existing','pending','ambiguous'])
def test_one_shot_external_boundary_keeps_actual_handles_and_never_rerolls(tmp_path,monkeypatch,scenario):
    # Fake API only: no Kaggle connection, uploaded notebook, or actual submission.
    archive=tmp_path/'fixture.zip';archive.write_bytes(b'CPU fixture archive')
    control=tmp_path/'control.json'
    control.write_text(json.dumps({'schema':'gemma4-finalizing-official-control-preparation-v2',
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'notebook_status':'COMPLETE',
        'notebook_gpu_enabled':False,'notebook_model_inference':False,'submission_id':None,
        'byte_identical_to_first_actual_DEV_OFF':True,'notebook':'https://www.kaggle.com/code/mikkel255/cpu-fixture',
        'notebook_version':1,'verified_gates':{'next_daily_window':'2000-01-01T00:00:00Z'}}))
    message='Matched finalization-v2 OFF control; exact full-budget DEV artifact; no learned adapter'
    calls=[];reads=[];sleeps=[]
    def submissions(competition):
        reads.append(competition)
        if scenario=='existing':return [SimpleNamespace(description=message,ref='CPU-existing',status='SubmissionStatus.PENDING')]
        if scenario=='pending':return [SimpleNamespace(description='another fixture',ref='CPU-other',
            status='SubmissionStatus.PENDING' if len(reads)==1 else 'SubmissionStatus.COMPLETE')]
        return []
    def submit(**kwargs):
        calls.append(kwargs)
        if scenario=='ambiguous':raise TimeoutError('CPU fixture ambiguous response')
        return SimpleNamespace(ref='CPU-created',message='fixture only')
    monkeypatch.setitem(sys.modules,'kaggle',SimpleNamespace(api=SimpleNamespace(
        competition_submissions=submissions,competition_submit_code=submit)))
    monkeypatch.setattr('time.sleep',lambda seconds:sleeps.append(seconds))
    out=tmp_path/'out';script=Path(__file__).resolve().parents[1]/'scripts/wait_and_submit_off_control.py'
    monkeypatch.setattr(sys,'argv',[str(script),'--control-receipt',str(control),'--archive',str(archive),'--out',str(out)])
    runpy.run_path(str(script),run_name='__main__')
    result=json.loads((out/'progress.json').read_text())
    if scenario=='existing':
        assert not calls and result['submission_id']=='CPU-existing'
        assert result['state']=='existing_matching_submission_found'
    else:
        assert len(calls)==1 and calls[0]['kernel']=='mikkel255/cpu-fixture' and calls[0]['kernel_version']==1
        assert calls[0]['file_name']=='submission.zip' and (out/'submission-intent.json').is_file()
        if scenario=='ambiguous':
            assert result['state']=='request_failed_requires_reconciliation' and result['submission_id'] is None
        else:assert result['state']=='created' and result['submission_id']=='CPU-created'
    if scenario=='pending':assert len(reads)==2 and sleeps==[300]
