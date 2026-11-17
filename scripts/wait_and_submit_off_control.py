"""Submit the already-verified matched OFF control after actual Kaggle gates.

This is a one-shot continuation of the current experiment. It never rebuilds
the archive, changes a notebook version, trains, or automatically retries an
ambiguous/failed submission request.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-receipt',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('one-shot control handoff is immutable; reconcile existing intent/result')
    control=json.loads(args.control_receipt.read_text());expected=control['archive_sha256']
    if (control.get('schema')!='gemma4-finalizing-official-control-preparation-v2'
            or control['notebook_status']!='COMPLETE' or control['notebook_gpu_enabled'] is not False
            or control['notebook_model_inference'] is not False or control['submission_id'] is not None
            or not control['byte_identical_to_first_actual_DEV_OFF'] or sha(args.archive)!=expected):
        raise ValueError('already-compiled, notebook-emitted, exact matched OFF control required')
    url=urlsplit(control['notebook'])
    if url.scheme!='https' or url.netloc!='www.kaggle.com' or not url.path.startswith('/code/mikkel255/'):
        raise ValueError('exact existing owned notebook URL required')
    kernel=url.path.removeprefix('/code/');window=datetime.fromisoformat(control['verified_gates']['next_daily_window'].replace('Z','+00:00'))
    message='Matched finalization-v2 OFF control; exact full-budget DEV artifact; no learned adapter'
    args.out.mkdir(parents=True,mode=0o700)
    declaration={'schema':'gemma4-deferred-off-submit-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(__file__),'control_receipt_sha256':sha(args.control_receipt),
        'process_pid':os.getpid(),'process_birth_ticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        'archive_sha256':expected,'kernel':kernel,'version':control['notebook_version'],
        'earliest_utc':window.isoformat(),'message':message,'submission_attempt_limit':1,
        'on_unknown_status':'wait; never infer terminal completion','on_submission_failure':'preserve intent/error and stop for reconciliation',
        'scope':'one exact matched OFF control; not a learned result or SOTA claim'}
    write(args.out/'plan.json',declaration)
    status={'state':'prepared' if args.plan_only else 'waiting_daily_window','submission_id':None,'new_submission_requests':0}
    def publish(state,**values):
        status.update(state=state,as_of=datetime.now(timezone.utc).isoformat(),**values)
        write(args.out/'progress.json',status);print(json.dumps(status),flush=True)
    publish(status['state'])
    if args.plan_only:return
    while datetime.now(timezone.utc)<window:
        time.sleep(min(300,max(1,(window-datetime.now(timezone.utc)).total_seconds())))
    from kaggle import api
    while True:
        try:
            submissions=api.competition_submissions('gemma-4-developer-agent')
        except Exception as exc:
            publish('waiting_read_access',error_type=type(exc).__name__)
            time.sleep(1800);continue
        existing=next((row for row in submissions if row.description==message),None)
        if existing is not None:
            publish('existing_matching_submission_found',submission_id=existing.ref,observed_status=str(existing.status));return
        pending=[row.ref for row in submissions if str(row.status).split('.')[-1] not in {'COMPLETE','ERROR','CANCELLED'}]
        if pending:
            if status['state']!='waiting_prior_submission':publish('waiting_prior_submission',pending_ids=pending)
            time.sleep(300);continue
        break
    if sha(args.archive)!=expected or sha(args.control_receipt)!=declaration['control_receipt_sha256']:
        publish('inputs_changed_no_submission');return
    intent={'as_of':datetime.now(timezone.utc).isoformat(),'kernel':kernel,'version':control['notebook_version'],
        'competition':'gemma-4-developer-agent','file_name':'submission.zip','archive_sha256':expected,'message':message}
    write(args.out/'submission-intent.json',intent)
    publish('one_submission_request',new_submission_requests=1)
    try:
        result=api.competition_submit_code(file_name='submission.zip',message=message,competition='gemma-4-developer-agent',
            kernel=kernel,kernel_version=control['notebook_version'],quiet=True)
    except Exception as exc:
        response=getattr(exc,'response',None);failure={'error_type':type(exc).__name__}
        if response is not None:
            failure['http_status']=response.status_code
            try:failure['server_error']=response.json()
            except ValueError:failure['server_error']='non-JSON response retained by exception type/status only'
        publish('request_failed_requires_reconciliation',**failure);return
    publish('created',submission_id=str(result.ref),message=result.message,
        scope='official evaluation queued; score and completion require subsequent observation')


if __name__=='__main__':main()
