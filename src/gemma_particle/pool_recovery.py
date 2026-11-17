"""Reconcile a stopped TRAIN prefix without rerunning or grading partial work."""
from copy import deepcopy
import json
import os
from pathlib import Path

from .harness_identity import sha256_file


def stable_server_identity(server):
    result=deepcopy(server)
    # Available memory is a startup observation; every model/launch/source and
    # physical slot field remains exact. The serving launcher rechecks headroom.
    result.pop('observed_free_bytes',None)
    return result


def reconcile_pool_prefix(previous,current_plan,interruption):
    previous=Path(previous).resolve();interruption=Path(interruption).resolve()
    original=json.loads((previous/'plan.json').read_text())
    progress=json.loads((previous/'progress.json').read_text())
    receipt=json.loads(interruption.read_text())
    legacy_image_fields={}
    if 'image' not in original:
        original['image']='gemma-particle-sandbox:fastapi-deps-v2'
        legacy_image_fields['image']='original collector default; fresh raw manifests bind actual image ID'
    if 'image_id' not in original:
        interrupted=receipt.get('interrupted_train_task')
        if interrupted not in original['task_ids']:raise ValueError('interrupted task must belong to original study')
        original['image_id']=json.loads((previous/interrupted/'off/cohort.json').read_text())['image_id']
        legacy_image_fields['image_id']='original interrupted raw cohort manifest'
    keys=('behavior_study','protocol_sha256','task_ids','excluded_tasks','qualification_sha256',
        'visible_tasks_sha256','raw_tasks_sha256','image','image_id','endpoint','executor',
        'executor_commit','prompt_profile','curator','curator_commit','stop_rule')
    if (any(original[k]!=current_plan[k] for k in keys)
            or stable_server_identity(original['server_identity'])!=stable_server_identity(current_plan['server_identity'])):
        raise ValueError('recovery must preserve the original fixed study and serving contract')
    owners=[r for r in receipt.get('historical_supervisors',[])
        if r['role']=='collect_behavior_pool.py']
    if (receipt.get('schema')!='gemma4-runtime-interruption-observation-v1' or len(owners)!=1
            or owners[0].get('proc_exists') is not False
            or Path(owners[0]['historical_binding']['cwd']).resolve()!=Path(original['curator']).resolve()):
        raise ValueError('explicit missing original collector observation required')
    binding=owners[0]['historical_binding'];proc=Path(f"/proc/{binding['pid']}")
    if proc.exists() and (proc/'stat').read_text().split(') ',1)[1].split()[19]==binding['birth_ticks']:
        raise ValueError('original collector is still live')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            command=[os.fsdecode(s) for s in (process/'cmdline').read_bytes().split(b'\0') if s]
            if (any(s.endswith(('/collect_behavior_pool.py','/collect_arms.py')) for s in command)
                    and '--root' in command):
                output=Path(command[command.index('--root')+1])
                if not output.is_absolute():output=(process/'cwd').resolve()/output
                if output.resolve()==previous or previous in output.resolve().parents:
                    raise ValueError('a live collector still owns the interrupted pool')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    finished=progress.get('finished',[]);ids=original['task_ids']
    if (progress.get('state')!='collecting' or not finished
            or [r['instance_id'] for r in finished]!=ids[:len(finished)] or len(finished)>=len(ids)
            or sum(r['admitted_contexts'] for r in finished)!=progress.get('admitted_contexts')
            or progress['admitted_contexts']>=2):
        raise ValueError('only a complete contiguous fixed-order prefix can be recovered')
    task=ids[len(finished)];cohort=previous/task
    if (receipt.get('interrupted_train_task')!=task or receipt.get('interrupted_train_arm')!='off'
            or receipt.get('train_capture_terminal') is not False
            or receipt.get('unattempted_train_arms')!=['reproduce','localize','verify']):
        raise ValueError('the next interrupted task and unattempted arms must be explicit')
    raw=json.loads((cohort/'off/cohort.json').read_text())
    if (raw['task_ids']!=[task] or raw['arm']!='off' or raw['runs']
            or (cohort/'off'/task/'trajectory.json').exists()
            or (cohort/'off'/task/'agent.patch').exists()
            or not list((cohort/'off'/task/'requests').glob('*-request.json'))
            or any((cohort/arm).exists() for arm in ['reproduce','localize','verify'])
            or (previous/(task+'-grades')).exists() or (cohort/'outcomes.jsonl').exists()):
        raise ValueError('interrupted OFF must remain ungraded and other arms unattempted')
    for row in finished:
        if (sha256_file(Path(row['grades'])/'result.json')!=row['grade_result_sha256']
                or sha256_file(Path(row['grades'])/'jev-contexts.jsonl')!=row['contexts_sha256']):
            raise ValueError('original one-pass grade or admitted contexts changed')
    suffix=ids[len(finished)+1:]
    if any((previous/t).exists() or (previous/(t+'-grades')).exists() for t in suffix):
        raise ValueError('recovery may only continue the original unattempted suffix')
    inventory={str(p.relative_to(cohort)):sha256_file(p) for p in sorted(cohort.rglob('*')) if p.is_file()}
    return finished,{'previous_pool':str(previous),'previous_plan_sha256':sha256_file(previous/'plan.json'),
        'previous_progress_sha256':sha256_file(previous/'progress.json'),
        'interruption_receipt':str(interruption),'interruption_receipt_sha256':sha256_file(interruption),
        'legacy_image_field_sources':legacy_image_fields,
        'interrupted_task':task,'interrupted_status':'ungraded; no terminal OFF; three other arms unattempted',
        'interrupted_artifact_sha256':inventory,'unattempted_suffix':suffix,
        'policy':'Reuse every complete prefix grade by offline revalidation; no new judgments or rerolls. '
            'Preserve the interrupted task, then continue the original suffix in order with all four arms.'}
