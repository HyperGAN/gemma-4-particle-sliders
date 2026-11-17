"""Bind the full released confirmation environment panel, retaining failures."""
from pathlib import Path

from .confirmation import confirmation_asset_names
from .conversion_handoff import read
from .harness_identity import sha256_file, wheel_identity


def confirmation_input_hashes(scope, inputs):
    inputs = Path(inputs)
    names=confirmation_asset_names(scope)
    hashes={name: sha256_file(inputs / name) for name in [*names, 'sandbox/setup.py']}
    receipt=read(inputs/'download-receipt.json')
    batch=receipt.get('confirmation_batches',{}).get(scope.identity['access_receipt_sha256'],{})
    if (receipt.get('schema')!='gemma4-official-download-v1' or receipt.get('competition')!='gemma-4-developer-agent'
            or batch.get('access')!=scope.identity or not set(names)<=set(batch.get('requested_assets',[]))
            or receipt['files'].get('tasks.jsonl',{}).get('sha256')!=scope.identity['official_tasks_sha256']
            or any(receipt['files'].get(name,{}).get('sha256')!=value
                or receipt['files'].get(name,{}).get('source')!='competition' for name,value in hashes.items())):
        raise ValueError('complete official post-access download provenance required for confirmation assets')
    return hashes


def control_qualified(receipt, returncode):
    if receipt is None or returncode != 0 or receipt.get('status') != 'passed': return False
    empty, reference = receipt['arms']['empty'], receipt['arms']['reference']
    origins = receipt.get('origin_checks', [])
    return (empty['resolved'] is False and type(empty['test_exit_code']) is int and empty['test_exit_code'] != 0 and empty['error'] is None
        and reference['resolved'] is True and type(reference['test_exit_code']) is int and reference['test_exit_code'] == 0 and reference['error'] is None
        and len(origins) == 2 and all(o['workspace'] is True and o['origin'].startswith('/workspace/') for o in origins))


def audit_confirmation_control(record, plan, scope):
    task = record['instance_id']
    if (task not in plan['task_ids'] or type(record['returncode']) is not int
            or sha256_file(record['log']) != record['log_sha256']):
        raise ValueError('actual terminal confirmation environment attempt/log required')
    receipt = None
    if record['receipt'] is None:
        if record['returncode'] == 0 or record['qualified'] is not False:
            raise ValueError('pre-receipt failure requires actual nonzero exit evidence')
    else:
        receipt = read(record['receipt'])
        sources = [v for p, v in plan['execution_sources'].items() if p.endswith('/scripts/verify_controls.py')]
        if (len(sources) != 1 or sha256_file(record['receipt']) != record['receipt_sha256']
                or receipt.get('schema') != 'gemma4-real-task-verifier-controls-v1'
                or receipt.get('split') != 'holdout' or receipt.get('gpu_used') is not False
                or receipt.get('task') != task or receipt.get('confirmation') != scope.identity
                or receipt['image_id'] != plan['image_id']
                or receipt['tasks_sha256'] != plan['raw_tasks_sha256']
                or receipt['snapshot_sha256'] != plan['input_files'][f'snapshots/{task}.tgz']
                or receipt['wheel_selection_sha256'] != plan['wheel_profile_sha256']
                or receipt['execution_script_sha256'] != sources[0] or record['status'] != receipt['status']):
            raise ValueError('confirmation control differs from original source/access/environment')
    if record['qualified'] is not bool(control_qualified(receipt, record['returncode'])):
        raise ValueError('environment qualification differs from actual negative/reference evidence')
    return receipt


def load_confirmation_environments(path, scope):
    path = Path(path); result = read(path); plan_path = Path(result['plan']); plan = read(plan_path)
    expected_ids = [row['instance_id'] for row in scope.rows]
    if (result.get('schema') != 'gemma4-confirmation-environments-v1' or result.get('state') != 'complete'
            or plan.get('schema') != 'gemma4-confirmation-environment-plan-v1' or plan.get('planned') != 49
            or plan.get('workers') != 2 or plan.get('gpu_used') is not False or plan.get('model_inference') is not False
            or result.get('gpu_used') is not False or result.get('model_inference') is not False
            or result.get('planned') != 49 or result.get('access_identity') != scope.identity
            or result['plan_sha256'] != sha256_file(plan_path) or plan['access_identity'] != scope.identity
            or plan['task_ids'] != expected_ids or [r['instance_id'] for r in result['controls']] != expected_ids
            or plan['raw_tasks_sha256'] != scope.identity['official_tasks_sha256']
            or sha256_file(plan['access_receipt']) != scope.identity['access_receipt_sha256']
            or sha256_file(plan['raw_tasks']) != plan['raw_tasks_sha256']
            or result['image_id'] != plan['image_id'] or result['wheels'] != plan['wheel_profile']
            or wheel_identity(plan['wheel_profile']) != plan['wheels']
            or sha256_file(Path(plan['wheel_profile']) / 'receipt.json') != plan['wheel_profile_sha256']
            or read(Path(plan['wheel_profile']) / 'receipt.json')['wheels'] != plan['wheels']
            or confirmation_input_hashes(scope, plan['inputs']) != plan['input_files']
            or sha256_file(Path(plan['inputs'])/'download-receipt.json')!=plan['download_receipt_sha256']
            or any(sha256_file(p) != value for p, value in plan['execution_sources'].items())
            or any(sha256_file(p) != value for p, value in plan['controller_sources'].items())):
        raise ValueError('complete unchanged original forty-nine-task confirmation environment panel required')
    for record in result['controls']: audit_confirmation_control(record, plan, scope)
    qualified = [r['instance_id'] for r in result['controls'] if r['qualified']]
    if result['qualified'] != len(qualified): raise ValueError('qualified environment count differs')
    return {'plan': plan, 'qualification': result, 'qualified_task_ids': qualified}
