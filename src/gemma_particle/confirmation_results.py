"""Reaudit the one fixed, complete confirmation pair; never select a model."""
import math
from pathlib import Path

from .confirmation import (confirmation_pair_identity, validate_confirmation_claim,
    validate_confirmation_reservation, validate_confirmation_server)
from .conversion_handoff import read
from .data import digest
from .dev_evidence import audit_verified_task, journal_digest, validate_matching_contract
from .dev_selection import _summarize_rows, paired_comparison, EMPTY_PATCH_SHA256
from .harness_identity import sha256_file
from .lora_serving import BASE_ALIAS


def summarize_confirmation_rows(tasks, outcomes, *, qualified_ids):
    if len(tasks)!=49 or any(t['split']!='holdout' for t in tasks):
        raise ValueError('complete original forty-nine-task holdout required')
    expected={t['instance_id']:t for t in tasks};rows={}
    if len(expected)!=49:raise ValueError('unique original confirmation tasks required')
    for row in outcomes:
        task=row['instance_id']
        if (task not in expected or task in rows or row.get('split')!='holdout' or row.get('generator_kind')!='gemma4'
                or type(row.get('resolved')) is not bool or type(row.get('capture_complete')) is not bool
                or row.get('visible_task_sha256')!=digest(expected[task])):
            raise ValueError('actual unique verified confirmation outcome required')
        receipt=row['verification']
        if (receipt.get('stage')!='fresh-container-tests' or receipt.get('instance_id')!=task
                or receipt.get('patch_sha256')!=row['patch_sha256']
                or receipt.get('requests_sha256')!=digest(row['requests'])
                or receipt.get('first_request_sha256')!=digest(row['first_request'])
                or receipt.get('resolved') is not row['resolved']
                or receipt.get('evaluation_contract_sha256')!=row['evaluation_contract_sha256']):
            raise ValueError('confirmation outcome differs from its actual fresh verification')
        if row['resolved'] and (receipt.get('error') is not None or receipt.get('test_exit_code')!=0
                or not receipt.get('origin_checks') or not all(o.get('workspace') is True for o in receipt['origin_checks'])):
            raise ValueError('resolved confirmation needs clean fresh workspace tests')
        if (type(row.get('tool_calls')) is not int or row['tool_calls']<0
                or type(row.get('elapsed_seconds')) not in (int,float) or not math.isfinite(row['elapsed_seconds'])
                or row['elapsed_seconds']<0 or row.get('agent_error') is not None and not isinstance(row['agent_error'],str)):
            raise ValueError('finite actual confirmation tool/time/error accounting required')
        official=row['resolved'] and not (row.get('agent_error') and row['patch_sha256']==EMPTY_PATCH_SHA256)
        rows[task]={'resolved':bool(official),'fresh_test_resolved':row['resolved'],'agent_error':row.get('agent_error'),
            'capture_complete':row['capture_complete'],'tool_proposals':row['tool_calls'],
            'agent_seconds':row['elapsed_seconds'],'repo':expected[task]['repo']}
    if set(rows)!=set(expected) or not set(qualified_ids)<=set(expected):
        raise ValueError('every fixed confirmation task needs its terminal result')
    return _summarize_rows(rows,qualified_ids)


def audit_confirmation_condition(path, scope, claim_path, environments):
    path=Path(path);result=read(path);claim=validate_confirmation_claim(scope,claim_path)
    identity=confirmation_pair_identity(scope,claim_path);condition=result['condition_id']
    expected_ids=[t['instance_id'] for t in scope.rows]
    if (result.get('schema')!='gemma4-confirmation-condition-v1' or result.get('status')!='complete'
            or condition not in scope.receipt['permitted_conditions'] or result.get('confirmation')!=identity
            or result.get('environment_qualification_sha256')!=environments['qualification_sha256']
            or [r['instance_id'] for r in result['finished']]!=expected_ids
            or any(sha256_file(p)!=value for p,value in result['sources'].items())):
        raise ValueError('one complete unchanged permitted confirmation condition required')
    server=read(result['server_identity']);learned=condition!='off'
    validate_confirmation_server(scope,server,learned=learned)
    if digest(server)!=result['server_identity_sha256']:raise ValueError('confirmation server identity changed')
    records=[];canonical=None;plan=environments['plan']
    for record in result['finished']:
        task=record['instance_id'];cohort=Path(record['cohort']);verification=Path(record['verification'])
        if (not cohort.resolve().is_relative_to(Path(claim['pair_root']).resolve())
                or not verification.resolve().is_relative_to(Path(claim['pair_root']).resolve())
                or sha256_file(verification/'outcomes.jsonl')!=record['outcomes_sha256']
                or journal_digest(cohort,task)!=record['model_journal_sha256']):
            raise ValueError('original paired confirmation paths/journal/outcomes changed')
        row,contract=audit_verified_task(cohort,verification,task,expected_arm='off',
            expected_alias='particle_lora' if learned else BASE_ALIAS)
        validate_matching_contract(contract,scope.plan)
        reservation=validate_confirmation_reservation(scope,claim_path,condition,task,cohort)
        environment=contract['environment']
        if (row.get('confirmation')!=identity or contract.get('confirmation')!=identity
                or row.get('confirmation_task_reservation_sha256')!=reservation
                or row['snapshot_sha256']!=plan['input_files'][f'snapshots/{task}.tgz']
                or contract['image_id']!=plan['image_id'] or environment['wheels']!=plan['wheels']
                or environment['sandbox_setup_sha256']!=plan['input_files']['sandbox/setup.py']
                or environment['packages']!=plan['harness']):
            raise ValueError('confirmation row differs from fixed claim/qualified environment')
        fingerprint=digest(contract)
        if canonical is not None and canonical!=fingerprint:raise ValueError('within-condition confirmation contracts differ')
        canonical=fingerprint;records.append(row)
    summary=summarize_confirmation_rows(scope.rows,records,qualified_ids=environments['qualified_task_ids'])
    if result['summary']!=summary or result['canonical_contract_sha256']!=canonical:
        raise ValueError('confirmation aggregate differs from actual audited cases')
    return result,canonical


def compare_confirmation_pair(paths, scope, claim_path, environments):
    if len(paths)!=2:raise ValueError('exactly the original OFF and selected conditions required')
    results={};files={};canonical=None;sources=None
    for path in paths:
        result,fingerprint=audit_confirmation_condition(path,scope,claim_path,environments)
        name=result['condition_id']
        if name in results:raise ValueError('duplicate confirmation condition')
        if canonical is not None and canonical!=fingerprint:raise ValueError('unmatched paired confirmation contracts')
        if sources is not None and sources!=result['sources']:raise ValueError('paired confirmation implementations differ')
        canonical=fingerprint;sources=result['sources'];results[name]=result
        files[name]={'path':str(Path(path).resolve()),'sha256':sha256_file(path)}
    if list(results)!=scope.receipt['permitted_conditions']:raise ValueError('fixed OFF-then-selected condition order required')
    off=results['off']['summary'];selected=results[scope.receipt['selected_condition']['id']]['summary']
    return {'schema':'gemma4-confirmation-comparison-v1','status':'complete',
        'confirmation':confirmation_pair_identity(scope,claim_path),'condition_results':files,
        'canonical_contract_sha256':canonical,'environment_qualification_sha256':environments['qualification_sha256'],
        'off':off,'selected':selected,'paired':paired_comparison(selected,off),
        'paired_resolution_gain':(selected['resolved']-off['resolved'])/49,
        'qualified_environment_tasks':len(environments['qualified_task_ids']),'declared_total':49,
        'new_candidate_selected':False,'official_score':None,
        'scope':'one locked repository-level confirmation pair; environment failures retained; no retuning or official SOTA claim'}
