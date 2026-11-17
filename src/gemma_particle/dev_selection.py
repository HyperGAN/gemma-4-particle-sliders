"""Complete-panel DEV accounting and prospective learned-candidate selection.

Inputs must come from the source-bound local runner/verifier. These checks
validate internal receipt consistency; they do not authenticate invented JSON
as an actual model or test execution.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .data import digest
from .harness_identity import sha256_file

EMPTY_PATCH_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def empf_only_selection(base, origin):
    """Keep the original controls and full EMPF checkpoint/strength grid."""
    if 'selection_scope' in base or set(base['training']['presets'])!={'empf','rhbi_optimized'}:
        raise ValueError('an original two-recipe selection plan is required')
    plan=deepcopy(base)
    plan['training']['presets']=['empf']
    plan['conditions']=[row for row in plan['conditions'] if row['role']!='learned' or row['preset']=='empf']
    plan['expected_conditions']=len(plan['conditions'])
    plan['selection_scope']={'kind':'empf_only','original_plan':str(origin),'original_plan_sha256':sha256_file(origin)}
    return plan


def baseline_selection_sha256(selection, plan_sha256):
    """Return the baseline provenance for a validated selection plan."""
    scope=selection.get('selection_scope')
    return scope['original_plan_sha256'] if scope else plan_sha256


def load_selection_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('schema')!='gemma4-dev-selection-plan-v1':raise ValueError('explicit DEV selection plan required')
    training=plan['training']
    scope=plan.get('selection_scope')
    if scope is not None:
        origin=Path(scope['original_plan'])
        if scope.get('kind')!='empf_only' or sha256_file(origin)!=scope['original_plan_sha256']:
            raise ValueError('exact original selection scope required')
        base=json.loads(origin.read_text())
        if plan!=empf_only_selection(base,origin):
            raise ValueError('EMPF-only scope may change only the recipe grid')
    elif set(training['presets'])!={'empf','rhbi_optimized'}:
        raise ValueError('an explicit user-selected scope revision is required')
    conditions=plan['conditions'];ids=[r['id'] for r in conditions]
    learned=[r for r in conditions if r['role']=='learned']
    expected={(preset,step,strength) for preset in plan['training']['presets']
        for step in plan['training']['checkpoints'] for strength in plan['training']['strengths'] if strength!=0}
    observed={(r['preset'],r['completed_steps'],r['strength']) for r in learned}
    if (len(ids)!=len(expected)+4 or len(set(ids))!=len(ids) or len(learned)!=len(expected) or observed!=expected
            or {r['id'] for r in conditions if r['role']!='learned'}!={'off','prompt-reproduce','prompt-localize','prompt-verify'}):
        raise ValueError('complete declared recipe/checkpoint/strength grid and four controls required')
    if (training['horizon']!=1024
            or training['checkpoints']!=[256,512,1024] or training['strengths']!=[0,.5,1,1.5]
            or training['batch']!=2 or training['response_width']!=5376 or training['weights']!='raw'
            or training['ema_decay'] is not None or plan['promotion']['full_grid_required'] is not True):
        raise ValueError('declared full-state public-recipe selection law changed')
    for name,count in [('primary',42),('secondary',13)]:
        panel=plan[name]
        if (panel['count']!=count or len(panel['task_ids'])!=count or len(set(panel['task_ids']))!=count
                or sha256_file(panel['manifest'])!=panel['manifest_sha256']
                or sha256_file(panel['qualification'])!=panel['qualification_sha256']):
            raise ValueError('complete fixed DEV panel or qualification changed')
    if set(plan['primary']['task_ids'])&set(plan['secondary']['task_ids']):
        raise ValueError('primary and secondary DEV tasks must be distinct')
    if not set(plan['secondary']['calibrated_task_ids'])<=set(plan['secondary']['task_ids']):
        raise ValueError('secondary calibrated tasks must remain in the full denominator')
    return plan


def wilson_interval(successes,total):
    if type(successes) is not int or type(total) is not int or not 0<=successes<=total or total<1:
        raise ValueError('positive denominator and integer success count required')
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [0. if successes==0 else max(0.,center-half),
            1. if successes==total else min(1.,center+half)]


def summarize_verified_panel(tasks,outcomes,*,calibrated_ids=()):
    if not tasks or any(t.get('split')!='dev' for t in tasks):
        raise ValueError('only the fixed DEV tasks can supply selection results')
    expected={t['instance_id']:t for t in tasks}
    if len(expected)!=len(tasks):raise ValueError('duplicate DEV task')
    rows={}
    for row in outcomes:
        task_id=row['instance_id']
        if task_id not in expected or task_id in rows:raise ValueError('unknown or duplicate DEV outcome')
        if (row.get('schema') not in ('gemma4-verified-trajectory-v3','gemma4-verified-trajectory-v4')
                or row.get('generator_kind')!='gemma4' or row.get('split')!='dev'
                or type(row.get('resolved')) is not bool or type(row.get('capture_complete')) is not bool
                or row.get('visible_task_sha256')!=digest(expected[task_id])):
            raise ValueError('actual verified Gemma DEV outcome required; grades and CPU fixtures are forbidden')
        for index,packet in enumerate(row['requests']):
            if (packet.get('instance_id')!=task_id or packet.get('decision')!=index
                    or packet.get('request_sha256')!=digest(packet['request'])):
                raise ValueError('DEV request identity/order changed')
        receipt=row['verification']
        if (receipt.get('stage')!='fresh-container-tests' or receipt.get('instance_id')!=task_id
                or receipt.get('patch_sha256')!=row['patch_sha256']
                or receipt.get('requests_sha256')!=digest(row['requests'])
                or receipt.get('first_request_sha256')!=digest(row['first_request'])
                or receipt.get('resolved') is not row['resolved']
                or receipt.get('evaluation_contract_sha256')!=row['evaluation_contract_sha256']):
            raise ValueError('DEV outcome differs from its verified patch/trajectory')
        if row['resolved'] and (receipt.get('error') is not None or receipt.get('test_exit_code')!=0
                or not receipt.get('origin_checks') or not all(o.get('workspace') is True for o in receipt['origin_checks'])):
            raise ValueError('resolved DEV result requires clean fresh tests of edited workspace')
        if (type(row.get('tool_calls')) is not int or row['tool_calls']<0
                or type(row.get('elapsed_seconds')) not in (int,float)
                or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0
                or row.get('agent_error') is not None and not isinstance(row['agent_error'],str)):
            raise ValueError('finite actual tool/time/error accounting required')
        official_resolved=row['resolved'] and not (row.get('agent_error') and row['patch_sha256']==EMPTY_PATCH_SHA256)
        rows[task_id]={'resolved':bool(official_resolved),'fresh_test_resolved':row['resolved'],'agent_error':row.get('agent_error'),
            'capture_complete':row.get('capture_complete') is True,'tool_proposals':row['tool_calls'],
            'agent_seconds':row['elapsed_seconds'],'repo':expected[task_id]['repo']}
    if set(rows)!=set(expected):raise ValueError('every fixed DEV task needs a terminal result; missing tasks cannot be inferred')
    if not set(calibrated_ids)<=set(expected):raise ValueError('unknown calibrated task')
    return _summarize_rows(rows,calibrated_ids)


def _summarize_rows(rows,calibrated_ids):
    solved=sum(r['resolved'] for r in rows.values());repos={}
    for task_id,row in sorted(rows.items()):
        group=repos.setdefault(row['repo'],{'resolved':0,'total':0})
        group['resolved']+=row['resolved'];group['total']+=1
    return {'total':len(rows),'resolved':solved,'resolution_rate':solved/len(rows),
        'wilson_95':wilson_interval(solved,len(rows)),
        'agent_errors':sum(r['agent_error'] is not None for r in rows.values()),
        'incomplete_captures':sum(not r['capture_complete'] for r in rows.values()),
        'tool_proposals':sum(r['tool_proposals'] for r in rows.values()),
        'agent_seconds':sum(r['agent_seconds'] for r in rows.values()),
        'calibrated_resolved':sum(rows[task]['resolved'] for task in calibrated_ids),
        'calibrated_total':len(calibrated_ids),'per_repository':repos,'tasks':rows}


def paired_comparison(candidate,off):
    if candidate['tasks'].keys()!=off['tasks'].keys():raise ValueError('paired comparisons require the same complete DEV tasks')
    wins=sum(c['resolved'] and not off['tasks'][task]['resolved'] for task,c in candidate['tasks'].items())
    losses=sum(not c['resolved'] and off['tasks'][task]['resolved'] for task,c in candidate['tasks'].items())
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
    return {'wins':wins,'losses':losses,'ties':len(candidate['tasks'])-n,'exact_discordance_p':p}


def choose_learned_candidate(plan,results):
    conditions={r['id']:r for r in plan['conditions']}
    if set(results)!=set(conditions):raise ValueError('all declared conditions need complete results or explicit preflight rejection')
    for name,result in results.items():
        if result.get('status')=='unqualified':
            evidence=result.get('preflight_error_sha256')
            if conditions[name]['role']!='learned' or not isinstance(evidence,str) or len(evidence)!=64:
                raise ValueError('actual learned-serving preflight error evidence required')
            continue
        if result.get('status')!='complete':raise ValueError('unfinished DEV condition cannot be selected')
        for panel in ('primary','secondary'):
            if set(result[panel]['tasks'])!=set(plan[panel]['task_ids']):
                raise ValueError('complete declared primary and secondary panels required for every scored condition')
            calibrated=plan['secondary']['calibrated_task_ids'] if panel=='secondary' else ()
            if _summarize_rows(result[panel]['tasks'],calibrated)!=result[panel]:
                raise ValueError('DEV aggregate differs from its complete per-task results')
    off=results['off']
    if off.get('status')!='complete':raise ValueError('complete matched OFF required')
    def core(result):
        p=result['primary'];return (-p['resolved'],p['agent_errors'],p['tool_proposals'])
    eligible=[]
    for name,condition in conditions.items():
        result=results[name]
        if condition['role']!='learned' or result['status']!='complete':continue
        if (core(result)<core(off) and result['secondary']['calibrated_resolved']>=off['secondary']['calibrated_resolved']):
            eligible.append(name)
    def key(name):
        c=conditions[name];r=results[name]
        return (*core(r),r['primary']['agent_seconds'],c['completed_steps'],c['strength'],c['preset'])
    selected=min(eligible,key=key) if eligible else None
    return {'selected':selected,'eligible':sorted(eligible,key=key),'advance_to_locked_confirmation':selected is not None,
        'primary_paired_vs_off':{name:paired_comparison(result['primary'],off['primary'])
            for name,result in results.items() if result.get('status')=='complete' and name!='off'},
        'scope':'complete DEV selection only; not holdout confirmation or an official SOTA result'}
