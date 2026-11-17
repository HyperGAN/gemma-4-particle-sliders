"""Reaudit a complete declared DEV grid and lock its learned choice offline."""
import argparse
import json
from pathlib import Path

from gemma_particle.data import digest,read_jsonl
from gemma_particle.dev_evidence import audit_verified_task,journal_digest,validate_matching_contract
from gemma_particle.dev_selection import load_selection_plan,summarize_verified_panel,choose_learned_candidate
from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import BASE_ALIAS
from gemma_particle.resource_supplement import (load_resource_supplement, normalize_qualified_contract,
    audit_condition_slot, audit_task_slot)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-plan',type=Path,default=Path('research/dev-selection-v1.json'))
    p.add_argument('--condition-result',type=Path,action='append',required=True)
    p.add_argument('--resource-supplement',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plan=load_selection_plan(args.selection_plan);plan_sha=sha256_file(args.selection_plan)
    if args.out.exists():raise FileExistsError('DEV choice is immutable')
    resource=None if args.resource_supplement is None else load_resource_supplement(args.resource_supplement)
    if resource is not None and resource['selection_plan_sha256']!=plan_sha:
        raise ValueError('resource policy must preserve the exact original selection plan')
    physical_contracts={};resource_evidence={}
    declarations={c['id']:c for c in plan['conditions']};results={};canonical={};sources=None;cache_hashes=set();files={}
    for path in args.condition_result:
        result=json.loads(path.read_text());condition=result['condition'];name=condition['id']
        if (name in results or declarations.get(name)!=condition or result['selection_plan_sha256']!=plan_sha):
            raise ValueError('one complete declared actual condition result required')
        if result.get('status')=='unqualified':
            evidence=Path(result['preflight_error_path'])
            if (condition['role']!='learned' or result.get('schema')!='gemma4-dev-condition-rejected-v1'
                    or sha256_file(evidence)!=result['preflight_error_sha256']):
                raise ValueError('actual preflight error file required for rejected learned condition')
            failure=json.loads(evidence.read_text())
            if failure.get('status')!='error' or failure.get('condition_id')!=name or failure.get('exit_code',0)==0:
                raise ValueError('terminal matching model-preflight failure required')
            results[name]=result;files[name]={'path':str(path.resolve()),'sha256':sha256_file(path)}
            continue
        if result.get('status')!='complete' or result.get('schema')!='gemma4-dev-condition-result-v1':
            raise ValueError('complete actual DEV condition or evidenced rejection required')
        if sources is not None and result['sources']!=sources:raise ValueError('DEV execution implementations differ across conditions')
        sources=result['sources'];packet=result.get('packet')
        if condition['role']=='learned':
            if not packet:raise ValueError('learned checkpoint provenance required')
            for field,hash_field in [('native_metadata','native_metadata_sha256'),('cache_index','cache_index_sha256'),('conversion_report','conversion_report_sha256')]:
                if sha256_file(packet[field])!=packet[hash_field]:raise ValueError('learned packet provenance changed')
            cache_hashes.add(packet['cache_index_sha256'])
        elif packet:raise ValueError('unadapted control contains a learned packet')
        if resource is not None:
            resource_evidence[name]=audit_condition_slot(path,result,args.resource_supplement,resource)
            physical_contracts[name]=dict(result['canonical_contracts'])
        expected_alias='particle_lora' if packet else BASE_ALIAS
        for panel in ('primary','secondary'):
            visible=Path(plan[panel]['manifest'])
            if panel=='primary':visible=visible.parent/'dev.jsonl'
            tasks=read_jsonl(visible);records=[r for r in result['finished'] if r['panel']==panel]
            if [r['instance_id'] for r in records]!=plan[panel]['task_ids']:raise ValueError('complete fixed DEV task order required')
            def outcomes():
                for record in records:
                    if (sha256_file(Path(record['verification'])/'outcomes.jsonl')!=record['outcomes_sha256']
                            or journal_digest(record['cohort'],record['instance_id'])!=record['model_journal_sha256']):
                        raise ValueError('DEV task evidence changed after collection')
                    row,contract=audit_verified_task(record['cohort'],record['verification'],record['instance_id'],
                        expected_arm=condition['arm'],expected_alias=expected_alias)
                    validate_matching_contract(contract,plan);fingerprint=digest(contract)
                    if result['canonical_contracts'][panel]!=fingerprint:raise ValueError('recorded DEV contract differs')
                    if resource is not None:
                        manifest=json.loads((Path(record['cohort'])/'cohort.json').read_text())
                        audit_task_slot(path.parent,result,record,manifest,resource_evidence[name],resource)
                        contract=normalize_qualified_contract(contract,name,resource);fingerprint=digest(contract)
                    if panel in canonical and canonical[panel]!=fingerprint:raise ValueError('unmatched DEV condition contracts')
                    canonical[panel]=fingerprint
                    yield row
            calibrated=plan['secondary']['calibrated_task_ids'] if panel=='secondary' else ()
            summary=summarize_verified_panel(tasks,outcomes(),calibrated_ids=calibrated)
            if summary!=result[panel]:raise ValueError('DEV metrics differ from actual verified outcomes')
        results[name]=result;files[name]={'path':str(path.resolve()),'sha256':sha256_file(path)}
    if len(cache_hashes)!=1:raise ValueError('one exact training cache is required across learned conditions')
    selection=choose_learned_candidate(plan,results)
    args.out.mkdir(parents=True)
    result={'schema':'gemma4-locked-dev-selection-v1','selection_plan_sha256':plan_sha,
        'condition_results':files,'canonical_contracts':canonical,'cache_index_sha256':next(iter(cache_hashes)),
        **selection,'model_dev_conditions':len(results),'holdout_accessed':False,'official_score':None}
    if resource is not None:
        result.update(resource_supplement={'path':str(args.resource_supplement.resolve()),'sha256':sha256_file(args.resource_supplement)},
            raw_physical_canonical_contracts=physical_contracts,resource_evidence=resource_evidence)
    (args.out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected':selection['selected'],'advance_to_locked_confirmation':selection['advance_to_locked_confirmation']}))


if __name__=='__main__':main()
