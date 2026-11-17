import json
import pytest
from gemma_particle.data import prepare_tasks, read_jsonl
from gemma_particle.curation import select_contexts, TARGET_INSTRUCTIONS,instruction_set,DEFAULT_INSTRUCTION_PROTOCOL
from gemma_particle.data import digest


def test_references_removed_and_splits_immutable(tmp_path):
    source = tmp_path / "raw.jsonl"
    row = dict(instance_id="fastapi_1", repo="fastapi/fastapi", base_commit="a"*40,
               problem_statement="Repair a bug", hints_text="", created_at="2026-01-01",
               patch="PRIVATE REFERENCE", test_patch="PRIVATE TEST")
    source.write_text(json.dumps(row)+"\n")
    out = tmp_path / "split"
    manifest = prepare_tasks(source, out)
    assert manifest["splits"]["train"]["count"] == 1
    assert "PRIVATE" not in (out / "train.jsonl").read_text()
    with pytest.raises(FileExistsError):
        prepare_tasks(source, out)


def outcome(arm, resolved, protocol=DEFAULT_INSTRUCTION_PROTOCOL):
    messages = [{"role": "system", "content": "base" + ("\n\n"+instruction_set(protocol)[arm] if arm != "off" else "")+'\n\nADK agent identity'},
                {"role": "user", "content": "same issue"}]
    request={'model':'gemma','messages':messages,'tools':None,'generation':{'seed':0}}
    requests=[dict(instance_id='task',arm=arm,decision=0,request=request,request_sha256=digest(request))]
    receipt = dict(instance_id="task", resolved=resolved, patch_sha256="a"*64,
                   requests_sha256=digest(requests),first_request_sha256=digest(request),stage="fresh-container-tests")
    result=dict(schema='gemma4-verified-trajectory-v3',generator_kind='gemma4',frozen_base_generator=True,instance_id="task", arm=arm,
                first_request=request,task_budget_exhausted=False,
                resolved=resolved,requests=requests,capture_complete=True,evaluation_contract_sha256='matched',
                model_identity={'repo':'google/gemma-4-E2B-it','revision':'b'*40,'dtype':'bfloat16'},
                patch_sha256="a"*64, verification=receipt, tool_calls=3, elapsed_seconds=5)
    if protocol!=DEFAULT_INSTRUCTION_PROTOCOL:result['instruction_protocol']=protocol
    return result


def arms(*,off=False,protocol=DEFAULT_INSTRUCTION_PROTOCOL):
    return [outcome('off',off,protocol),outcome('verify',True,protocol),outcome('reproduce',False,protocol),outcome('localize',False,protocol)]


def test_curation_requires_actual_paired_win_and_train_membership():
    tasks = [dict(instance_id="task", split="train")]
    assert not select_contexts(tasks, arms(off=True))
    pairs = select_contexts(tasks, arms())
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["neutral_messages"][0]["content"] == "base\n\nADK agent identity"
    assert pair["positive_messages"][1:] == pair["neutral_messages"][1:]
    assert all(m["role"] != "assistant" for m in pair["neutral_messages"])
    with pytest.raises(ValueError):
        select_contexts([dict(instance_id="task", split="dev")], [outcome("off", False)])
    bad = outcome("verify", True)
    bad["verification"]["requests_sha256"] = "bad"
    with pytest.raises(ValueError):
        select_contexts(tasks, [bad])


def test_fixture_incomplete_cohort_and_contract_changes_cannot_supply_targets():
    tasks=[dict(instance_id='task',split='train')]
    rows=arms();rows[1]['generator_kind']='cpu_protocol_fixture'
    with pytest.raises(ValueError,match='CPU protocol'):
        select_contexts(tasks,rows)
    with pytest.raises(ValueError,match='four declared'):
        select_contexts(tasks,arms()[:2])
    rows=arms();rows[1]['evaluation_contract_sha256']='different'
    with pytest.raises(ValueError,match='unmatched'):
        select_contexts(tasks,rows)


def test_curation_rejects_different_sampling_and_quantized_teacher():
    tasks=[dict(instance_id='task',split='train')]
    rows=arms();winner=rows[1];packet=winner['requests'][0]
    packet['request']['generation']['seed']=42
    packet['request_sha256']=digest(packet['request'])
    winner['verification']['requests_sha256']=digest(winner['requests'])
    winner['verification']['first_request_sha256']=digest(winner['first_request'])
    with pytest.raises(ValueError,match='generation differ'):
        select_contexts(tasks,rows)
    rows=arms();rows[1]['model_identity']['dtype']='w4a16'
    with pytest.raises(ValueError,match='BF16'):
        select_contexts(tasks,rows)


def test_task_timeout_off_control_retains_real_first_request_but_winner_must_be_complete():
    tasks=[dict(instance_id='task',split='train')]
    rows=arms();baseline=rows[0]
    baseline.update(capture_complete=False,requests=[],task_budget_exhausted=True)
    baseline['verification']['requests_sha256']=digest([])
    assert len(select_contexts(tasks,rows))==1
    baseline['task_budget_exhausted']=False
    assert select_contexts(tasks,rows)==[]
    baseline['task_budget_exhausted']=True;rows[1]['capture_complete']=False
    assert select_contexts(tasks,rows)==[]


def test_versioned_tool_instruction_pairs_and_mixed_protocol_rejection():
    tasks=[dict(instance_id='task',split='train')]
    rows=arms(protocol='tools-v2');pairs=select_contexts(tasks,rows)
    assert pairs[0]['instruction_protocol']=='tools-v2'
    assert 'Keep filepath a plain path' in pairs[0]['positive_messages'][0]['content']
    assert 'Keep filepath a plain path' not in pairs[0]['neutral_messages'][0]['content']
    assert 'instruction_protocol' not in select_contexts(tasks,arms())[0]
    rows[0].pop('instruction_protocol')
    with pytest.raises(ValueError,match='unmatched instruction protocols'):select_contexts(tasks,rows)
    rows=arms()
    for row in rows:row['instruction_protocol']='tools-v2'
    with pytest.raises(ValueError,match='declared instruction'):select_contexts(tasks,rows)
