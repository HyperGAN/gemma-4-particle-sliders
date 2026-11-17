import json
import pytest
from gemma_particle.data import prepare_tasks, read_jsonl
from gemma_particle.curation import select_contexts, TARGET_INSTRUCTIONS
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


def outcome(arm, resolved):
    messages = [{"role": "system", "content": "base" + ("\n\n"+TARGET_INSTRUCTIONS[arm] if arm != "off" else "")},
                {"role": "user", "content": "same issue"}, {"role": "assistant", "content": "finished patch"}]
    receipt = dict(instance_id="task", resolved=resolved, patch_sha256="a"*64,
                   trajectory_sha256=digest(messages), stage="fresh-container-tests")
    return dict(instance_id="task", arm=arm, resolved=resolved, messages=messages,
                patch_sha256="a"*64, verification=receipt, tool_calls=3, elapsed_seconds=5)


def test_curation_requires_actual_paired_win_and_train_membership():
    tasks = [dict(instance_id="task", split="train")]
    assert not select_contexts(tasks, [outcome("off", True), outcome("verify", True)])
    pairs = select_contexts(tasks, [outcome("off", False), outcome("verify", True)])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["neutral_messages"][0]["content"] == "base"
    assert pair["positive_messages"][1:] == pair["neutral_messages"][1:]
    assert all(m["role"] != "assistant" for m in pair["neutral_messages"])
    with pytest.raises(ValueError):
        select_contexts([dict(instance_id="task", split="dev")], [outcome("off", False)])
    bad = outcome("verify", True)
    bad["verification"]["trajectory_sha256"] = "bad"
    with pytest.raises(ValueError):
        select_contexts(tasks, [bad])
