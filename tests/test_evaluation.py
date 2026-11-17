import pytest
from gemma_particle.evaluation import compare_matched, MATCH_FIELDS


def test_errors_remain_in_denominator_and_contracts_match():
    identity = dict.fromkeys(MATCH_FIELDS, "same")
    before = [dict(instance_id="a", status="completed", resolved=False),
              dict(instance_id="b", status="completed", resolved=True, verification_ref="test-log")]
    after = [dict(instance_id="a", status="completed", resolved=True, verification_ref="test-log"),
             dict(instance_id="b", status="timeout", resolved=False)]
    result = compare_matched(["a", "b"], before, after, identity, identity)
    assert result["paired_gain"] == 0 and result["candidate_timeouts"] == 1
    with pytest.raises(ValueError):
        compare_matched(["a", "b"], before, after[:1], identity, identity)
    with pytest.raises(ValueError):
        compare_matched(["a", "b"], before, after, identity, dict(identity, model="different"))
