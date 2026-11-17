"""Paired resolution metrics; all declared tasks stay in the denominator."""
import math
import numpy as np

MATCH_FIELDS = ("task_manifest_sha256", "model", "quantization", "harness_sha256", "budget", "sampling", "seed")


def compare_matched(expected_ids, baseline, candidate, baseline_identity, candidate_identity):
    for key in MATCH_FIELDS:
        if key not in baseline_identity or baseline_identity[key] != candidate_identity.get(key):
            raise ValueError(f"unmatched evaluation contract: {key}")
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise ValueError("nonempty unique declared task IDs required")
    def index(rows):
        result = {}
        for row in rows:
            key = row["instance_id"]
            if key in result or type(row.get("resolved")) is not bool:
                raise ValueError("duplicate result or nonboolean resolved field")
            if row["status"] not in ("completed", "error", "timeout"):
                raise ValueError("result is not terminal")
            if row["resolved"] and (row["status"] != "completed" or not row.get("verification_ref")):
                raise ValueError("resolved result needs completed verification evidence")
            result[key] = row
        if set(result) != expected:
            raise ValueError("evaluation missing/adding declared tasks")
        return result
    a, b = index(baseline), index(candidate)
    ids = sorted(expected)
    before = np.array([a[k]["resolved"] for k in ids], dtype=float)
    after = np.array([b[k]["resolved"] for k in ids], dtype=float)
    gain = after - before
    wins, losses = int((gain > 0).sum()), int((gain < 0).sum())
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.
    rng = np.random.default_rng(0)
    bootstrap = gain[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
    return {"tasks": len(ids), "baseline_resolved": int(before.sum()), "candidate_resolved": int(after.sum()),
            "baseline_rate": float(before.mean()), "candidate_rate": float(after.mean()),
            "paired_gain": float(gain.mean()), "wins": wins, "losses": losses,
            "gain_ci95_task_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist(),
            "mcnemar_exact_p_two_sided": p,
            "candidate_errors": sum(r["status"] == "error" for r in b.values()),
            "candidate_timeouts": sum(r["status"] == "timeout" for r in b.values()),
            "scope": "declared local tasks; does not establish a Kaggle leaderboard result"}
