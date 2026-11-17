"""Separate agent-visible tasks from reference patches and freeze repo holdouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VISIBLE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "hints_text", "created_at")
REPO_SPLITS = {"fastapi/fastapi": "train", "psf/requests": "dev", "Textualize/rich": "holdout", "encode/httpx": "holdout"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare_tasks(source, out):
    rows = read_jsonl(source)
    identifiers = [row["instance_id"] for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("duplicate task IDs")
    prepared = {split: [] for split in ("train", "dev", "holdout")}
    for row in sorted(rows, key=lambda row: row["instance_id"]):
        split = REPO_SPLITS[row["repo"]]
        visible = {key: row[key] for key in VISIBLE_FIELDS}
        visible["split"] = split
        prepared[split].append(visible)
    manifest = {"schema": "gemma-repo-disjoint-pilot-v1", "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "policy": REPO_SPLITS, "reference_patches_removed": True, "splits": {}}
    for split, tasks in prepared.items():
        manifest["splits"][split] = {"count": len(tasks), "ids": [r["instance_id"] for r in tasks], "sha256": digest(tasks)}
    out = Path(out)
    if out.exists():
        raise FileExistsError("split directory is immutable; use the existing split or a new protocol")
    out.mkdir(parents=True)
    for split, tasks in prepared.items():
        write_jsonl(out / f"{split}.jsonl", tasks)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
