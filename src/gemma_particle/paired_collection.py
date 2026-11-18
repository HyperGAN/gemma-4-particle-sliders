"""Validate the fixed TRAIN pilot and expose its declared arms per task."""
from collections import Counter
import json
from pathlib import Path

from .data import read_jsonl
from .harness_identity import sha256_file

PAIR = ('off', 'verify')
QUARTET = ('off', 'reproduce', 'localize', 'verify')
PILOT_QUARTET = (*PAIR, 'reproduce', 'localize')


def load_collection_pilot(path, expected_sha, visible_tasks, qualification=None):
    path = Path(path).resolve()
    if sha256_file(path) != expected_sha:
        raise ValueError('TRAIN pilot declaration changed')
    plan = json.loads(path.read_text())
    root = path.parent.parent
    pins = plan['pinned_inputs']
    manifest_name = 'data/splits-v1/train.jsonl'
    qualification_name = 'reports/train-environment-qualification-v2.json'
    prior_plan_name = 'runs/jev-finalize-train-pool-v4/plan.json'
    prior_progress_name = 'runs/jev-finalize-train-pool-v4/progress.json'
    for name in (manifest_name, qualification_name, prior_plan_name, prior_progress_name,
                 'runs/empf-only-schedule-and-train-collection-audit-20261117-v1.json'):
        if sha256_file(root / name) != pins[name]:
            raise ValueError('Original TRAIN pilot inputs changed')
    if sha256_file(visible_tasks) != pins[manifest_name]:
        raise ValueError('TRAIN pilot requires the original visible TRAIN manifest')
    qualification_path = root / qualification_name if qualification is None else qualification
    if sha256_file(qualification_path) != pins[qualification_name]:
        raise ValueError('TRAIN pilot requires the original environment qualification')
    controls = json.loads(Path(qualification_path).read_text())['controls']
    qualified = {r['instance_id'] for r in controls if r['status'] == 'passed'}
    tasks = {r['instance_id']: r for r in read_jsonl(visible_tasks)}
    rows = plan['planned_tasks']
    if (plan['schema'] != 'gemma4-train-paired-collection-proposal-v1'
            or len(rows) != 12 or len({row['base_commit'] for row in rows}) != 12
            or len({row['instance_id'] for row in rows}) != 12
            or sum(len(row['planned_arms']) for row in rows) != 32
            or plan['planned_model_trajectories'] != 32):
        raise ValueError('Expected the declared twelve-task paired TRAIN experiment')
    prior_plan = json.loads((root / prior_plan_name).read_text())
    prior_progress = json.loads((root / prior_progress_name).read_text())
    excluded = {row['instance_id'] for row in prior_progress['finished']} | set(prior_plan['excluded_tasks'])
    excluded.add(prior_plan['recovery']['interrupted_task'])
    if excluded != set(plan['excluded_attempted_or_reserved_ids']):
        raise ValueError('TRAIN pilot exclusions differ from the original pool')
    groups = Counter()
    for row in rows:
        task_id = row['instance_id']
        if task_id in excluded or task_id not in qualified or task_id not in tasks:
            raise ValueError('Pilot tasks must be fresh and environment-qualified')
        task = tasks[task_id]
        arms = tuple(row['planned_arms'])
        if (task['split'] != 'train' or task['repo'] != 'fastapi/fastapi'
                or any(task[k] != row[k] for k in ['base_commit', 'created_at']) or arms not in (PAIR, PILOT_QUARTET)):
            raise ValueError('Pilot scope differs from its TRAIN task and arm declarations')
        groups[(row['chronological_group'], arms)] += 1
    expected_groups = {(group, arms): count for group in range(1, 5)
        for arms, count in [(PAIR, 2), (PILOT_QUARTET, 1)]}
    if groups != Counter(expected_groups):
        raise ValueError('Expected two pairs and one quartet per chronological group')
    return rows
