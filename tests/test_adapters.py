import json
import os
from pathlib import Path
import subprocess


def test_actual_official_compiler_and_adk_use_adapter_alias_for_decisions_and_summaries(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out=tmp_path/'candidate-routing'
    result=subprocess.run([str(root/'.harness-venv/bin/python'),'scripts/qualify_candidate_routing.py','--out',str(out)],
        cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr[-5000:]
    receipt=json.loads((out/'result.json').read_text())
    assert receipt['official_compiler_selected_adapter_alias']
    assert receipt['all_captured_calls_use_exact_adapter_alias']
    assert receipt['summary_uses_adapter_alias']
    assert receipt['seed_and_thinking_controls_preserved']
    assert receipt['server_alias_and_bytes_mismatches_rejected']
    assert receipt['synthetic_adapter'] and not receipt['pretrained_inference']


def test_candidate_cli_rejects_teacher_prompt_combination_before_execution(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([str(root/'.harness-venv/bin/python'),'scripts/run_trajectories.py',
        '--tasks',str(tmp_path/'not-read'),'--task-id','not-read','--arm','localize',
        '--adapter',str(tmp_path/'not-read'),'--wheels',str(tmp_path/'not-read'),'--out',str(tmp_path/'out')],
        cwd=root,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'candidate cohort uses the OFF system prompt' in result.stderr
    assert not (tmp_path/'out').exists()
