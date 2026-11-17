from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from gemma_particle.curation import select_contexts
from gemma_particle.data import digest
from gemma_particle.history import NO_COMPACTION,OFFICIAL_COMPACTION,history_contract,is_assistant_decision
from test_data import arms


def compacted_arms():
    rows=arms()
    for row in rows:
        row['history_profile']=OFFICIAL_COMPACTION
        row['verification']['history_profile']=OFFICIAL_COMPACTION
    winner=rows[1]
    summary={'model':'gemma','messages':[{'role':'user','content':
        'The following is a conversation history between a user and an AI agent. Please summarize it.'}],
        'tools':None,'generation':{}}
    winner['requests'].append(dict(instance_id='task',arm=winner['arm'],decision=1,purpose='history_summary',
        request=summary,request_sha256=digest(summary)))
    after=deepcopy(winner['requests'][0]);after['decision']=2
    after['request']['messages'].append({'role':'assistant','content':'Actual observed compacted history'})
    after['request_sha256']=digest(after['request']);winner['requests'].append(after)
    winner['verification']['requests_sha256']=digest(winner['requests'])
    return rows


def test_compaction_keeps_summary_in_journal_but_only_decisions_enter_targets():
    rows=compacted_arms();pairs=select_contexts([dict(instance_id='task',split='train')],rows)
    assert [p['decision'] for p in pairs]==[0,2]
    assert pairs[1]['positive_messages'][1:]==pairs[1]['neutral_messages'][1:]
    assert pairs[1]['neutral_messages'][-1]['content']=='Actual observed compacted history'
    assert all(p['history_profile']==OFFICIAL_COMPACTION for p in pairs)
    assert len(rows[1]['requests'])==3


def test_auxiliary_tag_cannot_hide_tool_decisions_or_cross_history_profiles():
    rows=compacted_arms();packet=rows[1]['requests'][0]
    packet['purpose']='history_summary'
    with pytest.raises(ValueError,match='public history summarizer'):
        is_assistant_decision(packet,OFFICIAL_COMPACTION)
    with pytest.raises(ValueError,match='undeclared'):
        is_assistant_decision(rows[1]['requests'][1],NO_COMPACTION)
    rows=compacted_arms();rows[0].pop('history_profile');rows[0]['verification'].pop('history_profile')
    with pytest.raises(ValueError,match='unmatched history'):
        select_contexts([dict(instance_id='task',split='train')],rows)
    assert history_contract(NO_COMPACTION)['compaction'] is None
    assert history_contract(OFFICIAL_COMPACTION)['compaction']['token_threshold']==14336


def test_actual_compiler_clones_preserve_one_history_journal(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    artifact=tmp_path/'artifact'
    built=subprocess.run([str(root/'.harness-venv/bin/python'),'scripts/build_submission.py','--out',str(artifact)],
        cwd=root,env=env,text=True,capture_output=True)
    assert built.returncode==0,built.stderr[-3000:]
    out=tmp_path/'compiled-history'
    result=subprocess.run([str(root/'.harness-venv/bin/python'),'scripts/qualify_history.py','--artifact',str(artifact/'agent'),'--out',str(out)],
        cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr[-5000:]
    receipt=json.loads((out/'result.json').read_text())
    assert receipt['official_compiler_model_clones_share_journal']
    assert receipt['summary_uses_compiled_seed_and_thinking_config']
    assert receipt['assistant_decisions']==8 and receipt['summary_calls']>0
