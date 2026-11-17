from copy import deepcopy

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
