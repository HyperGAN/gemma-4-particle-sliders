"""Explicit ADK history profiles, using the official starter's public APIs."""
from contextvars import ContextVar

NO_COMPACTION='uncompacted-v1'
ARCHIVED_COMPACTION='official-compaction-v1'
OFFICIAL_COMPACTION='official-compaction-v2'
HISTORY_PROFILES=(NO_COMPACTION,OFFICIAL_COMPACTION)
CALL_PURPOSE=ContextVar('gemma_call_purpose',default=None)


def history_contract(profile):
    if profile==NO_COMPACTION:return {'profile':profile,'compaction':None,'cache':None}
    if profile not in (ARCHIVED_COMPACTION,OFFICIAL_COMPACTION):raise ValueError('unknown history profile')
    return {'profile':profile,
        'compaction':{'compaction_interval':15,'overlap_size':2,'token_threshold':14336,'event_retention_size':5},
        'cache':{'min_tokens':2048,'ttl_seconds':1800,'cache_intervals':10},
        'summarizer':('public LlmEventSummarizer, same registered model and unchanged default prompt; capture-only purpose tag'
            if profile==ARCHIVED_COMPACTION else
            'public LlmEventSummarizer, same compiled canonical root model and unchanged default prompt; capture-only purpose tag')}


def eval_history_options(profile,llm):
    contract=history_contract(profile)
    if profile==NO_COMPACTION:return {}
    if profile==ARCHIVED_COMPACTION:raise ValueError('archived history profile requires its original frozen executor')
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps.app import EventsCompactionConfig
    from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
    from google.adk.apps.base_events_summarizer import BaseEventsSummarizer

    class TaggedSummarizer(BaseEventsSummarizer):
        async def maybe_summarize_events(self,*,events):
            bound=llm.llm_client.bound_model
            if bound is None:raise RuntimeError('compiled root model has not been observed before compaction')
            token=CALL_PURPOSE.set('history_summary')
            try:return await LlmEventSummarizer(llm=bound).maybe_summarize_events(events=events)
            finally:CALL_PURPOSE.reset(token)

    return {'context_cache_config':ContextCacheConfig(**contract['cache']),
        'events_compaction_config':EventsCompactionConfig(**contract['compaction'],summarizer=TaggedSummarizer())}


def is_assistant_decision(packet,profile):
    purpose=packet.get('purpose')
    if purpose is None:return True
    if purpose!='history_summary' or profile not in (ARCHIVED_COMPACTION,OFFICIAL_COMPACTION):
        raise ValueError('undeclared auxiliary model request')
    request=packet['request'];messages=request['messages']
    # The public summarizer submits only a user history prompt without tools.
    # A purpose tag cannot silently remove an ordinary tool decision.
    if (request.get('tools') or len(messages)!=1 or messages[0].get('role')!='user'
            or not isinstance(messages[0].get('content'),str)
            or not messages[0]['content'].startswith('The following is a conversation history between a user and an AI agent.')):
        raise ValueError('auxiliary request differs from public history summarizer')
    return False
