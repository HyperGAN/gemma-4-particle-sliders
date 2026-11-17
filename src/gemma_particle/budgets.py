"""Documented competition defaults; short profiles are historical opt-ins."""
PILOT_BUDGET = 'pilot-5m-v1'
OFFICIAL_BUDGET = 'official-default-60m-v1'
DEFAULT_BUDGET = OFFICIAL_BUDGET
BUDGET_PROFILES = {
    PILOT_BUDGET:dict(timeout_seconds=60,max_tool_calls=40,max_time_minutes=5,max_turns=70),
    OFFICIAL_BUDGET:dict(timeout_seconds=300,max_tool_calls=100,max_time_minutes=60,max_turns=500),
}


def budget_values(profile=DEFAULT_BUDGET):
    if profile not in BUDGET_PROFILES:raise ValueError('unknown operational budget profile')
    return BUDGET_PROFILES[profile].copy()
