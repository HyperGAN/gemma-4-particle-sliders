"""Explicit common prompts; historical defaults remain byte-identical."""
import hashlib

DEFAULT_PROMPT_PROFILE = 'repair-v1'
FINALIZING_PROMPT_PROFILE = 'repair-finalize-v2'
PROMPT_PROFILES = (DEFAULT_PROMPT_PROFILE, FINALIZING_PROMPT_PROFILE)
FINALIZATION_CUE = ('After submit_patch returns status ok, finish with one short final text response. '
    'Do not call submit_patch again unless you made another code change after the previous submission.')


def profile_prompt(base, profile=DEFAULT_PROMPT_PROFILE):
    if profile == DEFAULT_PROMPT_PROFILE:
        return base
    if profile == FINALIZING_PROMPT_PROFILE:
        return base.rstrip() + '\n\n' + FINALIZATION_CUE + '\n'
    raise ValueError('unknown common system-prompt profile')


def prompt_identity(base, profile=DEFAULT_PROMPT_PROFILE):
    return {'profile': profile, 'sha256': hashlib.sha256(profile_prompt(base, profile).encode()).hexdigest()}
