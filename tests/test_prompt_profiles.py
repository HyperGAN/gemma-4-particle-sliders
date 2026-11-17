import importlib.util
from pathlib import Path

import pytest

from gemma_particle.prompt_profiles import (DEFAULT_PROMPT_PROFILE,FINALIZING_PROMPT_PROFILE,
    FINALIZATION_CUE,profile_prompt,prompt_identity)
from gemma_particle.curation import instruction_set

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_prompt_profile_test',ROOT/'scripts/build_submission.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_original_prompt_archive_is_exactly_preserved_and_profile_is_explicit(tmp_path):
    base=(ROOT/'submission/prompts/system.md').read_text()
    assert profile_prompt(base,DEFAULT_PROMPT_PROFILE)==base
    original=builder.build(tmp_path/'old',builder.MODELS[0],budget_profile='official-default-60m-v1')
    assert original['sha256']=='05a2bee7bc26199e3b260c20548ecb6a545951d59bce85e77627ada46daa0a02'
    assert 'prompt_profile' not in original
    with pytest.raises(ValueError,match='unknown'):profile_prompt(base,'silent-change')


def test_new_common_cue_is_identical_across_all_arms_and_separate_from_teacher_cue(tmp_path):
    base=(ROOT/'submission/prompts/system.md').read_text()
    common=profile_prompt(base,FINALIZING_PROMPT_PROFILE)
    assert common.count(FINALIZATION_CUE)==1
    identity=prompt_identity(base,FINALIZING_PROMPT_PROFILE)
    assert identity['profile']==FINALIZING_PROMPT_PROFILE
    assert identity['sha256']!=prompt_identity(base)['sha256']
    for arm in ('off','reproduce','localize','verify'):
        out=tmp_path/arm
        receipt=builder.build(out,builder.MODELS[0],instruction=None if arm=='off' else arm,
            instruction_protocol='tools-v2',prompt_profile=FINALIZING_PROMPT_PROFILE)
        observed=(out/'agent/prompts/system.md').read_text()
        expected=common if arm=='off' else common.rstrip()+'\n\n'+instruction_set('tools-v2')[arm]+'\n'
        assert observed==expected and observed.count(FINALIZATION_CUE)==1
        assert receipt['prompt_profile']==FINALIZING_PROMPT_PROFILE
