from copy import deepcopy
import json

import pytest

from gemma_particle.decisions import load_key, validate_request, validate_response


def test_private_literal_key_loader_never_executes_and_file_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv('NANOGPT_API_KEY', 'old-test-value')
    file = tmp_path/'startup'
    file.write_text('export NANOGPT_API_KEY="new-test-value" # harmless\n')
    assert load_key(file) == 'new-test-value'
    assert load_key() == 'old-test-value'
    for value in ['$(touch sentinel)', '`touch sentinel`', 'a b', '', 'a\nexport NANOGPT_API_KEY=b']:
        file.write_text('export NANOGPT_API_KEY='+value+'\n')
        with pytest.raises(ValueError):
            load_key(file)
    assert not (tmp_path/'sentinel').exists()


def test_refusals_missing_answers_nan_and_inconsistent_scores_never_become_judgments():
    request = {'model': 'typesafe/jev-1.14', 'state': 'test', 'questions': {
        'repair': {'type': 'score', 'instructions': 'Repair evidence?', 'criteria': ['No evidence', 'Observed evidence']},
        'ran': {'type': 'noul', 'instructions': 'Test ran?'},
    }}
    response = {'model': request['model'], 'answers': {
        'repair': {'type': 'score', 'score': .8, 'confidence': .6, 'probabilities': {'0': .2, '1': .8}},
        'ran': {'type': 'noul', 'noul': .9},
    }}
    assert json.loads(validate_request(request)) == request
    assert validate_response(request, response) == response
    variants = []
    bad = deepcopy(response); bad['model'] = 'typesafe/jev-latest'; variants.append(bad)
    bad = deepcopy(response); del bad['answers']['ran']; variants.append(bad)
    bad = deepcopy(response); bad['answers']['ran'] = {'type': 'refusal'}; variants.append(bad)
    bad = deepcopy(response); bad['answers']['ran']['noul'] = float('nan'); variants.append(bad)
    bad = deepcopy(response); bad['answers']['repair']['score'] = .5; variants.append(bad)
    bad = deepcopy(response); bad['answers']['repair']['probabilities']['1'] = .7; variants.append(bad)
    for bad in variants:
        with pytest.raises(ValueError):
            validate_response(request, bad)


def test_independently_rounded_live_score_and_probabilities_are_retained():
    request = {'model': 'typesafe/jev-1.14', 'state': 'test', 'questions': {
        'repair': {'type': 'score', 'instructions': 'Repair evidence?', 'criteria': ['Absent', 'Proposed', 'Implemented', 'Checked']},
    }}
    response = {'model': request['model'], 'answers': {'repair': {
        'type': 'score', 'score': 1.44, 'confidence': .17,
        'probabilities': {'0': .19, '1': .48, '2': .02, '3': .31},
    }}}
    assert validate_response(request, response)['answers']['repair']['score'] == 1.44


def test_latest_gateway_alias_is_preserved_without_accepting_a_different_model():
    request = {'model': 'typesafe/jev-latest', 'state': 'test', 'questions': {
        'ran': {'type': 'noul', 'instructions': 'Was a test run?'},
    }}
    response = {'model': '~typesafe/jev-latest', 'answers': {'ran': {'type': 'noul', 'noul': .95}}}
    assert validate_response(request, response)['model'] == '~typesafe/jev-latest'
    for model in ['typesafe/jev-1.14', 'liquid/d1']:
        with pytest.raises(ValueError):
            validate_response(request, dict(response, model=model))
