from dataclasses import asdict
import importlib
from pathlib import Path

import torch

from gemma_particle.model import ContextRow, Native, make_game
from test_gemma_binding import assert_equal, tiny_model


def test_rate_screen_uses_fixed_projection_distribution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    screen = importlib.import_module('screen_empf_lr')
    directions = screen.projections_for(3)
    a = torch.tensor([[0, 1, 2], [4, 3, 2]], dtype=torch.float64)
    assert torch.equal(directions, screen.projections_for(3))
    assert screen.sliced_wasserstein(a, a.flip(0), directions)['mean'] == 0.0
    assert screen.sliced_wasserstein(a, a + 1, directions)['mean'] > 0.0


def test_only_the_base_rate_changes_the_eight_update_empf_recipe():
    natives = []

    def game(rate):
        native = Native(tiny_model(), identity={'fixture': 'cpu-rate-screen'})
        natives.append(native)
        rows = [native.prepare(ContextRow((2, 3, i), (2, 8, 3, i), str(i), 0)) for i in range(4, 6)]
        override = {'lr': rate} if rate is not None else None
        return make_game(native, rows, preset='empf', steps=1024, batch=2, recipe_overrides=override)

    try:
        default = game(None)
        explicit = game(0.012)
        alternative = game(0.006)
        assert default.signature == explicit.signature
        assert asdict(alternative.recipe) == {**asdict(default.recipe), 'lr': 0.006}
        assert alternative.recipe.reg_every == 1
        for _ in range(8):
            assert_equal(default.step(), explicit.step())
            assert alternative.step()['step'] == default.completed_steps
        assert_equal(default.state_dict(), explicit.state_dict())
    finally:
        for native in natives:
            native.close()
