from copy import deepcopy

import pytest
import torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM

from gemma_particle.model import ContextRow, Native, make_game


def tiny_model():
    config = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, max_position_embeddings=128,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=64,
        layer_types=["sliding_attention", "full_attention", "full_attention"], num_kv_shared_layers=1)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Gemma4ForCausalLM(config).eval()


def fixture(preset):
    model = tiny_model()
    native = Native(model, identity={"model": "CPU-random-Gemma4Text-fixture", "pretrained": False})
    rows = [native.prepare(ContextRow((2, 3, 4+i), (2, 8, 9, 3, 4+i), f"fixture-{i}", 0)) for i in range(4)]
    game = make_game(native, rows, preset=preset, steps=3, batch=2)
    return model, native, rows, game


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("preset", ["rhbi_optimized", "empf"])
def test_native_gemma4_updates_resume_and_off_parity(preset):
    model, native, rows, game = fixture(preset)
    try:
        frozen = deepcopy(model.state_dict())
        initial = deepcopy(native.network.state_dict())
        off = native.hidden(rows[0].row.neutral_ids, strength=0).detach()
        assert len(native.slider.hosts) == 10  # shared final layer has Q/O only
        assert torch.equal(off, native.generated(rows[0]))  # zero-up constructor
        assert game.recipe.ema_decay == 0
        game.step()
        packet = game.state_dict()
        expected = game.step()
        expected_state = game.state_dict()
        _, other, _, resumed = fixture(preset)
        try:
            resumed.load_state_dict(packet)
            assert_equal(expected, resumed.step())
            assert_equal(expected_state, resumed.state_dict())
        finally:
            other.close()
        assert torch.equal(off, native.hidden(rows[0].row.neutral_ids, strength=0))
        assert any(not torch.equal(initial[k], v) for k, v in native.network.state_dict().items())
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())
        assert all(not p.requires_grad for p in model.parameters())
        assert native.generated(rows[0]).requires_grad
    finally:
        native.close()
