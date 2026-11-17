import json
from copy import deepcopy
import torch
import pytest
from test_gemma_binding import tiny_model, assert_equal
from gemma_particle.model import Native, ContextRow, make_game
from gemma_particle.runtime import save_cache, load_cache, save_native, restore_native


def test_cache_raw_export_and_full_disk_continuation(tmp_path):
    identity = {'fixture':'CPU-random-Gemma4'}
    native = Native(tiny_model(), identity=identity, checkpointing=True)
    other = None
    try:
        prepared = [native.prepare(ContextRow((2,3,4+i),(2,8,9,3,4+i),str(i),0)) for i in range(4)]
        cache = tmp_path/'cache'
        save_cache(cache, prepared, {'model_identity':identity})
        rows, _ = load_cache(cache, identity=identity, train_ids={'0','1','2','3'})
        with pytest.raises(ValueError):
            load_cache(cache, identity=identity, train_ids={'0'})
        game = make_game(native, rows, preset='empf', steps=3, batch=2)
        game.step()
        checkpoint = tmp_path/'full.pt'
        torch.save(game.state_dict(), checkpoint)
        packet = torch.load(checkpoint, weights_only=True)
        raw = tmp_path/'raw'
        save_native(raw, native, game_signature=game.signature, completed_steps=1)
        expected_raw = native.hidden((2,3,4)).detach().clone()
        other = Native(tiny_model(), identity=identity, checkpointing=True)
        restore_native(other, raw)
        assert torch.equal(expected_raw, other.hidden((2,3,4)))
        resumed = make_game(other, rows, preset='empf', steps=3, batch=2)
        resumed.load_state_dict(packet)
        assert_equal(game.step(), resumed.step())
        assert_equal(game.state_dict(), resumed.state_dict())
    finally:
        native.close()
        if other:
            other.close()
