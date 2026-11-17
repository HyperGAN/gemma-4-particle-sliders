import torch
from peft import PeftModel
from gemma_particle.model import ContextRow, Native, make_game
from gemma_particle.export import collect_activations, export_peft
from test_gemma_binding import tiny_model


def test_measured_conversion_loads_through_real_peft(tmp_path):
    base = tiny_model()
    native = Native(base, identity={"fixture": "random-Gemma4"})
    try:
        rows = [ContextRow((2, 3, 4+i), (2, 8, 9, 3, 4+i), f"fixture-{i}", 0) for i in range(4)]
        game = make_game(native, [native.prepare(row) for row in rows], preset="empf", steps=1, batch=2)
        game.step()
        fit = collect_activations(native, rows[:2], tokens_per_layer=5)
        probe = collect_activations(native, rows[2:], tokens_per_layer=5)
        path = tmp_path / "adapter"
        receipt = export_peft(native, fit, probe, path, base_model_name="CPU-fixture",
                              provenance={"fit_task_ids": ["fixture-0", "fixture-1"], "probe_task_ids": ["fixture-2", "fixture-3"]})
        assert receipt["exact_merge"] is False
        assert all(m["relative_delta_rmse"] is not None for m in receipt["layer_metrics"].values())
        fresh = tiny_model()
        loaded = PeftModel.from_pretrained(fresh, path).eval()
        ids = torch.tensor([[2, 3, 4]])
        with torch.no_grad():
            logits = loaded(ids).logits
        assert torch.isfinite(logits).all()
        assert not torch.equal(logits, tiny_model()(ids).logits)
    finally:
        native.close()
