import json
import os
import subprocess
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


@pytest.mark.skipif(os.environ.get('GEMMA_RUN_TINY_CUDA_RESUME')!='1',reason='explicit shared-GPU resume qualification only')
@pytest.mark.parametrize('preset',['empf','rhbi_optimized'])
def test_cuda_disk_resume_keeps_rng_bytes_on_cpu(tmp_path,preset):
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in ('0','1'):raise ValueError('explicit single physical shared GPU required')
    free,total=[int(v.strip())*1024**2 for v in subprocess.check_output(['nvidia-smi',f'--id={physical}',
        '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    if free<.05*total+8*1024**3:pytest.skip('shared micro-check headroom unavailable')
    torch.cuda.set_per_process_memory_fraction(.05,0)
    native=Native(tiny_model().to('cuda:0'),identity={'fixture':'CUDA random Gemma4'},checkpointing=True)
    other=None
    try:
        rows=[native.prepare(ContextRow((2,3,4+i),(2,8,9,3,4+i),str(i),0)) for i in range(4)]
        options={'optimizer_dense_profile':'gpu_native_polar' if preset=='rhbi_optimized' else 'recipe'}
        game=make_game(native,rows,preset=preset,steps=3,batch=2,**options)
        game.step();checkpoint=tmp_path/'full.pt';torch.save(game.state_dict(),checkpoint)
        expected=game.step();expected_state=game.state_dict()
        other=Native(tiny_model().to('cuda:0'),identity={'fixture':'CUDA random Gemma4'},checkpointing=True)
        resumed=make_game(other,rows,preset=preset,steps=3,batch=2,**options)
        with pytest.raises(TypeError,match='torch.ByteTensor'):
            resumed.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True))
        resumed.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        assert_equal(expected,resumed.step())
        assert_equal(expected_state,resumed.state_dict())
    finally:
        native.close()
        if other:other.close()
