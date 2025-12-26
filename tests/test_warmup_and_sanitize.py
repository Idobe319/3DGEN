import sys
import types
import os
import tempfile
from PIL import Image
import numpy as np

import pytest

# Import the helpers
from neoforge_core import GeometryGenerator
from run_local import flash_attn_available_and_patch


class FakeModel:
    def __init__(self):
        self.call_count = 0
    
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, *a, **k):
        # simulate expensive operation
        self.call_count += 1
        return [{'mesh_path': 'fake.obj', 'output_dir': os.getcwd()}]


def test_geometry_generator_does_not_invoke_call_on_init(monkeypatch, tmp_path):
    # Monkeypatch TSR to return a FakeModel that would increment call_count on __call__
    fake_pkg = types.SimpleNamespace(TSR=FakeModel)
    monkeypatch.setitem(sys.modules, 'tsr.system', fake_pkg)

    # Initialize GeometryGenerator: previously this invoked __call__ once during warmup
    gen = GeometryGenerator(trellis_weight_path=None, tsr_options={}, use_cache=False)
    # Ensure the fake model has not performed a heavy __call__ during init
    assert isinstance(gen.model, FakeModel)
    assert gen.model.call_count == 0

    # Calling generate should invoke the model and increment call_count
    img = Image.new('RGBA', (128, 128), (255, 0, 0, 255))
    p = tmp_path / 'in.png'
    img.save(p)
    out = gen.generate(str(p))
    assert gen.model.call_count >= 1


def test_sanitize_for_trellis_handles_empty_alpha(tmp_path):
    # Build an RGBA image with empty alpha channel (all zeros)
    from neoforge_core import sanitize_for_trellis
    img = Image.new('RGBA', (64, 64), (255, 255, 255, 0))
    res = sanitize_for_trellis(img, min_size=32)
    # The returned image must be RGBA and alpha should have some non-zero pixels (fallback mask should set full or heuristic mask)
    assert res.mode == 'RGBA'
    arr = np.array(res)
    alpha = arr[:, :, 3]
    assert alpha.max() > 0


if __name__ == '__main__':
    pytest.main([__file__])