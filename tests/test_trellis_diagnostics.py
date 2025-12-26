import os
import sys
import tempfile
# ensure project root is importable for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.trellis_utils import discover_expected_ckpts, check_model_dir_for_ckpts


def test_discover_returns_set():
    s = discover_expected_ckpts(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    assert isinstance(s, set)


def test_check_model_dir_for_ckpts_empty(tmp_path):
    # create an empty model dir
    md = tmp_path / 'model'
    md.mkdir()
    missing = check_model_dir_for_ckpts(str(md), expected={'foo.ckpt', 'bar.safetensors'})
    assert set(missing) == {'foo.ckpt', 'bar.safetensors'}


def test_check_model_dir_for_ckpts_present(tmp_path):
    md = tmp_path / 'model'
    ckpt_dir = md / 'ckpts'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'foo.ckpt').write_bytes(b'd')
    missing = check_model_dir_for_ckpts(str(md), expected={'foo.ckpt', 'bar.safetensors'})
    assert set(missing) == {'bar.safetensors'}
