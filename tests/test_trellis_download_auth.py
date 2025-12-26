import os
from unittest import mock
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.trellis_utils import download_missing_ckpts


def test_download_missing_ckpts_auth_required(monkeypatch, tmp_path):
    md = tmp_path / 'model'
    md.mkdir()

    def fake_hf_hub_download(repo_id, filename, token=None):
        raise RuntimeError('401 Unauthorized')

    # ensure huggingface_hub import succeeds by injecting a dummy module
    import types
    fake_mod = types.ModuleType('huggingface_hub')
    fake_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_mod)

    res = download_missing_ckpts(str(md), expected={'a.ckpt'}, hf_token=None)
    assert isinstance(res, dict)
    assert 'a.ckpt' in res['failed']
    assert res.get('auth_required') is True


def test_download_missing_ckpts_failure_with_token(monkeypatch, tmp_path):
    md = tmp_path / 'model'
    md.mkdir()

    def fake_hf_hub_download(repo_id, filename, token=None):
        raise RuntimeError('Some network error')

    import types
    fake_mod = types.ModuleType('huggingface_hub')
    fake_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_mod)

    res = download_missing_ckpts(str(md), expected={'b.ckpt'}, hf_token='fake')
    assert isinstance(res, dict)
    assert 'b.ckpt' in res['failed']
    assert res.get('auth_required') is not True
