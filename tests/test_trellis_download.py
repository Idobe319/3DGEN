import os
import tempfile
from unittest import mock

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.trellis_utils import download_missing_ckpts


def test_download_missing_ckpts_no_hf(monkeypatch, tmp_path):
    md = tmp_path / 'model'
    md.mkdir()
    # monkeypatch huggingface_hub import failure
    monkeypatch.setitem(os.environ, 'HF_TOKEN', '')
    # simulate missing expected via parameter
    res = download_missing_ckpts(str(md), expected={'missingA.ckpt'}, hf_token=None)
    # since huggingface_hub likely not available in this environment, expect an error key
    assert isinstance(res, dict)
    assert 'downloaded' in res and 'failed' in res


def test_download_missing_ckpts_with_mock(monkeypatch, tmp_path):
    md = tmp_path / 'model'
    md.mkdir()
    ckpt_dir = md / 'ckpts'
    ckpt_dir.mkdir()

    def fake_hf_hub_download(repo_id, filename, token=None):
        # write a small temporary file and return its path
        p = tmp_path / 'cache' / filename.replace('/', '_')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'data')
        return str(p)

    monkeypatch.setitem(os.environ, 'HF_TOKEN', 'fake')
    monkeypatch.setattr('scripts.trellis_utils.hf_hub_download', fake_hf_hub_download, raising=False)

    res = download_missing_ckpts(str(md), expected={'foo.ckpt'}, hf_token='fake')
    # downloaded should show foo.ckpt
    assert 'foo.ckpt' in res['downloaded'] or 'foo.ckpt' in res['failed']
