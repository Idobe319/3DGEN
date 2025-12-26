import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.trellis_utils import discover_ckpt_sources, ckpt_hint


def test_discover_ckpt_sources_returns_dict():
    m = discover_ckpt_sources(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    assert isinstance(m, dict)


def test_ckpt_hint_format():
    m = discover_ckpt_sources(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # If mapping is empty, ckpt_hint should still return a string for a sample name
    h = ckpt_hint('model.ckpt', m)
    assert isinstance(h, str)
    assert 'ckpts/model.ckpt' in h
