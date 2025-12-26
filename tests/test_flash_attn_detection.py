import importlib
import sys
import types

import pytest

from run_local import flash_attn_available_and_patch


def make_module(has_func: bool = False):
    m = types.ModuleType('flash_attn')
    if has_func:
        def dummy(*a, **k):
            return None
        m.flash_attn_func = dummy
    return m


def test_no_flash_attn(monkeypatch):
    # Ensure function returns False when flash_attn not present
    monkeypatch.setitem(sys.modules, 'flash_attn', None)
    if 'flash_attn' in sys.modules:
        del sys.modules['flash_attn']
    assert flash_attn_available_and_patch() is False


def test_flash_attn_present(monkeypatch):
    m = make_module(has_func=True)
    monkeypatch.setitem(sys.modules, 'flash_attn', m)
    assert flash_attn_available_and_patch() is True


def test_monkey_patch_interface(monkeypatch):
    # Simulate package where flash_attn exists but function is in submodule
    m = make_module(has_func=False)
    monkeypatch.setitem(sys.modules, 'flash_attn', m)

    fake_sub = types.ModuleType('flash_attn.flash_attn_interface')
    def fake_func(*a, **k):
        return None
    fake_sub.flash_attn_func = fake_func
    monkeypatch.setitem(sys.modules, 'flash_attn.flash_attn_interface', fake_sub)

    # Reload to ensure behavior occurs on import
    assert flash_attn_available_and_patch() is True


if __name__ == '__main__':
    pytest.main([__file__])