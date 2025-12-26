from neoforge_core import _model_cache


def test_model_cache_basic(tmp_path):
    calls = {'n': 0}

    def loader():
        calls['n'] += 1
        return {'created': calls['n']}

    # Clear cache
    _model_cache.clear()

    a = _model_cache.get('engine', 'ident', loader)
    b = _model_cache.get('engine', 'ident', loader)
    assert a is b
    assert calls['n'] == 1

    # Different key -> loader called again
    c = _model_cache.get('engine', 'ident2', loader)
    assert calls['n'] == 2

    _model_cache.clear()
    d = _model_cache.get('engine', 'ident', loader)
    assert calls['n'] == 3
