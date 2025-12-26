import sys, traceback
sys.path.insert(0, 'external/TRELLIS')
print('PYTHON', sys.executable)
# 1) Import check
try:
    from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_FAIL', repr(e))
    traceback.print_exc()

# 2) Check expected ckpts
try:
    import importlib.util, os
    try:
        import scripts.trellis_utils as t
    except Exception:
        util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(os.path.dirname(__file__), 'trellis_utils.py'))
        if util_spec is None or util_spec.loader is None:
            raise
        t = importlib.util.module_from_spec(util_spec)
        util_spec.loader.exec_module(t)
    expected = t.discover_expected_ckpts()
    print('EXPECTED_CKPTS_COUNT', len(expected))
    if expected:
        print('EXPECTED_CKPTS_SAMPLE', list(expected)[:5])
    missing = t.check_model_dir_for_ckpts('tsr', expected=expected)
    print('MISSING_IN_TSR_DIR', missing)
except Exception as e:
    print('TRELLIS_UTILS_FAIL', repr(e))
    traceback.print_exc()

# 3) Simple HF download dry-run (no token):
try:
    res = None
    import importlib.util, os
    try:
        import scripts.trellis_utils as t
    except Exception:
        util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(os.path.dirname(__file__), 'trellis_utils.py'))
        if util_spec is None or util_spec.loader is None:
            raise
        t = importlib.util.module_from_spec(util_spec)
        util_spec.loader.exec_module(t)
    res = t.download_missing_ckpts('tsr', expected=expected, hf_token=None)
    print('DOWNLOAD_HELPER', res)
except Exception as e:
    print('DOWNLOAD_HELPER_FAIL', repr(e))
    traceback.print_exc()
