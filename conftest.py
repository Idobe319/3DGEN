import sys
import os
import pytest
from pathlib import Path

def pytest_ignore_collect(collection_path: Path, config):
    # Prevent collecting tests under external/spconv_build entirely to avoid build attempts at import-time
    p = str(collection_path)
    if os.path.normpath('external' + os.sep + 'spconv_build') in os.path.normpath(p):
        return True
    return False


def pytest_collection_modifyitems(config, items):
    # General approach: skip tests that require spconv if 'spconv' isn't importable or if running on Windows
    spconv_available = True
    try:
        import spconv  # type: ignore
    except Exception:
        spconv_available = False

    for item in items:
        fn = item.location[0]
        # If the test file references spconv or is under the external spconv build, skip when not available
        if 'spconv' in fn or 'spconv_build' in fn or os.path.normpath('external\\spconv_build') in os.path.normpath(fn):
            if not spconv_available or sys.platform.startswith('win'):
                skip_marker = pytest.mark.skip(reason="skipped: spconv not available or not supported on this platform")
                item.add_marker(skip_marker)
