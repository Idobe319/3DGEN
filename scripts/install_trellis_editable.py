#!/usr/bin/env python3
"""Install external/TRELLIS as an editable package into the active Python env.
Usage:
  python scripts/install_trellis_editable.py
"""
import os
import sys
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TRELLIS_DIR = ROOT / 'external' / 'TRELLIS'
if not TRELLIS_DIR.exists():
    print('external/TRELLIS directory not found:', TRELLIS_DIR)
    sys.exit(2)

print('Installing external/TRELLIS as editable package into current env...')
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', str(TRELLIS_DIR)])
    print('Installed editable. Verifying import...')
    try:
        import importlib
        trellis = importlib.import_module('trellis')
        print('OK: trellis imported from', getattr(trellis, '__file__', '(unknown)'))
    except Exception as e:
        print('Installation succeeded but import failed:', e)
        print('Try activating the target environment and rerunning this script inside it:')
        print('  conda activate trellis')
        print('  python scripts/install_trellis_editable.py')
        sys.exit(1)
except subprocess.CalledProcessError as e:
    print('pip install failed:', e)
    sys.exit(1)

print('Done.')
