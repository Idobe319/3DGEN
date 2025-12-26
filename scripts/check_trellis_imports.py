#!/usr/bin/env python3
"""Quick diagnostics for TRELLIS import behaviour.
- Tests import trellis in current interpreter
- Prints guidance for `python -c "import trellis"` checks and editable install
"""
import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TRELLIS_DIR = ROOT / 'external' / 'TRELLIS'

print('== TRELLIS import check (current interpreter) ==')
try:
    import trellis
    print('OK: trellis imported from', getattr(trellis, '__file__', '(unknown)'))
except Exception as e:
    print('FAIL: import trellis failed in current interpreter:', e)
    if TRELLIS_DIR.exists():
        print('\nHints:')
        print(f' - Add local TRELLIS to PYTHONPATH (PowerShell):')
        print(f'     $env:PYTHONPATH = "{str(TRELLIS_DIR)}"')
        print('   Then run:')
        print('     conda run -n trellis python -c "import trellis; print(trellis.__file__)"')
        print('\n - Or install as editable in your conda env:')
        print('     conda run -n trellis python -m pip install -e .\\external\\TRELLIS')
    else:
        print('\nexternal/TRELLIS not found in repository; ensure the TRELLIS code is present.')

print('\n== Quick python -c sanity check (run in trellis env) ==')
print('Suggested (PowerShell):')
print("$env:PYTHONPATH='{}'; conda run -n trellis python -c \"import trellis; print(trellis.__file__)\"".format(str(TRELLIS_DIR)))
print('\nOr install editable package as shown in the hints above.')
