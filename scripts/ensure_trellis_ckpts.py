#!/usr/bin/env python3
"""Check and optionally download missing TRELLIS checkpoint files.
Usage:
  python scripts/ensure_trellis_ckpts.py [--model-dir <path>] [--download]

If --download is set and an HF token is present in HUGGINGFACE_TOKEN/HF_TOKEN,
attempts to download missing ckpts using the trellis_utils helper.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument('--model-dir', help='Local model directory (pipeline.json + ckpts/) to check. Defaults to external/TRELLIS or tsr/TRELLIS-image-large', default=None)
p.add_argument('--download', action='store_true', help='Attempt to download missing ckpts using HF token in env')
args = p.parse_args()

# locate default model dir candidates
candidates = []
if args.model_dir:
    candidates.append(args.model_dir)
candidates.extend([str(ROOT / 'external' / 'TRELLIS'), str(ROOT / 'tsr' / 'TRELLIS-image-large'), str(ROOT / 'tsr')])

found = None
for c in candidates:
    if c and Path(c).exists() and Path(c).joinpath('pipeline.json').exists():
        found = c
        break

try:
    import importlib.util
    util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(str(ROOT), 'scripts', 'trellis_utils.py'))
    if util_spec is None or util_spec.loader is None:
        raise RuntimeError('Failed to locate trellis_utils module')
    trellis_utils = importlib.util.module_from_spec(util_spec)
    util_spec.loader.exec_module(trellis_utils)
except Exception as e:
    print('Failed to load scripts/trellis_utils.py:', e)
    sys.exit(2)

expected = trellis_utils.discover_expected_ckpts(str(ROOT))
print('Expected checkpoint stems (from configs):')
for s in sorted(expected):
    print(' -', s)

if not found:
    print('\nNo local model directory with pipeline.json found in candidates:')
    for c in candidates:
        print(' -', c)
    print('\nPlace the TRELLIS repo (pipeline.json + ckpts/) in one of those locations, or pass --model-dir')
    sys.exit(1)

print('\nUsing model dir:', found)
missing = trellis_utils.check_model_dir_for_ckpts(found, expected)
if not missing:
    print('All expected checkpoint candidates are present in', found)
    sys.exit(0)

print('Missing checkpoint candidates:')
for m in sorted(missing):
    print(' -', m)

# Provide hints
sources = trellis_utils.discover_ckpt_sources(str(ROOT))
print('\nHints to obtain missing ckpts:')
for m in sorted(missing):
    try:
        hint = trellis_utils.ckpt_hint(m, sources)
    except Exception:
        hint = 'See scripts/trellis_utils.py for hints'
    print(' -', hint)

if args.download:
    token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if not token:
        print('\nNo HUGGINGFACE_TOKEN/HF_TOKEN found in environment; cannot auto-download. Export token and re-run with --download.')
        sys.exit(2)
    print('\nAttempting auto-download of missing ckpts to', found)
    res = trellis_utils.download_missing_ckpts(found, expected=set(missing), hf_token=token)
    print('Download result:', res)
    if res.get('downloaded'):
        print('Downloaded:', res['downloaded'])
    if res.get('failed'):
        print('Failed:', res['failed'])
    if res.get('error'):
        print('Error:', res['error'])

print('\nDone. After placing/downloading ckpts, re-run the verification command to confirm TRELLIS loads and runs.')
