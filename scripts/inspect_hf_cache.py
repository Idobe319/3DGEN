#!/usr/bin/env python3
import os
from pathlib import Path

print('ENV VARS:')
print(' HF_HOME =', os.environ.get('HF_HOME'))
print(' HUGGINGFACE_HUB_CACHE =', os.environ.get('HUGGINGFACE_HUB_CACHE'))
print(' TRANSFORMERS_CACHE =', os.environ.get('TRANSFORMERS_CACHE'))

print('\nDefault HF cache candidates:')
candidates = [
    Path.home() / '.cache' / 'huggingface',
    Path(os.environ.get('LOCALAPPDATA', '')) / 'huggingface',
    Path(os.environ.get('USERPROFILE', '')) / '.cache' / 'huggingface',
]
for p in candidates:
    print(' -', p, '(exists)' if p.exists() else '(missing)')

# Search for TRELLIS-related paths inside these candidates
roots = [p for p in candidates if p.exists()]
print('\nSearching candidate caches for TRELLIS*...')
hits = []
for root in roots:
    try:
        for f in root.rglob('*'):
            try:
                if 'trellis' in f.name.lower():
                    hits.append(str(f))
            except Exception:
                continue
    except Exception:
        continue

if hits:
    print('\nTRELLIS-related hits (first 200):')
    for h in sorted(set(hits))[:200]:
        print(' -', h)
else:
    print('\nNo TRELLIS-related hits found in default cache locations.')

# Also print local tsr snapshot path and its ckpts contents if present
local_snapshot = Path('tsr') / 'TRELLIS-image-large'
print('\nLocal model snapshot path:', local_snapshot.resolve())
if local_snapshot.exists():
    print(' Snapshot exists; top entries:', sorted([e.name for e in local_snapshot.iterdir()])[:40])
    ckpts = local_snapshot / 'ckpts'
    if ckpts.exists():
        print(' CKPTS files:', sorted([p.name for p in ckpts.iterdir() if p.is_file()])[:200])
    else:
        print(' No ckpts/ directory under snapshot')
else:
    print(' Snapshot directory not found')
