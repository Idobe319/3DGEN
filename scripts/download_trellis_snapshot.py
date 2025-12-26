#!/usr/bin/env python3
"""Download a minimal TRELLIS model snapshot into tsr/TRELLIS-image-large.
This script uses huggingface_hub.snapshot_download with allow_patterns to avoid fetching unnecessary files.
"""
import os, sys
from huggingface_hub import snapshot_download

repo = 'microsoft/TRELLIS-image-large'
out = os.path.abspath(os.path.join('tsr','TRELLIS-image-large'))
print('Downloading', repo, 'into', out)

patterns = [
    'pipeline.json',
    'README.md',
    'ckpts/*',
    'configs/**',
    '**/*.json',
    'models/**',
]

try:
    path = snapshot_download(repo_id=repo, local_dir=out, local_dir_use_symlinks=False, allow_patterns=patterns, repo_type='model')
    print('SNAPSHOT_DONE', path)
    # list top-level files
    entries = sorted(os.listdir(out)) if os.path.exists(out) else []
    print('TOP_ENTRIES:', entries[:50])
    # check ckpts
    ckpt_dir = os.path.join(out, 'ckpts')
    if os.path.isdir(ckpt_dir):
        print('CKPTS:', sorted(entry for entry in os.listdir(ckpt_dir) if entry.endswith('.safetensors') or entry.endswith('.pt'))[:50])
    else:
        print('CKPTS dir not found in snapshot (download may be gated or incomplete).')
except Exception as e:
    print('SNAPSHOT_ERROR', type(e).__name__, str(e))
    sys.exit(1)
