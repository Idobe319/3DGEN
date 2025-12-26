#!/usr/bin/env python3
"""Prepare a local TRELLIS model folder for offline use.

- Copies any matching checkpoint files from tsr/ckpts into tsr/TRELLIS-image-large/ckpts
- Warns about missing pipeline.json and recommends downloading the full HF snapshot if needed
- Prints clear next steps for the user
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSR = ROOT / 'tsr'
SRC_CKPTS = TSR / 'ckpts'
MODEL_DIR = TSR / 'TRELLIS-image-large'
DEST_CKPTS = MODEL_DIR / 'ckpts'

if not SRC_CKPTS.exists():
    print('Source ckpts dir not found:', SRC_CKPTS)
    raise SystemExit(1)

os.makedirs(DEST_CKPTS, exist_ok=True)

moved = []
for p in sorted(SRC_CKPTS.glob('*')):
    if p.suffix.lower() in ('.safetensors', '.safetensor', '.pt', '.pth', '.ckpt'):
        dest = DEST_CKPTS / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
            moved.append(p.name)

print('Copied checkpoint files into:', DEST_CKPTS)
for m in moved:
    print('  -', m)

# Check for pipeline.json
if not (MODEL_DIR / 'pipeline.json').exists():
    print('\nNote: pipeline.json not found in the model dir.')
    print('For a fully-functional offline TRELLIS model, download the full "microsoft/TRELLIS-image-large" repository snapshot from Hugging Face and place it under:')
    print('  ', MODEL_DIR)
    print('Required layout:')
    print('  - pipeline.json')
    print('  - ckpts/ (contains the checkpoint .safetensors files)')
    print('  - model subfolders with .json configs for each model referenced by pipeline.json')
    print('\nIf you only want to run certain stages or to validate that the weights are present, this copy step is sufficient for now.')

print('\nNext steps:')
print('  - Run the diagnostic in offline mode to see any remaining missing components:')
print('      $env:HF_HUB_OFFLINE="1"')
print('      conda run -n trellis python -u scripts/check_trellis.py')
print('  - Or run the pipeline in offline mode (pass --trellis-weights to point to the model dir):')
print('      conda run -n trellis python -u run_local.py --input samples/example.jpg --engine trellis --offline --trellis-weights tsr\\TRELLIS-image-large --fast-preview --quality low -o workspace_trellis_smoke')
