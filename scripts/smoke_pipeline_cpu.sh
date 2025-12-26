#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test that runs run_local.py on a tiny synthetic image using CPU-friendly flags
PY=$(which python || echo python)
TMPDIR=$(mktemp -d)
IMG="$TMPDIR/test_in.png"
python - <<'PY'
from PIL import Image
img = Image.new('RGBA', (64,64), (255,0,0,255))
img.save('"$IMG"')
PY

echo "Running smoke pipeline (CPU, fast-preview)..."
# Run with fast preview, low quality, offline, and force TripoSR to avoid TRELLIS GPU requirements
$PY run_local.py -i "$IMG" -o "$TMPDIR/out" --engine tripo --fast-preview --quality low --offline --no-flash-attn --cache-models

echo "Smoke run completed. Output dir: $TMPDIR/out"
ls -la "$TMPDIR/out" || true
