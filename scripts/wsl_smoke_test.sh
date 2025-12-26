#!/usr/bin/env bash
set -euo pipefail

# Run a thorough WSL smoke test: GPU, torch, flash_attn, and a short TRELLIS run

echo "[wsl_smoke_test] nvidia-smi:"
nvidia-smi || echo "nvidia-smi failed (check NVIDIA driver / WSL GPU passthrough)"

echo "[wsl_smoke_test] torch probe:"
python - <<'PY'
import torch, sys
print('python', sys.version.splitlines()[0])
print('torch', getattr(torch,'__version__',None))
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

echo "[wsl_smoke_test] flash_attn probe:"
python - <<'PY'
import importlib
try:
    fa = importlib.import_module('flash_attn')
    print('flash_attn file:', getattr(fa,'__file__',None))
except Exception as e:
    print('flash_attn import error:', e)
PY

# Activate venv or conda env
if [ -f ".venv/bin/activate" ]; then
  echo "Activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif command -v conda >/dev/null 2>&1 && conda env list | grep -q "^trellis\b"; then
  echo "Activating conda trellis env"
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate trellis
else
  echo "No .venv or conda trellis env found — please activate your environment manually and re-run the script"
  exit 2
fi

# find sample input
IMG=""
if [ -f "samples/example.jpg" ]; then
  IMG="samples/example.jpg"
else
  candidate=$(find samples -type f \( -iname '*.jpg' -o -iname '*.png' \) -size +1k -print | head -n 1 || true)
  if [ -n "$candidate" ]; then
    IMG="$candidate"
  else
    echo "No sample image found under samples/ — please supply an image to test" && exit 2
  fi
fi

OUTDIR="workspace_trellis_wsl_smoke"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

echo "[wsl_smoke_test] Running run_local.py with --engine trellis (low quality, fast-preview)"
python -u run_local.py --input "$IMG" --engine trellis --fast-preview --quality low -o "$OUTDIR" 2>&1 | tee "$LOG"

# look for success markers
if grep -E "Loading.*TRELLIS|Running.*TRELLIS|TRELLIS OK|TripoSR completed|TripoSR: found" "$LOG" -n -m 5; then
  echo "[wsl_smoke_test] Found TRELLIS/TripoSR markers in log — check $LOG for details"
else
  echo "[wsl_smoke_test] No TRELLIS markers found in the log. Print head of log for debugging:"
  head -n 200 "$LOG"
fi

# check outputs
if ls "$OUTDIR" | grep -E "clean_quads|temp_raw|_uv" >/dev/null 2>&1; then
  echo "[wsl_smoke_test] Smoke artifacts found in $OUTDIR"
else
  echo "[wsl_smoke_test] No expected artifacts were found; the run might have failed. See $LOG"
fi

echo "WSL smoke test complete."