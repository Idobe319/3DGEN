#!/usr/bin/env bash
set -euo pipefail

echo "Attempting to install flash-attn (best effort)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja packaging
set +e
python -m pip install flash-attn --no-build-isolation
EXIT=$?
set -e
if [ $EXIT -eq 0 ]; then
    echo "flash-attn installed via pip"
    exit 0
fi

echo "flash-attn pip install failed; attempting to build from source"
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git
