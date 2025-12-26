#!/usr/bin/env bash
set -euo pipefail

echo "WSL2 setup script for TRELLIS/TripoSR — run this inside Ubuntu on WSL2"

echo "1) Update and install system packages"
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget git build-essential curl ca-certificates python3-venv python3-dev gcc g++ make cmake ninja-build pkg-config libopenmpi-dev libssl-dev libffi-dev rsync

# Choose environment backend: default to venv for simplicity, allow --use-conda to use Miniconda
USE_CONDA=false
for arg in "$@"; do
    if [ "$arg" = "--use-conda" ]; then
        USE_CONDA=true
    fi
done

if [ "$USE_CONDA" = "true" ]; then
    echo "Using Miniconda path (user requested --use-conda)"
    if [ ! -d "$HOME/miniconda3" ]; then
        echo "2) Installing Miniconda..."
        cd "$HOME"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p "$HOME/miniconda3"
        rm miniconda.sh
    else
        echo "Miniconda already present at $HOME/miniconda3"
    fi

    # Initialize conda in this shell
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base || true

    ENV_NAME="trellis"
    if ! conda env list | grep -q "^$ENV_NAME\b"; then
        echo "3) Creating conda environment: $ENV_NAME (python 3.10)"
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    conda activate "$ENV_NAME"

    echo "4) Upgrade pip and install PyTorch (CU121) in conda env"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121

    echo "5) Install build deps for FlashAttention and common python deps"
    python -m pip install ninja packaging cython

else
    echo "Using system Python + venv (recommended for reproducibility)"
    # Ensure python3.10 is available
    if ! command -v python3.10 >/dev/null 2>&1; then
        echo "python3.10 not found — installing"
        sudo apt install -y python3.10 python3.10-venv python3.10-dev
    fi

    echo "3) Creating and activating venv (.venv)"
    python3.10 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel

    echo "4) Install PyTorch (CU121)"
    python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121 || true

    echo "5) Install build deps for FlashAttention and common python deps"
    python -m pip install ninja packaging cython || true
fi

# Try installing flash-attn and spconv conditionally (best-effort)
# Only attempt heavy installs if CUDA appears available and torch reports a CUDA build
python - <<'PY'
import subprocess, sys
try:
    import torch
    cuda = torch.cuda.is_available()
    print('torch', getattr(torch, '__version__', None), 'cuda_available', cuda)
except Exception as e:
    print('torch import failed:', e)
    cuda = False

if cuda:
    print('Attempting pip install flash-attn and spconv (best-effort)')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flash-attn', '--no-build-isolation'], check=False)
    # Try spconv wheel for cu121 where available
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'spconv-cu121'], check=False)
else:
    print('Skipping flash-attn and spconv install (no CUDA)')
PY

# Project placement: recommend copying to native FS for performance
if [ -d "/mnt/c/Users/Admin/Desktop/3DGEN" ]; then
    echo "6) Copying project into WSL filesystem for better IO performance"
    mkdir -p "$HOME/projects"
    rsync -a --delete /mnt/c/Users/Admin/Desktop/3DGEN "$HOME/projects/" || cp -r /mnt/c/Users/Admin/Desktop/3DGEN "$HOME/projects/"
    echo "Project copied to: $HOME/projects/3DGEN"
    cd "$HOME/projects/3DGEN"
else
    echo "6) Project not found under /mnt/c/Users/Admin/Desktop/3DGEN — please clone your repo into ~/projects/3DGEN and re-run the script"
    exit 1
fi

# Install project Python deps (best effort)
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
else
    echo "No requirements.txt found — attempting editable install"
    python -m pip install -e . || echo "Editable install failed; install dependencies manually"
fi

# Run environment probe
echo "7) Environment probe: torch / flash_attn / GPU"
python -c "import torch; print('torch', torch.__version__, 'cuda', getattr(torch.version,'cuda',None), 'cuda_avail', torch.cuda.is_available(), 'device', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
python -c "import importlib, sys
try:
    fa = importlib.import_module('flash_attn')
    print('flash_attn file:', getattr(fa,'__file__',None), 'has_func=', hasattr(fa,'flash_attn_func'))
except Exception as e:
    print('flash_attn import error:', e)"

cat <<'EOF'
Done. If flash-attn failed to install you have two options:
 - try a different version: pip install flash-attn==2.6.3 --no-build-isolation
 - or use the built-in SDPA (already supported by the pipeline)
EOF

echo "WSL setup script finished. To run a TRELLIS smoke test, run:"
echo "  python run.py --help"
echo "or run your project-specific entry command and look for 'FlashAttention available:' logs"
