# WSL2 + TRELLIS Setup Guide (Ubuntu on Windows 11)

This document shows a recommended, reproducible way to run TRELLIS/TripoSR under WSL2 with GPU passthrough for reliable FlashAttention support.

> High-level steps (Windows host):
> 1. Install WSL2 + Ubuntu (PowerShell as Administrator)
> 2. Inside Ubuntu, run the provided script to install Miniconda, create `trellis` env, install PyTorch+CUDA and flash-attn
> 3. Copy your project into the WSL filesystem for better IO and run smoke tests

## 1) Install WSL2 (Windows)
Open PowerShell as Administrator and run one of the following approaches (both work):

Option A — Manual (recommended):

```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
shutdown /r /t 0
```

After reboot, run:

```
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Option B — Automated script (run as Administrator):

```
# from an elevated PowerShell prompt in the repo root
.\scripts\enable_wsl.ps1
```

Restart if requested. Confirm:

```
wsl -l -v
```

If Ubuntu shows version 1, upgrade:

```
wsl --set-version Ubuntu 2
```

## 2) Verify NVIDIA drivers
On Windows (PowerShell):

```
nvidia-smi
```

Inside WSL Ubuntu:

```
nvidia-smi
```

Both should show your GPU details (e.g., NVIDIA GeForce RTX 4080 Laptop GPU).

## 3) Run the WSL setup script (inside Ubuntu)
Copy the project into WSL (recommended) and run the bundled script (defaults to using a Python venv for portability):

```
# from inside WSL
bash /mnt/c/Users/Admin/Desktop/3DGEN/scripts/wsl_setup.sh

# or, if you copied the project into ~/projects/3DGEN in WSL
bash ~/projects/3DGEN/scripts/wsl_setup.sh

# To use Miniconda instead of venv, pass --use-conda:
bash ~/projects/3DGEN/scripts/wsl_setup.sh --use-conda
```

The script will:
- Install Miniconda (non-interactive)
- Create `conda env create -n trellis python=3.10`
- Install PyTorch (CU121 wheels) and common deps
- Try to install `flash-attn` (best-effort); fallback to source build if pip fails
- Copy your project to `~/projects/3DGEN` for better I/O
- Run a probe that prints torch, CUDA and `flash_attn` availability

## 4) Helpful follow-ups
- If `flash-attn` build fails, try a pinned version:

```
pip install flash-attn==2.6.3 --no-build-isolation
```

- If you prefer not to install FlashAttention, the repo falls back to PyTorch SDPA (slower but stable).

## 5) Smoke tests
After setup, from the project directory in WSL run:

```
bash scripts/wsl_smoke_test.sh
```

Look for `has_func=True` for `flash_attn` and `torch.cuda.is_available()` to be `True`.

---
If you'd like, I can also script an automated WSL installer that attempts `wsl --install` from PowerShell (requires Administrator), but I recommend running that command manually to review prompts.
