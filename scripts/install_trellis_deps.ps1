# Run from project root (PowerShell)
# This script attempts to install commonly required packages for TRELLIS on Windows.
# Use with caution — binary wheels must match your PyTorch/CUDA version.

python - <<'PY'
import torch, sys
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version reported by torch:', torch.version.cuda)
else:
    print('No CUDA available; TRELLIS CPU runs may be very slow.')
PY

Write-Host "\nProceeding to install optional packages. If you have a specific CUDA/PyTorch combo, install matching wheels manually." -ForegroundColor Yellow

# SpConv example (only if you know your CUDA version supports it)
Write-Host "Installing xformers, imageio and imageio-ffmpeg..." -ForegroundColor Cyan
python -m pip install xformers imageio imageio-ffmpeg

Write-Host "Installing Kaolin (NVIDIA) - NOTE: this may require selecting a wheel for your CUDA version." -ForegroundColor Cyan
Write-Host "If the following fails, please install Kaolin manually following NVIDIA documentation." -ForegroundColor Yellow
python -m pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html

Write-Host "If you require spconv, install the wheel matching your CUDA (example spconv-cu120):" -ForegroundColor Cyan
Write-Host "python -m pip install spconv-cu120" -ForegroundColor Gray

Write-Host "\nScript finished. Verify installations and consider restarting your shell if needed." -ForegroundColor Green