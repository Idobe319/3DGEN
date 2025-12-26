# Attempt to install flash-attn into the project's venv on Windows (best-effort)
# Usage: run from project root in PowerShell: & .\scripts\try_install_flash_attn_windows.ps1

$venvPy = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $venvPy)) {
    Write-Host "Venv python not found at $venvPy" -ForegroundColor Red
    Write-Host "Activate your venv or adjust the path in this script." -ForegroundColor Yellow
    exit 2
}

# Use the proper invocation with & and the venv python
Write-Host "Using venv python: $venvPy" -ForegroundColor Cyan
& "$venvPy" -m pip install --upgrade pip setuptools wheel
# Try installing flash-attn (may fail on Windows)
try {
    & "$venvPy" -m pip install flash-attn --no-build-isolation
    Write-Host "flash-attn pip install attempted; check output above for success." -ForegroundColor Green
} catch {
    Write-Host "flash-attn pip install failed; attempting fallback to git source (may require build tools)." -ForegroundColor Yellow
    try {
        & "$venvPy" -m pip install git+https://github.com/Dao-AILab/flash-attention.git
        Write-Host "flash-attn source install attempted; review output for errors." -ForegroundColor Green
    } catch {
        Write-Host "All flash-attn install attempts failed. Consider using WSL2 for a more reliable build environment." -ForegroundColor Red
    }
}

Write-Host "Done. If flash-attn failed to install, follow the WSL instructions in docs/WSL_SETUP.md." -ForegroundColor Cyan