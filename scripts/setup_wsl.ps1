# Run as Administrator in PowerShell to install WSL and Ubuntu
# This script performs the Windows-side steps to enable WSL2 and ensure Ubuntu is available.

param()

Write-Host "WSL setup helper for TRELLIS/TripoSR"

if (-not (Test-Path Env:USERPROFILE)) {
    Write-Host "This script should be run on Windows PowerShell."
}

Write-Host "Step 1: Ensure WSL is installed (requires Administrator privileges)"
Write-Host "If you haven't installed WSL yet, run (as Administrator): wsl --install -d Ubuntu"
Write-Host "After installation, open Ubuntu from the Start menu and run the WSL setup script:"
Write-Host "  bash /mnt/c/Users/Admin/Desktop/3DGEN/scripts/wsl_setup.sh"

Write-Host "If you prefer the script to attempt to run wsl --install automatically, run PowerShell as Administrator and execute:"
Write-Host "  wsl --install -d Ubuntu"

Write-Host "Once Ubuntu is installed, open it and run the WSL setup script above to finish setting up Miniconda, conda env, and flash-attn."
