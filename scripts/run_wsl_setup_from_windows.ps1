<#
Run this from Windows (not as Admin necessary). It attempts to run the repo's WSL setup
script inside the Ubuntu distro if WSL and the distro are already initialized.
If Ubuntu hasn't been initialized (first-run user creation), it will tell you to open it and finish setup manually.
#>

# Check WSL status
$st = wsl --status 2>&1
if ($st -match "Default Version: 2") {
    Write-Host "WSL appears to be installed and configured (Default Version 2)" -ForegroundColor Green
} else {
    Write-Warning "WSL doesn't look ready. Run enable script or check 'wsl --status' output manually."
    Write-Host $st
    exit 1
}

# Quick attempt to run inside Ubuntu
Write-Host "Attempting to run the WSL setup script inside Ubuntu-22.04 (non-interactive)" -ForegroundColor Cyan
try {
    wsl -d Ubuntu-22.04 -- bash -lc "if [ ! -f /root/.neoforge_wsl_setup_done ]; then cd /mnt/c/Users/Admin/Desktop/3DGEN && chmod +x scripts/wsl_setup.sh && bash scripts/wsl_setup.sh --use-conda && touch /root/.neoforge_wsl_setup_done; else echo 'WSL setup already performed'; fi" | Tee-Object -Variable out
    Write-Host "WSL setup attempt finished. Inspect output above for errors." -ForegroundColor Green
} catch {
    Write-Warning "Invocation failed — the distro may not be initialized. Try opening Ubuntu from Start Menu and finish first-run user creation, then re-run this script."
    Write-Host $_
    exit 2
}

Write-Host "If the script completed successfully, run: wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/projects/3DGEN/scripts/wsl_smoke_test.sh' or run scripts/wsl_smoke_test.sh from inside the distro." -ForegroundColor Cyan
