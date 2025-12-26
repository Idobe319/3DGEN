<#
Run this PowerShell script as Administrator to finalize WSL setup.
It runs DISM health checks, SFC scan, and optionally reboots.
#>

if (-not ([bool](net session 2>$null))) {
    Write-Error "This script must be run as Administrator. Right-click PowerShell -> Run as Administrator"
    exit 1
}

Write-Host "[finalize_wsl] Running DISM restorehealth (this can take several minutes)..." -ForegroundColor Cyan
# Run DISM health checks
dism.exe /Online /Cleanup-Image /ScanHealth
$d1 = dism.exe /Online /Cleanup-Image /RestoreHealth
if ($LASTEXITCODE -ne 0) {
    Write-Warning "DISM returned non-zero exit code ($LASTEXITCODE) — check output and consider running again manually"
} else {
    Write-Host "DISM restorehealth completed." -ForegroundColor Green
}

Write-Host "[finalize_wsl] Running SFC /scannow (this may take several minutes)..." -ForegroundColor Cyan
sfc /scannow
if ($LASTEXITCODE -ne 0) {
    Write-Warning "SFC finished with exit code $LASTEXITCODE. Review CBS log for details."
} else {
    Write-Host "SFC completed (exit code 0)." -ForegroundColor Green
}

Write-Host "Final step: system restart is recommended to apply changes." -ForegroundColor Yellow
$ans = Read-Host "Reboot now? (Y/n)"
if ($ans -eq "" -or $ans.ToLower().StartsWith('y')) {
    Write-Host "Rebooting now..." -ForegroundColor Cyan
    shutdown /r /t 5
} else {
    Write-Host "Skipping reboot. Please reboot when convenient." -ForegroundColor Yellow
}
