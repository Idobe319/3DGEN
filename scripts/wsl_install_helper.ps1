# WSL Install Helper (needs Administrator privileges)
# Run in an elevated PowerShell session

Write-Host "WSL Install Helper for NeoForge (Ubuntu)" -ForegroundColor Cyan

try {
    $wslStatus = wsl -l -v 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "WSL already present:" -ForegroundColor Green
        Write-Host $wslStatus
        Write-Host "If you need to change default version: wsl --set-default-version 2" -ForegroundColor Yellow
        return
    }
} catch {
    # proceed to install
}

Write-Host "Attempting to install WSL (Ubuntu). This requires Administrator privileges." -ForegroundColor Yellow
Write-Host "If UAC prompts appear, accept them."

try {
    wsl --install -d Ubuntu
    Write-Host "WSL install invoked successfully. Please restart your machine if prompted." -ForegroundColor Green
} catch {
    Write-Host "WSL install failed or is unavailable. Please run the following commands manually as Administrator:" -ForegroundColor Red
    Write-Host "  wsl --install -d Ubuntu" -ForegroundColor Gray
    Write-Host "See docs/WSL_SETUP.md for details." -ForegroundColor Gray
}
