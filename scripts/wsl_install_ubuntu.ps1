# Run in a normal PowerShell (Admin not required if WSL is already enabled)
# Installs Ubuntu-22.04 on WSL and sets WSL2 as default
Write-Host "Setting WSL default version to 2..." -ForegroundColor Cyan
wsl --set-default-version 2

Write-Host "Installing Ubuntu 22.04..." -ForegroundColor Cyan
wsl --install -d Ubuntu-22.04

Write-Host 'If Ubuntu did not auto-start, launch it from Start Menu or run wsl -d Ubuntu-22.04' -ForegroundColor Yellow
Write-Host 'Once in Ubuntu, create a user, then run: ./scripts/wsl_setup.sh inside the Ubuntu shell (see docs).' -ForegroundColor Green