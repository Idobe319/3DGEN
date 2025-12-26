# Run this PowerShell script as Administrator
# Enables WSL and VirtualMachinePlatform, then reboots the machine.
try {
    Write-Host "Enabling Microsoft-Windows-Subsystem-Linux..." -ForegroundColor Cyan
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    Write-Host "Enabling VirtualMachinePlatform..." -ForegroundColor Cyan
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    Write-Host "Features enabled. Rebooting now..." -ForegroundColor Green
    shutdown /r /t 0
} catch {
    Write-Error "Failed to enable WSL/VirtualMachinePlatform: $_"
    exit 1
}