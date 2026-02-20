# PowerShell script to set up port forwarding from Windows to WSL2
# Run this script in PowerShell as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WSL2 Dashboard - Windows Access Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get WSL2 IP address
$wslIP = wsl hostname -I
$wslIP = $wslIP.Split(" ")[0]

Write-Host "WSL2 IP Address: $wslIP" -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Remove existing port forwarding rules (if any)
Write-Host "Removing existing port forwarding rules..." -ForegroundColor Yellow
try {
    netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0 2>$null
    netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0 2>$null
    Write-Host "Done." -ForegroundColor Green
} catch {
    Write-Host "No existing rules found." -ForegroundColor Gray
}
Write-Host ""

# Add new port forwarding rules
Write-Host "Adding port forwarding rules..." -ForegroundColor Yellow
try {
    netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=$wslIP
    Write-Host "✓ Port 3000 (Dashboard) forwarded" -ForegroundColor Green

    netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wslIP
    Write-Host "✓ Port 8000 (Backend API) forwarded" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to add port forwarding rules!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
Write-Host ""

# Show current rules
Write-Host "Current port forwarding rules:" -ForegroundColor Yellow
netsh interface portproxy show all
Write-Host ""

# Windows Firewall configuration
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Windows Firewall Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If you still can't access, you may need to allow ports through Windows Firewall:"
Write-Host ""
Write-Host "Run these commands in Administrator PowerShell:" -ForegroundColor Yellow
Write-Host 'netsh advfirewall firewall add rule name="WSL2 Dashboard" dir=in action=allow protocol=TCP localport=3000' -ForegroundColor White
Write-Host 'netsh advfirewall firewall add rule name="WSL2 Backend" dir=in action=allow protocol=TCP localport=8000' -ForegroundColor White
Write-Host ""

# Access URLs
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Access URLs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "From Windows Browser:" -ForegroundColor Green
Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "Or use WSL2 IP directly:" -ForegroundColor Green
Write-Host "  Dashboard: http://$wslIP`:3000" -ForegroundColor White
Write-Host "  Backend API: http://$wslIP`:8000" -ForegroundColor White
Write-Host ""

# Test connection
Write-Host "Testing connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://$wslIP`:3000" -UseBasicParsing -TimeoutSec 2
    if ($response.StatusCode) {
        Write-Host "✓ Dashboard is accessible!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Dashboard may not be running yet. Start it with: cd ~/ai_os_final/services/dashboard_v2 && npm run dev" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
