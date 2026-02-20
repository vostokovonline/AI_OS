@echo off
REM Quick start script for Windows users
REM This script opens the dashboard in your default browser

echo ========================================
echo AI-OS v2 Dashboard - Quick Start
echo ========================================
echo.

REM Get WSL2 IP address
echo Getting WSL2 IP address...
for /f "tokens=*" %%i in ('wsl hostname -I') do set WSL_IP=%%i
set WSL_IP=%WSL_IP: =%

echo WSL2 IP: %WSL_IP%
echo.

REM Check if services are running
echo Checking if services are running...
curl -s http://%WSL_IP%:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Dashboard is running
    echo.
    echo Opening dashboard in browser...
    start http://%WSL_IP%:3000
) else (
    echo [WARNING] Dashboard may not be running yet
    echo.
    echo Please start the dashboard in WSL2:
    echo   cd ~/ai_os_final/services/dashboard_v2
    echo   npm run dev
    echo.
    echo Then run this script again or open manually:
    echo   http://%WSL_IP%:3000
    echo.
)

pause
