# Запустить ЭТОТ ФАЙЛ в Windows PowerShell (Администратор)
# Правый клик -> "Запуск от имени администратора"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Настройка доступа к Dashboard из Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# WSL2 IP
$wslIP = "172.25.50.61"

Write-Host "Настройка проброса портов..." -ForegroundColor Yellow

# Удалить старые правила
netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0 2>$null
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0 2>$null

# Добавить новые правила
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=$wslIP
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wslIP

Write-Host "✓ Готово!" -ForegroundColor Green
Write-Host ""

# Показать правила
Write-Host "Активные правила:" -ForegroundColor Yellow
netsh interface portproxy show all
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ОТКРОЙТЕ В БРАУЗЕРЕ:" -ForegroundColor Green
Write-Host "   http://localhost:3000" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Открыть браузер
Start-Process "http://localhost:3000"
