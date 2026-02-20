# Quick Start Guide - WSL2 Dashboard

## 🚀 Быстрый запуск (из Windows)

### Шаг 1: Установка зависимостей (в WSL2)

```bash
cd ~/ai_os_final/services/dashboard_v2
npm install
```

### Шаг 2: Запуск сервера (в WSL2)

```bash
npm run dev
```

### Шаг 3: Открытие в браузере (из Windows)

**Способ 1: Прямой доступ по IP (Рекомендуется)**

```bash
# Узнать IP-адрес WSL2
npm run wsl-ip
```

Или выполните в WSL2:
```bash
bash ./get-wsl-ip.sh
```

Затем откройте в Windows браузере:
- `http://172.25.50.61:3000` (замените на ваш IP из скрипта)

---

**Способ 2: Автоматический запуск (из Windows)**

Дважды кликните на файл `start.bat` в папке dashboard_v2 в проводнике Windows.

---

**Способ 3: Настройка постоянного доступа**

1. Откройте **PowerShell от имени администратора** в Windows
2. Запустите скрипт:
   ```powershell
   cd \\wsl.localhost\Ubuntu\home\onor\ai_os_final\services\dashboard_v2
   .\setup-windows-access.ps1
   ```

После этого вы сможете открывать `http://localhost:3000` в браузере Windows.

---

## 🔧 Ручная настройка портов (если скрипты не работают)

### В PowerShell (Administrator):

```powershell
# 1. Получить IP WSL2
wsl hostname -I

# 2. Настроить проброс портов (замените <WSL_IP> на реальный IP)
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=<WSL_IP>
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>

# 3. Проверить правила
netsh interface portproxy show all
```

### Открыть порты в брандмауэре Windows (если нужно):

```powershell
netsh advfirewall firewall add rule name="WSL2 Dashboard" dir=in action=allow protocol=TCP localport=3000
netsh advfirewall firewall add rule name="WSL2 Backend" dir=in action=allow protocol=TCP localport=8000
```

---

## 📝 Полезные команды

### В WSL2:

```bash
# Запуск dashboard
npm run dev

# Получить IP адрес
npm run wsl-ip

# Сборка для продакшена
npm run build

# Предпросмотр сборки
npm run preview
```

### В Windows PowerShell:

```powershell
# Просмотр всех правил проброса портов
netsh interface portproxy show all

# Удаление правил
netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0

# Проверка подключения
Test-NetConnection -ComputerName 172.25.50.61 -Port 3000
```

---

## ❓ Частые проблемы

### 1. Браузер не может открыть страницу

**Решение:**
- Убедитесь, что `npm run dev` запущен в WSL2
- Проверьте IP-адрес: `npm run wsl-ip`
- Попробуйте другой способ доступа из списка выше

### 2. Скрипт PowerShell не работает

**Решение:**
- Убедитесь, что PowerShell запущен от имени администратора
- Выполните: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. IP-адрес WSL2 изменился после перезагрузки

**Решение:**
- Запустите `npm run wsl-ip` для получения нового IP
- Или перенастройте правила проброса портов с новым IP

### 4. Firewall блокирует соединение

**Решение:**
```powershell
# Разрешить порты
netsh advfirewall firewall add rule name="WSL2 Dashboard" dir=in action=allow protocol=TCP localport=3000
```

---

## 📚 Дополнительная информация

- `README.md` - Полная документация
- `get-wsl-ip.sh` - Скрипт для получения IP-адреса WSL2
- `setup-windows-access.ps1` - PowerShell скрипт для настройки доступа
- `start.bat` - Быстрый запуск из Windows
