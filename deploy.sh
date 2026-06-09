#!/bin/bash
##############################################################################
# AI_OS Auto-Deploy Script
# Автоматическое развертывание изменений Python кода в Docker контейнеры
##############################################################################

set -e  # Exit on error
    # Цвета для вывода
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color

    # Конфигурация
    CORE_DIR="/home/onor/ai_os_final/services/core"
    CORE_CONTAINERS=("ns_core" "ns_core_worker")
    PYTHON_FILES=("*.py")

    # Список файлов для синхронизации (новые personality файлы)
    SYNC_FILES=(
        "personality_engine.py"
        "personality_decision_integration.py"
        "personality_agent_prompts.py"
        "retroactive_artifacts.py"
        "models.py"  # <-- Added models.py to sync
        "main.py"
        "goal_decomposer.py"
    )

# =============================================================================
# Функции для логирования
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Основная логика
# =============================================================================

case "${1:-fast}" in
    fast)
        MODE="Fast"
        ;;
    *)
        MODE="Full"
        ;;
esac

# Проверка что контейнер существует
if ! docker ps | grep -q "ns_core"; then
    log_error "Container ns_core NOT running"
    echo "Please start containers first: ./startup.sh"
    exit 1
fi

log_info "Starting deployment (MODE: ${MODE})..."

# =============================================================================
# Синхронизация core директории
# =============================================================================

log_info "Syncing core directory..."

for file in "${SYNC_FILES[@]}"; do
    if [ -f "${CORE_DIR}/${file}" ]; then
        log_success "✓ ${file}"
    else
        log_warning "⚠ ${file} NOT FOUND (will skip)"
    fi
done

# =============================================================================
# Копирование Python файлов в контейнеры
# =============================================================================

for container in "${CORE_CONTAINERS[@]}"; do
    log_info "Copying files to ${container}..."

    # Используем docker cp для каждого файла
    for file in ${CORE_DIR}/*.py; do
        if [ "${MODE}" = "Fast" ]; then
            # Fast mode: копируем только изменённые файлы
            docker cp "${file}" ${container}:/app/ 2>/dev/null
        else
            # Full mode: копируем все файлы
            docker cp "${file}" ${container}:/app/
        fi
    done

    # Копируем поддиректории (ВСЕ Python-пакеты, кроме __pycache__)
    for dir_path in $(find "${CORE_DIR}" -maxdepth 1 -type d ! -name __pycache__); do
        # Извлекаем только имя директории
        dir=$(basename "${dir_path}")

        if [ "${dir}" = "__pycache__" ]; then
            continue
        fi

        if [ -d "${CORE_DIR}/${dir}" ]; then
            # Use tar with exclude to avoid problematic files
            tar -c \
                --exclude='__pycache__' \
                --exclude='*.pyc' \
                --exclude='.git' \
                --exclude='.pytest_cache' \
                --exclude='*.pyo' \
                --exclude='*.egg-info' \
                --exclude='tests/' \
                -C "${CORE_DIR}" "${dir}" | \
            docker exec -i ${container} tar -x -C /app/
            log_success "✓ ${dir}/"
        else
            log_warning "⚠ ${dir} NOT FOUND (will skip)"
        fi
    done

    log_success "Files synced to ${container}"
done

# =============================================================================
# Перезапуск контейнеров для применения изменений
# =============================================================================

if [ "${MODE}" = "Fast" ]; then
    # Fast mode: перезапускаем только core контейнеры
    log_info "Fast deploy to ${CORE_CONTAINERS[@]}..."

    for container in "${CORE_CONTAINERS[@]}"; do
        docker restart "${container}" 2>&1 | grep -q "container" >/dev/null

        if [ $? -eq 0 ]; then
            log_success "✓ ${container} restarted"
        else
            log_error "✗ ${container} restart FAILED"
        fi
    done
else
    # Full mode: перезапускаем все контейнеры
    log_info "Full deploy - restarting all services..."

    docker-compose restart 2>&1 | tail -5

    log_success "Deployment completed"
fi

log_info "MODE: ${MODE} deployment finished"
