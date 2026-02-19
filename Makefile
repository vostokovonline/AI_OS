.PHONY: help deploy deploy-fast deploy-single status logs restart test-goal

# AI_OS Makefile - Удобное управление проектом

help:
	@echo "AI_OS Makefile Commands:"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy         - Full deployment (sync files + clear cache + restart)"
	@echo "  make deploy-fast    - Fast deployment (sync files + restart, no cache clear)"
	@echo "  make deploy-core    - Deploy only to ns_core container"
	@echo "  make deploy-worker  - Deploy only to ns_core_worker container"
	@echo ""
	@echo "Status & Logs:"
	@echo "  make status         - Show container status"
	@echo "  make logs           - Show ns_core logs (tail -f)"
	@echo "  make logs-worker    - Show ns_core_worker logs (tail -f)"
	@echo ""
	@echo "Management:"
	@echo "  make restart        - Restart all containers"
	@echo "  make restart-core   - Restart ns_core only"
	@echo "  make restart-worker - Restart ns_core_worker only"
	@echo ""
	@echo "Tests:"
	@echo "  make test-goal      - Create test atomic goal"
	@echo "  make test-llm       - Test LLM fallback system"
	@echo ""
	@echo "Data Model Governance:"
	@echo "  make verify-models  - Verify models.py integrity"
	@echo "  make detect-drift   - Check models vs database drift"
	@echo "  make check          - Run all pre-deployment checks"
	@echo ""
	@echo "LLM Management:"
	@echo "  make llm-status     - Check LLM fallback status"
	@echo "  make llm-reset      - Reset Groq cooldown"

# Деплой
deploy:
	@./deploy.sh full

deploy-fast:
	@./deploy.sh fast

deploy-core:
	@./deploy.sh single ns_core

deploy-worker:
	@./deploy.sh single ns_core_worker

# Статус
status:
	@./deploy.sh status

logs:
	@./deploy.sh logs ns_core

logs-worker:
	@./deploy.sh logs ns_core_worker

# Перезапуск
restart:
	@docker restart ns_core ns_core_worker
	@echo "✅ All containers restarted"

restart-core:
	@docker restart ns_core
	@echo "✅ ns_core restarted"

restart-worker:
	@docker restart ns_core_worker
	@echo "✅ ns_core_worker restarted"

# LLM Management
llm-status:
	@docker exec ns_core curl -s http://localhost:8000/llm/status | python3 -m json.tool

llm-reset:
	@docker exec ns_core curl -s -X POST http://localhost:8000/llm/reset_groq | python3 -m json.tool

test-llm:
	@echo "Testing LLM with fallback..."
	@docker exec ns_core curl -s -X POST http://localhost:8000/llm/test \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Say Hello World!"}' | python3 -m json.tool

# Тесты
test-goal:
	@echo "Creating test atomic goal..."
	@docker exec ns_core curl -s -X POST http://localhost:8000/goals/create \
		-H "Content-Type: application/json" \
		-d '{"title": "Test Goal from Makefile", "description": "Testing deployment script", "goal_type": "achievable", "is_atomic": true, "depth_level": 3}' \
		| python3 -m json.tool

# База данных
db-shell:
	@docker exec -it ns_postgres psql -U ns_admin -d ns_core_db

db-backup:
	@echo "Creating database backup..."
	@docker exec ns_postgres pg_dump -U ns_admin ns_core_db > backups/ns_core_db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup saved to backups/"

# Redis
redis-cli:
	@docker exec -it ns_redis redis-cli

# Model & Database Validation
verify-models:
	@echo "Verifying model integrity..."
	@cd services/core && python3 verify_models.py

detect-drift:
	@echo "Detecting database drift..."
	@cd services/core && python3 detect_drift.py

detect-drift-fix:
	@echo "Detecting database drift with fix suggestions..."
	@cd services/core && python3 detect_drift.py --fix

# Pre-deployment validation
check: verify-models
	@echo "✅ Pre-deployment checks passed"

# Очистка
clean:
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned Python cache"

# Сборка
build:
	@echo "Building Docker images..."
	@docker-compose build --no-cache core
	@echo "✅ Build complete"

# Полный перезапуск
rebuild: clean build deploy
	@echo "✅ Full rebuild complete!"
