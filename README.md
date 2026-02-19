# AI-OS - Goal Execution System v3.0

**Autonomous goal-execution system powered by AI agents**

AI-OS is a sophisticated multi-level goal management system that decomposes high-level goals into atomic subgoals, executes them through specialized AI agents, and produces verifiable artifacts.

## 🏗️ Architecture

### Core Services

- **ns_core** (port 8000) - FastAPI service: goal management, execution API
- **ns_core_worker** - Celery worker for async task processing
- **ns_postgres** (5432) - PostgreSQL database with audit trail
- **ns_redis** (6379) - Redis for Celery + caching
- **ns_litellm** (4000) - LLM proxy/router (Groq, Ollama, OpenAI, etc.)
- **ns_memory** (8001) - Memory service with Neo4j + Milvus (vector DB)
- **ns_dashboard** (8501) - Streamlit dashboard v1
- **dashboard_v2** (3000) - React dashboard v2 (operational thinking interface)
- **Temporal.io** (8088 UI, 7233 server) - Workflow orchestration for continuous goals

### Goal System v3.0

```
Mission (L0) → Strategic (L1) → Operational (L2) → Tactical/Atomic (L3)
```

**Key Features:**
- **Unit of Work Pattern** - Transaction management with atomic operations
- **Goal State Transitions** - All state changes through `transition_goal()` service
- **Audit Trail** - Every transition logged to `goal_status_transitions` table
- **Hard Invariants** - Business rules enforced at domain layer
- **Goal Types**: achievable, continuous, directional, exploratory, meta
- **Goal Contracts** - Formalized constraints on LLM behavior
- **Artifact Layer** - Verifiable results from atomic goals
- **Skill Manifests** - Explicit contracts for all skills

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- API Keys for LLM providers (Groq, OpenAI, etc.)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-os.git
cd ai-os

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env

# Start all services
docker-compose up -d

# Check status
make status
```

### First Goal

```bash
# Create a simple atomic goal
curl -X POST http://localhost:8000/goals/create \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test goal",
    "description": "Testing AI-OS",
    "goal_type": "achievable",
    "is_atomic": true
  }'

# Execute goal
curl -X POST http://localhost:8000/goals/{goal_id}/execute

# Check status
curl http://localhost:8000/goals/{goal_id}
```

## 📊 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

```
POST /goals/create                    # Create new goal
POST /goals/{goal_id}/execute         # Execute atomic goal
POST /goals/{goal_id}/decompose       # Decompose into subgoals
POST /goals/{goal_id}/strict_evaluate # Strict evaluation
POST /goals/{goal_id}/reflect         # Generate next goals
GET  /goals/{goal_id}/tree            # Goal hierarchy
GET  /goals/list                      # List all goals
GET  /goals/stats                     # Statistics
```

## 🎯 Goal System Architecture

### Goal Lifecycle

```
create → classify → decompose → execute → strict_evaluate → reflect → extract_patterns
```

### State Transitions (via UnitOfWork)

All state transitions go through `transition_goal()`:

```python
from goal_transition_service import transition_goal
from infrastructure.uow import UnitOfWork

async with uow_factory() as uow:
    result = await transition_goal(
        uow=uow,
        goal_id=goal.id,
        new_state="done",
        reason="Goal completed successfully",
        actor="goal_executor"
    )
```

### Audit Trail

Every transition is logged to `goal_status_transitions` table:

```sql
SELECT * FROM goal_status_transitions
WHERE goal_id = 'your-goal-id'
ORDER BY created_at DESC;
```

### Hard Invariants

The system enforces business rules at the domain layer:

1. **No direct status assignments** - All changes through `transition_goal()`
2. **Terminal state protection** - Cannot exit from done/frozen/permanent
3. **Goal type constraints** - Continuous/directional goals cannot be "done"
4. **Artifact requirement** - Atomic goals require passed artifacts
5. **Parent-child consistency** - Cannot complete parent with active children

## 🤖 Agent Graph (LangGraph)

Multi-agent system with role-based model selection:

- **SUPERVISOR** (qwen3-coder) - Routes tasks
- **CODER** (qwen3-coder) - Code generation
- **PM** (gpt-oss) - Goal management
- **RESEARCHER** (qwen3-coder) - Information gathering
- **INTELLIGENCE** (deepseek-v3.1) - Complex reasoning

## 🔄 Continuous Goals (Temporal.io)

Long-running goals managed by Temporal.io workflows:

```bash
# Start continuous goal
curl -X POST http://localhost:8000/goals/continuous/start \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Weekly system optimization",
    "description": "Optimize every Monday",
    "cron_schedule": "0 9 * * 1"
  }'

# Temporal Web UI
open http://localhost:8088
```

## 🛠️ Development

### Commands

```bash
make deploy-fast      # Fast deploy (~5s)
make deploy           # Full deploy with cache clear
make status           # Container status
make logs             # ns_core logs
make logs-worker      # Worker logs
make db-shell         # PostgreSQL shell
make test-goal        # Create test goal
```

### File Structure

```
services/core/
├── domain/              # Domain layer (business logic)
│   └── goal_domain_service.py
├── infrastructure/      # Infrastructure layer (UoW, repositories)
│   └── uow.py
├── agents/             # LangGraph agents
├── canonical_skills/   # Built-in skills
├── goal_executor.py    # Goal execution (v1)
├── goal_executor_v2.py # Goal execution with UoW
├── goal_decomposer.py  # Goal decomposition
├── goal_transition_service.py  # State transitions
├── audit_logger_v2.py  # Audit trail
└── main.py             # FastAPI app
```

### Adding New Skills

1. Create skill in `services/core/canonical_skills/`
2. Implement `SkillResult` return type
3. Register in `mvp_skills.py` or `production_skills.py`
4. Deploy with `make deploy-fast`

### Testing

```bash
# System tests
./services/core/test_goals.sh

# Browser skills
python3 services/core/skills/browser/test_basics.py
```

## 📈 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/docs

# Container status
docker ps

# Database stats
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c "SELECT COUNT(*) FROM goals;"
```

### Audit Trail

```sql
-- Recent transitions
SELECT created_at, to_status, triggered_by, reason
FROM goal_status_transitions
ORDER BY created_at DESC
LIMIT 20;

-- Goal statistics
SELECT status, COUNT(*) FROM goals GROUP BY status;
```

## 🔐 Security

**IMPORTANT:**
- Never commit `.env` file (contains API keys)
- Use `.env.example` as template
- Rotate credentials regularly
- Review audit logs for suspicious activity

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Support

For issues and questions:
- GitHub Issues: https://github.com/YOUR_USERNAME/ai-os/issues
- Documentation: http://localhost:8000/docs

---

**Built with ❤️ using FastAPI, LangGraph, Temporal.io, and Docker**
