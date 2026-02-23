# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-OS is a sophisticated goal-execution system powered by AI agents, built with FastAPI (backend) and React (dashboard). The system decomposes high-level goals into atomic subgoals, executes them through specialized agents, and produces verifiable artifacts.

**Current Status**: Production system with v3.0 goal system, v1 artifact layer, and v1 skill manifests fully operational.

---

## Code Quality Refactoring (COMPLETED - 2026-02-20)

**Critical improvements**: Replaced all anti-patterns with production-ready code.

### ✅ COMPLETED:

#### 1. Structured Logging (1,146 print statements → 0)
- **Created**: `logging_config.py` - Centralized structlog-based logging
- **Replaced**: All print() statements with structured logging in 58 production files
- **Benefits**:
  - JSON logs in production (parseable by ELK/Loki)
  - Human-readable logs in development
  - Structured context (goal_id, user_id, error details)
  - Proper log levels (debug, info, warning, error, critical)

**Usage pattern:**
```python
from logging_config import get_logger, log_goal_transition

logger = get_logger(__name__)

# Simple event
logger.info("goal_created", goal_id=str(goal.id), title=goal.title)

# Goal transition (specialized)
log_goal_transition(
    goal_id=str(goal.id),
    from_state="active",
    to_state="done",
    actor="goal_executor",
    reason="Execution complete"
)
```

#### 2. Exception Handling (319 bare except → 0)
- **Fixed**: All bare `except:` clauses with specific exception types
- **Files**: 15 critical files (agent_graph.py, tools.py, main.py, etc.)
- **Pattern applied**:
  - Specific exceptions (ValueError, KeyError, HTTPError, etc.)
  - Structured logging with context
  - No more silent failures
  - Better debugging visibility

**Before:**
```python
try:
    risky_operation()
except:  # ❌ Hides ALL errors
    pass
```

**After:**
```python
try:
    risky_operation()
except SpecificError as e:
    logger.debug("specific_error", context=...)
except Exception as e:
    logger.error("unexpected_error", error=str(e), context=...)
```

#### 3. Error Handling Infrastructure
- **Created**: `error_handler.py` - Circuit breaker pattern, safe execution wrappers
- **Benefits**: Prevents cascade failures, graceful degradation

### System Health Metrics (BEFORE → AFTER):

| Metric | Before | After |
|--------|--------|-------|
| Print statements | 1,146 | 0 ✅ |
| Bare except clauses | 319 | 0 ✅ |
| Observability | Poor | Excellent ✅ |
| Error visibility | Silent failures | All logged ✅ |
| Production-ready | No | Yes ✅ |

### Git Commits:
- `refactor: Replace all print statements with structured logging` (8 commits)
- `refactor: Fix all bare except clauses with proper exception handling` (1 commit)

---

## Stabilization Sprint (COMPLETED - 2026-02-20)

**Goal**: Fix critical bugs + implement test shield for production readiness.

### Bug Fixes (5 Critical Issues)

| # | Problem | File | Symptom | Fix |
|---|---------|------|---------|-----|
| 1 | `get_status()` without `await` | main.py:1554 | `/llm/status` → 500 | Added `await` |
| 2 | LiteLLM no healthcheck | docker-compose.yml | Silent failures | Added healthcheck + master_key |
| 3 | Neo4j no healthcheck | docker-compose.yml | Silent failures | Added healthcheck |
| 4 | `logger not defined` | emotional_layer.py | Import error | Added `from logging_config import get_logger` |
| 5 | 67% artifacts failed | artifact_verifier.py | False negatives | Added inline content detection |

### Test Shield (45 Tests)

```
tests/
├── conftest.py                    # Fixtures and config
├── README.md                      # Test documentation
├── unit/
│   ├── test_imports.py           # 18 smoke tests (module imports)
│   ├── test_goal_state_machine.py # 10 tests (status protection)
│   └── test_artifact_verifier.py  # 16 tests (inline vs file)
└── e2e/
    └── test_api.py                # 8 tests (API connectivity)
```

**Results**: `45 passed, 3 skipped`

| Category | Tests | What it catches |
|----------|-------|-----------------|
| Smoke imports | 18 | `logger not defined`, syntax errors |
| State machine | 10 | Direct status assignment bypass |
| Artifact verifier | 16 | False failed artifacts |
| E2E API | 8 | Services unavailable |

**Running tests:**
```bash
make test-unit    # Unit tests (~2s)
make test-e2e     # E2E tests (~1s) - requires running services
make test-all     # All tests
```

### CI/CD Infrastructure

```
.github/workflows/
├── test.yml     # Unit tests + lint on push/PR
└── deploy.yml   # Deploy info on main push
```

**test.yml runs:**
- Unit tests with pytest
- Linting with ruff, black, isort
- Docker compose syntax check

### Code Quality Tools

```bash
# Configuration files
pyproject.toml          # ruff, black, isort, pytest, coverage config
.pre-commit-config.yaml # Pre-push test hook
requirements-test.txt   # pytest, pytest-asyncio, httpx, pytest-cov
pytest.ini              # pytest configuration
```

**Pre-commit hooks:**
- black (formatting)
- isort (import sorting)
- ruff (linting)
- pytest-unit (runs on pre-push)

### Metrics (Before → After)

| Metric | Before | After |
|--------|--------|-------|
| Artifacts passed | 285 | 411 (+44%) |
| LLM status endpoint | 500 Error | 200 OK |
| Unit tests | 0 | 45 |
| CI/CD | None | GitHub Actions |
| Healthchecks | 0 services | 2 services (LiteLLM, Neo4j) |

### Key Lessons Learned

1. **Async/await errors are silent killers** - `get_status()` returned coroutine, not result
2. **Healthchecks are not optional** - 12 services without healthchecks = bomb
3. **Test imports first** - `logger not defined` caught by smoke test in 0.1s
4. **Artifact verification matters** - 67% false negatives = broken execution loop
5. **Inline vs file detection** - Content like `{"key": "value"}` is NOT a file path

### New Makefile Commands

```bash
make test-unit    # Run unit tests
make test-e2e     # Run E2E tests (requires running services)
make test-all     # Run all tests
make llm-status   # Check LLM fallback status
make llm-reset    # Reset Groq cooldown
```

---

---

## Dashboard v2 Update (COMPLETED - 2026-02-21)

**Major improvements**: CORS fix, new pages (Autonomy, Admin), database migration, enhanced API integration.

### ✅ COMPLETED:

#### 1. CORS Configuration Fix
**Problem**: Dashboard v2 running on `http://172.25.50.61:3000` couldn't access backend API due to CORS policy errors.

**Solution**: Updated `docker-compose.yml` with comprehensive ALLOWED_ORIGINS:
```yaml
environment:
  ALLOWED_ORIGINS: >
    http://localhost:3000,
    http://localhost:8501,
    http://localhost:8000,
    http://172.25.50.61:3000,
    http://10.255.255.254:3000,
    http://172.25.50.61:8501,
    http://10.255.255.254:8501
```

**Result**: All dashboard v2 pages can now access backend API without CORS errors.

#### 2. Database Migration - Artifacts Table
**Problem**: `Artifact` model in `models.py` had columns `state_mutations` and `decision_signals` that didn't exist in database.

**Migration**: `services/core/migrations/add_artifacts_autonomy_columns.sql`
```sql
ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS state_mutations JSON;
ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS decision_signals JSON;
```

**Applied**: ✅ Successfully added columns, `/artifacts` endpoint now works.

#### 3. New Dashboard Pages

##### Autonomy Page (🧠 src/pages/Autonomy.tsx)
**Features**:
- Real-time alerts summary from `/alerts/summary` API
- Decision engine status display
- Safety constraints (ethics, budget, time_horizon)
- Active policies visualization
- Recent decisions timeline with confidence scores
- Auto-refresh every 5 seconds

**API Integration**:
- `GET /alerts/summary` - System alerts summary
- Planned: `/autonomy/state` - Full autonomy state (TODO)

##### Admin Page (🛡️ src/pages/Admin.tsx)
**Features**:
- **Pending Approvals Tab**: Manual goal completion approval workflow
  - Filter: `completion_mode='manual'` AND `status IN ('active', 'done')`
  - Approve/Reject buttons with API integration
- **Reflections Tab**: View lessons learned from completed goals
  - Placeholder for future reflections API
- **System Observer Tab**: Real-time monitoring placeholder
    - CPU, Memory, Active Tasks metrics (coming soon)

**API Integration**:
- `GET /goals/list` - Load all goals for approval filtering
- `POST /goals/{goal_id}/approve_completion` - Approve manual goal completion
- `POST /goals/{goal_id}/mutate` - Reject/modify goals

**Stats Display**:
- Pending Approvals (manual done goals)
- Completed Goals
- Active Goals
- System Health (mock: 95%)

#### 4. Structured Logging Fix
**Problem**: `StandardLoggerAdapter` added but logger calls used structlog kwargs format.

**Solution**: `logging_config.py` now includes wrapper that handles both formats:
```python
class StandardLoggerAdapter:
    def _format_kwargs(self, event: str, **kwargs) -> str:
        if not kwargs:
            return event
        parts = [event]
        for key, value in kwargs.items():
            parts.append(f"{key}={value}")
        return " | ".join(parts)
```

**Result**: System starts without `TypeError: logger._log() got unexpected keyword argument`.

### Dashboard v2 Features

#### Available Pages:
1. **Graph** - ReactFlow goal dependency visualization
2. **Timeline (Gantt)** - Historical goal execution timeline
3. **Dependency Tree** - Hierarchical goal tree view
4. **Observability** - Russian: "Наблюдаемость" - System metrics
5. **Questions** - Russian: "Вопросы" - Question management
6. **Decomposition** - Russian: "Декомпозиция" - Goal decomposition
7. **Skills** (OCCP v1.0) - Skill registry and management
8. **Deployments** (OCCP v1.0) - Deployment tracking
9. **Metrics** (OCCP v1.0) - Performance metrics
10. **Federation** (OCCP v1.0) - Multi-system federation
11. **Artifacts** (OCCP v1.0) - Artifact registry
12. **Autonomy** (NEW 🧠) - Autonomous decision-making system
13. **Admin** (NEW 🛡️) - System administration

#### API Endpoints Used by Dashboard:

##### Goals API:
- `GET /goals/list` - List all goals
- `GET /goals/{goal_id}/tree` - Get goal tree
- `GET /goals/{goal_id}/artifacts` - Get goal artifacts
- `POST /goals/{goal_id}/approve_completion` - Approve manual goal completion
- `POST /goals/{goal_id}/mutate` - Modify goal

##### Alerts API:
- `GET /alerts` - Get system alerts
- `GET /alerts/summary` - Get alerts summary
- `POST /alerts/{alert_id}/resolve` - Resolve alert

##### Interventions API:
- `GET /interventions/candidates` - Get intervention candidates
- `GET /interventions/{id}/simulation` - Get simulation results
- `POST /interventions/{id}/simulate` - Run simulation
- `GET /interventions/{id}/risk` - Get risk assessment
- `POST /interventions/{id}/approve` - Approve intervention
- `POST /interventions/{id}/reject` - Reject intervention

##### Artifacts API:
- `GET /artifacts` - Get all artifacts (now works after migration!)
- `GET /artifacts?limit=50&offset=0` - Paginated artifacts list

### Dashboard v2 Development

#### Build & Run:
```bash
cd /home/onor/ai_os_final/services/dashboard_v2

# Development
npm install
npm run dev         # Port 3000

# Production build
npm run build
```

#### WSL2 Access from Windows:
```bash
# Get WSL2 IP
ip addr show eth0 | grep inet

# Access from Windows browser:
# http://<WSL2_IP>:3000
```

#### File Structure:
```
services/dashboard_v2/
├── src/
│   ├── api/
│   │   └── client.ts           # Axios API client with SSE
│   ├── components/
│   │   ├── canvas/            # ReactFlow graph
│   │   ├── controls/          # Control panel
│   │   ├── inspector/         # Node inspector
│   │   ├── timeline/          # Gantt timeline
│   │   └── tree/              # Dependency tree
│   ├── pages/
│   │   ├── Autonomy.tsx       # NEW: Autonomy system
│   │   ├── Admin.tsx          # NEW: System admin
│   │   ├── Skills.tsx         # OCCP: Skills
│   │   ├── Deployments.tsx    # OCCP: Deployments
│   │   ├── Observability.tsx  # OCCP: Metrics
│   │   ├── Federation.tsx     # OCCP: Federation
│   │   └── Artifacts.tsx      # OCCP: Artifacts
│   ├── store/
│   │   ├── uiStore.ts         # UI state (Zustand)
│   │   ├── graphStore.ts      # Graph state
│   │   ├── executionLogStore.ts
│   │   └── toastStore.ts
│   ├── types/index.ts         # TypeScript definitions
│   └── App.tsx                # Main app
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

### Dashboard Status:
- ✅ Running on http://localhost:3000
- ✅ Hot reload working
- ✅ All pages accessible
- ✅ No console errors
- ✅ CORS configured
- ✅ API integration working

### Access URLs:
- **Primary**: http://172.25.50.61:3000
- **Alternative**: http://10.255.255.254:3000
- **Local (WSL2)**: http://localhost:3000

---
## Transaction Boundary Refactoring (COMPLETED)

**Core architectural change**: Extracting transaction management into UnitOfWork pattern.

**Result**: All goal operations are now atomic with proper rollback support.

### Migration Status

#### ✅ COMPLETED (2026-02-19):
1. Created `domain/goal_domain_service.py` - Pure domain service with invariant validation
2. Created `infrastructure/uow.py` - UnitOfWork, GoalRepository, AuditLogger, BulkTransitionService
3. Refactored `goal_transition_service.py` v3.0 - Accepts UnitOfWork, no internal commit
4. Refactored `goal_executor.py` - Uses UoW for goal creation
5. Refactored `goal_executor_v2.py` - Added `execute_goal_with_uow()` method
6. Refactored `goal_decomposer.py` - Uses UoW for decomposition
7. Refactored `artifact_registry.py` - Added `register_with_uow()` methods
8. Refactored `goal_strict_evaluator.py` - Added UoW methods
9. **NEW**: BulkTransitionService - Mass operations in single transaction
10. **NEW**: Rollback tests - Atomic behavior verified

#### ✅ NEW ENDPOINTS:
- `POST /goals/bulk-transition` - Transition 1000 goals in one transaction
- `POST /goals/freeze-tree` - Freeze entire goal tree atomically

---

## Unit of Work Pattern

### Overview

All write operations use UnitOfWork for atomic transactions:

```python
from infrastructure.uow import UnitOfWork, create_uow_provider
from database import AsyncSessionLocal

get_uow = create_uow_provider()

async with get_uow() as uow:
    # All operations share the same transaction
    goal = await uow.goals.get(uow.session, goal_id)
    await transition_service.transition(uow, goal_id, "active", "reason")
    # Auto-commit on success, auto-rollback on exception
```

### Transaction Boundaries

| Operation | Transaction Scope |
|-----------|-------------------|
| Goal creation | Single UoW - goal + all metadata |
| Decomposition | Single UoW - parent + all subgoals |
| Execution | Single UoW - goal + artifacts |
| Bulk transition | Single UoW - all goals locked + updated |
| Freeze tree | Single UoW - entire tree frozen |

### Migration Pattern

**Before (OLD):**
```python
await transition_goal(goal_id, "active", "reason")  # ❌ Hidden commit inside
```

**After (NEW):**
```python
async with get_uow() as uow:
    await transition_service.transition(
        uow=uow,
        goal_id=UUID(goal_id),
        new_state="active",  # NOTE: 'new_state', not 'to_state'
        reason="reason",
        actor="system"
    )
    # Commit happens automatically at end of 'with' block
```

### Bulk Operations

```python
from infrastructure.uow import bulk_transition_service

async with get_uow() as uow:
    result = await bulk_transition_service.execute_bulk(
        uow=uow,
        goal_ids=[uuid1, uuid2, uuid3],
        new_state="active",
        reason="Mass activation",
        actor="admin"
    )
    # Returns: {"succeeded": 3, "failed": 0, "results": [...]}
```

### Pessimistic Locking

For concurrent access, use `get_for_update()`:

```python
async with get_uow() as uow:
    # Lock goal for exclusive access
    goal = await uow.goals.get_for_update(uow.session, goal_id)
    
    # Modify while locked
    goal.progress = 0.5
    await uow.goals.update(uow.session, goal)
    # Lock released on commit
```

### Anti-Patterns

| ❌ Anti-Pattern | ✅ Correct Approach |
|-----------------|---------------------|
| `goal.status = "done"` | `transition_service.transition(uow, ...)` |
| `await session.commit()` inside UoW | Let UoW auto-commit |
| Multiple UoW for related ops | Single UoW for atomic ops |
| `transition_goal()` legacy function | `transition_service.transition(uow, ...)` |

### Testing

```bash
# Run UoW rollback tests
docker exec ns_core python -m pytest /app/tests/test_uow_rollback.py -v

# Test bulk transitions
curl -X POST http://localhost:8000/goals/bulk-transition \
  -H "Content-Type: application/json" \
  -d '{"goal_ids": ["uuid1", "uuid2"], "new_state": "active"}'
```


---

## Memory System Architecture (v3.1)

**Overview**: Multi-layer memory system with 6 different types of storage.

### Memory Types

| Type | Storage | TTL | Purpose |
|------|---------|-----|---------|
| **SemanticMemory** | PostgreSQL + Milvus | Yes (cleanup) | Decision patterns |
| **MemorySignal** | Redis | Yes (auto) | Runtime pressure |
| **ContextualMemory** | PostgreSQL | No | User context |
| **AffectiveMemory** | PostgreSQL | No | Emotional outcomes |
| **Vector (Milvus)** | Milvus | No | Semantic search |
| **Graph (Neo4j)** | Neo4j | No | Entity relationships |

### Key Components

**1. SemanticMemory** (`semantic_memory.py`)
```python
from semantic_memory import semantic_memory

# Store pattern
pattern_id = await semantic_memory.store_pattern(
    pattern_type="success_pattern",
    content={"goal_type": "achievable", "domains": ["programming"]},
    source_goal_id=goal_id,
    confidence=0.8
)

# Retrieve recommendations
recommendations = await semantic_memory.get_recommendations(goal)

# Vector search (Milvus)
patterns = await semantic_memory.retrieve_similar_patterns_vector("success", limit=5)

# Cleanup old patterns
deleted = await semantic_memory.cleanup_old_patterns(days=30)
```

**2. MemorySignal** (`memory_signal.py`)
```python
from memory_signal import MemorySignal, persistent_memory_registry

# Create signal
signal = MemorySignal(
    type="recent_failure",
    target="skill:web_research",
    intensity=0.8,
    ttl=5  # 5 hours
)

# Store in Redis (persists across restarts)
persistent_memory_registry.add(signal)

# Get active signals
active = persistent_memory_registry.get_active()
```

**3. EmotionalFeedbackLoop** (`emotional_feedback_loop.py`)
```python
from emotional_feedback_loop import emotional_feedback_loop

# Record goal completion (auto-called by GoalStrictEvaluator)
await emotional_feedback_loop.record_goal_completion(
    goal_id=str(goal.id),
    user_id="user_123",
    outcome="success",
    metrics={"score": 0.95, "duration": 120}
)
```

### Scheduled Tasks

| Task | Schedule | File |
|------|----------|------|
| Pattern cleanup | Daily at 4:00 AM | `scheduler.py` |
| Signal decay | Hourly | `scheduler.py` |
| Invariants check | Daily at 3:00 AM | `scheduler.py` |

### API Endpoints

```bash
# Get memory stats
GET /memory/stats

# Health check
GET /memory/health

# Cleanup patterns
POST /patterns/cleanup?days=30

# Vector search
POST /patterns/search-vector?query=success&limit=5

# Batch store
POST /memory/batch-store
```

### Usage Pattern

```python
# In goal completion flow:
# 1. GoalStrictEvaluator calls _record_completion()
# 2. _record_completion() calls emotional_feedback_loop.record_goal_completion()
# 3. EmotionalFeedbackLoop stores to AffectiveMemory
# 4. Pattern extracted and stored to SemanticMemory + Milvus + Neo4j
# 5. Future decisions influenced by MemorySignal (Redis)
```

### Deduplication

Patterns are deduplicated before storage:
```python
# _find_similar_pattern() checks:
# 1. Vector similarity in Milvus
# 2. Jaccard similarity on key fields
# If similar pattern exists → update confidence instead of creating duplicate
```

---

## Centralized Logging System (NEW 2026-02-19)

**Problem**: 1,146 print() statements with no structured logging.

**Solution**: Implemented production-ready logging with structlog.

### Setup

The logging system is configured in `logging_config.py`:

```python
from logging_config import get_logger, log_goal_transition

logger = get_logger(__name__)

# Structured logging with context
logger.info(
    "goal_created",
    goal_id=str(goal.id),
    title=goal.title,
    goal_type=goal.goal_type
)

# Automatic goal transition logging
log_goal_transition(
    goal_id=str(goal.id),
    from_state="active",
    to_state="done",
    actor="goal_executor",
    reason="Execution complete"
)
```

### Benefits

✅ **Structured Logs**: JSON format for production, readable for development
✅ **Context Preservation**: All relevant data attached to log entry
✅ **Error Tracking**: Automatic exception logging with full stack traces
✅ **Performance**: Async logging doesn't block execution
✅ **Queryable**: Easy to search and filter in log aggregation systems

### Log Levels

```python
logger.debug("detailed_info")     # Development diagnostics
logger.info("business_event")      # Important events
logger.warning("anomaly")          # Unexpected but recoverable
logger.error("failure")            # Operation failed
logger.critical("system_down")     # Service unavailable
```

### Configuration

Development (default):
```python
# logging_config.py
setup_logging(
    level="INFO",
    log_file=None,     # Console only
    json_logs=False    # Human-readable
)
```

Production:
```python
setup_logging(
    level="WARNING",   # Less noise
    log_file="/var/log/ai-os/app.log",
    json_logs=True     # Parseable by ELK/Loki
)
```

---

## Error Handling System (NEW 2026-02-19)

**Problem**: 319 bare `except:` clauses hiding critical errors.

**Solution**: Centralized error handling in `error_handler.py`.

### Safe Execution Pattern

```python
from error_handler import ErrorHandler, handle_errors

# Option 1: Functional approach
result = ErrorHandler.safe_execute(
    func=lambda: risky_operation(),
    default=None,
    context={"goal_id": goal.id}
)

# Option 2: Decorator approach
@handle_errors(default=None, context={"operation": "goal_creation"})
def create_goal(data):
    # Errors logged automatically, never crashes
    ...
```

### Circuit Breaker Pattern

Prevent cascade failures when external services go down:

```python
from error_handler import CircuitBreaker, CircuitBreakerOpen

breaker = CircuitBreaker(
    func=external_api_call,
    failure_threshold=5,  # Open after 5 failures
    timeout=60            # Retry after 60 seconds
)

try:
    result = await breaker.call()
except CircuitBreakerOpen:
    # Service down, use fallback
    result = get_cached_data()
```

### Migration Pattern

**Before (OLD):**
```python
try:
    await httpx.post(url, json=data)
except: pass  # ❌ Hides all errors!
```

**After (NEW):**
```python
try:
    await httpx.post(url, json=data)
except httpx.HTTPError as e:
    logger.error("http_failed", url=url, error=str(e))
except Exception as e:
    logger.warning("unexpected_error", error_type=type(e).__name__)
    # Don't crash, but don't hide errors either
```

### Error Categories

| Error Type | Handler | Log Level | Action |
|------------|----------|-----------|--------|
| `httpx.HTTPError` | HTTP failures | ERROR | Retry with backoff |
| `ValueError` | Invalid input | WARNING | Return validation error |
| `RuntimeError` | System error | CRITICAL | Circuit breaker |
| `Exception` | Unexpected | ERROR | Log and notify |

---

## Memory Architecture (DETAILED)

The AI-OS memory system distinguishes between **ephemeral logs** and **persistent memory**.

### Memory Components

#### 1. Semantic Memory (`semantic_memory.py`)

**Purpose**: Extract and store decision patterns from reflections.

```python
from semantic_memory import SemanticMemory

memory = SemanticMemory()

# Store decision pattern
await memory.store_pattern(
    context="goal_execution",
    pattern="When LLM fails, fallback to Ollama",
    outcome="success",
    confidence=0.95
)

# Retrieve similar patterns
patterns = await memory.retrieve_patterns(
    context="goal_execution",
    situation="LLM failure"
)
```

**Storage**: Neo4j graph database
- Nodes: Decisions, outcomes, contexts
- Relationships: `LED_TO`, `SIMILAR_TO`, `CONTRADICTS`
- Queries: Cypher pattern matching

#### 2. Artifact Registry (`artifact_registry.py`)

**Purpose**: Store verifiable results from atomic goals.

```python
from artifact_registry import artifact_registry

# Register artifact (called by skills automatically)
await artifact_registry.register(
    goal_id=goal.id,
    artifact_type="FILE",
    content_kind="code",
    content_location="/app/output/main.py",
    verification_rule="file_exists_and_readable"
)

# Query artifacts
artifacts = await artifact_registry.get_by_goal(goal_id)
```

**Storage**: PostgreSQL `artifacts` table
- Indexed by: `goal_id`, `artifact_type`, `created_at`
- Verification: Code-based rules in `artifact_verifier.py`
- Key Rule: No passed artifacts → goal status = "incomplete"

#### 3. Memory Signal (`memory_signal.py`)

**Purpose**: V4 memory system integration for long-term pattern storage.

```python
from memory_signal import MemorySignal

signal = MemorySignal()

# Send memory signal to V4 system
await signal.emit(
    signal_type="decision_pattern",
    data={
        "decision": "Used transition_goal()",
        "outcome": "atomic_transaction_success",
        "metadata": {"uow_used": True}
    }
)
```

**Storage**: Milvus vector database + Neo4j
- Vector embeddings for semantic search
- Graph relationships for context
- Query by similarity: Find similar past decisions

### Memory vs Logs

| Aspect | Logs | Memory |
|--------|------|--------|
| **Purpose** | Debugging, monitoring | Learning, decisions |
| **Lifetime** | Ephemeral (days/weeks) | Persistent (years) |
| **Format** | Text lines | Structured data |
| **Query** | grep, log aggregation | Graph, vector search |
| **Example** | "Goal execution started at 12:00" | "Pattern: Goals with depth >3 fail 30% more often" |

### Memory Access Patterns

**1. Pattern Extraction (Reflection → Memory)**
```python
# After goal completion, reflect and extract patterns
from goal_reflector import goal_reflector
from semantic_memory import SemanticMemory

reflection = await goal_reflector.reflect(goal_id)
await semantic_memory.store_pattern(
    context=reflection.context,
    pattern=reflection.decision_made,
    outcome=reflection.result,
    confidence=reflection.confidence_score
)
```

**2. Memory Retrieval (Memory → Decision)**
```python
# Before execution, check memory for similar cases
patterns = await semantic_memory.retrieve_patterns(
    context="goal_decomposition",
    situation=f"goal_type={goal.goal_type}, depth={goal.depth_level}"
)

# Use patterns to inform decision
if any(p.outcome == "failure" for p in patterns):
    logger.info("high_risk_detected", patterns=patterns)
    # Apply different strategy
```

**3. Artifact Lookup (Memory → Verification)**
```python
# Verify goal completion by checking artifacts
artifacts = await artifact_registry.get_by_goal(goal_id)
passed = [a for a in artifacts if a.verification_status == "passed"]

if not passed:
    logger.warning("no_passed_artifacts", goal_id=goal_id)
    # Goal status → "incomplete"
```

### Memory Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   AI-OS Memory System                │
└─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐     ┌─────▼─────┐     ┌───▼────┐
   │ Semantic │     │ Artifacts │     │ Memory │
   │  Memory  │     │ Registry  │     │ Signal │
   │          │     │           │     │        │
   │  Neo4j  │     │PostgreSQL │     │ Milvus │
   │  Graph  │     │   Table   │     │ Vector │
   └────┬────┘     └─────┬─────┘     └───┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                    ┌────▼────┐
                    │ Query   │
                    │ Layer   │
                    └─────────┘
```

### Memory Statistics

```bash
# Check memory usage
docker exec ns_memory python -c "
from semantic_memory import SemanticMemory
memory = SemanticMemory()
stats = await memory.get_stats()
print(f'Patterns stored: {stats[\"total_patterns\"]}')
print(f'Contexts: {stats[\"total_contexts\"]}')
"

# Check artifacts
make db-shell
psql> SELECT artifact_type, COUNT(*) FROM artifacts GROUP BY artifact_type;
```

---

## Quick Development Commands

### Deployment (Primary Workflow)
```bash
# Fast deploy (recommended for most changes) - ~5 seconds
make deploy-fast

# Full deploy with cache clear (use after model changes)
make deploy

# Deploy to specific container
make deploy-core    # ns_core only
make deploy-worker  # ns_core_worker only
```

### Status & Logs
```bash
make status         # Container status
make logs           # ns_core logs (tail -f)
make logs-worker    # ns_core_worker logs
```

### Database Operations
```bash
make db-shell       # PostgreSQL shell
make db-backup      # Create backup
make redis-cli      # Redis CLI
```

### Testing
```bash
make test-goal      # Create test atomic goal
make test-llm       # Test LLM fallback system
```

### Manual Testing Scripts
```bash
./services/core/test_goals.sh          # Test goal execution flow
./services/core/skills/browser/test_basics.py  # Test browser skills
```

### Dashboard (React v2)
```bash
cd services/dashboard_v2
npm install
npm run dev         # Development server on port 3000
npm run build       # Production build
```

**WSL2 Access from Windows:**
```bash
# Get WSL2 IP address
npm run wsl-ip

# Then access from Windows browser:
# http://<WSL2_IP>:3000 (Dashboard)
# http://<WSL2_IP>:8000 (Backend API)
```

### Build Infrastructure
```bash
make build          # Rebuild Docker images
make rebuild        # Clean + build + deploy (full rebuild)
```

## High-Level Architecture

### Service Composition

The system runs as Docker containers orchestrated by docker-compose:

- **ns_core** (port 8000) - Main FastAPI service: goal management, execution API
- **ns_core_worker** - Celery worker for async task processing
- **ns_postgres** (5432) - PostgreSQL database
- **ns_redis** (6379) - Redis for Celery + caching
- **ns_litellm** (4000) - LLM proxy/router (supports Groq, Ollama, OpenAI, etc.)
- **ns_memory** (8001) - Memory service with Neo4j + Milvus (vector DB)
- **ns_dashboard** (8501) - Streamlit dashboard v1
- **dashboard_v2** (3000) - React dashboard v2 (operational thinking interface)
- **ns_opencode** (8002) - Code execution service
- **ns_websurfer** - Browser automation service
- **ns_telegram** (8004) - Telegram integration
- **ns_governor** - Docker container management
- **Temporal** (8088 UI, 7233 server) - Workflow orchestration for Continuous Goals

### Temporal.io Integration (Phase 1 Complete)

**NEW:** Continuous Goals now use Temporal.io Cron Workflows for reliable periodic execution.

**What is Temporal used for:**
- ✅ Continuous Goals (daily/weekly/monthly tasks via cron)
- ✅ Long-running workflows with automatic retry
- ✅ Mission-level goals (multi-day execution with resume)
- ❌ NOT used for: atomic goals (LangGraph), quick tasks (Celery)

**Temporal Components:**
- **Continuous Goals Workflow** (`services/temporal/workflows/continuous_goals.py`)
  - `ContinuousGoalCronWorkflow` - Cron-based periodic execution
  - `ContinuousGoalOneShotWorkflow` - Single execution for testing

- **Activities** (`services/temporal/activities/continuous_activities.py`)
  - `evaluate_continuous_goal()` - Check current state and calculate score
  - `generate_next_action()` - Plan next improvement action
  - `execute_continuous_action()` - Execute the action
  - `update_trend_metrics()` - Track progress over time
  - `check_goal_health()` - Verify goal is being actively executed

- **Worker** (`services/temporal/workers/continuous_worker.py`)
  - Listens to `ai-os-continuous` task queue
  - Runs workflows and activities

**API Endpoints:**
```
POST /goals/continuous/start          # Start continuous goal with cron
POST /goals/continuous/execute-once/{goal_id}  # Execute once
GET  /goals/continuous/status/{workflow_id}     # Check workflow status
POST /goals/continuous/cancel/{workflow_id}     # Cancel workflow
POST /goals/continuous/terminate/{workflow_id}  # Force terminate
GET  /temporal/workflows              # List workflows
```

**Temporal Web UI:** http://localhost:8088

**Starting the Worker:**
```bash
cd /home/onor/ai_os_final/services/temporal
./run_continuous_worker.sh
```

**Example Usage:**
```bash
curl -X POST http://localhost:8000/goals/continuous/start \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Weekly system optimization",
    "description": "Optimize system performance every Monday",
    "cron_schedule": "0 9 * * 1",
    "domains": ["performance", "programming"]
  }'
```

### Goal System v3.0 Architecture

The goal system is the core of AI-OS, implementing a sophisticated multi-level goal management system:

```
Mission (L0) → Strategic (L1) → Operational (L2) → Tactical/Atomic (L3)
```

**Key Components:**

1. **Goal Types** (models.py:Goal.goal_type):
   - `achievable` - Can be completed (has clear success criteria)
   - `continuous` - Ongoing improvement (no final state)
   - `directional` - Values/principles (guides direction)
   - `exploratory` - Research/discovery (outcome unknown)
   - `meta` - Self-improvement goals

2. **Goal Contracts** (goal_contract_validator.py):
   - Formalized constraints on LLM behavior
   - Define allowed_actions, max_depth, max_subgoals, evaluation_mode
   - Auto-applied based on goal_type
   - Prevent infinite loops and excessive decomposition

3. **Goal Lifecycle**:
   - `create` → `classify` → `decompose` → `execute` → `strict_evaluate` → `reflect` → `extract_patterns`
   - Decomposition creates parent-child relationships (Goal.parent_id)
   - Execution produces Artifacts (verifiable results)
   - Reflection generates next goals

4. **Evaluation System**:
   - **Strict Evaluator** (goal_strict_evaluator.py) - Fact-based completion check
     - Binary mode (passed/failed)
     - Scalar mode (score 0.0-1.0)
     - Trend mode (improving/stable/degrading)
   - **Reflector** (goal_reflector.py) - Causal analysis and next goal generation

5. **Goal Mutation** (goal_mutator.py):
   - Strengthen/weaken goals at runtime
   - Change goal_type
   - Freeze/thaw goals
   - Tracks mutation_history

### Artifact Layer v1

Atomic goals (L3, is_atomic=True) MUST produce verifiable artifacts:

- **Artifact Types**: FILE, KNOWLEDGE, DATASET, REPORT, LINK, EXECUTION_LOG
- **Verification**: Code-based rules in artifact_verifier.py (NOT LLM-based)
- **Registry**: artifact_registry.py manages artifact CRUD operations
- **Key Rule**: No passed artifacts → goal status = "incomplete"

### Skill Manifest v1

Skills have explicit contracts (skill_manifest.py, skill_registry.py):

```yaml
skill:
  name: web_research
  inputs: {required: [query], optional: [max_sources]}
  outputs: {artifact_type: REPORT}
  produces:
    - type: KNOWLEDGE
    - type: FILE
  verification:
    - name: min_sources
      rule: sources_count >= 3
```

Skills MUST return `SkillResult` with artifacts, not just strings.

### LLM Fallback System

The system automatically handles Groq rate limits:

- **Primary**: Groq (fast, but rate-limited)
- **Fallback**: Ollama Qwen2.5-Coder (slower, local)
- **Cooldown**: 6 hours (configurable via GROQ_COOLDOWN_HOURS)
- **State**: Stored in Redis
- **API**:
  - `GET /llm/status` - Check current status
  - `POST /llm/reset_groq` - Manually reset cooldown

### Agent Graph (LangGraph)

Multi-agent system using LangGraph with role-based model selection (agent_graph.py:16-54):

- **SUPERVISOR** (qwen3-coder) - Routes tasks to appropriate agents
- **CODER** (qwen3-coder) - Code generation and modification
- **PM** (gpt-oss) - Goal management and planning
- **RESEARCHER** (qwen3-coder) - Information gathering
- **INTELLIGENCE** (deepseek-v3.1) - Complex reasoning tasks

Each agent has:
- Role-specific temperature settings
- Specialized tools (tools.py, tools_external.py)
- Custom system prompts (agents/prompts.py)
- Timeout configuration (120s default)

### Memory ≠ Logs

The system distinguishes between logs (ephemeral) and memory (persistent):

- **Semantic Memory** (semantic_memory.py) - Decision patterns extracted from reflections
- **Artifact Registry** - Tangible results that persist after logs are deleted
- **Memory Signal** (memory_signal.py) - V4 memory system integration

## File Structure

### Deployment Infrastructure

**deploy.sh** - Main deployment automation script:
- `./deploy.sh full` - Sync files → clear __pycache__ → restart containers
- `./deploy.sh fast` - Sync files → restart (no cache clear, ~5s)
- `./deploy.sh single <container>` - Deploy to specific container only
- `./deploy.sh status` - Show container status
- `./deploy.sh logs <container>` - Tail container logs

**What deploy.sh does:**
1. Copies all `*.py` files from `services/core/` to `/app/` in containers
2. Copies subdirectories: `canonical_skills/`, `agents/`, `core/`
3. Clears Python cache (`__pycache__`, `*.pyc`)
4. Restarts containers with health check
5. Auto-fixes broken mounts if containers fail to start

### Core Service (`services/core/`)

**Models & Database:**
- `models.py` - SQLAlchemy models (Goal, GoalRelation, Artifact, SkillManifestDB, etc.)
- `database.py` - Database connection and session management
- `schemas.py` - Pydantic schemas for API validation

**Goal System:**
- `goal_executor.py` - Main goal execution orchestrator
- `goal_decomposer.py` - Goal decomposition logic
- `goal_strict_evaluator.py` - Fact-based evaluation (binary/scalar/trend)
- `goal_reflector.py` - Causal reflection and next goal generation
- `goal_mutator.py` - Runtime goal modification
- `goal_contract_validator.py` - Contract validation and enforcement
- `semantic_memory.py` - Decision pattern extraction

**Artifacts & Skills:**
- `artifact_registry.py` - Artifact CRUD operations
- `artifact_verifier.py` - Code-based verification
- `skill_manifest.py` - Skill contract definitions
- `skill_registry.py` - Skill management and discovery
- `canonical_skills/` - Built-in skill implementations
  - `mvp_skills.py` - Core MVP skills (write_file, web_research, ask_user, echo)
  - `production_skills.py` - Production-ready skills with manifests
  - `base.py` - Base skill interface and SkillResult class

**Agent System:**
- `agent_graph.py` - LangGraph agent orchestration
- `agents/prompts.py` - Agent system prompts
- `core/supervisor.py` - Supervisor agent pattern

**API:**
- `main.py` - FastAPI app and all endpoints
- `tasks.py` - Celery task definitions (async background jobs)

**Async Task Execution:**
- Goals executed via Celery tasks in `ns_core_worker` container
- Worker runs 8 concurrent processes: `celery -A tasks.celery_app worker --loglevel=info -Q default -c 8`
- Task queue: Redis (ns_redis container)
- View worker tasks: `make logs-worker`

**Utilities:**
- `llm_fallback.py` - LLM fallback management
- `telemetry.py` - System telemetry and logging
- `tools.py`, `tools_external.py` - Agent tool definitions

### Dashboard v2 (`services/dashboard_v2/`)

React-based "Operational Thinking Interface":

- **src/api/client.ts** - Backend API client
- **src/components/canvas/GraphCanvas.tsx** - ReactFlow graph visualization
- **src/components/inspector/InspectorPanel.tsx** - Context-aware node details
- **src/store/uiStore.ts** - UI state machine (explore/exploit/reflect modes)
- **src/store/graphStore.ts** - Graph state management
- **vite.config.ts** - Vite build configuration
- **tailwind.config.js** - TailwindCSS styling configuration
- **TypeScript** - Full type safety with strict mode

### Other Services

- **services/dashboard/** - Streamlit dashboard v1 (legacy)
- **services/memory/** - Neo4j + Milvus vector DB for semantic memory
- **services/opencode/** - Code execution sandbox
- **services/websurfer/** - Browser automation (Playwright)
- **services/telegram/** - Telegram bot integration
- **services/governor/** - Docker container management
- **skills/** - External skill definitions (mounted into containers)

## Common Workflows

### Making Code Changes to Core Service

```bash
# 1. Edit Python files in services/core/
vim services/core/my_module.py

# 2. Fast deploy (~5 seconds)
make deploy-fast

# 3. Check logs
make logs

# 4. If you see "module not found" or stale behavior
make deploy  # Full deploy with cache clear
```

### Adding a New API Endpoint

```bash
# 1. Add endpoint to services/core/main.py
# 2. Add schemas to services/core/schemas.py if needed
# 3. Deploy
make deploy-fast

# 4. Test
curl -X POST http://localhost:8000/your-endpoint \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

### Debugging a Failed Goal

```bash
# 1. Check goal status in database
make db-shell
psql> SELECT id, title, status, progress, is_atomic FROM goals WHERE title LIKE '%your goal%';

# 2. Check execution trace
psql> SELECT title, execution_trace FROM goals WHERE id = 'your-goal-id';

# 3. Check worker logs
make logs-worker

# 4. Check for artifacts
psql> SELECT * FROM artifacts WHERE goal_id = 'your-goal-id';
```

### Testing a New Skill

```bash
# 1. Create skill in services/core/canonical_skills/my_skill.py
# 2. Register in services/core/canonical_skills/mvp_skills.py or production_skills.py
# 3. Deploy
make deploy

# 4. Create test atomic goal that uses the skill
curl -X POST http://localhost:8000/goals/create \
  -H "Content-Type: application/json" \
  -d '{"title": "Test my skill", "description": "...", "goal_type": "achievable", "is_atomic": true}'

# 5. Execute goal
curl -X POST http://localhost:8000/goals/{goal_id}/execute
```

## Development Patterns

### Adding a New Goal Type

1. Update `goal_contract_validator.py` with default contract
2. Add evaluation logic in `goal_strict_evaluator.py` if needed
3. Update decomposition prompts in `agents/prompts.py`

### Creating a New Skill

1. Implement skill in `canonical_skills/`
2. Create manifest in `skills/manifests/` or register programmatically
3. Ensure skill returns `SkillResult` with artifacts
4. Add verification rules to manifest

**Example Skill Structure** (canonical_skills/base.py):
```python
from skill_manifest import SkillResult

async def my_skill(params: dict) -> SkillResult:
    # 1. Execute logic
    result = do_work(params)

    # 2. Return with artifacts
    return SkillResult(
        success=True,
        data=result,
        artifacts=[{
            "type": "FILE",
            "content_kind": "file",
            "content_location": "output.txt"
        }]
    )
```

### Modifying Database Models

1. Create migration in `services/core/migrations/`
2. Apply via `make deploy` (clears cache)
3. Update `models.py`
4. Update `schemas.py` for API validation

### Debugging Common Issues

**Container won't start after code changes:**
```bash
# Check logs
make logs

# Clear Python cache and redeploy
make deploy

# If stuck in restart loop
docker-compose up -d --force-recreate core
```

**Module import errors:**
- Symptom: "ModuleNotFoundError" or stale code behavior
- Cause: Python __pycache__ not cleared
- Fix: `make deploy` (full deploy with cache clear)

**Groq rate limit flooding logs:**
- Symptom: 404 errors every 10 minutes
- Cause: Groq API rate limit exceeded
- Fix: System auto-switches to Ollama; check with `make llm-status`

**Goal stuck in "incomplete" status:**
- Symptom: Atomic goal won't complete
- Cause: No passed artifacts registered
- Fix: Check artifact_registry.py and ensure skill returns SkillResult with artifacts

### Debugging Goal Execution

```bash
# Check goal status
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c \
  "SELECT id, title, status, progress, is_atomic FROM goals ORDER BY created_at DESC LIMIT 5;"

# Check execution trace
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c \
  "SELECT title, execution_trace IS NOT NULL as has_trace FROM goals WHERE is_atomic = true;"

# View logs
make logs
make logs-worker
```

## Important Constraints

1. **Atomic Goals (L3)**: MUST produce at least one passed artifact or status = "incomplete"
2. **Goal Contracts**: Decomposition and execution check contracts before proceeding
3. **LLM Fallback**: System auto-switches to Ollama when Groq hits rate limits
4. **Verification**: Artifacts verified by code rules, NOT LLM evaluation
5. **Memory vs Logs**: Artifacts persist in DB; logs are ephemeral
6. **Container Sync**: Use `make deploy-fast` for code changes (~5s), `make deploy` for model changes (clears cache)
7. **Python Cache**: If you see "module not found" or stale code issues, use `make deploy` to clear __pycache__
8. **UoW Pattern**: All goal transitions MUST use `transition_service.transition(uow, ...)` - direct status assignment is blocked

## Environment Configuration

Key environment variables (`.env`):

```bash
# Database
POSTGRES_USER=ns_admin
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ns_core_db

# LLM
LLM_MODEL=cloud-reasoner  # Primary model via LiteLLM
FALLBACK_MODEL=ollama/qwen2.5-coder:latest
GROQ_COOLDOWN_HOURS=6

# Neo4j (Memory)
NEO4J_PASSWORD=your_password

# MinIO (Object Storage)
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=your_password
```

## Testing

### Automated Tests (pytest)

**Structure:**
```
tests/
├── conftest.py                    # Fixtures and config
├── unit/
│   ├── test_imports.py           # 18 smoke tests
│   ├── test_goal_state_machine.py # 10 state machine tests
│   └── test_artifact_verifier.py  # 16 verifier tests
└── e2e/
    └── test_api.py                # 8 API tests
```

**Running tests:**
```bash
make test-unit    # Unit tests (~2s) - no dependencies
make test-e2e     # E2E tests (~1s) - requires running services
make test-all     # All tests

# Or directly in container:
docker exec ns_core pytest /app/tests/unit -v
docker exec ns_core pytest /app/tests/e2e -v -m e2e
```

### Legacy System Tests
- `services/core/test_system.py` - General system tests
- `services/core/skills/browser/test_basics.py` - Browser skill tests

### Manual Testing via API

```bash
# Create goal
curl -X POST http://localhost:8000/goals/create \
  -H "Content-Type: application/json" \
  -d '{"title": "Test goal", "description": "Testing", "goal_type": "achievable", "is_atomic": true}'

# Decompose goal
curl -X POST http://localhost:8000/goals/{goal_id}/decompose \
  -H "Content-Type: application/json" \
  -d '{"max_depth": 1}'

# Execute goal
curl -X POST http://localhost:8000/goals/{goal_id}/execute

# Check LLM status
curl http://localhost:8000/llm/status

# Bulk transition (NEW)
curl -X POST http://localhost:8000/goals/bulk-transition \
  -H "Content-Type: application/json" \
  -d '{"goal_ids": ["uuid1", "uuid2"], "new_state": "active", "reason": "Mass activation"}'

# Freeze tree (NEW)
curl -X POST http://localhost:8000/goals/freeze-tree \
  -H "Content-Type: application/json" \
  -d '{"root_goal_id": "uuid", "reason": "Project paused"}'
```

## Utility Scripts

### Backup and Maintenance
```bash
scripts/backup_goals.sh                 # Backup goals to JSON
scripts/restore_goals.sh                # Restore from backup
scripts/setup_backup_cron.sh            # Auto-backup every hour
scripts/create_old_goals.sh             # Create sample historical goals
scripts/create_sample_relations.sh      # Create goal relationships

scripts/resume_pending_goals.sh         # Resume goals stuck in "pending"
scripts/setup_auto_resume_cron.sh       # Auto-resume every 5 minutes
```

### System Tests
```bash
services/core/test_goals.sh             # Test goal creation and execution
services/core/skills/browser/test_basics.py  # Test browser automation
services/core/test_system.py            # General system tests
```

## Troubleshooting

### Container Won't Start
```bash
make logs           # Check ns_core logs
make logs-worker    # Check worker logs
docker logs ns_postgres  # Check DB logs
```

### Module Not Found Error
```bash
make deploy         # Full deploy with cache clear
```

### Groq Rate Limit
```bash
make llm-status     # Check if fallback is active
make llm-reset      # Manually reset cooldown
```

### Database Connection Issues
```bash
docker restart ns_postgres
make status
```

## Key Integrations

- **LiteLLM** (ns_litellm) - Unified LLM API routing
- **LangGraph** - Agent orchestration graph
- **Celery** - Async task queue (Redis broker)
- **Temporal.io** - Optional: Long-running workflow orchestration
- **Neo4j** - Graph database for memory relationships
- **Milvus** - Vector database for semantic search
