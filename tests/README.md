# AI-OS Tests

Minimal test shield for stabilization phase.

## Running Tests

### Unit Tests (no external dependencies)
```bash
# In container
docker exec ns_core pytest /app/tests/unit -v

# Or locally with pytest installed
pytest tests/unit -v
```

### E2E Tests (requires running services)
```bash
# Make sure services are running
make status

# Run E2E tests
pytest tests/e2e -v -m e2e
```

### All Tests
```bash
pytest tests/ -v
```

## Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| **Smoke** | Module imports, basic sanity | `tests/unit/test_imports.py` |
| **State Machine** | Goal transitions validation | `tests/unit/test_goal_state_machine.py` |
| **Artifact Verifier** | Inline vs file detection | `tests/unit/test_artifact_verifier.py` |
| **E2E API** | Endpoint connectivity | `tests/e2e/test_api.py` |

## Adding New Tests

1. Unit tests go in `tests/unit/`
2. E2E tests go in `tests/e2e/`
3. Use fixtures from `conftest.py`
4. Mock external services (LLM, DB, Redis) in unit tests
5. Only E2E tests should hit real services

## CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements-test.txt
          pytest tests/unit -v
```
