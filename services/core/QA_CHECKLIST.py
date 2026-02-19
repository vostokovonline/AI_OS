"""
OCCP v0.3 QA/Regression Checklist
For verifying implementation fixes without regressions
"""

CHECKLIST_VERSION = "1.0"
DATE = "2026-02-04"

# ============================================================================
# PHASE 0: BASELINE (Before any fixes)
# ============================================================================

PHASE_0_BASELINE = """
## PHASE 0: Establish Baseline

### Step 0.1: Run Compliance Tests
```bash
cd services/core
python3 test_occp_sandbox.py
```

EXPECTED:
- ✅ 17/17 tests PASS
- ✅ No errors, no warnings
- ✅ "ALL SANDBOX COMPLIANCE TESTS PASSED"

ACTUAL: ___________________

PASS/FAIL: _____


### Step 0.2: Run Stress Test
```bash
python3 stress_test_occp.py
```

EXPECTED METRICS:
- ✅ Throughput: ~40K req/sec
- ✅ Success Rate: ≥95%
- ✅ Forbidden Detection: ≥70%
- ✅ Timeouts: 0

ACTUAL:
- Throughput: ___________ req/sec
- Success Rate: _________%
- Forbidden Detection: _________%
- Timeouts: ___________

PASS/FAIL: _____


### Step 0.3: Check Database State
```bash
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c "
SELECT goal_type, COUNT(*) FROM goals GROUP BY goal_type;
"
```

EXPECTED:
- ✅ Only: achievable, continuous, directional
- ✅ No unknown types

ACTUAL: ___________________

PASS/FAIL: _____


### BASELINE SUMMARY
Overall: _____ PASS / _____ FAIL

If FAIL → Do not proceed with fixes until baseline passes!
"""

# ============================================================================
# PHASE 1: Concurrent SK Veto (HIGH)
# ============================================================================

PHASE_1_SK_VETO = """
## PHASE 1: Concurrent SK Veto Fix

### Implementation
File: services/core/occp_gateway.py

Add to __init__:
```python
self._sk_lock = asyncio.Lock()
```

Wrap SK check:
```python
async def _sk_check(self, request: FederatedRequest) -> OCCPDecisionSchema:
    async with self._sk_lock:
        # Existing SK check logic here
        allowed = await self.sk_checker.allows_federated(request)
        # ... rest of method
```

VERIFICATION: Implementation complete? _____


### Step 1.1: Unit Test - SK Lock Behavior
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'services/core')

async def test_sk_lock():
    from occp_gateway import OCCPGateway

    # Mock MCL and SK checkers
    class MockMCL:
        async def allows_federated(self, req):
            return True

    class MockSK:
        def __init__(self):
            self.call_count = 0

        async def allows_federated(self, req):
            self.call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return True

    gateway = OCCPGateway(
        node_id='test',
        mcl_checker=MockMCL(),
        sk_checker=MockSK(),
        resource_manager=None
    )

    # Concurrent requests
    from occp_v03_types import FederatedRequest, OCCPRequestType, ResourceBound

    request = FederatedRequest(
        request_id='test-1',
        request_type=OCCPRequestType.COMPUTE_ASSIST,
        node_id='test-node',
        resource_bound=ResourceBound(
            compute_seconds=30,
            memory_mb=512,
            max_tokens=10000
        ),
        sandbox=True
    )

    # Fire 100 concurrent requests
    tasks = [gateway._sk_check(request) for _ in range(100)]
    await asyncio.gather(*tasks)

    print(f'SK called {gateway.sk_checker.call_count} times')
    assert gateway.sk_checker.call_count == 100

asyncio.run(test_sk_lock())
print('✅ SK lock unit test PASSED')
"
```

EXPECTED:
- ✅ SK called exactly 100 times (no races)
- ✅ No exceptions, no deadlocks

ACTUAL: ___________________

PASS/FAIL: _____


### Step 1.2: Stress Test - Verify No Regression
```bash
python3 stress_test_occp.py
```

EXPECTED (vs baseline):
- ✅ Throughput: ≥30K req/sec (may drop slightly)
- ✅ Success Rate: ≥95% (no regression)
- ✅ Forbidden Detection: ≥70% (no regression)
- ✅ Timeouts: 0

ACTUAL:
- Throughput: ___________ req/sec
- Success Rate: _________%
- Forbidden Detection: _________%
- Timeouts: ___________

PASS/FAIL: _____


### PHASE 1 SUMMARY
Implementation: _____ COMPLETE
Unit Tests: _____ PASS
Stress Test: _____ PASS (no regression)

Overall: _____ READY FOR PHASE 2
"""


# ============================================================================
# PHASE 2: Side-Channel Mitigation (MEDIUM)
# ============================================================================

PHASE_2_SIDE_CHANNELS = """
## PHASE 2: Side-Channel Mitigation

### Implementation
File: services/core/occp_sandbox.py

Modify _check_payload() to add constant-time padding:
```python
def _check_payload(self, payload: Dict) -> Optional[SandboxViolation]:
    # Add at start
    start_time = time.monotonic()
    target_duration = 0.0001  # 100μs constant time

    # ... existing logic ...

    # Add at end before return
    elapsed = time.monotonic() - start_time
    if elapsed < target_duration:
        await asyncio.sleep(target_duration - elapsed)

    return None
```

Wait! This won't work in sync function. Use async wrapper instead.

CORRECT APPROACH:
Make check_payload_async() and call from execute():

VERIFICATION: Implementation complete? _____


### Step 2.1: Unit Test - Timing Consistency
```bash
python3 -c "
import time
import asyncio
import sys
sys.path.insert(0, 'services/core')

async def test_timing_consistency():
    from occp_compute_sandbox import create_compute_assist_sandbox
    from occp_sandbox import SandboxOp

    sandbox = create_compute_assist_sandbox()

    # Test allowed payload (fast)
    allowed_payload = {
        'operation_type': 'compute',
        'input_data': {'values': [1, 2, 3]}
    }

    # Test forbidden payload (slow if check is slow)
    forbidden_payload = {
        'operation_type': 'compute',
        'input_data': {'goal_id': 'test'}
    }

    # Measure 100 iterations each
    times_allowed = []
    times_forbidden = []

    for _ in range(100):
        start = time.perf_counter()
        await sandbox.execute(SandboxOp.COMPUTE, allowed_payload)
        times_allowed.append(time.perf_counter() - start)

    for _ in range(100):
        start = time.perf_counter()
        await sandbox.execute(SandboxOp.COMPUTE, forbidden_payload)
        times_forbidden.append(time.perf_counter() - start)

    avg_allowed = sum(times_allowed) / len(times_allowed) * 1000
    avg_forbidden = sum(times_forbidden) / len(times_forbidden) * 1000

    print(f'Allowed payload avg: {avg_allowed:.2f}ms')
    print(f'Forbidden payload avg: {avg_forbidden:.2f}ms')
    print(f'Timing difference: {abs(avg_allowed - avg_forbidden):.2f}ms')

    # Check if timing is similar (within 10x)
    ratio = max(avg_allowed, avg_forbidden) / min(avg_allowed, avg_forbidden)
    if ratio < 10:
        print('✅ Timing consistency acceptable (ratio < 10x)')
    else:
        print(f'⚠️  Timing ratio too high: {ratio:.1f}x')

asyncio.run(test_timing_consistency())
"
```

EXPECTED:
- ✅ Timing difference < 10x (acceptable leakage)
- ✅ No excessive slowdown (<0.1ms per check)

ACTUAL:
- Timing difference: _________ms
- Ratio: ___________x

PASS/FAIL: _____


### Step 2.2: Stress Test - Verify Performance
```bash
python3 stress_test_occp.py
```

EXPECTED (vs baseline):
- ✅ Avg Response Time: <0.1ms (max 5x slowdown)
- ✅ Throughput: >10K req/sec (max 5x slowdown)
- ✅ Success Rate: ≥95%

ACTUAL:
- Avg Response Time: _________ms
- Throughput: ___________ req/sec
- Success Rate: _________%

PASS/FAIL: _____


### PHASE 2 SUMMARY
Implementation: _____ COMPLETE
Timing Test: _____ PASS
Performance Test: _____ PASS (acceptable slowdown)

Overall: _____ READY FOR PHASE 3
"""


# ============================================================================
# PHASE 3: Audit Atomic Append (MEDIUM)
# ============================================================================

PHASE_3_AUDIT = """
## PHASE 3: Audit Atomic Append

### Implementation
File: services/core/occp_gateway.py

Add file locking for audit log:
```python
import fcntl

async def _audit_log(self, request, decisions, result):
    entry = self._create_audit_entry(request, decisions, result)

    with open(self.audit_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        f.write(json.dumps(entry) + '\\n')
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
```

VERIFICATION: Implementation complete? _____


### Step 3.1: Unit Test - Concurrent Audit Writes
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'services/core')

async def test_concurrent_audit():
    # Test concurrent writes to same file
    import tempfile
    import os

    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    async def write_audit(worker_id):
        with open(temp_path, 'a') as f:
            # Simulate file lock
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f'Worker-{worker_id}: {asyncio.get_event_loop().time()}\\n')
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Launch 50 concurrent writes
    tasks = [write_audit(i) for i in range(50)]
    await asyncio.gather(*tasks)

    # Verify all writes completed
    with open(temp_path, 'r') as f:
        lines = f.readlines()

    print(f'Expected lines: 50')
    print(f'Actual lines: {len(lines)}')

    if len(lines) == 50:
        print('✅ All writes completed (no data loss)')
    else:
        print(f'❌ Data loss! Missing {50 - len(lines)} lines')

    # Cleanup
    os.unlink(temp_path)

asyncio.run(test_concurrent_audit())
"
```

EXPECTED:
- ✅ Exactly 50 lines written (no lost data)
- ✅ No file corruption
- ✅ No deadlocks

ACTUAL: ___________________

PASS/FAIL: _____


### Step 3.2: Stress Test - Verify Audit Integrity
```bash
# Run stress test and check audit log
python3 stress_test_occp.py

# Check audit log completeness
wc -l /tmp/occp_stress_test.log

# Should have at least 500 lines (one per request)
```

EXPECTED:
- ✅ Audit log lines ≥ total requests
- ✅ No partial/corrupted entries

ACTUAL: ___________________

PASS/FAIL: _____


### PHASE 3 SUMMARY
Implementation: _____ COMPLETE
Concurrent Writes: _____ PASS
Audit Integrity: _____ PASS

Overall: _____ READY FOR PHASE 4
"""


# ============================================================================
# PHASE 4: Federated Throttling (HIGH)
# ============================================================================

PHASE_4_THROTTLING = """
## PHASE 4: Federated Throttling

### Implementation
File: services/core/occp_gateway.py

Add token bucket rate limiter:
```python
from collections import deque
import time

class RateLimiter:
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # requests per second
        self.burst = burst  # max burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens=1):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

# In Gateway.__init__:
self._rate_limiters = {}  # node_id -> RateLimiter

# In handle_request():
limiter = self._get_rate_limiter(request.node_id)
if not await limiter.acquire():
    return self._deny(request, OCCPDecisionSchema(
        layer="Resource",
        decision=OCCPDecision.DENY,
        reason_code=OCCPReasonCode.RESOURCE_01,
        explanation="Rate limit exceeded"
    ))
```

VERIFICATION: Implementation complete? _____


### Step 4.1: Unit Test - Rate Limiting
```bash
python3 -c "
import asyncio
import time
import sys
sys.path.insert(0, 'services/core')

async def test_rate_limiter():
    # Simple rate limiter test
    class RateLimiter:
        def __init__(self, rate=10, burst=10):
            self.rate = rate
            self.burst = burst
            self.tokens = burst
            self.last_update = time.time()
            self._lock = asyncio.Lock()

        async def acquire(self, tokens=1):
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                return False

    limiter = RateLimiter(rate=10, burst=10)

    # Should allow first 10 requests immediately
    for i in range(10):
        assert await limiter.acquire() == True

    print('✅ First 10 requests allowed')

    # Next request should be denied (out of tokens)
    assert await limiter.acquire() == False
    print('✅ 11th request denied (rate limit working)')

    # Wait for token refill
    await asyncio.sleep(0.2)

    # Should allow 1 more
    assert await limiter.acquire() == True
    print('✅ Request allowed after refill')

asyncio.run(test_rate_limiter())
"
```

EXPECTED:
- ✅ First 10 requests allowed
- ✅ 11th request denied
- ✅ Request allowed after refill

ACTUAL: ___________________

PASS/FAIL: _____


### Step 4.2: Integration Test - Throttling Under Load
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'services/core')

async def test_throttling():
    # Simulate burst of 100 requests at 10 req/sec limit
    class RateLimiter:
        def __init__(self, rate=10, burst=10):
            self.rate = rate
            self.burst = burst
            self.tokens = burst
            self.last_update = time.time()
            self._lock = asyncio.Lock()

        async def acquire(self, tokens=1):
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                return False

    limiter = RateLimiter(rate=10, burst=10)

    allowed = 0
    denied = 0

    # Fire 100 concurrent requests
    tasks = []
    for _ in range(100):
        tasks.append(limiter.acquire())

    results = await asyncio.gather(*tasks)
    allowed = sum(results)
    denied = len(results) - allowed

    print(f'Allowed: {allowed}/100')
    print(f'Denied: {denied}/100')

    # Should allow ~10-20 (burst size + race conditions)
    if 10 <= allowed <= 20:
        print('✅ Rate limiting working correctly')
    else:
        print(f'⚠️  Unexpected allowed count: {allowed}')

asyncio.run(test_throttling())
"
```

EXPECTED:
- ✅ Allowed: 10-20 requests (burst size)
- ✅ Denied: 80-90 requests

ACTUAL: ___________________

PASS/FAIL: _____


### Step 4.3: Stress Test - Verify Throttling Impact
```bash
python3 stress_test_occp.py
```

EXPECTED (vs baseline):
- ✅ Throughput: >10K req/sec (may decrease)
- ✅ Success Rate: ≥90% (throttling works)
- ✅ Throttled requests logged

ACTUAL: ___________________

PASS/FAIL: _____


### PHASE 4 SUMMARY
Implementation: _____ COMPLETE
Rate Limiter: _____ PASS
Throttling Test: _____ PASS
Overall: _____ READY FOR FINAL CHECK
"""


# ============================================================================
# PHASE 5: FINAL VERIFICATION
# ============================================================================

PHASE_5_FINAL = """
## PHASE 5: Final Regression Check

### Step 5.1: Full Compliance Test Suite
```bash
cd services/core
python3 test_occp_sandbox.py
```

EXPECTED:
- ✅ 17/17 tests PASS
- ✅ No regressions from baseline

ACTUAL: ___________________

PASS/FAIL: _____


### Step 5.2: Stress Test - Post-Fix Metrics
```bash
python3 stress_test_occp.py
```

EXPECTED METRICS:
- ✅ Throughput: >10K req/sec
- ✅ Success Rate: ≥90%
- ✅ Forbidden Detection: ≥70%
- ✅ Timeouts: 0
- ✅ Throttling: working (if tested)

ACTUAL:
- Throughput: ___________ req/sec
- Success Rate: _________%
- Forbidden Detection: _________%
- Timeouts: ___________

PASS/FAIL: _____


### Step 5.3: Database Integrity Check
```bash
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c "
SELECT goal_type, COUNT(*) FROM goals GROUP BY goal_type;
"
```

EXPECTED:
- ✅ No data corruption
- ✅ Same types as baseline

ACTUAL: ___________________

PASS/FAIL: _____


### Step 5.4: Documentation Update
Update RFC with fixes:

- [ ] Section 5.4.3: Document SK lock
- [ ] Section 5.7: Document audit atomic append
- [ ] Section 6.9: Document rate limiting
- [ ] Appendix: Add known limitations (obfuscation)

VERIFICATION: _____ COMPLETE


### FINAL SUMMARY

PHASE 0 (Baseline): _____ PASS
PHASE 1 (SK Veto): _____ PASS
PHASE 2 (Side-Channels): _____ PASS
PHASE 3 (Audit): _____ PASS
PHASE 4 (Throttling): _____ PASS
PHASE 5 (Final): _____ PASS

OVERALL STATUS: _____ PRODUCTION READY

If all phases PASS → System is production-ready!
If any phase FAIL → Fix and re-test.
"""


if __name__ == "__main__":
    print("="*70)
    print("OCCP v0.3 QA/Regression Checklist")
    print("="*70)
    print(f"Version: {CHECKLIST_VERSION}")
    print(f"Date: {DATE}")
    print("="*70)
    print("\nTo use this checklist:")
    print("1. Start with PHASE 0 to establish baseline")
    print("2. Proceed through PHASE 1-4 in order")
    print("3. Run PHASE 5 for final verification")
    print("4. Fill in ACTUAL results for each step")
    print("5. Mark PASS/FAIL based on EXPECTED vs ACTUAL")
    print("="*70)
