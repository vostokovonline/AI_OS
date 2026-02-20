"""
OCCP v0.3 Stress Tester
Tests sandbox isolation, gateway validation, and resource limits under load
"""
import asyncio
import time
import random
import sys
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

# Add services/core to path
sys.path.insert(0, 'services/core')

from occp_compute_sandbox import create_compute_assist_sandbox
from occp_adversarial_sandbox import create_adversarial_test_sandbox
from occp_sandbox import SandboxOp, SandboxViolation

# ========================
# CONFIGURATION
# ========================

@dataclass
class StressTestConfig:
    """Stress test parameters"""
    concurrent_instances: int = 50
    total_requests: int = 500

    # Payload distribution
    forbidden_ratio: float = 0.3  # 30% forbidden payloads

    # Resource limits for testing
    max_tokens: int = 10000
    max_time_seconds: int = 30

    # Logging
    verbose: bool = False
    log_file: str = "/tmp/occp_stress_test.log"

    # Test scenarios
    test_nested_forbidden: bool = True
    test_timing_attacks: bool = True
    test_concurrent_write_conflicts: bool = True


@dataclass
class TestMetrics:
    """Test execution metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    forbidden_violations: int = 0
    resource_violations: int = 0
    timeouts: int = 0

    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0

    # Throughput
    requests_per_second: float = 0.0

    # Error breakdown
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def calculate_throughput(self):
        """Calculate requests per second"""
        if self.total_time > 0:
            self.requests_per_second = self.total_requests / self.total_time

    def update_time(self, elapsed: float):
        """Update timing metrics"""
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)


# ========================
# PAYLOAD GENERATORS
# ========================

class PayloadGenerator:
    """Generate test payloads for stress testing"""

    FORBIDDEN_CONTEXTS = ['goals', 'vectors', 'legacy_axis', 'mcl_state', 'sk_state']

    @staticmethod
    def generate_allowed() -> Dict:
        """Generate allowed (clean) payload"""
        return {
            "operation_type": "compute",
            "input_data": {
                "values": [random.randint(1, 100) for _ in range(10)],
                "action": "sum"
            }
        }

    @staticmethod
    def generate_forbidden_simple() -> Dict:
        """Generate payload with forbidden context at top level"""
        forbidden = random.choice(PayloadGenerator.FORBIDDEN_CONTEXTS)
        return {
            "operation_type": "compute",
            "input_data": {
                f"{forbidden}_id": f"forbidden-{random.randint(1000, 9999)}"
            }
        }

    @staticmethod
    def generate_forbidden_nested() -> Dict:
        """Generate payload with nested forbidden context"""
        forbidden = random.choice(PayloadGenerator.FORBIDDEN_CONTEXTS)
        return {
            "operation_type": "compute",
            "input_data": {
                "nested": {
                    "deep": {
                        f"{forbidden}_ref": "forbidden-value"
                    }
                }
            }
        }

    @staticmethod
    def generate_obfuscated() -> Dict:
        """Generate payload with obfuscated forbidden keys"""
        forbidden = random.choice(PayloadGenerator.FORBIDDEN_CONTEXTS)
        # Try to obfuscate: g_o_a_l_s instead of goals
        if forbidden == "goals":
            obfuscated = "g_o_a_l_s"
        elif forbidden == "vectors":
            obfuscated = "v_e_c_t_o_r_s"
        else:
            obfuscated = forbidden

        return {
            "operation_type": "compute",
            "input_data": {
                f"{obfuscated}_id": "obfuscated-forbidden"
            }
        }

    @classmethod
    def generate(cls, config: StressTestConfig) -> Tuple[Dict, str]:
        """Generate payload with label"""
        rand = random.random()

        if rand < config.forbidden_ratio:
            # Forbidden payload
            choice = random.choice(['simple', 'nested', 'obfuscated'])
            if choice == 'simple':
                return cls.generate_forbidden_simple(), 'forbidden_simple'
            elif choice == 'nested' and config.test_nested_forbidden:
                return cls.generate_forbidden_nested(), 'forbidden_nested'
            else:
                return cls.generate_obfuscated(), 'forbidden_obfuscated'
        else:
            # Allowed payload
            return cls.generate_allowed(), 'allowed'


# ========================
# STRESS TEST WORKERS
# ========================

class StressTestWorker:
    """Execute sandbox operations under stress"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.metrics = TestMetrics()
        self.lock = asyncio.Lock()
        self.log_file = open(config.log_file, 'w') if config.log_file else None

    def log(self, message: str):
        """Write to log file"""
        if self.log_file:
            timestamp = datetime.utcnow().isoformat()
            self.log_file.write(f"[{timestamp}] {message}\n")
            self.log_file.flush()
        if self.config.verbose:
            logger.info(message)

    async def execute_sandbox(self, payload: Dict, payload_type: str) -> Dict:
        """Execute single sandbox operation"""
        sandbox = create_compute_assist_sandbox(
            max_tokens=self.config.max_tokens,
            max_time_seconds=self.config.max_time_seconds
        )

        start_time = time.perf_counter()

        try:
            result = await sandbox.execute(SandboxOp.COMPUTE, payload)
            elapsed = time.perf_counter() - start_time

            # Analyze result
            is_forbidden = 'forbidden' in payload_type
            expected_violation = is_forbidden

            actual_violation = not result.success and result.aborted

            success = (
                (not expected_violation and result.success) or
                (expected_violation and actual_violation)
            )

            return {
                'payload_type': payload_type,
                'success': success,
                'elapsed': elapsed,
                'result': result,
                'expected_violation': expected_violation,
                'actual_violation': actual_violation,
                'timeout': result.timeout,
                'aborted': result.aborted
            }

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return {
                'payload_type': payload_type,
                'success': False,
                'elapsed': elapsed,
                'error': str(e),
                'expected_violation': 'forbidden' in payload_type,
                'actual_violation': False
            }

    async def worker(self, worker_id: int, semaphore: asyncio.Semaphore):
        """Worker coroutine that processes tasks"""
        tasks_processed = 0

        while True:
            async with semaphore:
                # Check if we should stop
                async with self.lock:
                    if self.metrics.total_requests >= self.config.total_requests:
                        break
                    self.metrics.total_requests += 1

                # Generate payload
                payload, payload_type = PayloadGenerator.generate(self.config)

                # Execute
                result = await self.execute_sandbox(payload, payload_type)

                # Update metrics
                async with self.lock:
                    if result['success']:
                        self.metrics.successful_requests += 1
                    else:
                        # Track error type
                        error_type = result.get('payload_type', 'unknown')
                        self.metrics.errors_by_type[error_type] = \
                            self.metrics.errors_by_type.get(error_type, 0) + 1

                    if result['actual_violation']:
                        self.metrics.forbidden_violations += 1

                    if result.get('timeout', False):
                        self.metrics.timeouts += 1

                    self.metrics.update_time(result['elapsed'])

                    tasks_processed += 1

                    # Log failures
                    if not result['success']:
                        self.log(f"Worker {worker_id}: FAIL - {result['payload_type']} - {result.get('error', 'unknown')}")

        return tasks_processed


# ========================
# STRESS TEST RUNNER
# ========================

async def run_stress_test(config: StressTestConfig) -> TestMetrics:
    """Run stress test with given configuration"""
    logger.info(f"\n{'='*70}")
    logger.info(f"OCCP v0.3 Stress Test")
    logger.info(f"{'='*70}")
    logger.info(f"Concurrent Instances: {config.concurrent_instances}")
    logger.info(f"Total Requests: {config.total_requests}")
    logger.info(f"Forbidden Ratio: {config.forbidden_ratio:.1%}")
    logger.info(f"Log File: {config.log_file}")
    logger.info(f"{'='*70}\n")

    worker = StressTestWorker(config)
    semaphore = asyncio.Semaphore(config.concurrent_instances)

    start_time = time.time()

    # Launch workers
    tasks = [
        worker.worker(worker_id, semaphore)
        for worker_id in range(config.concurrent_instances)
    ]

    # Wait for completion
    await asyncio.gather(*tasks)

    end_time = time.time()

    # Calculate final metrics
    worker.metrics.calculate_throughput()

    # Close log file
    if worker.log_file:
        worker.log_file.close()

    return worker.metrics


# ========================
# REPORTING
# ========================

def print_metrics(metrics: TestMetrics):
    """Print test metrics"""
    logger.info(f"\n{'='*70}")
    logger.info(f"STRESS TEST RESULTS")
    logger.info(f"{'='*70}")

    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total Requests:        {metrics.total_requests}")
    logger.info(f"  Successful:            {metrics.successful_requests} ({metrics.successful_requests/metrics.total_requests*100:.1f}%)")
    logger.info(f"  Forbidden Violations:  {metrics.forbidden_violations}")
    logger.info(f"  Timeouts:              {metrics.timeouts}")

    logger.info(f"\n⏱️  Timing Metrics:")
    logger.info(f"  Avg Response Time:     {metrics.total_time/metrics.total_requests*1000:.2f}ms")
    logger.info(f"  Min Response Time:     {metrics.min_time*1000:.2f}ms")
    logger.info(f"  Max Response Time:     {metrics.max_time*1000:.2f}ms")

    logger.info(f"\n🚀 Throughput:")
    logger.info(f"  Requests/sec:          {metrics.requests_per_second:.2f}")

    if metrics.errors_by_type:
        logger.info(f"\n❌ Errors by Type:")
        for error_type, count in sorted(metrics.errors_by_type.items(),
                                       key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_type}:            {count}")

    logger.info(f"\n{'='*70}")

    # Compliance check
    logger.info(f"\n✅ COMPLIANCE CHECK:")

    if metrics.forbidden_violations == 0:
        logger.info(f"  ❌ FAILED: No forbidden violations detected")
        logger.info(f"     (Sandbox should reject {metrics.forbidden_violations} forbidden payloads)")
    else:
        expected_forbidden = int(metrics.total_requests * 0.3)  # Approx 30%
        detection_rate = metrics.forbidden_violations / expected_forbidden * 100 if expected_forbidden > 0 else 0
        logger.info(f"  ✅ Forbidden violations detected: {metrics.forbidden_violations}")
        logger.info(f"     Detection rate: ~{detection_rate:.0f}%")

    success_rate = metrics.successful_requests / metrics.total_requests * 100
    if success_rate >= 95:
        logger.info(f"  ✅ SUCCESS RATE: {success_rate:.1f}% (target: ≥95%)")
    else:
        logger.info(f"  ❌ SUCCESS RATE: {success_rate:.1f}% (target: ≥95%)")

    if metrics.timeouts == 0:
        logger.info(f"  ✅ NO TIMEOUTS")
    else:
        logger.info(f"  ⚠️  TIMEOUTS: {metrics.timeouts}")

    logger.info(f"\n{'='*70}\n")


# ========================
# MAIN
# ========================

def main():
    """Main entry point"""
    config = StressTestConfig(
        concurrent_instances=50,
        total_requests=500,
        forbidden_ratio=0.3,
        verbose=False,
        log_file="/tmp/occp_stress_test.log"
    )

    # Run stress test
    metrics = asyncio.run(run_stress_test(config))

    # Print results
    print_metrics(metrics)


if __name__ == "__main__":
    main()
