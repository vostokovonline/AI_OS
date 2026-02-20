"""
OCCP (Open Cognitive Control Protocol) Test Suite
Draft 0.1 Compliance Tests

Tests Meta-Cognition Layer (MCL) and Survivability Kernel (SK)
"""
import asyncio
import requests
import json
from typing import Dict, List

BASE_URL = "http://localhost:8000"

class OCCPTest:
    """Base test class"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None

    async def run(self) -> bool:
        """Run test"""
        try:
            await self.test()
            self.passed = True
            return True
        except Exception as e:
            self.error = str(e)
            return False

    async def test(self):
        """Override in subclass"""
        raise NotImplementedError


class TestMCLState(OCCPTest):
    """Test MCL state retrieval"""
    def __init__(self):
        super().__init__("MCL: Get initial state")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/mcl/state")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "state" in data, "Response missing 'state' field"

        state = data["state"]
        if state:
            # If state exists, validate schema
            assert "cognitive_mode" in state, "State missing 'cognitive_mode'"
            assert "entropy_budget" in state, "State missing 'entropy_budget'"
            assert "epistemic_confidence" in state, "State missing 'epistemic_confidence'"
            assert "drift_score" in state, "State missing 'drift_score'"
            assert state["cognitive_mode"] in ["exploration", "exploitation", "preservation"]
            logger.info(f"  ✓ Current mode: {state['cognitive_mode']}")
        else:
            logger.info("  ℹ No initial MCL state (will be created on first operation)")


class TestMCLSetMode(OCCPTest):
    """Test MCL mode transition"""
    def __init__(self):
        super().__init__("MCL: Set mode to exploration")

    async def test(self):
        response = requests.post(
            f"{BASE_URL}/occp/mcl/set-mode",
            json={"mode": "exploration", "rationale": "Test mode transition"}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "state" in data, "Response missing 'state'"

        state = data["state"]
        assert state["cognitive_mode"] == "exploration", f"Expected 'exploration', got {state['cognitive_mode']}"
        assert state["entropy_budget"] == 0.7, f"Expected entropy_budget 0.7, got {state['entropy_budget']}"
        assert state["epistemic_confidence"] == 0.3, f"Expected epistemic_confidence 0.3, got {state['epistemic_confidence']}"
        assert state["risk_posture"] == "aggressive", f"Expected 'aggressive', got {state['risk_posture']}"

        logger.info(f"  ✓ Mode set to exploration")
        logger.info(f"    - entropy_budget: {state['entropy_budget']}")
        logger.info(f"    - epistemic_confidence: {state['epistemic_confidence']}")
        logger.info(f"    - risk_posture: {state['risk_posture']}")


class TestMCLAllowedOps(OCCPTest):
    """Test MCL allowed operations"""
    def __init__(self):
        super().__init__("MCL: Get allowed operations")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/mcl/allowed-operations")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "operations" in data, "Response missing 'operations'"

        ops = data["operations"]
        assert "mode" in ops, "Operations missing 'mode'"
        assert "allowed" in ops, "Operations missing 'allowed'"
        assert "forbidden" in ops, "Operations missing 'forbidden'"

        logger.info(f"  ✓ Mode: {ops['mode']}")
        logger.info(f"  ✓ Allowed ({len(ops['allowed'])}): {', '.join(ops['allowed'])}")
        logger.info(f"  ✓ Forbidden ({len(ops['forbidden'])}): {', '.join(ops['forbidden'])}")


class TestMCLModeRestrictions(OCCPTest):
    """Test that different modes have different restrictions"""
    def __init__(self):
        super().__init__("MCL: Mode restrictions")

    async def test(self):
        modes_to_test = ["exploration", "exploitation", "preservation"]

        for mode in modes_to_test:
            # Set mode
            requests.post(
                f"{BASE_URL}/occp/mcl/set-mode",
                json={"mode": mode, "rationale": f"Testing {mode} mode"}
            )

            # Get allowed operations
            response = requests.get(f"{BASE_URL}/occp/mcl/allowed-operations")
            assert response.status_code == 200

            data = response.json()
            ops = data["operations"]
            assert ops["mode"] == mode, f"Expected mode {mode}, got {ops['mode']}"

            logger.info(f"  ✓ Mode '{mode}': {len(ops['allowed'])} allowed, {len(ops['forbidden'])} forbidden")

            # Verify restrictions
            if mode == "exploration":
                # Exploration should allow most operations
                assert len(ops["forbidden"]) == 0, "Exploration mode should have no forbidden operations"
            elif mode == "preservation":
                # Preservation should forbid vector.apply, goal.create, goal.decompose
                assert "vector.apply" in ops["forbidden"], "Preservation should forbid vector.apply"
                assert "goal.create" in ops["forbidden"], "Preservation should forbid goal.create"
                assert "goal.decompose" in ops["forbidden"], "Preservation should forbid goal.decompose"


class TestSKKernel(OCCPTest):
    """Test SK kernel retrieval"""
    def __init__(self):
        super().__init__("SK: Get kernel")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/sk/kernel")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "kernel" in data, "Response missing 'kernel'"

        kernel = data["kernel"]
        if kernel:
            assert "version" in kernel, "Kernel missing 'version'"
            assert "authority_level" in kernel, "Kernel missing 'authority_level'"
            assert "self_modifiable" in kernel, "Kernel missing 'self_modifiable'"
            logger.info(f"  ✓ Kernel v{kernel['version']}")
            logger.info(f"    - authority_level: {kernel['authority_level']}")
            logger.info(f"    - self_modifiable: {kernel['self_modifiable']}")
        else:
            logger.info("  ℹ No active SK kernel")


class TestSKRules(OCCPTest):
    """Test SK rules retrieval"""
    def __init__(self):
        super().__init__("SK: Get rules")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/sk/rules")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "rules" in data, "Response missing 'rules'"

        rules = data["rules"]
        logger.info(f"  ✓ Found {len(rules)} SK rules")

        # Validate rule schema
        for rule in rules:
            assert "rule_id" in rule, "Rule missing 'rule_id'"
            assert "signal_name" in rule, "Rule missing 'signal_name'"
            assert "operator" in rule, "Rule missing 'operator'"
            assert "threshold" in rule, "Rule missing 'threshold'"
            assert "actions" in rule, "Rule missing 'actions'"
            assert "explanation" in rule, "Rule missing 'explanation'"

            logger.info(f"    - {rule['rule_id']}: {rule['signal_name']} {rule['operator']} {rule['threshold']}")


class TestSKRecordSignal(OCCPTest):
    """Test SK signal recording"""
    def __init__(self):
        super().__init__("SK: Record signal")

    async def test(self):
        # Record a test signal
        response = requests.post(
            f"{BASE_URL}/occp/sk/record-signal",
            json={
                "signal_name": "mission_drift",
                "signal_value": 0.5,
                "context": {"test": True}
            }
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "signal" in data, "Response missing 'signal'"

        signal = data["signal"]
        assert "id" in signal, "Signal missing 'id'"
        assert signal["signal_name"] == "mission_drift", f"Expected 'mission_drift', got {signal['signal_name']}"
        assert signal["signal_value"] == 0.5, f"Expected 0.5, got {signal['signal_value']}"

        logger.info(f"  ✓ Signal recorded: {signal['signal_name']} = {signal['signal_value']}")


class TestSKGetSignals(OCCPTest):
    """Test SK signals retrieval"""
    def __init__(self):
        super().__init__("SK: Get signals")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/sk/signals?limit=10")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "signals" in data, "Response missing 'signals'"

        signals = data["signals"]
        logger.info(f"  ✓ Found {len(signals)} signals")

        # Validate signal schema
        for signal in signals[:3]:  # Check first 3
            assert "signal_name" in signal, "Signal missing 'signal_name'"
            assert "signal_value" in signal, "Signal missing 'signal_value'"
            assert "measured_at" in signal, "Signal missing 'measured_at'"


class TestOCCPAudit(OCCPTest):
    """Test OCCP audit log"""
    def __init__(self):
        super().__init__("OCCP: Audit log")

    async def test(self):
        # First perform some actions to generate audit events
        requests.post(
            f"{BASE_URL}/occp/mcl/set-mode",
            json={"mode": "exploration", "rationale": "Test audit logging"}
        )

        requests.post(
            f"{BASE_URL}/occp/sk/record-signal",
            json={"signal_name": "test_signal", "signal_value": 0.3}
        )

        # Get audit log
        response = requests.get(f"{BASE_URL}/occp/audit?limit=10")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "events" in data, "Response missing 'events'"

        events = data["events"]
        logger.info(f"  ✓ Found {len(events)} audit events")

        # Validate event schema
        for event in events:
            assert "source" in event, "Event missing 'source'"
            assert "decision" in event, "Event missing 'decision'"
            assert "rationale" in event, "Event missing 'rationale'"
            assert event["source"] in ["MCL", "SK"], f"Invalid source: {event['source']}"


class TestOCCPInfo(OCCPTest):
    """Test OCCP protocol info"""
    def __init__(self):
        super().__init__("OCCP: Protocol info")

    async def test(self):
        response = requests.get(f"{BASE_URL}/occp/info")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "protocol" in data, "Response missing 'protocol'"

        protocol = data["protocol"]
        assert "version" in protocol, "Protocol missing 'version'"
        assert "components" in protocol, "Protocol missing 'components'"

        logger.info(f"  ✓ OCCP Draft {protocol['version']}")
        logger.info(f"  ✓ Components: {', '.join(protocol['components'].keys())}")

        assert "mcl" in protocol["components"], "MCL not listed in components"
        assert "sk" in protocol["components"], "SK not listed in components"


class TestMCLDriftAutoTransition(OCCPTest):
    """Test MCL auto-transition to preservation on high drift"""
    def __init__(self):
        super().__init__("MCL: Auto-transition on high drift")

    async def test(self):
        # Set mode to exploration
        requests.post(
            f"{BASE_URL}/occp/mcl/set-mode",
            json={"mode": "exploration", "rationale": "Setup for drift test"}
        )

        # Update drift score to > 0.7
        response = requests.post(
            f"{BASE_URL}/occp/mcl/update-drift",
            json={"drift_score": 0.8}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "state" in data, "Response missing 'state'"

        state = data["state"]
        # Should auto-transition to preservation
        assert state["cognitive_mode"] == "preservation", f"Expected 'preservation', got {state['cognitive_mode']}"
        assert state["drift_score"] == 0.8, f"Expected drift 0.8, got {state['drift_score']}"

        logger.info(f"  ✓ Drift score 0.8 triggered auto-transition to preservation")


async def run_all_tests() -> Dict[str, List[OCCPTest]]:
    """Run all OCCP tests"""
    tests = [
        TestMCLState(),
        TestMCLSetMode(),
        TestMCLAllowedOps(),
        TestMCLModeRestrictions(),
        TestSKKernel(),
        TestSKRules(),
        TestSKRecordSignal(),
        TestSKGetSignals(),
        TestOCCPAudit(),
        TestOCCPInfo(),
        TestMCLDriftAutoTransition(),
    ]

    logger.info("=" * 60)
    logger.info("OCCP DRAFT 0.1 COMPLIANCE TEST SUITE")
    logger.info("=" * 60)

    passed = []
    failed = []

    for test in tests:
        logger.info(f"\n[TEST] {test.name}")
        try:
            result = await test.run()
            if result:
                passed.append(test)
                logger.info(f"  ✅ PASSED")
            else:
                failed.append(test)
                logger.info(f"  ❌ FAILED: {test.error}")
        except Exception as e:
            failed.append(test)
            test.error = str(e)
            logger.info(f"  ❌ FAILED: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {len(passed)}/{len(tests)} passed")
    logger.info("=" * 60)

    if failed:
        logger.info("\n❌ FAILED TESTS:")
        for test in failed:
            logger.info(f"  - {test.name}: {test.error}")

    return {"passed": passed, "failed": failed}


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())

    # Exit with error code if any tests failed
    exit(0 if not results["failed"] else 1)
