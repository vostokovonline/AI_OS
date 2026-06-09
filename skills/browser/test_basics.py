"""
Browser Skills - Basic Tests

Проверяет decision matrix без реального browser.
"""

import sys
from pathlib import Path

# Add services/core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.browser.base import BrowserAction, BrowserActionType
from skills.browser.selector import (
    select_browser_executor,
    should_use_vibium,
    get_executor_capability_matrix
)


def test_decision_matrix():
    """Тест decision matrix"""
    print("=" * 70)
    print("Test 1: Decision Matrix")
    print("=" * 70)

    # ========================================================================
    # Test 1.1: Semantic instruction → Vibium
    # ========================================================================
    action1 = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Найди кнопку входа и авторизуйся",
        url="https://app.example.com/login"
    )

    result1 = select_browser_executor(action1)
    assert result1.value == "vibium", "Semantic instruction should select Vibium"
    print("✅ Test 1.1: Semantic instruction → Vibium")

    # ========================================================================
    # Test 1.2: SaaS URL → Vibium
    # ========================================================================
    action2 = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Открой настройки",
        url="https://dashboard.saas.com/settings"
    )

    result2 = select_browser_executor(action2)
    assert result2.value == "vibium", "SaaS URL should select Vibium"
    print("✅ Test 1.2: SaaS URL → Vibium")

    # ========================================================================
    # Test 1.3: Deterministic context → Playwright
    # ========================================================================
    action3 = BrowserAction(
        type=BrowserActionType.EXTRACT,
        instruction="Extract price",
        url="https://example.com/product",
        context={"deterministic": True}
    )

    result3 = select_browser_executor(action3)
    assert result3.value == "playwright", "Deterministic context should select Playwright"
    print("✅ Test 1.3: Deterministic context → Playwright")

    # ========================================================================
    # Test 1.4: Bulk operations → Playwright
    # ========================================================================
    action4 = BrowserAction(
        type=BrowserActionType.EXTRACT,
        instruction="Extract all products",
        context={"bulk": True}
    )

    result4 = select_browser_executor(action4)
    assert result4.value == "playwright", "Bulk operations should select Playwright"
    print("✅ Test 1.4: Bulk operations → Playwright")

    # ========================================================================
    # Test 1.5: Explicit request
    # ========================================================================
    action5 = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Some instruction",
        context={"executor": "playwright"}
    )

    result5 = select_browser_executor(action5)
    assert result5.value == "playwright", "Explicit request should override"
    print("✅ Test 1.5: Explicit request → Playwright")

    # ========================================================================
    # Test 1.6: Failure history
    # ========================================================================
    action6 = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Some instruction"
    )

    result6 = select_browser_executor(
        action6,
        failure_history={"playwright": 3}  # Playwright failed 3 times
    )

    assert result6.value == "vibium", "Failure history should switch to Vibium"
    print("✅ Test 1.6: Failure history → Vibium")

    print("\n✅ All decision matrix tests passed!")


def test_should_use_vibium():
    """Тест convenience function"""
    print("\n" + "=" * 70)
    print("Test 2: Convenience Function")
    print("=" * 70)

    # Semantic → Vibium
    assert should_use_vibium("Найди кнопку") == True
    print("✅ Test 2.1: 'Найди кнопку' → Vibium")

    # Deterministic → Playwright
    assert should_use_vibium("Extract data", context={"deterministic": True}) == False
    print("✅ Test 2.2: Deterministic → Playwright")

    print("\n✅ All convenience function tests passed!")


def test_capability_matrix():
    """Тест capability matrix"""
    print("\n" + "=" * 70)
    print("Test 3: Capability Matrix")
    print("=" * 70)

    matrix = get_executor_capability_matrix()

    # Check structure
    assert "vibium" in matrix
    assert "playwright" in matrix
    print("✅ Test 3.1: Matrix has both executors")

    # Check Vibium capabilities
    vibium = matrix["vibium"]
    assert vibium["semantic_navigation"] == True
    assert vibium["deterministic"] == False
    assert vibium["ui_robust"] == True
    assert vibium["speed"] == "slow"
    print("✅ Test 3.2: Vibium capabilities correct")

    # Check Playwright capabilities
    playwright = matrix["playwright"]
    assert playwright["semantic_navigation"] == False
    assert playwright["deterministic"] == True
    assert playwright["ui_robust"] == False
    assert playwright["speed"] == "fast"
    print("✅ Test 3.3: Playwright capabilities correct")

    print("\n✅ All capability matrix tests passed!")


def test_browser_action():
    """Тест BrowserAction dataclass"""
    print("\n" + "=" * 70)
    print("Test 4: BrowserAction")
    print("=" * 70)

    action = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Test instruction",
        url="https://example.com",
        context={"key": "value"},
        timeout_ms=10000,
        screenshot=True
    )

    # Test serialization
    action_dict = action.to_dict()
    assert action_dict["type"] == "semantic"
    assert action_dict["instruction"] == "Test instruction"
    assert "context_keys" in action_dict
    print("✅ Test 4.1: BrowserAction serialization works")

    # Test validation
    try:
        invalid_action = BrowserAction(
            type=BrowserActionType.SEMANTIC,
            instruction=""  # Empty instruction
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Test 4.2: Empty instruction validation works")

    print("\n✅ All BrowserAction tests passed!")


def test_real_world_scenarios():
    """Тест реальных сценариев"""
    print("\n" + "=" * 70)
    print("Test 5: Real-World Scenarios")
    print("=" * 70)

    scenarios = [
        {
            "name": "SaaS login",
            "instruction": "Войди в аккаунт и открой dashboard",
            "url": "https://app.saas.com",
            "expected": "vibium"
        },
        {
            "name": "Mass scraping",
            "instruction": "Extract all product prices",
            "url": "https://shop.com/products",
            "context": {"bulk": True},
            "expected": "playwright"
        },
        {
            "name": "Admin panel navigation",
            "instruction": "Найди в админке настройки биллинга",
            "url": "https://admin.example.com",
            "expected": "vibium"
        },
        {
            "name": "Docs scraping",
            "instruction": "Extract documentation content",
            "url": "https://docs.example.com/api",
            "context": {"deterministic": True},
            "expected": "playwright"
        },
        {
            "name": "Complex auth",
            "instruction": "Разберись с авторизацией через OAuth",
            "context": {"auth_complex": True},
            "expected": "vibium"
        }
    ]

    for scenario in scenarios:
        action = BrowserAction(
            type=BrowserActionType.SEMANTIC,
            instruction=scenario["instruction"],
            url=scenario.get("url"),
            context=scenario.get("context", {})
        )

        result = select_browser_executor(action)
        expected = scenario["expected"]

        assert result.value == expected, \
            f"Scenario '{scenario['name']}' expected {expected}, got {result.value}"

        print(f"✅ Scenario '{scenario['name']}' → {expected}")

    print("\n✅ All real-world scenario tests passed!")


if __name__ == "__main__":
    test_decision_matrix()
    test_should_use_vibium()
    test_capability_matrix()
    test_browser_action()
    test_real_world_scenarios()

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Install Vibium: pip install vibium")
    print("   2. Install Playwright: pip install playwright && playwright install chromium")
    print("   3. Run integration example: python -m skills.browser.integration")
    print("   4. Start using in Goal Executor!")
