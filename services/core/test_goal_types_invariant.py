"""
Goal Type Invariant Test v3.1

КРИТИЧЕСКИЙ ИНВАРИАНТ:
В базе данных НЕ должно быть goal_type без контракта.

Это защита от тихой деградации системы.
"""
from sqlalchemy import select, func
from models import Goal
from database import AsyncSessionLocal
from goal_contract_validator import GoalContractValidator


class TestGoalTypeInvariant:
    """Проверка инварианта: все goal_type в БД имеют контракты"""

    async def test_no_goal_types_without_contracts(self):
        """
        ИНВАРИАНТ #1: В БД нет goal_type без контракта

        Если этот тест падает → система в опасном состоянии
        Кто-то создал цель с типом, для которого нет контракта.
        Такая цель будет вечно в pending.
        """
        async with AsyncSessionLocal() as db:
            # Получаем все уникальные goal_type из БД
            result = await db.execute(
                select(Goal.goal_type).distinct()
            )
            db_types = set([row[0] for row in result])

            # Получаем все типы с контрактами
            contract_types = set([
                "achievable",
                "continuous",
                "directional",
                "exploratory",
                "meta"
            ])

            # Проверяем что все типы из БД имеют контракты
            types_without_contracts = db_types - contract_types

            assert len(types_without_contracts) == 0, (
                f"CRITICAL: Found goal_type without contract: {types_without_contracts}\n"
                f"Database types: {db_types}\n"
                f"Contract types: {contract_types}\n"
                f"Это опасно! Цели с этими типами будут вечно в pending.\n"
                f"Либо добавьте контракт, либо мигрируйте цели в другой тип."
            )

    async def test_no_fallback_contract_behavior(self):
        """
        ИНВАРИАНТ #2: create_default_contract падает на неизвестном типе

        Проверяем что система больше не использует fallback к 'achievable'.
        """
        # Пытаемся создать контракт для несуществующего типа
        try:
            GoalContractValidator.create_default_contract("nonexistent_type")
            raise AssertionError("Should have raised ValueError for unknown type")
        except ValueError as e:
            assert "has no contract defined" in str(e)
            assert "Refusing to create goal" in str(e)

    
    async def test_canonical_five_types_only(self):
        """
        ИНВАРИАНТ #3: В БД только канонические 5 типов

        v3.1 canonical types:
        - directional (ценности)
        - achievable (конечные цели)
        - continuous (бесконечное улучшение)
        - exploratory (исследование)
        - meta (саморефлексия)
        """
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Goal.goal_type, func.count(Goal.id))
                .group_by(Goal.goal_type)
            )
            db_types = [(row[0], row[1]) for row in result]

            canonical_types = {
                "directional",
                "achievable",
                "continuous",
                "exploratory",
                "meta"
            }

            for goal_type, count in db_types:
                assert goal_type in canonical_types, (
                    f"Non-canonical goal_type found: {goal_type}\n"
                    f"Count: {count} goals\n"
                    f"Allowed types: {canonical_types}"
                )

    
    async def test_no_bounded_or_philosophical(self):
        """
        ИНВАРИАНТ #4: Устаревшие типы удалены из БД

        bounded и philosophical были мигрированы в v3.1:
        - bounded → achievable
        - philosophical → directional
        """
        async with AsyncSessionLocal() as db:
            # Проверяем что bounded нет
            bounded_result = await db.execute(
                select(func.count(Goal.id)).where(Goal.goal_type == "bounded")
            )
            bounded_count = bounded_result.scalar()

            # Проверяем что philosophical нет
            philosophical_result = await db.execute(
                select(func.count(Goal.id)).where(Goal.goal_type == "philosophical")
            )
            philosophical_count = philosophical_result.scalar()

            assert bounded_count == 0, (
                f"Found {bounded_count} goals with type 'bounded'. "
                f"Этот тип должен быть мигрирован в 'achievable' или 'continuous'."
            )

            assert philosophical_count == 0, (
                f"Found {philosophical_count} goals with type 'philosophical'. "
                f"Этот тип должен быть мигрирован в 'directional'."
            )

    def test_all_five_contracts_defined(self):
        """
        ИНВАРИАНТ #5: Контракты определены для всех 5 канонических типов

        Каждый тип должен иметь:
        - allowed_actions
        - forbidden
        - max_depth
        - max_subgoals
        - evaluation_mode
        - timeout_seconds
        - resource_limits
        """
        canonical_types = ["directional", "achievable", "continuous", "exploratory", "meta"]

        required_fields = [
            "allowed_actions",
            "forbidden",
            "max_depth",
            "max_subgoals",
            "evaluation_mode",
            "timeout_seconds",
            "resource_limits"
        ]

        for goal_type in canonical_types:
            contract = GoalContractValidator.create_default_contract(goal_type)

            for field in required_fields:
                assert field in contract, (
                    f"Contract for '{goal_type}' missing field: {field}"
                )

    def test_contracts_are_strict(self):
        """
        ИНВАРИАНТ #6: Контракты действительно запрещают опасные операции

        - directional НЕ может execute (ценности нельзя выполнить)
        - continuous (non-mission) НЕ может decompose (бесконечное улучшение уже декомпозировано)
        - meta НЕ может execute напрямую (мета-цели требуют мутацию)
        """
        # directional: NO execute
        directional = GoalContractValidator.create_default_contract("directional")
        assert "execute" not in directional["allowed_actions"], (
            "directional goals cannot be executed (they are values, not tasks)"
        )
        assert "execute" in directional["forbidden"], (
            "directional must explicitly forbid execute"
        )

        # continuous (depth > 0): NO decompose
        # Mission-level continuous (depth=0) can decompose, but non-mission cannot
        continuous = GoalContractValidator.create_default_contract("continuous", depth_level=1)
        assert "decompose" not in continuous["allowed_actions"], (
            "non-mission continuous goals cannot be decomposed (already decomposed)"
        )
        assert "decompose" in continuous["forbidden"], (
            "non-mission continuous must explicitly forbid decompose"
        )

        # meta: NO direct execute
        meta = GoalContractValidator.create_default_contract("meta")
        assert "execute" not in meta["allowed_actions"], (
            "meta goals cannot be executed directly (require mutation)"
        )
        assert "execute" in meta["forbidden"], (
            "meta must explicitly forbid execute"
        )


if __name__ == "__main__":
    # Быстрая проверка без pytest
    import asyncio

    async def run_invariant_tests():
        test = TestGoalTypeInvariant()

        logger.info("🔍 Running Goal Type Invariant Tests...")
        logger.info("=" * 60)

        tests = [
            ("No goal types without contracts", test.test_no_goal_types_without_contracts),
            ("No fallback contract behavior", test.test_no_fallback_contract_behavior),
            ("Canonical five types only", test.test_canonical_five_types_only),
            ("No bounded or philosophical", test.test_no_bounded_or_philosophical),
            ("All five contracts defined", test.test_all_five_contracts_defined),
            ("Contracts are strict", test.test_contracts_are_strict),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                logger.info(f"✅ {name}")
                passed += 1
            except Exception as e:
                logger.info(f"❌ {name}")
                logger.info(f"   {e}")
                failed += 1

        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{len(tests)} passed")

        if failed > 0:
            logger.info("⚠️  SYSTEM IN DANGEROUS STATE")
            return 1
        else:
            logger.info("✅ ALL INVARIANTS HOLD")
            return 0

    exit(asyncio.run(run_invariant_tests()))
