"""
SMOKE TESTS - Module Import Validation

These tests verify that all core modules can be imported without errors.
They catch issues like:
- Missing imports
- Undefined variables (logger not defined)
- Syntax errors
- Circular imports

If these fail, DO NOT DEPLOY.
"""
import pytest
import sys
import os

# Add app directory to path (modules are in /app/ in container)
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)


class TestModuleImports:
    """Verify all critical modules can be imported."""

    def test_import_main(self):
        """Main FastAPI app must import without errors."""
        import main
        assert hasattr(main, 'app')

    def test_import_models(self):
        """Database models must import."""
        import models
        assert hasattr(models, 'Goal')
        assert hasattr(models, 'Artifact')

    def test_import_schemas(self):
        """Pydantic schemas must import."""
        import schemas
        # GoalRequest is in main.py, check other schemas
        assert hasattr(schemas, 'EmotionalSignals') or hasattr(schemas, 'MessageCreate')

    def test_import_database(self):
        """Database connection must import."""
        import database
        assert hasattr(database, 'AsyncSessionLocal')

    def test_import_llm_fallback(self):
        """LLM fallback system must import."""
        import llm_fallback
        assert hasattr(llm_fallback, 'llm_fallback')
        assert hasattr(llm_fallback, 'LLMFallbackManager')

    def test_import_emotional_layer(self):
        """Emotional layer must import (catches 'logger not defined')."""
        import emotional_layer
        assert hasattr(emotional_layer, 'emotional_layer')

    def test_import_emotional_helpers(self):
        """Emotional helpers must import."""
        import emotional_helpers
        assert hasattr(emotional_helpers, 'collect_emotional_signals')

    def test_import_artifact_verifier(self):
        """Artifact verifier must import."""
        import artifact_verifier
        assert hasattr(artifact_verifier, 'ArtifactVerifier')
        assert hasattr(artifact_verifier, 'artifact_verifier')

    def test_import_artifact_registry(self):
        """Artifact registry must import."""
        import artifact_registry
        assert hasattr(artifact_registry, 'ArtifactRegistry')

    def test_import_goal_executor(self):
        """Goal executor must import."""
        import goal_executor
        assert hasattr(goal_executor, 'GoalExecutor')

    def test_import_goal_decomposer(self):
        """Goal decomposer must import."""
        import goal_decomposer
        assert hasattr(goal_decomposer, 'goal_decomposer')

    def test_import_goal_transition_service(self):
        """Goal transition service must import."""
        import goal_transition_service
        assert hasattr(goal_transition_service, 'transition_service')

    def test_import_agent_graph(self):
        """Agent graph must import."""
        import agent_graph
        assert hasattr(agent_graph, 'app_graph')

    def test_import_infrastructure_uow(self):
        """Unit of Work infrastructure must import."""
        from infrastructure.uow import UnitOfWork, GoalRepository
        assert UnitOfWork is not None
        assert GoalRepository is not None

    def test_import_legacy_policy(self):
        """Legacy policy must import."""
        from policies.legacy_policy import LegacyPolicy
        assert LegacyPolicy is not None


class TestLoggingConfig:
    """Verify logging is properly configured."""

    def test_logging_config_imports(self):
        """Logging config must be available."""
        from logging_config import get_logger, log_goal_transition
        assert get_logger is not None
        assert log_goal_transition is not None

    def test_logger_works(self):
        """Logger must be callable."""
        from logging_config import get_logger
        logger = get_logger(__name__)
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
