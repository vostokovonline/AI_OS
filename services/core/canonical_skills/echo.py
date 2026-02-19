"""
ECHO SKILL - Simplest MVP Skill for Testing
Echo input text as artifact - идеально для smoke-test системы
"""
import os
from canonical_skills.base import Skill, SkillResult, Artifact


class EchoSkill(Skill):
    """
    Простейший навык для тестирования

    Вход: текст
    Выход: артефакт с тем же текстом
    Назначение: smoke-test Goal → Skill → Artifact → Verification
    """

    # ---- REQUIRED METADATA ----
    id = "core.echo"
    version = "1.0"
    description = "Echo input text as artifact - simplest skill for testing"

    capabilities = ["echo", "text-output", "test"]
    requirements = []

    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo back"
            }
        },
        "required": ["text"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "echoed": {
                "type": "string",
                "description": "The text that was echoed"
            },
            "length": {
                "type": "integer",
                "description": "Length of echoed text"
            }
        }
    }

    produces_artifacts = ["FILE"]

    # ---- EXECUTION ----
    def execute(self, input_data: dict, context: dict) -> SkillResult:
        """
        Echo input text as artifact

        Args:
            input_data: {"text": "some text"}
            context: {"goal_id": "...", "session_id": "...", ...}

        Returns:
            SkillResult with text artifact
        """
        # Validate input
        text = input_data.get("text")
        if not text:
            return self._error_result("No text provided in input")

        try:
            # Create artifact with echoed text
            artifact = self._artifact(
                type_="FILE",
                content=text,
                metadata={
                    "source": "EchoSkill",
                    "goal_id": context.get("goal_id"),
                    "length": len(text)
                }
            )

            # Prepare output
            output = {
                "echoed": text,
                "length": len(text)
            }

            return self._success_result(output, [artifact])

        except Exception as e:
            return self._error_result(f"Echo execution failed: {str(e)}")

    # ---- VERIFICATION ----
    def verify(self, result: SkillResult) -> bool:
        """
        Verify echo result

        Checks:
        1. Result is successful
        2. Has exactly 1 artifact
        3. Artifact content matches echoed text
        """
        # Must be successful
        if not result.success:
            return False

        # Must have artifacts
        if not result.artifacts:
            return False

        # Must have exactly 1 artifact
        if len(result.artifacts) != 1:
            return False

        # Artifact content must match output
        artifact = result.artifacts[0]
        if artifact.content != result.output.get("echoed"):
            return False

        # Metadata must be present
        if "length" not in artifact.metadata:
            return False

        return True
