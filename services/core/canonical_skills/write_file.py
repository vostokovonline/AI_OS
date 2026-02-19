"""
WRITE FILE SKILL - Write text content to file
Записывает текстовый контент в файл - базовый навык производства
"""
import os
from pathlib import Path
from canonical_skills.base import Skill, SkillResult, Artifact


class WriteFileSkill(Skill):
    """
    Записывает текст в файл

    Вход: text + filename
    Выход: FILE artifact
    Назначение: базовое производство артефактов
    """

    # ---- REQUIRED METADATA ----
    id = "core.write_file"
    version = "1.0"
    description = "Write text content to file - basic production skill"

    capabilities = ["write", "file-production", "text-output"]
    requirements = ["filesystem"]

    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Content to write to file"
            },
            "filename": {
                "type": "string",
                "description": "Name of the file to create"
            },
            "directory": {
                "type": "string",
                "description": "Directory to write to (optional, default: from ARTIFACTS_PATH env)"
            }
        },
        "required": ["text", "filename"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Full path to created file"
            },
            "size": {
                "type": "integer",
                "description": "Size of file in bytes"
            }
        }
    }

    produces_artifacts = ["file", "text"]

    # ---- EXECUTION ----
    def execute(self, input_data: dict, context: dict) -> SkillResult:
        """
        Write text content to file

        Args:
            input_data: {"text": "...", "filename": "...", "directory": "..."}
            context: {"goal_id": "...", ...}

        Returns:
            SkillResult with file artifact
        """
        # Validate input
        text = input_data.get("text", "")
        filename = input_data.get("filename")
        directory = input_data.get("directory") or os.getenv("ARTIFACTS_PATH", "/data/artifacts")

        if not filename:
            return self._error_result("filename is required")

        if not text:
            return self._error_result("text is required")

        try:
            # Create directory if not exists
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)

            # Full file path
            file_path = dir_path / filename

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Get file size
            file_size = file_path.stat().st_size

            # Create artifact
            artifact = self._artifact(
                type_="file",
                content=str(file_path),
                metadata={
                    "filename": filename,
                    "size": file_size,
                    "encoding": "utf-8",
                    "goal_id": context.get("goal_id"),
                    "content_kind": "file"
                }
            )

            # Prepare output
            output = {
                "file_path": str(file_path),
                "size": file_size
            }

            return self._success_result(output, [artifact])

        except Exception as e:
            return self._error_result(f"Failed to write file: {str(e)}")

    # ---- VERIFICATION ----
    def verify(self, result: SkillResult) -> bool:
        """
        Verify file was written successfully

        Checks:
        1. Result is successful
        2. Has 1 artifact
        3. File exists
        4. File has content
        """
        if not result.success:
            return False

        if not result.artifacts or len(result.artifacts) != 1:
            return False

        artifact = result.artifacts[0]
        file_path = artifact.content

        # Check file exists
        if not Path(file_path).exists():
            return False

        # Check file has content
        if Path(file_path).stat().st_size == 0:
            return False

        # Check metadata
        if "size" not in artifact.metadata:
            return False

        return True
