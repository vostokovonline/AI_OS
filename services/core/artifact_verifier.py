"""
ARTIFACT VERIFIER - v1
Code-based verification of artifacts (NOT LLM-based)

Key principle: Code decides, not LLM suggestions
"""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class VerificationResult:
    """Результат одной проверки"""
    def __init__(self, name: str, passed: bool, details: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.details = details

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details
        }


class ArtifactVerifier:
    """
    Верификатор артефактов - CODE-BASED проверки

    Verification types v1:
    - file_exists: Файл существует
    - file_not_empty: Файл не пустой
    - min_length: Минимальная длина контента
    - json_valid: JSON валиден по схеме
    - markdown_not_empty: Markdown содержит контент
    - has_references: Есть ссылки/источники
    """

    ARTIFACT_TYPES = ["FILE", "KNOWLEDGE", "DATASET", "REPORT", "LINK", "EXECUTION_LOG"]

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.getenv("ARTIFACTS_PATH", "/data/artifacts")
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def verify(self, artifact_data: Dict) -> List[VerificationResult]:
        """
        Верифицирует артефакт

        Args:
            artifact_data: {
                "type": "FILE|KNOWLEDGE|...",
                "content_kind": "file|db|vector|external",
                "content_location": "path or URL",
                ...
            }

        Returns:
            Список результатов проверок
        """
        artifact_type = artifact_data.get("type", "FILE")
        content_kind = artifact_data.get("content_kind", "file")
        content_location = artifact_data.get("content_location", "")

        results = []

        # Базовая проверка: тип артефакта валиден
        results.append(self._verify_type(artifact_type))

        if content_kind == "file":
            # Файловые проверки
            results.extend(self._verify_file_artifact(artifact_type, content_location))
        elif content_kind == "external":
            # Внешние проверки (URL, repo)
            results.extend(self._verify_external_artifact(artifact_type, content_location))
        elif content_kind == "db":
            # DB проверки
            results.extend(self._verify_db_artifact(artifact_type, content_location))
        elif content_kind == "vector":
            # Vector DB проверки
            results.extend(self._verify_vector_artifact(artifact_type, content_location))

        return results

    def _verify_type(self, artifact_type: str) -> VerificationResult:
        """Проверяет валидность типа артефакта"""
        if artifact_type in self.ARTIFACT_TYPES:
            return VerificationResult("valid_type", True, f"Type {artifact_type} is valid")
        else:
            return VerificationResult("valid_type", False, f"Unknown type: {artifact_type}")

    def _verify_file_artifact(self, artifact_type: str, location: str) -> List[VerificationResult]:
        """Проверяет файловый артефакт"""
        results = []
        full_path = os.path.join(self.base_path, location)

        # 1. Файл существует
        if os.path.exists(full_path):
            results.append(VerificationResult("file_exists", True, f"File exists: {full_path}"))
        else:
            results.append(VerificationResult("file_exists", False, f"File not found: {full_path}"))
            return results  # Если файла нет - дальше нет смысла проверять

        # 2. Файл не пустой
        file_size = os.path.getsize(full_path)
        if file_size > 0:
            results.append(VerificationResult("file_not_empty", True, f"File size: {file_size} bytes"))
        else:
            results.append(VerificationResult("file_not_empty", False, "File is empty"))

        # 3. Специфичные проверки по типу
        if artifact_type == "FILE":
            results.extend(self._verify_generic_file(full_path))
        elif artifact_type == "KNOWLEDGE":
            results.extend(self._verify_knowledge_file(full_path))
        elif artifact_type == "DATASET":
            results.extend(self._verify_dataset_file(full_path))
        elif artifact_type == "REPORT":
            results.extend(self._verify_report_file(full_path))

        return results

    def _verify_generic_file(self, path: str) -> List[VerificationResult]:
        """Проверка обычного файла"""
        results = []

        # Читаем файл
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Минимальная длина (не менее 10 символов)
            if len(content) >= 10:
                results.append(VerificationResult("min_length", True, f"Content length: {len(content)}"))
            else:
                results.append(VerificationResult("min_length", False, f"Content too short: {len(content)}"))

            # Определяем тип по расширению
            if path.endswith('.json'):
                results.extend(self._verify_json_content(content))
            elif path.endswith('.md'):
                results.extend(self._verify_markdown_content(content))

        except Exception as e:
            results.append(VerificationResult("readable", False, f"Cannot read file: {e}"))

        return results

    def _verify_json_content(self, content: str) -> List[VerificationResult]:
        """Проверяет JSON контент"""
        results = []

        try:
            data = json.loads(content)
            results.append(VerificationResult("json_valid", True, "JSON is valid"))
        except json.JSONDecodeError as e:
            results.append(VerificationResult("json_valid", False, f"JSON invalid: {e}"))

        return results

    def _verify_markdown_content(self, content: str) -> List[VerificationResult]:
        """Проверяет Markdown контент"""
        results = []

        # Не пустой после удаления пробелов
        stripped = content.strip()
        if len(stripped) > 20:
            results.append(VerificationResult("markdown_not_empty", True, "Markdown has content"))
        else:
            results.append(VerificationResult("markdown_not_empty", False, "Markdown too short"))

        return results

    def _verify_knowledge_file(self, path: str) -> List[VerificationResult]:
        """Проверка knowledge файла"""
        results = []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Минимум 100 символов для knowledge chunk
            if len(content) >= 100:
                results.append(VerificationResult("min_knowledge_length", True, f"Length: {len(content)}"))
            else:
                results.append(VerificationResult("min_knowledge_length", False, f"Too short: {len(content)}"))

        except Exception as e:
            results.append(VerificationResult("knowledge_readable", False, str(e)))

        return results

    def _verify_dataset_file(self, path: str) -> List[VerificationResult]:
        """Проверка dataset файла"""
        results = []

        try:
            import pandas as pd

            # Пытаемся прочитать как CSV
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                results.append(VerificationResult("csv_readable", True, f"Rows: {len(df)}, Columns: {len(df.columns)}"))

                # Минимум 1 строка + header
                if len(df) >= 1:
                    results.append(VerificationResult("dataset_not_empty", True, f"Dataset has {len(df)} rows"))
                else:
                    results.append(VerificationResult("dataset_not_empty", False, "Dataset is empty"))

        except Exception as e:
            results.append(VerificationResult("dataset_valid", False, str(e)))

        return results

    def _verify_report_file(self, path: str) -> List[VerificationResult]:
        """Проверка report файла"""
        results = []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Минимум 200 символов для отчета
            if len(content) >= 200:
                results.append(VerificationResult("report_min_length", True, f"Length: {len(content)}"))
            else:
                results.append(VerificationResult("report_min_length", False, f"Report too short: {len(content)}"))

        except Exception as e:
            results.append(VerificationResult("report_readable", False, str(e)))

        return results

    def _verify_external_artifact(self, artifact_type: str, location: str) -> List[VerificationResult]:
        """Проверяет внешний артефакт (URL, repo)"""
        results = []

        # Проверка валидности URL
        if location.startswith('http://') or location.startswith('https://'):
            results.append(VerificationResult("url_format", True, "Valid URL format"))
        elif location.startswith('git@') or location.startswith('https://github.com/') or location.startswith('https://gitlab.com/'):
            results.append(VerificationResult("repo_format", True, "Valid repo format"))
        else:
            results.append(VerificationResult("external_format", False, "Invalid external format"))

        return results

    def _verify_db_artifact(self, artifact_type: str, location: str) -> List[VerificationResult]:
        """Проверяет DB артефакт"""
        results = []

        # Базовая проверка: location не пустой
        if location:
            results.append(VerificationResult("db_reference_exists", True, f"DB ref: {location}"))
        else:
            results.append(VerificationResult("db_reference_exists", False, "Empty DB reference"))

        return results

    def _verify_vector_artifact(self, artifact_type: str, location: str) -> List[VerificationResult]:
        """Проверяет vector DB артефакт"""
        results = []

        # Базовая проверка: location не пустой (vector ID)
        if location:
            results.append(VerificationResult("vector_id_exists", True, f"Vector ID: {location}"))
        else:
            results.append(VerificationResult("vector_id_exists", False, "Empty vector ID"))

        return results

    def get_overall_status(self, verification_results: List[VerificationResult]) -> str:
        """
        Возвращает общий статус верификации

        Returns:
            "passed" - все проверки прошли
            "failed" - есть критические ошибки
            "partial" - есть некритические предупреждения
        """
        if not verification_results:
            return "failed"

        # Критические проверки (должны пройти обязательно)
        critical_checks = ["file_exists", "valid_type", "json_valid", "csv_readable"]
        critical_results = [r for r in verification_results if r.name in critical_checks]

        # Если хоть одна критическая не прошла - failed
        if any(not r.passed for r in critical_results):
            return "failed"

        # Если все прошли - passed
        if all(r.passed for r in verification_results):
            return "passed"

        # Иначе - partial (есть warnings)
        return "partial"


# Глобальный экземпляр
artifact_verifier = ArtifactVerifier()
