"""
ARTIFACT VERIFICATION ENGINE v1
Executes verification rules defined in skill manifests

Key principle:
- Verification is CODE, not YAML ideas
- Rules are EXECUTED, not just described
- LLM cannot override verification results
"""
import os
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class VerificationEngine:
    """
    Движок верификации артефактов

    Выполняет правила из манифестов:
    - min_sources: sources_count >= 3
    - min_length: len(content) > 500
    - has_citations: "## References" in content
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.getenv("ARTIFACTS_PATH", "/data/artifacts")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def verify(
        self,
        artifact: Dict,
        rules: List[Dict]
    ) -> List[Dict]:
        """
        Верифицирует артефакт по списку правил

        Args:
            artifact: {
                "type": "FILE",
                "content_kind": "file",
                "content_location": "path/to/file.md",
                ...
            }
            rules: [
                {"name": "min_sources", "rule": "sources_count >= 3"},
                {"name": "min_length", "rule": "len(content) > 500"}
            ]

        Returns:
            [
                {"name": "min_sources", "passed": true, "details": "..."},
                {"name": "min_length", "passed": false, "details": "..."}
            ]
        """
        results = []

        # Загружаем контент артефакта
        artifact_content = self._load_artifact_content(artifact)

        for rule in rules:
            rule_name = rule.get("name", "unknown")
            rule_expression = rule.get("rule", "")

            try:
                # Выполняем правило
                passed, details = self._execute_rule(
                    rule_name,
                    rule_expression,
                    artifact,
                    artifact_content
                )

                results.append({
                    "name": rule_name,
                    "passed": passed,
                    "details": details
                })

            except Exception as e:
                # При ошибке выполнения правила - считаем что не пройдено
                results.append({
                    "name": rule_name,
                    "passed": False,
                    "details": f"Rule execution error: {str(e)}"
                })

        return results

    def _load_artifact_content(self, artifact: Dict) -> Dict:
        """
        Загружает контент артефакта для верификации

        Returns:
            {
                "text": "...",
                "data": {...},
                "sources_count": 3,
                "file_exists": True,
                "file_size": 1234
            }
        """
        content_kind = artifact.get("content_kind", "file")
        location = artifact.get("content_location", "")

        result = {
            "file_exists": False,
            "file_size": 0,
            "text": "",
            "data": {},
            "sources_count": 0
        }

        if content_kind == "file":
            # Файловый артефакт
            file_path = self.base_path / location

            if file_path.exists():
                result["file_exists"] = True
                result["file_size"] = file_path.stat().st_size

                # Читаем контент
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        result["text"] = text

                        # Извлекаем метаданные
                        result["sources_count"] = self._count_sources(text)
                        result["has_citations"] = self._has_citations(text)
                        result["line_count"] = len(text.split('\n'))
                        result["word_count"] = len(text.split())

                except Exception as e:
                    result["error"] = str(e)

        elif content_kind == "vector_db":
            # Vector DB артефакт
            result["vector_id"] = location
            result["file_exists"] = True  # Вектор существует если ID не пустой

        elif content_kind == "external":
            # Внешний артефакт
            result["url"] = location
            result["file_exists"] = location.startswith("http")

        return result

    def _execute_rule(
        self,
        rule_name: str,
        rule_expression: str,
        artifact: Dict,
        content: Dict
    ) -> tuple[bool, str]:
        """
        Выполняет одно правило верификации

        Args:
            rule_name: Имя правила
            rule_expression: Выражение правила (например, "sources_count >= 3")
            artifact: Артефакт
            content: Загруженный контент артефакта

        Returns:
            (passed, details)
        """
        # Создаем контекст для выполнения правила
        context = {
            "artifact": artifact,
            "content": content,
            "sources_count": content.get("sources_count", 0),
            "file_size": content.get("file_size", 0),
            "line_count": content.get("line_count", 0),
            "word_count": content.get("word_count", 0),
            "text": content.get("text", ""),
        }

        # Специальные правила
        if rule_name == "min_sources":
            return self._check_min_sources(context, rule_expression)

        elif rule_name == "min_length":
            return self._check_min_length(context, rule_expression)

        elif rule_name == "citations_present":
            return self._check_citations_present(context, rule_expression)

        elif rule_name == "file_exists":
            return self._check_file_exists(context, rule_expression)

        elif rule_name == "non_empty":
            return self._check_non_empty(context, rule_expression)

        # Общее правило - выполняем как Python expression
        else:
            try:
                # Безопасное выполнение выражения
                passed = eval(rule_expression, {"__builtins__": {}}, context)
                details = f"Rule: {rule_expression}"
                return passed, details
            except Exception as e:
                return False, f"Rule evaluation error: {e}"

    def _check_min_sources(self, context: Dict, rule_expression: str) -> tuple[bool, str]:
        """Проверяет минимальное количество источников"""
        # Парсим выражение: "sources_count >= 3"
        match = re.match(r'sources_count\s*>=\s*(\d+)', rule_expression)
        if match:
            min_sources = int(match.group(1))
            actual = context.get("sources_count", 0)
            passed = actual >= min_sources
            details = f"Sources: {actual}/{min_sources}"
            return passed, details
        return False, "Invalid rule format"

    def _check_min_length(self, context: Dict, rule_expression: str) -> tuple[bool, str]:
        """Проверяет минимальную длину контента"""
        # Парсим выражение: "len(summary) > 300" или "len(content) > 500"
        match = re.match(r'len\((\w+)\)\s*[>]\s*(\d+)', rule_expression)
        if match:
            field = match.group(1)
            min_length = int(match.group(2))

            if field == "summary":
                # Извлекаем summary из текста (первые N символов)
                text = context.get("text", "")
                # Предполагаем что summary в начале
                summary = text[:500]  # Первые 500 символов
                actual = len(summary)
            else:
                actual = context.get("word_count", 0)

            passed = actual >= min_length
            details = f"Length: {actual}/{min_length}"
            return passed, details
        return False, "Invalid rule format"

    def _check_citations_present(self, context: Dict, rule_expression: str) -> tuple[bool, str]:
        """Проверяет наличие цитирований"""
        text = context.get("text", "")
        has_citations = self._has_citations(text)
        passed = has_citations
        details = f"Citations: {'present' if has_citations else 'not found'}"
        return passed, details

    def _check_file_exists(self, context: Dict, rule_expression: str) -> tuple[bool, str]:
        """Проверяет что файл существует"""
        passed = context.get("file_exists", False)
        details = f"File exists: {passed}"
        return passed, details

    def _check_non_empty(self, context: Dict, rule_expression: str) -> tuple[bool, str]:
        """Проверяет что контент не пустой"""
        text = context.get("text", "")
        actual_length = len(text.strip())
        passed = actual_length > 0
        details = f"Content length: {actual_length} chars"
        return passed, details

    def _count_sources(self, text: str) -> int:
        """Подсчитывает количество источников в markdown"""
        # Ищем ссылки в формате markdown: [Title](URL)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
        return len(links)

    def _has_citations(self, text: str) -> bool:
        """Проверяет наличие цитирований"""
        # Ищем либо ссылки, либо секцию References
        has_links = bool(re.search(r'\[([^\]]+)\]\(([^)]+)\)', text))
        has_references = bool(re.search(r'##?\s*References?', text, re.IGNORECASE))
        return has_links or has_references


# ============= USAGE =============

class SkillVerificationEngine:
    """
    Движок верификации для навыков

    Интегрирует верификацию с навыками
    """

    def __init__(self):
        self.engine = VerificationEngine()

    def verify_skill_result(
        self,
        skill_name: str,
        skill_result: Dict,
        manifest: Dict
    ) -> List[Dict]:
        """
        Верифицирует результаты выполнения навыка

        Args:
            skill_name: Имя навыка
            skill_result: SkillResult
            manifest: Манифест навыка

        Returns:
            Результаты верификации для каждого артефакта
        """
        all_results = []

        # Получаем правила из манифеста
        verification_rules = manifest.get("verification", [])

        # Верифицируем каждый артефакт
        for artifact_data in skill_result.get("artifacts", []):
            artifact_results = self.engine.verify(
                artifact=artifact_data,
                rules=verification_rules
            )

            all_results.extend(artifact_results)

        return all_results


# Глобальные экземпляры
verification_engine = VerificationEngine()
skill_verification_engine = SkillVerificationEngine()


# ============= EXAMPLE =============

def example_verification():
    """Пример верификации"""
    engine = VerificationEngine()

    artifact = {
        "type": "FILE",
        "content_kind": "file",
        "content_location": "research.md"
    }

    rules = [
        {"name": "min_sources", "rule": "sources_count >= 3"},
        {"name": "min_length", "rule": "len(content) > 500"},
        {"name": "citations_present", "rule": "has_citations == true"}
    ]

    results = engine.verify(artifact, rules)

    print("Verification Results:")
    for result in results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {result['name']}: {result['details']}")
