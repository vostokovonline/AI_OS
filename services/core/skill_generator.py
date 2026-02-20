from logging_config import get_logger
logger = get_logger(__name__)

"""
SKILL GENERATOR - Self-Writing Skills System
Генерирует навыки на основе требований используя LLM
"""
import os
import ast
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess

# LLM integration
import httpx


class SkillValidator:
    """Валидирует сгенерированный код skill"""

    def __init__(self):
        self.required_methods = ["execute", "verify"]
        self.required_attributes = ["id", "version", "description", "capabilities", "requirements", "input_schema", "output_schema", "produces_artifacts"]

    def validate(self, skill_code: str) -> tuple[bool, List[str]]:
        """
        Валидирует код skill

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # 1. Syntax check
        try:
            ast.parse(skill_code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # 2. AST parsing
        try:
            tree = ast.parse(skill_code)
        except Exception as e:
            return False, [f"Parse error: {e}"]

        # 3. Check for class definition
        class_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_found = True
                # Check class inherits from Skill
                if not any(
                    isinstance(base, ast.Name) and base.id == "Skill"
                    for base in node.bases
                ):
                    errors.append("Class must inherit from Skill")

                # Check for required methods
                method_names = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                for method in self.required_methods:
                    if method not in method_names:
                        errors.append(f"Missing required method: {method}()")

                # Check for required attributes
                assignments = [
                    n.targets[0].id if isinstance(n.targets[0], ast.Name) else None
                    for n in node.body
                    if isinstance(n, ast.Assign) or isinstance(n, ast.AnnAssign)
                ]

                break

        if not class_found:
            errors.append("No class definition found")

        # 4. Import check
        if "from canonical_skills.base import" not in skill_code:
            errors.append("Missing import: from canonical_skills.base import Skill, SkillResult, Artifact")

        # 5. Structure checks
        if "def execute(" not in skill_code:
            errors.append("execute() method not found")

        if "def verify(" not in skill_code:
            errors.append("verify() method not found")

        # 6. Check for dangerous patterns
        dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "compile(",
            "open("  # file ops should be controlled
        ]

        for pattern in dangerous_patterns:
            if pattern in skill_code and pattern not in ["def execute(", "def verify("]:
                errors.append(f"Potentially dangerous pattern: {pattern}")

        is_valid = len(errors) == 0
        return is_valid, errors


class SkillTester:
    """Тестирует сгенерированный skill"""

    def __init__(self):
        self.test_results = []

    def test_skill(self, skill_path: str) -> tuple[bool, List[str]]:
        """
        Тестирует basic functionality skill

        Returns:
            (all_passed, list_of_results)
        """
        results = []

        # 1. Import test
        try:
            # Add to path
            skill_dir = Path(skill_path).parent
            sys.path.insert(0, str(skill_dir))

            # Import module
            module_name = Path(skill_path).stem
            spec = __import__(f"canonical_skills.{module_name}", fromlist=[""])

            results.append("✅ Import: PASSED")

        except Exception as e:
            results.append(f"❌ Import: FAILED - {e}")
            return False, results

        # 2. Instantiation test
        try:
            # Find skill class
            for attr_name in dir(spec):
                attr = getattr(spec, attr_name)
                if isinstance(attr, type) and hasattr(attr, '__bases__'):
                    # Try to instantiate
                    instance = attr()
                    results.append(f"✅ Instantiation: PASSED ({attr.__name__})")

                    # Check required attributes
                    for req_attr in ["id", "version", "description"]:
                        if hasattr(instance, req_attr):
                            results.append(f"✅ Attribute {req_attr}: PASSED")
                        else:
                            results.append(f"❌ Attribute {req_attr}: MISSING")

                    break
        except Exception as e:
            results.append(f"❌ Instantiation: FAILED - {e}")

        # 3. Basic execution test
        try:
            # Test with minimal input
            result = instance.execute(
                input_data={},
                context={"goal_id": "test", "session_id": "test"}
            )
            results.append(f"✅ Execute method: PASSED (status: {result.success})")
        except Exception as e:
            results.append(f"❌ Execute method: FAILED - {e}")

        all_passed = all("PASSED" in r for r in results)
        return all_passed, results


class SkillGenerator:
    """
    Генерирует Skills на основе требований используя LLM

    Flow:
    1. Анализирует требования
    2. Генерирует код через LLM
    3. Валидирует код
    4. Тестирует skill
    5. Сохраняет файл
    6. Регистрирует в registry
    """

    def __init__(self, llm_base_url: str = None):
        self.llm_base_url = llm_base_url or os.getenv(
            "LLM_BASE_URL",
            "http://litellm:4000"
        )
        self.llm_model = os.getenv("LLM_MODEL", "smart-model")

        self.validator = SkillValidator()
        self.tester = SkillTester()

        # Use container path (not host path)
        self.skills_dir = Path("/app/canonical_skills")

    def generate_skill(
        self,
        requirements: Dict[str, Any],
        goal_context: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Генерирует skill по требованиям

        Args:
            requirements: {"artifacts": ["FILE"], "capabilities": ["write"]}
            goal_context: {"title": "...", "description": "..."}

        Returns:
            {
                "success": bool,
                "skill_path": str,
                "skill_id": str,
                "validation_errors": list,
                "test_results": list
            }
        """
        logger.info(f"\n🤖 SKILL GENERATION STARTED")
        logger.info(f"   Requirements: {requirements}")
        logger.info(f"   Goal: {goal_context.get('title', 'Unknown')}")

        # 1. Infer skill type
        skill_type = self._infer_skill_type(requirements, goal_context)
        logger.info(f"   🔮 Inferred skill type: {skill_type}")

        # 1.5. Check if skill already exists
        skill_path = self.skills_dir / f"{skill_type}.py"
        skill_id = f"core.{skill_type}"

        if skill_path.exists():
            logger.info(f"   ♻️  Skill already exists: {skill_path}")
            # Verify it's in registry
            from canonical_skills.registry import skill_registry
            existing_skill = skill_registry.get(skill_id)
            if existing_skill:
                logger.info(f"   ✅ Using existing skill: {skill_id}")
                return {
                    "success": True,
                    "skill_path": str(skill_path),
                    "skill_id": skill_id,
                    "validation_errors": [],
                    "test_results": ["Skill already exists"],
                    "all_tests_passed": True,
                    "cached": True
                }
            else:
                logger.info(f"   ⚠️  File exists but not in registry, will reload")

        # 2. Build prompt
        prompt = self._build_generation_prompt(
            skill_type=skill_type,
            requirements=requirements,
            goal_context=goal_context
        )

        # 3. Generate code with retries
        skill_code = None
        validation_errors = []

        for attempt in range(max_retries):
            logger.info(f"   📝 Generation attempt {attempt + 1}/{max_retries}")

            try:
                skill_code = self._call_llm(prompt)

                # Validate
                is_valid, errors = self.validator.validate(skill_code)

                if is_valid:
                    logger.info(f"   ✅ Validation: PASSED")
                    break
                else:
                    logger.info(f"   ❌ Validation: FAILED")
                    for error in errors:
                        logger.info(f"      - {error}")
                    validation_errors = errors

                    # Retry with feedback
                    prompt = self._build_retry_prompt(prompt, errors)

            except Exception as e:
                logger.info(f"   ❌ Generation error: {e}")
                validation_errors.append(str(e))

        if not skill_code:
            return {
                "success": False,
                "error": "Failed to generate valid code",
                "validation_errors": validation_errors
            }

        # 4. Save skill
        skill_path = self._save_skill_file(skill_type, skill_code)
        logger.info(f"   💾 Saved: {skill_path}")

        # 5. Test skill
        all_passed, test_results = self.tester.test_skill(str(skill_path))

        logger.info(f"   🧪 Tests: {sum(1 for r in test_results if 'PASSED' in r)}/{len(test_results)} passed")

        if not all_passed:
            for result in test_results:
                logger.info(f"      {result}")

        # 6. Reload registry
        self._reload_skill_registry(skill_type)

        skill_id = f"core.{skill_type}"

        logger.info(f"   ✅ SKILL GENERATED: {skill_id}")

        return {
            "success": True,
            "skill_path": str(skill_path),
            "skill_id": skill_id,
            "validation_errors": [],
            "test_results": test_results,
            "all_tests_passed": all_passed
        }

    def _infer_skill_type(self, requirements: Dict, goal_context: Dict) -> str:
        """Определяет тип skill по требованиям"""
        artifacts = requirements.get("artifacts", [])
        capabilities = requirements.get("capabilities", [])
        title = goal_context.get("title", "").lower()

        # Analyze patterns
        if "research" in title or "web" in title or "search" in title:
            return "web_research"

        elif "summarize" in title or "condense" in title or "summary" in title:
            return "summarize_knowledge"

        elif "structured" in title or "plan" in title or "json" in title:
            return "structured_generation"

        elif "verify" in title or "check" in title or "validation" in title:
            return "self_check"

        elif "write" in title or "create" in title or "file" in title:
            return "write_file"

        elif "FILE" in artifacts:
            return "write_file"

        elif "KNOWLEDGE" in artifacts and "FILE" in artifacts:
            return "web_research"

        else:
            return "generic_skill"

    def _build_generation_prompt(
        self,
        skill_type: str,
        requirements: Dict,
        goal_context: Dict
    ) -> str:
        """Строит промпт для генерации skill"""

        return f"""You are an expert Python developer specializing in AI_OS skills.

TASK: Generate a Python Skill class for AI_OS system

REQUIREMENTS:
- Skill must follow canonical interface (see canonical_skills/base.py)
- Must inherit from Skill base class
- Must implement execute() and verify() methods
- Must return SkillResult with artifacts

SKILL DETAILS:
- Skill Name: {skill_type.replace('_', ' ').title()}Skill
- Skill ID: "core.{skill_type}"
- Version: "1.0"
- Artifacts to produce: {requirements.get('artifacts', [])}
- Capabilities: {requirements.get('capabilities', [])}

GOAL CONTEXT:
- Title: {goal_context.get('title', '')}
- Description: {goal_context.get('description', '')}

CODE TEMPLATE (follow this structure):

```python
from canonical_skills.base import Skill, SkillResult, Artifact

class {skill_type.replace('_', ' ').title()}Skill(Skill):
    # Metadata
    id = "core.{skill_type}"
    version = "1.0"
    description = "..."
    capabilities = [...]
    requirements = [...]

    input_schema = {{
        "type": "object",
        "properties": {{
            ...
        }}
    }}

    output_schema = {{
        "type": "object",
        "properties": {{
            ...
        }}
    }}

    produces_artifacts = [...]

    def execute(self, input_data: dict, context: dict) -> SkillResult:
        # Implementation here
        ...

        return self._success_result(output, [artifact])

    def verify(self, result: SkillResult) -> bool:
        # Verification here
        # IMPORTANT: Use result.success (NOT result.status)
        # IMPORTANT: Use artifact.type (NOT artifact.type_)
        # Check result.success is True
        # Check result.artifacts exists
        # Check artifact.type == "FILE" or "KNOWLEDGE"
        # Check artifact.content is not empty
        return True/False
```

CRITICAL REMINDERS:
1. result.success (boolean) - NOT result.status
2. artifact.type (string) - NOT artifact.type_
3. artifact.content (any) - the actual content

IMPORTANT:
1. Code must be syntactically correct
2. All methods must be implemented
3. Return self._success_result() or self._error_result()
4. Create artifacts using self._artifact(type_="FILE", content=..., metadata={{...}})
5. Do NOT use external libraries unless necessary (prefer built-in: json, os, pathlib, re)
6. Handle errors gracefully
7. NO dangerous operations (eval, exec, etc.)

EXAMPLE OF CORRECT VERIFICATION:
```python
def verify(self, result: SkillResult) -> bool:
    if not result.success:  # Use .success NOT .status
        return False
    if not result.artifacts:
        return False
    for artifact in result.artifacts:
        if artifact.type not in ['FILE', 'KNOWLEDGE']:  # Use .type NOT .type_
            return False
        if not artifact.content:
            return False
    return True
```

EXAMPLE OF CORRECT ARTIFACT CREATION:
```python
# CORRECT
artifact = self._artifact(
    type_="FILE",
    content="Some text content",
    metadata={{
        "source": "MySkill",
        "goal_id": context.get("goal_id")
    }}
)

# WRONG - do not do this
artifact = Artifact(id="...", type="...", content="...")
```

Generate ONLY the Python code, no explanations.
"""

    def _build_retry_prompt(self, original_prompt: str, errors: List[str]) -> str:
        """Строит промпт для повторной генерации с учётом ошибок"""
        return f"""{original_prompt}

PREVIOUS ATTEMPT HAD ERRORS:
{chr(10).join(f'- {e}' for e in errors)}

Please fix these errors and regenerate the code.
Ensure all validation checks pass.
"""

    def _call_llm(self, prompt: str) -> str:
        """Вызывает LLM для генерации кода"""
        try:
            with httpx.Client(timeout=300.0) as client:  # 5 min timeout for LLM skill generation
                response = client.post(
                    f"{self.llm_base_url}/chat/completions",
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Python developer. Generate clean, correct, production-ready code. Return ONLY the code, no explanations."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4000
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Extract code from markdown if present
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                return content

        except Exception as e:
            raise Exception(f"LLM call failed: {e}")

    def _save_skill_file(self, skill_type: str, code: str) -> Path:
        """Сохраняет skill в файл"""
        filename = f"{skill_type}.py"
        filepath = self.skills_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        return filepath

    def _reload_skill_registry(self, skill_type: str):
        """Перезагружает registry с новым skill"""
        try:
            import importlib
            module_name = f"canonical_skills.{skill_type}"

            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                __import__(module_name)

            # Re-import skill class
            from canonical_skills.registry import skill_registry

            # Find and register new skill
            module = sys.modules[module_name]
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, '__bases__'):
                    try:
                        instance = attr()
                        if hasattr(instance, 'execute') and hasattr(instance, 'verify'):
                            skill_registry.register(instance)
                            logger.info(f"   ✅ Skill registered in registry")
                            break
                    except:
                        pass

        except Exception as e:
            logger.info(f"   ⚠️  Registry reload: {e}")


# Global instance
skill_generator = SkillGenerator()
