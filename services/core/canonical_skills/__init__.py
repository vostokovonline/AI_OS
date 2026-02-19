"""
CANONICAL SKILLS PACKAGE
Canonical Skills for AI_OS
"""
from canonical_skills.base import Skill, SkillResult, Artifact
from canonical_skills.registry import skill_registry
from canonical_skills.echo import EchoSkill
from canonical_skills.write_file import WriteFileSkill

__all__ = [
    "Skill",
    "SkillResult",
    "Artifact",
    "skill_registry",
    "EchoSkill",
    "WriteFileSkill"
]
