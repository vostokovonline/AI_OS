"""
Skill Registry - Single source of truth for skill resolution

All skill access MUST go through this registry.
No string-based skill IDs - only ResolvedSkill objects.
"""
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Any
from datetime import datetime


@dataclass(frozen=True)
class ResolvedSkill:
    """Immutable resolved skill object"""
    canonical_id: str
    version: str
    capabilities: List[str]
    metadata: Dict[str, Any]
    registered_at: str


class SkillRegistryError(Exception):
    """Raised when skill cannot be resolved"""
    pass


class SkillRegistry:
    """
    Single source of truth for skill resolution.
    
    Usage:
        registry = SkillRegistry()
        registry.register("core.echo", executor=echo_handler, version="v1")
        
        resolved = registry.resolve("core.echo")
        # Returns ResolvedSkill or raises SkillRegistryError
    """
    
    def __init__(self):
        self._skills: Dict[str, ResolvedSkill] = {}
        self._executors: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical
    
    def register(
        self,
        skill_id: str,
        executor: Callable,
        version: str = "v1",
        capabilities: List[str] = None,
        metadata: Dict[str, Any] = None,
        aliases: List[str] = None
    ) -> ResolvedSkill:
        """
        Register a skill in the registry.
        Must be called before skill can be resolved.
        """
        if not skill_id.startswith("core."):
            raise ValueError(f"Skill ID must start with 'core.': {skill_id}")
        
        resolved = ResolvedSkill(
            canonical_id=skill_id,
            version=version,
            capabilities=capabilities or [],
            metadata=metadata or {},
            registered_at=datetime.utcnow().isoformat()
        )
        
        self._skills[skill_id] = resolved
        self._executors[skill_id] = executor
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = skill_id
        
        return resolved
    
    def resolve(self, skill_id: str) -> ResolvedSkill:
        """
        Resolve a skill ID to ResolvedSkill.
        
        Raises:
            SkillRegistryError: If skill not registered
        """
        # Check alias first
        if skill_id in self._aliases:
            skill_id = self._aliases[skill_id]
        
        # Validate format
        if not skill_id.startswith("core."):
            raise SkillRegistryError(f"Invalid skill ID format (must start with 'core.'): {skill_id}")
        
        # Check if registered
        if skill_id not in self._skills:
            raise SkillRegistryError(f"Skill not registered: {skill_id}")
        
        return self._skills[skill_id]
    
    def resolve_or_raise(self, skill_id: str) -> ResolvedSkill:
        """Same as resolve() but re-raises as RuntimeError for cleaner handling"""
        try:
            return self.resolve(skill_id)
        except SkillRegistryError as e:
            raise RuntimeError(f"Skill resolution failed: {e}")
    
    def get_executor(self, skill_id: str) -> Callable:
        """Get executor function for skill"""
        resolved = self.resolve(skill_id)
        return self._executors[resolved.canonical_id]
    
    def is_registered(self, skill_id: str) -> bool:
        """Check if skill is registered"""
        if skill_id in self._aliases:
            return True
        return skill_id in self._skills
    
    def list_skills(self) -> List[ResolvedSkill]:
        """List all registered skills"""
        return list(self._skills.values())
    
    def get_stats(self) -> Dict:
        """Get registry statistics"""
        return {
            "total_skills": len(self._skills),
            "total_aliases": len(self._aliases),
            "skills": [s.canonical_id for s in self._skills.values()]
        }


# Global registry
_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get or create global skill registry"""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry


def resolve_skill(skill_id: str) -> ResolvedSkill:
    """Convenience function for skill resolution"""
    return get_skill_registry().resolve(skill_id)