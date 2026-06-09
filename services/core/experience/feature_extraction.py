"""
Immutable Feature Extraction - Deterministic feature extraction for replay

CRITICAL: Feature extraction must be a PURE FUNCTION with:
- feature_schema_version
- deterministic canonicalization
- normalized ordering
- stable hashing

This is the foundation for deterministic replay.
"""
import json
import hashlib
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


FEATURE_SCHEMA_VERSION = "v1.0"


@dataclass(frozen=True)
class CanonicalFeaturePayload:
    """
    Immutable canonical feature payload - used for hash/comparison.
    This is the deterministic, replay-safe part.
    """
    schema_version: str
    goal_type: str
    domain: str
    goal_complexity: float
    input_size: int
    requires_network: bool
    requires_filesystem: bool
    candidate_count: int
    
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "goal_type": self.goal_type,
            "domain": self.domain,
            "goal_complexity": self.goal_complexity,
            "input_size": self.input_size,
            "requires_network": self.requires_network,
            "requires_filesystem": self.requires_filesystem,
            "candidate_count": self.candidate_count
        }
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of canonical payload"""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    @staticmethod
    def from_dict(data: dict) -> "CanonicalFeaturePayload":
        return CanonicalFeaturePayload(
            schema_version=data["schema_version"],
            goal_type=data["goal_type"],
            domain=data["domain"],
            goal_complexity=data["goal_complexity"],
            input_size=data["input_size"],
            requires_network=data["requires_network"],
            requires_filesystem=data["requires_filesystem"],
            candidate_count=data["candidate_count"]
        )


@dataclass(frozen=True)
class FeatureExtractionMetadata:
    """Non-deterministic metadata - NOT used for hashing"""
    extracted_at: str
    source_execution_id: str
    extraction_latency_ms: float


@dataclass(frozen=True)
class FeatureVector:
    """
    Immutable feature vector for policy decision.
    
    Separates:
    - CanonicalFeaturePayload: deterministic, hashable, replay-safe
    - FeatureExtractionMetadata: runtime info, not for comparison
    """
    canonical: CanonicalFeaturePayload
    metadata: FeatureExtractionMetadata
    feature_hash: str  # Hash of canonical payload only
    
    @property
    def schema_version(self) -> str:
        return self.canonical.schema_version
    
    @property
    def goal_type(self) -> str:
        return self.canonical.goal_type
    
    @property
    def domain(self) -> str:
        return self.canonical.domain
    
    @property
    def goal_complexity(self) -> float:
        return self.canonical.goal_complexity
    
    @property
    def input_size(self) -> int:
        return self.canonical.input_size
    
    @property
    def requires_network(self) -> bool:
        return self.canonical.requires_network
    
    @property
    def requires_filesystem(self) -> bool:
        return self.canonical.requires_filesystem
    
    @property
    def candidate_count(self) -> int:
        return self.canonical.candidate_count
    
    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical.to_dict(),
            "metadata": {
                "extracted_at": self.metadata.extracted_at,
                "source_execution_id": self.metadata.source_execution_id,
                "extraction_latency_ms": self.metadata.extraction_latency_ms
            },
            "feature_hash": self.feature_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "FeatureVector":
        return FeatureVector(
            canonical=CanonicalFeaturePayload.from_dict(data["canonical"]),
            metadata=FeatureExtractionMetadata(
                extracted_at=data["metadata"]["extracted_at"],
                source_execution_id=data["metadata"]["source_execution_id"],
                extraction_latency_ms=data["metadata"]["extraction_latency_ms"]
            ),
            feature_hash=data["feature_hash"]
        )


class FeatureExtractor:
    """
    Pure function for deterministic feature extraction.
    
    Given the same inputs, ALWAYS produces the same output.
    No side effects, no random values, no external state.
    """
    
    @staticmethod
    def extract(
        goal_description: str,
        goal_type: str,
        domain: str,
        inputs: Dict[str, Any],
        candidate_skills: List[str]
    ) -> FeatureVector:
        """
        Extract deterministic feature vector from goal context.
        
        This is a PURE FUNCTION - no side effects, no external state.
        """
        # Canonicalize goal type
        canonical_goal_type = goal_type.lower().strip() if goal_type else "achievable"
        
        # Canonicalize domain
        canonical_domain = domain.lower().strip() if domain else "general"
        
        # Compute complexity from description length
        description_len = len(goal_description or "")
        goal_complexity = min(1.0, description_len / 1000.0)  # Normalize to [0, 1]
        
        # Input size (count, not values - deterministic)
        input_size = len(inputs) if inputs else 0
        
        # Requirements detection (deterministic keyword matching)
        requires_network = any(kw in (goal_description or "").lower() 
                               for kw in ["web", "http", "search", "api", "fetch", "url"])
        requires_filesystem = any(kw in (goal_description or "").lower()
                                  for kw in ["file", "write", "read", "directory", "folder"])
        
        # Candidate count
        candidate_count = len(candidate_skills) if candidate_skills else 0
        
        # Create canonical payload (deterministic, hashable)
        canonical = CanonicalFeaturePayload(
            schema_version=FEATURE_SCHEMA_VERSION,
            goal_type=canonical_goal_type,
            domain=canonical_domain,
            goal_complexity=round(goal_complexity, 4),
            input_size=input_size,
            requires_network=requires_network,
            requires_filesystem=requires_filesystem,
            candidate_count=candidate_count
        )
        
        # Compute stable hash from canonical payload only
        feature_hash = canonical.compute_hash()
        
        # Create metadata (non-deterministic, NOT for hashing)
        metadata = FeatureExtractionMetadata(
            extracted_at=datetime.utcnow().isoformat(),
            source_execution_id="",
            extraction_latency_ms=0.0
        )
        
        return FeatureVector(
            canonical=canonical,
            metadata=metadata,
            feature_hash=feature_hash
        )
    
    @staticmethod
    def extract_from_envelope(envelope: "ExecutionEnvelope") -> FeatureVector:
        """Extract features from execution envelope"""
        context = envelope.context_features or {}
        
        result = FeatureExtractor.extract(
            goal_description=context.get("goal_description", ""),
            goal_type=context.get("goal_type", envelope.goal_type),
            domain=context.get("domain", envelope.domain),
            inputs={"keys": context.get("input_keys", [])},  # No raw inputs
            candidate_skills=envelope.candidate_skill_ids or []
        )
        
        # Update metadata with envelope info
        from dataclasses import replace
        result = replace(
            result,
            metadata=FeatureExtractionMetadata(
                extracted_at=result.metadata.extracted_at,
                source_execution_id=envelope.execution_id,
                extraction_latency_ms=0.0
            )
        )
        
        return result


# Backward compatibility alias
def extract_features(
    goal_description: str,
    goal_type: str,
    domain: str,
    inputs: Dict[str, Any],
    candidate_skills: List[str]
) -> FeatureVector:
    """Convenience function"""
    return FeatureExtractor.extract(
        goal_description, goal_type, domain, inputs, candidate_skills
    )