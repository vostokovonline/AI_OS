"""
Enforcement Configuration for Execution Envelope Migration

Controls how strictly we enforce envelope-based execution.

Modes:
    warn: Log violations but allow execution (observability first)
    quarantine: Legacy executions don't train policy (data separation)
    hard_fail: Raw execution path completely blocked
"""
import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class EnforcementMode(str, Enum):
    WARN = "warn"
    QUARANTINE = "quarantine"
    HARD_FAIL = "hard_fail"


@dataclass(frozen=True)
class EnforcementConfig:
    """Immutable enforcement configuration"""
    mode: EnforcementMode
    quarantine_dir: str
    metrics_dir: str
    legacy_policy_version: str
    
    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "EnforcementConfig":
        """Load from environment with optional overrides"""
        mode_str = os.environ.get("ENFORCEMENT_MODE", "warn").lower()
        
        try:
            mode = EnforcementMode(mode_str)
        except ValueError:
            mode = EnforcementMode.WARN
        
        config = cls(
            mode=mode,
            quarantine_dir=os.environ.get(
                "ENFORCEMENT_QUARANTINE_DIR",
                "/app/decision_traces_invalid"
            ),
            metrics_dir=os.environ.get(
                "ENFORCEMENT_METRICS_DIR",
                "/app/enforcement_metrics"
            ),
            legacy_policy_version="legacy_v1"
        )
        
        if overrides:
            return cls(
                mode=overrides.get("mode", config.mode),
                quarantine_dir=overrides.get("quarantine_dir", config.quarantine_dir),
                metrics_dir=overrides.get("metrics_dir", config.metrics_dir),
                legacy_policy_version=overrides.get("legacy_policy_version", config.legacy_policy_version)
            )
        
        return config
    
    def is_envelope_required(self) -> bool:
        """Returns True if envelope is mandatory"""
        return self.mode == EnforcementMode.HARD_FAIL
    
    def can_train_policy(self, is_envelope_based: bool) -> bool:
        """Returns True if execution can train policy"""
        if self.mode == EnforcementMode.WARN:
            return True  # All executions train policy
        elif self.mode == EnforcementMode.QUARANTINE:
            return is_envelope_based  # Only envelope-based trains
        else:
            return True  # Hard fail means we only get envelope-based


class EnforcementMetrics:
    """Tracks enforcement compliance metrics"""
    
    def __init__(self, config: EnforcementConfig):
        self.config = config
        self._counts = {
            "envelope_executions": 0,
            "legacy_executions": 0,
            "violations_warn": 0,
            "violations_quarantine": 0,
            "violations_rejected": 0,
        }
        self._by_mode = {m.value: 0 for m in EnforcementMode}
    
    def record_envelope_execution(self):
        """Record successful envelope-based execution"""
        self._counts["envelope_executions"] += 1
    
    def record_legacy_execution(self):
        """Record legacy (non-envelope) execution"""
        self._counts["legacy_executions"] += 1
    
    def record_violation(self, mode: EnforcementMode):
        """Record enforcement violation"""
        key = f"violations_{mode.value}"
        if key in self._counts:
            self._counts[key] += 1
    
    def get_summary(self) -> dict:
        """Get metrics summary"""
        return {
            **self._counts,
            "mode": self.config.mode.value,
            "total_executions": self._counts["envelope_executions"] + self._counts["legacy_executions"],
            "compliance_rate": (
                self._counts["envelope_executions"] / 
                max(1, self._counts["envelope_executions"] + self._counts["legacy_executions"])
            )
        }
    
    def save(self):
        """Persist metrics to disk"""
        import json
        from pathlib import Path
        
        path = Path(self.config.metrics_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        filename = path / "enforcement_metrics.json"
        with open(filename, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


# Global config
_enforcement_config: Optional[EnforcementConfig] = None
_enforcement_metrics: Optional[EnforcementMetrics] = None


def get_enforcement_config() -> EnforcementConfig:
    """Get or create global enforcement config"""
    global _enforcement_config
    if _enforcement_config is None:
        _enforcement_config = EnforcementConfig.from_env()
    return _enforcement_config


def get_enforcement_metrics() -> EnforcementMetrics:
    """Get or create global enforcement metrics"""
    global _enforcement_metrics
    if _enforcement_metrics is None:
        _enforcement_metrics = EnforcementMetrics(get_enforcement_config())
    return _enforcement_metrics


def reload_enforcement_config(overrides: Optional[dict] = None):
    """Reload enforcement config (useful for testing)"""
    global _enforcement_config, _enforcement_metrics
    _enforcement_config = EnforcementConfig.from_env(overrides)
    _enforcement_metrics = EnforcementMetrics(_enforcement_config)