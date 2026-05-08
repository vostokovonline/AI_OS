"""
Dataset Extractor - Converts JSONL traces to normalized learning samples

Extracts from: /app/decision_traces/trace_YYYYMMDD.jsonl
Outputs: LearningSample with bounded rewards [-1, 1]
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .trajectory_dataset import (
    LearningSample, TraceContext, TraceSchemaValidator, TRACE_SCHEMA_VERSION
)


class RewardNormalizer:
    """
    Composite reward calculation with bounded range [-1, 1]
    
    reward = success_reward - latency_penalty - retry_penalty + artifact_bonus
    """
    
    # Configurable weights
    SUCCESS_REWARD = 1.0
    LATENCY_PENALTY_THRESHOLD_MS = 30000  # 30 seconds
    MAX_LATENCY_PENALTY = 0.3
    RETRY_PENALTY_PER_RETRY = 0.1
    ARTIFACT_BONUS_PER_ARTIFACT = 0.05
    MAX_ARTIFACT_BONUS = 0.2
    
    @staticmethod
    def normalize(
        success: bool,
        latency_ms: int,
        retry_count: int,
        artifacts_count: int,
        error: Optional[str] = None
    ) -> float:
        """
        Calculate composite bounded reward in range [-1, 1]
        """
        # Base reward from success/failure
        if success:
            reward = RewardNormalizer.SUCCESS_REWARD
        else:
            reward = -1.0
        
        # Latency penalty (normalized to max 0.3)
        if latency_ms > 0:
            latency_ratio = min(latency_ms / RewardNormalizer.LATENCY_PENALTY_THRESHOLD_MS, 1.0)
            latency_penalty = latency_ratio * RewardNormalizer.MAX_LATENCY_PENALTY
            reward -= latency_penalty
        
        # Retry penalty
        retry_penalty = min(retry_count * RewardNormalizer.RETRY_PENALTY_PER_RETRY, 0.5)
        reward -= retry_penalty
        
        # Artifact quality bonus
        if success and artifacts_count > 0:
            artifact_bonus = min(
                artifacts_count * RewardNormalizer.ARTIFACT_BONUS_PER_ARTIFACT,
                RewardNormalizer.MAX_ARTIFACT_BONUS
            )
            reward += artifact_bonus
        
        # If error occurred, additional penalty
        if error:
            reward -= 0.2
        
        # Bound to [-1, 1]
        return max(-1.0, min(1.0, reward))


class TrajectoryDatasetExtractor:
    """
    Extracts normalized learning samples from JSONL trace files.
    """
    
    def __init__(self, traces_dir: str = "/app/decision_traces"):
        self.traces_dir = Path(traces_dir)
        self.validator = TraceSchemaValidator()
    
    def extract_sample(self, trace: dict) -> Optional[LearningSample]:
        """
        Convert single trace to LearningSample.
        Returns None if trace is incomplete or invalid.
        """
        try:
            # Skip if incomplete (no completion yet)
            if not trace.get("completed_at"):
                return None
            
            # Validate schema
            self.validator.validate(trace)
            
            # Extract context
            ctx_data = trace.get("context", {})
            context = TraceContext(
                goal_type=ctx_data.get("goal_type", "unknown"),
                goal_length=ctx_data.get("goal_length", 0),
                domain=ctx_data.get("domain", "unknown"),
                input_tokens=ctx_data.get("input_tokens", 0),
                output_tokens=ctx_data.get("output_tokens", 0),
                latency_ms=ctx_data.get("latency_ms", 0.0),
                attempt=ctx_data.get("attempt", 1),
                previous_failures=ctx_data.get("previous_failures", 0),
                depth_level=ctx_data.get("depth_level", 0),
                has_subgoals=ctx_data.get("has_subgoals", False),
                completion_criteria_exists=ctx_data.get("completion_criteria_exists", False),
                constraints_count=ctx_data.get("constraints_count", 0),
                execution_mode=ctx_data.get("execution_mode", "auto"),
                goal_description=ctx_data.get("goal_description", ""),
                candidates=ctx_data.get("candidates", []),
                planner_depth=ctx_data.get("planner_depth", 0),
                retry_count=ctx_data.get("retry_count", 0)
            )
            
            # Get candidates from trace
            candidates = trace.get("candidates", [])
            if not candidates and "legacy_choice" in trace:
                # If only one choice, use legacy as the chosen one
                candidates = [trace.get("legacy_choice", "unknown")]
            
            # Determine chosen skill
            chosen_skill = trace.get("legacy_choice", "unknown")
            if not chosen_skill and candidates:
                chosen_skill = candidates[0]
            
            # Calculate normalized reward
            reward = RewardNormalizer.normalize(
                success=trace.get("success", False),
                latency_ms=trace.get("latency_ms", 0),
                retry_count=ctx_data.get("retry_count", 0),
                artifacts_count=trace.get("artifacts_count", 0),
                error=trace.get("error")
            )
            
            return LearningSample(
                trace_id=trace.get("trace_id", ""),
                goal_id=trace.get("goal_id", ""),
                context=context,
                candidates=candidates,
                chosen_skill=chosen_skill,
                reward=reward,
                latency_ms=trace.get("latency_ms", 0),
                success=trace.get("success", False),
                timestamp=trace.get("completed_at", "")
            )
            
        except ValueError as e:
            print(f"[DATASET_EXTRACTOR] Schema validation failed: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[DATASET_EXTRACTOR] Extract failed: {e}", flush=True)
            return None
    
    def extract_from_file(self, filepath: Path) -> List[LearningSample]:
        """
        Extract all valid samples from a single JSONL file.
        Merges start and completion records by trace_id.
        """
        samples = []
        trace_starts = {}  # trace_id -> start record
        trace_completions = {}  # trace_id -> completion record
        
        if not filepath.exists():
            return samples
        
        # First pass: collect starts and completions
        with open(filepath, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    trace_id = record.get("trace_id", "")
                    
                    if record.get("completed_at"):
                        # This is a completion record
                        trace_completions[trace_id] = record
                    else:
                        # This is a start record
                        trace_starts[trace_id] = record
                except json.JSONDecodeError:
                    continue
        
        # Second pass: merge and extract
        for trace_id, start in trace_starts.items():
            completion = trace_completions.get(trace_id)
            
            if not completion:
                # Skip incomplete traces
                continue
            
            # Merge records
            merged = {**start, **completion}
            
            try:
                sample = self.extract_sample(merged)
                if sample:
                    samples.append(sample)
            except Exception as e:
                print(f"[DATASET_EXTRACTOR] Failed to extract {trace_id}: {e}", flush=True)
        
        return samples
    
    def extract_all(self, date_pattern: str = "*") -> List[LearningSample]:
        """
        Extract samples from all matching trace files.
        
        Args:
            date_pattern: Glob pattern for filenames (default: all)
        """
        samples = []
        
        if not self.traces_dir.exists():
            print(f"[DATASET_EXTRACTOR] Traces dir not found: {self.traces_dir}", flush=True)
            return samples
        
        for filepath in self.traces_dir.glob(f"trace_{date_pattern}.jsonl"):
            file_samples = self.extract_from_file(filepath)
            samples.extend(file_samples)
            print(f"[DATASET_EXTRACTOR] Extracted {len(file_samples)} from {filepath.name}", flush=True)
        
        return samples
    
    def get_dataset_stats(self, samples: List[LearningSample]) -> Dict[str, Any]:
        """
        Get statistics about extracted dataset.
        """
        if not samples:
            return {"total_samples": 0}
        
        success_count = sum(1 for s in samples if s.success)
        rewards = [s.reward for s in samples]
        
        return {
            "total_samples": len(samples),
            "success_rate": success_count / len(samples),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "unique_skills": len(set(s.chosen_skill for s in samples)),
            "unique_goals": len(set(s.goal_id for s in samples))
        }