"""
Temporal Metrics - EMA, variance, trend, confidence evolution

CRITICAL: Without temporal state, bandit will just "count wins" without 
understanding evolution, stability, or confidence.
"""
import json
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class TemporalMetric:
    """
    Temporal metric with EMA, variance, trend, and confidence.
    
    Unlike stateless snapshots, this tracks evolution over time.
    """
    metric_name: str
    current_value: float = 0.0
    
    # EMA tracking
    ema: float = 0.0
    ema_alpha: float = 0.1  # Smoothing factor
    
    # Variance tracking
    variance: float = 0.0
    sample_count: int = 0
    
    # Trend detection
    trend: str = "stable"  # stable, improving, degrading
    trend_samples: int = 5
    
    # Confidence
    confidence: float = 0.0  # Based on sample count and variance
    
    # Timestamps
    last_updated: str = ""
    first_seen: str = ""
    
    def update(self, new_value: float) -> "TemporalMetric":
        """Update metric with new value"""
        now = datetime.utcnow().isoformat()
        
        if self.sample_count == 0:
            self.first_seen = now
            self.ema = new_value
            self.current_value = new_value
            self.last_updated = now
            self.sample_count = 1
            self.confidence = 0.1  # Low confidence initially
            return self
        
        # Update EMA
        self.ema = self.ema_alpha * new_value + (1 - self.ema_alpha) * self.ema
        
        # Update variance (Welford's algorithm)
        self.sample_count += 1
        delta = new_value - self.ema
        self.variance = ((self.sample_count - 1) * self.variance + delta * delta) / self.sample_count
        
        # Update current value
        self.current_value = new_value
        self.last_updated = now
        
        # Update trend
        self._update_trend()
        
        # Update confidence
        self._update_confidence()
        
        return self
    
    def _update_trend(self):
        """Detect trend based on recent values"""
        # Simple trend: if current > ema → improving
        if self.current_value > self.ema + 0.1:
            self.trend = "improving"
        elif self.current_value < self.ema - 0.1:
            self.trend = "degrading"
        else:
            self.trend = "stable"
    
    def _update_confidence(self):
        """Compute confidence based on sample count and variance"""
        # Confidence increases with samples, decreases with variance
        sample_factor = min(1.0, self.sample_count / 50)  # Max out at 50 samples
        variance_factor = max(0, 1 - (self.variance * 4))  # Variance > 0.25 → 0 confidence
        
        self.confidence = sample_factor * variance_factor
    
    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "ema": self.ema,
            "variance": self.variance,
            "trend": self.trend,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "first_seen": self.first_seen
        }
    
    @staticmethod
    def from_dict(data: dict) -> "TemporalMetric":
        metric = TemporalMetric(metric_name=data["metric_name"])
        metric.current_value = data.get("current_value", 0)
        metric.ema = data.get("ema", 0)
        metric.variance = data.get("variance", 0)
        metric.trend = data.get("trend", "stable")
        metric.confidence = data.get("confidence", 0)
        metric.sample_count = data.get("sample_count", 0)
        metric.last_updated = data.get("last_updated", "")
        metric.first_seen = data.get("first_seen", "")
        return metric


class TemporalMetricsStore:
    """
    Store for temporal metrics with persistence.
    
    Tracks evolution of key metrics over time.
    """
    
    def __init__(self, store_dir: str = "/app/temporal_metrics"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._metrics: Dict[str, TemporalMetric] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    metric = TemporalMetric.from_dict(data)
                    self._metrics[metric.metric_name] = metric
            except:
                continue
    
    def _save_metric(self, metric: TemporalMetric):
        """Save metric to disk"""
        filename = self.store_dir / f"{metric.metric_name}.json"
        with open(filename, "w") as f:
            json.dump(metric.to_dict(), f, indent=2)
    
    def get_metric(self, name: str) -> TemporalMetric:
        """Get or create metric"""
        if name not in self._metrics:
            self._metrics[name] = TemporalMetric(metric_name=name)
        return self._metrics[name]
    
    def record(self, name: str, value: float) -> TemporalMetric:
        """Record value for metric"""
        metric = self.get_metric(name)
        metric.update(value)
        self._save_metric(metric)
        return metric
    
    def get_all_metrics(self) -> Dict[str, TemporalMetric]:
        """Get all metrics"""
        return self._metrics
    
    def get_statistics(self) -> Dict:
        """Get metrics statistics"""
        return {
            "total_metrics": len(self._metrics),
            "metrics": {
                name: {
                    "current": m.current_value,
                    "ema": m.ema,
                    "variance": m.variance,
                    "trend": m.trend,
                    "confidence": m.confidence,
                    "samples": m.sample_count
                }
                for name, m in self._metrics.items()
            }
        }


# Global store
_temporal_store: Optional[TemporalMetricsStore] = None


def get_temporal_store() -> TemporalMetricsStore:
    """Get temporal metrics store"""
    global _temporal_store
    if _temporal_store is None:
        _temporal_store = TemporalMetricsStore()
    return _temporal_store


def record_temporal_metric(name: str, value: float) -> TemporalMetric:
    """Convenience function to record metric"""
    return get_temporal_store().record(name, value)