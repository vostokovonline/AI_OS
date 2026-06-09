"""Strategy Evolution - Self-improvement through strategy mutation"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """A strategy for achieving goals"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    rules: List[str] = field(default_factory=list)  # Rule-based conditions
    priority: float = 0.5
    success_rate: float = 0.5
    usage_count: int = 0
    last_used: Optional[datetime] = None
    performance_history: List[float] = field(default_factory=list)


@dataclass
class StrategyMutation:
    """A mutation operation on a strategy"""
    strategy_id: str
    operation: str  # add_rule, remove_rule, modify_priority, crossover
    description: str
    expected_impact: float = 0


class StrategyEvolution:
    """
    Strategy Evolution - generates and evolves strategies.
    
    Operations:
    1. Mutation - random changes to existing strategies
    2. Crossover - combine two strategies
    3. Selection - keep best performing
    4. Generation - create new strategies from scratch
    """
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.generation: int = 0
        self._load_builtin_strategies()
        logger.info("strategy_evolution_initialized")
    
    def _load_builtin_strategies(self) -> None:
        """Load initial strategies"""
        builtin = [
            Strategy(
                name="sequential_execution",
                description="Execute one goal at a time, fully complete before next",
                rules=["focus_on_single_goal", "no_parallel_execution"],
                priority=0.8,
                success_rate=0.75
            ),
            Strategy(
                name="parallel_decomposition",
                description="Decompose into subgoals and execute in parallel",
                rules=["decompose_fully", "parallel_execution", "aggregate_results"],
                priority=0.6,
                success_rate=0.65
            ),
            Strategy(
                name="conservative_approach",
                description="Take safe, proven path with low risk",
                rules=["minimize_complexity", "use_familiar_tools", "gradual_progress"],
                priority=0.7,
                success_rate=0.8
            ),
            Strategy(
                name="aggressive_exploration",
                description="Try new approaches, explore creative solutions",
                rules=["novel_combinations", "accept_higher_risk", "learn_from_failures"],
                priority=0.4,
                success_rate=0.5
            ),
        ]
        for s in builtin:
            self.strategies[s.id] = s
    
    def mutate(self, strategy_id: str) -> Optional[Strategy]:
        """Mutate an existing strategy"""
        if strategy_id not in self.strategies:
            return None
        
        original = self.strategies[strategy_id]
        mutated = Strategy(
            name=f"{original.name}_v{self.generation}",
            description=original.description,
            rules=original.rules.copy(),
            priority=original.priority,
            success_rate=original.success_rate,
        )
        
        operation = random.choice(["add_rule", "remove_rule", "modify_priority"])
        
        if operation == "add_rule":
            new_rules = ["seek_feedback", "break_into_parts", "use_simpler_approach",
                        "add_verification", "document_progress", "anticipate_complications"]
            available = [r for r in new_rules if r not in mutated.rules]
            if available:
                mutated.rules.append(random.choice(available))
        
        elif operation == "remove_rule" and mutated.rules:
            mutated.rules.pop(random.randint(0, len(mutated.rules) - 1))
        
        elif operation == "modify_priority":
            delta = random.uniform(-0.2, 0.2)
            mutated.priority = max(0.1, min(0.9, original.priority + delta))
        
        self.strategies[mutated.id] = mutated
        self.generation += 1
        
        logger.info("strategy_mutated", original=original.name, new=mutated.name)
        return mutated
    
    def crossover(self, strategy_id1: str, strategy_id2: str) -> Optional[Strategy]:
        """Combine two strategies into one"""
        if strategy_id1 not in self.strategies or strategy_id2 not in self.strategies:
            return None
        
        s1 = self.strategies[strategy_id1]
        s2 = self.strategies[strategy_id2]
        
        child = Strategy(
            name=f"{s1.name}+{s2.name}",
            description=f"Hybrid: {s1.description} + {s2.description}",
            rules=list(set(s1.rules + s2.rules))[:5],  # Max 5 rules
            priority=(s1.priority + s2.priority) / 2,
            success_rate=(s1.success_rate + s2.success_rate) / 2,
        )
        
        self.strategies[child.id] = child
        self.generation += 1
        
        logger.info("strategy_crossover", s1=s1.name, s2=s2.name, child=child.name)
        return child
    
    def select_best(self, limit: int = 3) -> List[Strategy]:
        """Select top performing strategies"""
        sorted_strategies = sorted(
            self.strategies.values(),
            key=lambda s: s.success_rate * s.priority,
            reverse=True
        )
        return sorted_strategies[:limit]
    
    def update_performance(self, strategy_id: str, score: float) -> None:
        """Update strategy performance after execution"""
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        strategy.usage_count += 1
        strategy.last_used = datetime.utcnow()
        
        history = strategy.performance_history
        history.append(score)
        
        # Keep last 20 scores
        if len(history) > 20:
            history = history[-20:]
        
        # Rolling success rate
        strategy.success_rate = sum(1 for s in history if s >= 0.6) / len(history)
        strategy.performance_history = history
        
        logger.debug("strategy_performance_updated", 
                    name=strategy.name, 
                    score=score,
                    new_rate=strategy.success_rate)
    
    def generate_recommendation(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategy recommendations based on context"""
        recommendations = []
        
        for strategy in self.select_best(5):
            score = strategy.success_rate * strategy.priority
            
            # Adjust based on context
            if context.get("complexity", 0.5) > 0.7:
                if "break_into_parts" in strategy.rules:
                    score *= 1.2
            
            if context.get("urgency", 0.5) > 0.7:
                if "sequential_execution" in strategy.name or "focus_on_single_goal" in strategy.rules:
                    score *= 0.8
            
            if context.get("novelty", 0.5) > 0.7:
                if "novel_combinations" in strategy.rules:
                    score *= 1.3
            
            recommendations.append({
                "strategy_id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "rules": strategy.rules,
                "score": score,
                "confidence": strategy.success_rate,
            })
        
        return sorted(recommendations, key=lambda r: r["score"], reverse=True)
    
    def evolve_generation(self) -> List[Strategy]:
        """Run one evolution cycle"""
        new_strategies = []
        
        # Mutate top 2
        top = self.select_best(2)
        for s in top:
            mutated = self.mutate(s.id)
            if mutated:
                new_strategies.append(mutated)
        
        # Crossover top 2
        if len(top) >= 2:
            crossed = self.crossover(top[0].id, top[1].id)
            if crossed:
                new_strategies.append(crossed)
        
        logger.info("evolution_cycle", generation=self.generation, new_count=len(new_strategies))
        return new_strategies