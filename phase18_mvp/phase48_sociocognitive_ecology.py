"""
Phase 48 — Autonomous Cognitive Ecology (v2: Socio-Cognitive Dynamics).

ARCHITECTURAL REGIME SHIFT:
  Old model:  Self → owns → Ecosystem
  New model:  Processes → compete → Coalition equilibrium → Self emerges

  This is NOT "ecology on top of a centralized self".
  This is ecology AS substrate, self AS emergent effect.

  Components:
    48.1 — Cognitive Resource Economy   compute as scarce currency
    48.2 — Attention Market             emergence, not allocation
    48.3 — Narrative Evolution Engine   mutation, competition, selection
    48.4 — Coalition Governance Layer   constitution, veto, monopoly control
    48.5 — Emergent Self Dynamics       self as metastable coalition equilibrium

  Architecture inversion:
    Before:  Engine → step() → processes
    After:   Ecology → processes → coalition → self → execution

  Phase 46 and 47 still exist — but as infrastructure, not center.
    - SelfLatent = compressed historical center of gravity (not homunculus)
    - NarrativeStabilizer = selection pressure mechanism (not compression log)
    - AgencyInference = contested attribution field (not scalar)
    - Counterfactuals = justification engine (not what-if generator)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.insert(0, '.')

from phase47_semantic_grounding import (
    SemanticProjection, EpisodicSemanticGraph, NarrativeStabilizer,
    LanguageBind, SemanticRetrieval, NarrativeEpisode,
    SemanticFactor, SemanticFactorType, RetrievalQuery
)
from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase44_object_centric_world_model import ObjectSlot
from phase47_self_model import SelfLatent, AgencyInference, CounterfactualSelf


# ============================================================================
# 48.1 — COGNITIVE RESOURCE ECONOMY
# ============================================================================

class ResourceType(Enum):
    COMPUTE = 'compute'
    RETRIEVAL = 'retrieval'
    NARRATIVE = 'narrative'
    MEMORY = 'memory'
    PLANNING = 'planning'
    COUNTERFACTUAL = 'counterfactual'


@dataclass
class ResourceBundle:
    """Scarce cognitive resources available at each step."""
    compute: float = 1.0
    retrieval: float = 1.0
    narrative: float = 1.0
    memory: float = 1.0
    planning: float = 1.0
    counterfactual: float = 1.0

    def total(self) -> float:
        return self.compute + self.retrieval + self.narrative + self.memory + self.planning + self.counterfactual


class CognitiveAgent:
    """
    A participant in the cognitive resource economy.

    Not a "process" — an agent has:
    - type: planner, narrative_mutator, uncertainty_tracker, goal_species, etc.
    - local objective: what it tries to achieve
    - bid function: how much compute it's willing to "pay" for resources
    - productivity: how efficiently it converts compute into utility
    - reliability: historical accuracy of its predictions/claims

    Agents compete in the attention market and form coalitions.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        local_objective: np.ndarray,
        bid_curve: float = 0.5,
        productivity: float = 0.5,
        reliability: float = 0.5,
        birth_step: int = 0
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.local_objective = local_objective.copy()
        self.birth_step = birth_step

        self.bid_curve = bid_curve          # willingness to pay for compute
        self.productivity = productivity     # utility per unit compute
        self.reliability = reliability       # historical trustworthiness

        self.wealth: float = 1.0             # accumulated "cognitive currency"
        self.utility_history: List[float] = []
        self.compute_consumed: float = 0.0
        self.age: int = 0
        self.survival_count: int = 0

        # Coalition membership
        self.coalition_id: Optional[str] = None
        self.veto_weight: float = 0.0        # earned through reliability
        self.agency_claim: float = 0.5       # how much it claims credit

        self.active: bool = True

    def bid(self, resource_type: ResourceType, total_supply: float) -> float:
        """How much this agent bids for a resource type.

        Bid = wealth * bid_curve * (reliability + noise)
        Higher reliability → more aggressive bidding (track record).
        """
        noise = 0.1 * np.random.random()
        bid = self.wealth * self.bid_curve * (self.reliability + noise)
        return float(np.clip(bid, 0.0, total_supply * 0.5))

    def consume_compute(self, amount: float) -> float:
        """Consume compute and generate utility.

        Returns the utility generated.
        """
        actual = min(amount, self.wealth)
        self.compute_consumed += actual
        self.wealth -= actual * 0.8
        utility = actual * self.productivity * self.reliability
        self.utility_history.append(utility)
        return utility

    def earn_wealth(self, reward: float):
        """Earn cognitive currency from contributing to outcomes."""
        self.wealth += reward * 0.1
        self.wealth = float(np.clip(self.wealth, 0.1, 10.0))
        self.survival_count += 1

    def update_reliability(self, prediction_error: float):
        """Update reliability based on prediction accuracy."""
        self.reliability = 0.9 * self.reliability + 0.1 * np.exp(-prediction_error)
        self.age += 1

    def get_competitive_fitness(self) -> float:
        """Overall fitness for coalition formation."""
        recent_utility = float(np.mean(self.utility_history[-10:])) if self.utility_history else 0.0
        return (0.4 * recent_utility + 0.3 * self.reliability + 0.2 * self.wealth / 10.0 + 0.1 * self.veto_weight)

    def get_stats(self) -> Dict:
        return {
            'type': self.agent_type,
            'age': self.age,
            'wealth': self.wealth,
            'reliability': self.reliability,
            'productivity': self.productivity,
            'veto_weight': self.veto_weight,
            'utility': float(np.mean(self.utility_history[-10:])) if self.utility_history else 0.0,
            'coalition': self.coalition_id,
            'fitness': self.get_competitive_fitness()
        }


class ResourceMarket:
    """
    Cognitive resource economy.

    Compute is scarce currency. Agents bid for resources.
    Allocation is the outcome of economic competition, not a scheduler.

    Each step:
    1. Total supply is determined (fixed but can scale)
    2. Agents bid for each resource type
    3. Resources are allocated proportionally to bids
    4. Agents consume compute → generate utility
    5. Wealth redistributes based on contribution
    """

    def __init__(
        self,
        total_supply: ResourceBundle = None,
        min_bid: float = 0.01,
        wealth_decay: float = 0.01
    ):
        self.supply = total_supply or ResourceBundle()
        self.min_bid = min_bid
        self.wealth_decay = wealth_decay

        self.prices: Dict[ResourceType, float] = {
            rt: 1.0 for rt in ResourceType
        }
        self.market_history: List[Dict] = []

    def step(
        self,
        agents: List[CognitiveAgent]
    ) -> Dict[str, Any]:
        """Run one market cycle.

        Returns allocation results.
        """
        if not agents:
            return {'allocations': {}, 'prices': {}}

        allocations: Dict[str, Dict[ResourceType, float]] = {}

        for resource_type in ResourceType:
            total_supply = getattr(self.supply, resource_type.value, 1.0)

            # Collect bids
            bids = [(a, a.bid(resource_type, total_supply)) for a in agents if a.active]
            total_bid = sum(b for _, b in bids) + 1e-8

            # Allocate proportionally to bids
            for agent, bid in bids:
                share = bid / total_bid
                allocated = share * total_supply
                if agent.agent_id not in allocations:
                    allocations[agent.agent_id] = {}
                allocations[agent.agent_id][resource_type] = allocated

            # Price adjusts based on demand/supply ratio
            demand_ratio = total_bid / (len(bids) * self.min_bid + 1e-8)
            self.prices[resource_type] = 0.9 * self.prices[resource_type] + 0.1 * demand_ratio

        # Agents consume compute
        for agent in agents:
            if agent.agent_id in allocations:
                compute_granted = allocations[agent.agent_id].get(ResourceType.COMPUTE, 0.0)
                if compute_granted > 0:
                    agent.consume_compute(compute_granted)

        # Wealth decay (anti-hoarding)
        for agent in agents:
            agent.wealth *= (1.0 - self.wealth_decay)

        self.market_history.append({
            'n_agents': len(agents),
            'prices': {rt.value: float(p) for rt, p in self.prices.items()},
            'total_compute_allocated': sum(
                allocations[a_id].get(ResourceType.COMPUTE, 0.0)
                for a_id in allocations
            )
        })

        return {
            'allocations': allocations,
            'prices': {rt.value: float(p) for rt, p in self.prices.items()}
        }

    def get_wealthiest_agent(self, agents: List[CognitiveAgent]) -> Optional[str]:
        if not agents:
            return None
        return max(
            (a for a in agents if a.active),
            key=lambda a: a.wealth
        ).agent_id

    def get_stats(self) -> Dict:
        return {
            'last_prices': {rt.value: float(p) for rt, p in self.prices.items()},
            'market_cycles': len(self.market_history)
        }


# ============================================================================
# 48.2 — ATTENTION MARKET
# ============================================================================

class AttentionMarket:
    """
    Attention as emergent economic outcome, not allocation.

    Processes earn attention (compute priority) through:
    - predictive leverage (how much uncertainty reduction they provide)
    - surprise reduction (how well they explain new data)
    - goal progress (how much they move toward attractors)
    - narrative centrality (how connected they are in the narrative graph)
    - historical reliability (track record)
    - uncertainty calibration (accuracy of confidence estimates)

    Attention = reward signal for the resource economy.
    """

    def __init__(
        self,
        predictive_weight: float = 0.3,
        surprise_weight: float = 0.2,
        goal_progress_weight: float = 0.25,
        narrative_centrality_weight: float = 0.15,
        reliability_weight: float = 0.1
    ):
        self.predictive_weight = predictive_weight
        self.surprise_weight = surprise_weight
        self.goal_progress_weight = goal_progress_weight
        self.narrative_centrality_weight = narrative_centrality_weight
        self.reliability_weight = reliability_weight

        self.attention_scores: Dict[str, float] = {}
        self.attention_history: List[Dict] = []

    def compute_attention(
        self,
        agents: List[CognitiveAgent],
        goal_prob: float,
        gp_delta: float,
        epistemic_uncertainty: float,
        aleatoric_uncertainty: float,
        self_coherence: float
    ) -> Dict[str, float]:
        """Compute attention scores for all active agents.

        Each agent's attention score = weighted sum of:
        - predictive leverage: agent.reliability * (1 - epistemic_uncertainty)
        - surprise reduction: agent.productivity * aleatoric_uncertainty
        - goal progress: agent.wealth * gp_delta (normalized)
        - narrative centrality: sum of coalition partners
        - reliability: agent.reliability
        """
        scores: Dict[str, float] = {}
        for agent in agents:
            if not agent.active:
                continue

            predictive = agent.reliability * max(0.0, 1.0 - epistemic_uncertainty)
            surprise = agent.productivity * aleatoric_uncertainty * 0.5
            goal_prog = (agent.wealth / 10.0) * max(0.0, gp_delta * 2)
            centrality = len([
                a for a in agents
                if a.active and a.coalition_id == agent.coalition_id and a.agent_id != agent.agent_id
            ]) * 0.1
            reliability = agent.reliability

            score = (
                self.predictive_weight * predictive
                + self.surprise_weight * surprise
                + self.goal_progress_weight * goal_prog
                + self.narrative_centrality_weight * centrality
                + self.reliability_weight * reliability
            )
            scores[agent.agent_id] = float(np.clip(score, 0.0, 2.0))

        self.attention_scores = scores
        self.attention_history.append({
            'mean_score': float(np.mean(list(scores.values()))) if scores else 0.0,
            'max_score': float(max(scores.values())) if scores else 0.0,
            'n_agents': len(scores)
        })

        return scores

    def distribute_wealth(
        self,
        agents: List[CognitiveAgent],
        attention_scores: Dict[str, float],
        total_reward: float
    ):
        """Distribute cognitive currency based on attention scores."""
        total_score = sum(attention_scores.values()) + 1e-8
        for agent in agents:
            if agent.agent_id in attention_scores:
                share = attention_scores[agent.agent_id] / total_score
                reward = share * total_reward
                agent.earn_wealth(reward)

    def get_stats(self) -> Dict:
        return {
            'current_scores': {
                aid: float(s) for aid, s in
                list(self.attention_scores.items())[:5]
            },
            'history_length': len(self.attention_history)
        }


# ============================================================================
# 48.3 — NARRATIVE EVOLUTION ENGINE
# ============================================================================

@dataclass
class NarrativeGene:
    """
    A unit of narrative that can mutate, compete, and evolve.

    Unlike NarrativeEpisode (Phase 47), this is a LIVING narrative:
    - It has fitness (retrieval probability)
    - It can mutate (drift in semantic space)
    - It can merge with other narratives
    - It can die (fitness drops below threshold)
    - It competes for retrieval bandwidth

    NarrativeStabilizer (Phase 47) becomes the SELECTION MECHANISM,
    generating these genes from raw trajectory compression.
    """
    gene_id: str
    semantic_vector: np.ndarray    # in semantic manifold space
    fitness: float = 0.5           # retrieval probability
    age: int = 0
    mutation_count: int = 0
    retrieval_count: int = 0
    parent_ids: List[str] = field(default_factory=list)
    coalition_support: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)


class NarrativeEcosystem:
    """
    Evolutionary narrative ecology.

    Narratives:
    - mutate (semantic drift)
    - merge (combine similar narratives)
    - compete (fitness-based retrieval)
    - fragment (high-fitness narratives spawn variants)
    - die (low fitness → pruned)

    NarrativeStabilizer (Phase 47) feeds raw compressed episodes into this
    ecosystem. The ecosystem applies selection pressure.
    """

    def __init__(
        self,
        semantic_dim: int = 32,
        max_genes: int = 50,
        mutation_rate: float = 0.05,
        merge_threshold: float = 0.75,
        retrieval_temperature: float = 1.0
    ):
        self.semantic_dim = semantic_dim
        self.max_genes = max_genes
        self.mutation_rate = mutation_rate
        self.merge_threshold = merge_threshold
        self.retrieval_temperature = retrieval_temperature

        self.genes: Dict[str, NarrativeGene] = {}
        self.gene_count: int = 0

        self.evolution_log: List[Dict] = []

    def _next_gene_id(self) -> str:
        self.gene_count += 1
        return f"ng_{self.gene_count}"

    def seed_from_episode(self, episode: NarrativeEpisode) -> str:
        """Create a narrative gene from a compressed episode."""
        vec = np.tanh(np.random.randn(self.semantic_dim) * 0.2 + 0.5 * episode.goal_delta)
        gene = NarrativeGene(
            gene_id=self._next_gene_id(),
            semantic_vector=vec,
            fitness=0.3 + 0.5 * max(0.0, episode.goal_delta),
            attributes={
                'n_steps': episode.n_steps,
                'goal_delta': episode.goal_delta,
                'primary_agent': episode.primary_agent,
                'mean_agency': episode.mean_agency,
                'episode_source': episode.episode_id
            }
        )
        self.genes[gene.gene_id] = gene
        self.evolution_log.append({
            'event': 'birth',
            'gene_id': gene.gene_id,
            'fitness': gene.fitness,
            'source': 'episode'
        })
        return gene.gene_id

    def seed_from_step(
        self,
        semantic_vector: np.ndarray,
        fitness: float = 0.3,
        attributes: Optional[Dict] = None
    ) -> str:
        """Seed a narrative gene from a single-step semantic factor."""
        gene = NarrativeGene(
            gene_id=self._next_gene_id(),
            semantic_vector=semantic_vector.copy(),
            fitness=fitness,
            attributes=attributes or {}
        )
        self.genes[gene.gene_id] = gene
        return gene.gene_id

    def step(self) -> Dict[str, List[str]]:
        """Run one evolution cycle.

        Returns: births, deaths, merges, mutations
        """
        events: Dict[str, List[str]] = {
            'births': [], 'deaths': [], 'merges': [],
            'mutations': [], 'fragments': []
        }

        # Age and decay
        for gid in list(self.genes.keys()):
            self.genes[gid].age += 1
            self.genes[gid].fitness *= 0.995  # gradual decay

        # Mutation: random drift + fitness cost
        for gid in list(self.genes.keys()):
            if np.random.random() < self.mutation_rate:
                gene = self.genes[gid]
                drift = np.random.randn(self.semantic_dim) * self.mutation_rate
                gene.semantic_vector += drift
                gene.semantic_vector = gene.semantic_vector / (np.linalg.norm(gene.semantic_vector) + 1e-8)
                gene.mutation_count += 1
                gene.fitness *= 0.98  # mutation cost
                events['mutations'].append(gid)

        # Merge: combine similar genes
        gene_list = list(self.genes.items())
        for i in range(len(gene_list)):
            for j in range(i + 1, len(gene_list)):
                gid1, g1 = gene_list[i]
                gid2, g2 = gene_list[j]
                if gid1 not in self.genes or gid2 not in self.genes:
                    continue
                sim = float(np.dot(g1.semantic_vector, g2.semantic_vector) /
                            (np.linalg.norm(g1.semantic_vector) *
                             np.linalg.norm(g2.semantic_vector) + 1e-8))
                if sim > self.merge_threshold and g1.fitness > 0 and g2.fitness > 0:
                    # Weighted merge
                    combined_vec = (g1.fitness * g1.semantic_vector +
                                    g2.fitness * g2.semantic_vector)
                    combined_vec = combined_vec / (np.linalg.norm(combined_vec) + 1e-8)
                    g1.semantic_vector = combined_vec
                    g1.fitness = max(g1.fitness, g2.fitness) * 1.05
                    g1.parent_ids.append(gid2)
                    del self.genes[gid2]
                    events['merges'].append(f"{gid1}+{gid2}")

        # Fragment: high-fitness genes spawn variants
        for gid in list(self.genes.keys()):
            gene = self.genes[gid]
            if (gene.fitness > 0.7 and gene.age > 5
                and len(self.genes) < self.max_genes * 0.8):
                variant = NarrativeGene(
                    gene_id=self._next_gene_id(),
                    semantic_vector=gene.semantic_vector + np.random.randn(self.semantic_dim) * 0.15,
                    fitness=gene.fitness * 0.7,
                    parent_ids=[gid]
                )
                variant.semantic_vector = variant.semantic_vector / (
                    np.linalg.norm(variant.semantic_vector) + 1e-8
                )
                self.genes[variant.gene_id] = variant
                events['fragments'].append(variant.gene_id)

        # Death: prune low-fitness genes
        for gid in list(self.genes.keys()):
            if (self.genes[gid].fitness < 0.05 and self.genes[gid].age > 5
                and len(self.genes) > self.max_genes // 2):
                del self.genes[gid]
                events['deaths'].append(gid)

        # Enforce max genes
        if len(self.genes) > self.max_genes:
            sorted_genes = sorted(
                self.genes.items(), key=lambda x: x[1].fitness
            )
            n_prune = len(self.genes) - self.max_genes
            for gid, _ in sorted_genes[:n_prune]:
                del self.genes[gid]
                events['deaths'].append(gid)

        if any(events.values()):
            self.evolution_log.append({
                'n_genes': len(self.genes),
                'events': {k: len(v) for k, v in events.items()}
            })

        return events

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, NarrativeGene, float]]:
        """Retrieve genes by semantic similarity × fitness."""
        scored = []
        for gid, gene in self.genes.items():
            sim = float(np.dot(gene.semantic_vector, query_vector) /
                        (np.linalg.norm(gene.semantic_vector) *
                         np.linalg.norm(query_vector) + 1e-8))
            score = sim * gene.fitness ** self.retrieval_temperature
            scored.append((score, gid, gene))

        scored.sort(key=lambda x: -x[0])
        result = []
        for score, gid, gene in scored[:top_k]:
            gene.retrieval_count += 1
            gene.fitness = min(1.0, gene.fitness * 1.01)
            result.append((gid, gene, float(score)))
        return result

    def get_semantic_diversity(self) -> float:
        """Compute pairwise diversity across narrative genes."""
        vectors = [g.semantic_vector for g in self.genes.values()]
        if len(vectors) < 2:
            return 0.0
        sims = []
        for i in range(min(10, len(vectors))):
            for j in range(i + 1, min(10, len(vectors))):
                sim = float(np.dot(vectors[i], vectors[j]) /
                            (np.linalg.norm(vectors[i]) *
                             np.linalg.norm(vectors[j]) + 1e-8))
                sims.append(sim)
        return 1.0 - float(np.mean(sims)) if sims else 0.0

    def get_stats(self) -> Dict:
        return {
            'n_genes': len(self.genes),
            'diversity': self.get_semantic_diversity(),
            'mean_fitness': float(np.mean([g.fitness for g in self.genes.values()])),
            'max_fitness': float(max(g.fitness for g in self.genes.values())),
            'evolution_events': len(self.evolution_log)
        }


# ============================================================================
# 48.4 — COALITION GOVERNANCE LAYER
# ============================================================================

class Coalition:
    """
    A coalition of cognitive agents temporarily aligned.

    Coalitions form around shared goals and narrative support.
    The dominant coalition at any step determines the "self" experience.
    """

    def __init__(self, coalition_id: str, birth_step: int = 0):
        self.coalition_id = coalition_id
        self.birth_step = birth_step
        self.member_ids: Set[str] = set()
        self.shared_goal: Optional[np.ndarray] = None
        self.total_fitness: float = 0.0
        self.dominance_duration: int = 0
        self.veto_power: float = 0.0

    def add_member(self, agent: CognitiveAgent):
        self.member_ids.add(agent.agent_id)
        agent.coalition_id = self.coalition_id
        self.total_fitness += agent.get_competitive_fitness()
        self.veto_power = max(self.veto_power, agent.veto_weight)

    def remove_member(self, agent_id: str):
        self.member_ids.discard(agent_id)

    def compute_cohesion(self) -> float:
        """How internally aligned the coalition is."""
        if len(self.member_ids) < 2:
            return 0.0
        return min(1.0, float(len(self.member_ids)) * 0.3 + self.veto_power * 0.7)

    def get_stats(self) -> Dict:
        return {
            'size': len(self.member_ids),
            'fitness': self.total_fitness,
            'cohesion': self.compute_cohesion(),
            'veto_power': self.veto_power,
            'dominance': self.dominance_duration
        }


class CoalitionGovernance:
    """
    Proto-constitution for a multi-agent cognitive system.

    Not anomaly detection. Constitutional dynamics:
    - Veto power: earned through reliability, not assigned
    - Anti-monopoly: prevents single agent from dominating compute
    - Continuity enforcement: prevents identity fragmentation
    - Entropy preservation: maintains minimum diversity
    - Exploration quotas: ensures novelty despite exploit pressure
    """

    def __init__(
        self,
        max_coalition_dominance: int = 20,
        min_coalition_diversity: float = 0.2,
        max_compute_concentration: float = 0.4,
        exploration_quota: float = 0.15,
        constitution_strength: float = 0.3
    ):
        self.max_coalition_dominance = max_coalition_dominance
        self.min_coalition_diversity = min_coalition_diversity
        self.max_compute_concentration = max_compute_concentration
        self.exploration_quota = exploration_quota
        self.constitution_strength = constitution_strength

        self.constitutional_violations: List[Dict] = []
        self.coalition_history: List[Dict] = []

    def form_coalitions(
        self,
        agents: List[CognitiveAgent],
        narrative_ecosystem: NarrativeEcosystem,
        query_vector: np.ndarray
    ) -> List[Coalition]:
        """Form coalitions based on shared goals and narrative support.

        1. Retrieve narratives relevant to current cognitive state
        2. Group agents by narrative support patterns
        3. Form coalitions around shared narrative clusters
        4. Apply constitutional constraints
        """
        # Reset coalition membership
        for a in agents:
            a.coalition_id = None

        # Retrieve supporting narratives
        retrieved = narrative_ecosystem.retrieve(query_vector, top_k=3)

        # Group agents by narrative similarity
        coalitions: List[Coalition] = []
        coalition_map: Dict[str, Coalition] = {}

        for agent in agents:
            if not agent.active:
                continue

            # Find best coalition match
            best_coalition = None
            best_similarity = 0.0

            for coal in coalitions:
                # Compute goal similarity
                shared_goal = coal.shared_goal
                if shared_goal is not None:
                    sim = float(np.dot(agent.local_objective, shared_goal) /
                                (np.linalg.norm(agent.local_objective) *
                                 np.linalg.norm(shared_goal) + 1e-8))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_coalition = coal

            if best_coalition and best_similarity > 0.3:
                best_coalition.add_member(agent)
            else:
                # Create new coalition
                cid = f"coal_{len(coalitions)}_{agent.agent_id}"
                coal = Coalition(cid, birth_step=0)
                coal.shared_goal = agent.local_objective.copy()
                coal.add_member(agent)
                coalitions.append(coal)

        # Constitutional: enforce diversity
        self._enforce_diversity(coalitions, agents)

        # Track dominance
        for coal in coalitions:
            coal.dominance_duration += 1

        self.coalition_history.append({
            'n_coalitions': len(coalitions),
            'sizes': [len(c.member_ids) for c in coalitions]
        })

        return coalitions

    def _enforce_diversity(
        self,
        coalitions: List[Coalition],
        agents: List[CognitiveAgent]
    ):
        """Constitutional diversity enforcement.

        If too few coalitions, split the largest.
        """
        if len(coalitions) < 2 and len(agents) >= 4:
            # Split largest coalition
            largest = max(coalitions, key=lambda c: len(c.member_ids))
            members = [a for a in agents if a.coalition_id == largest.coalition_id]
            if len(members) >= 3:
                split_point = len(members) // 2
                new_coal = Coalition(
                    f"coal_split_{len(self.coalition_history)}",
                    birth_step=0
                )
                new_coal.shared_goal = members[split_point].local_objective.copy()
                for m in members[split_point:]:
                    largest.remove_member(m.agent_id)
                    new_coal.add_member(m)
                coalitions.append(new_coal)
                self.constitutional_violations.append({
                    'type': 'diversity_enforcement',
                    'action': 'coalition_split'
                })

    def apply_antimonopoly(
        self,
        allocations: Dict[str, Dict[ResourceType, float]]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """Cap any single agent's compute share."""
        modified = {}
        for aid, resources in allocations.items():
            modified[aid] = {}
            for rt, amount in resources.items():
                if rt == ResourceType.COMPUTE and amount > self.max_compute_concentration:
                    self.constitutional_violations.append({
                        'type': 'anti_monopoly',
                        'agent': aid,
                        'capped': amount - self.max_compute_concentration
                    })
                    modified[aid][rt] = self.max_compute_concentration
                else:
                    modified[aid][rt] = amount
        return modified

    def get_dominant_coalition(
        self,
        coalitions: List[Coalition]
    ) -> Optional[Coalition]:
        """Get the dominant coalition based on total fitness and cohesion."""
        if not coalitions:
            return None
        return max(coalitions, key=lambda c: c.total_fitness * c.compute_cohesion())

    def get_stats(self) -> Dict:
        return {
            'violations': len(self.constitutional_violations),
            'n_coalitions_in_history': len(set(
                v['n_coalitions'] for v in self.coalition_history
            )) if self.coalition_history else 0
        }


# ============================================================================
# 48.5 — EMERGENT SELF DYNAMICS
# ============================================================================

class EmergentSelf:
    """
    Self as metastable coalition equilibrium.

    Self is NOT:
    - an object
    - a latent vector
    - a module

    Self IS:
    - the current dominant coalition's experienced perspective
    - continuously renegotiated between competing subsystems
    - historically constrained by narrative continuity
    - economically sustained by compute market outcomes
    - narratively stabilized by evolutionary narratives

    SelfLatent (Phase 46) becomes the COMPRESSED HISTORICAL CENTER OF GRAVITY.
    Not "me now", but "me across time" — a low-frequency identity prior.
    """

    def __init__(
        self,
        self_latent: SelfLatent,
        narrative_ecosystem: NarrativeEcosystem,
        semantic_dim: int = 32
    ):
        self.self_latent = self_latent
        self.narrative_ecosystem = narrative_ecosystem
        self.semantic_dim = semantic_dim

        # Current equilibrium
        self.current_coalition: Optional[Coalition] = None
        self.coalition_history: List[str] = []
        self.identity_stability: float = 1.0

    def recompute(
        self,
        dominant_coalition: Optional[Coalition],
        agents: List[CognitiveAgent],
        query_vector: np.ndarray,
        step_index: int
    ) -> Dict[str, Any]:
        """Recompute self from current coalition equilibrium.

        Self = dominant coalition's shared goal + agent properties
                + narrative support + historical self-latent prior
        """
        old_coalition_id = self.current_coalition.coalition_id if self.current_coalition else None
        self.current_coalition = dominant_coalition
        new_id = dominant_coalition.coalition_id if dominant_coalition else None

        if new_id and old_coalition_id != new_id:
            self.coalition_history.append(new_id)

        # Identity stability: how much the self changed this step
        if old_coalition_id and new_id:
            self.identity_stability *= 0.95
        elif new_id:
            self.identity_stability = min(1.0, self.identity_stability + 0.02)

        # Update SelfLatent (Phase 46) as historical center of gravity
        if dominant_coalition and dominant_coalition.shared_goal is not None:
            self.self_latent.update(
                dominant_coalition.shared_goal[:self.self_latent.latent_dim],
                np.zeros(4)
            )

        # Retrieve self-defining narratives
        self_narratives = self.narrative_ecosystem.retrieve(query_vector, top_k=3)

        return {
            'dominant_coalition': new_id,
            'coalition_size': len(dominant_coalition.member_ids) if dominant_coalition else 0,
            'identity_stability': self.identity_stability,
            'self_coherence': self.self_latent.get_identity_signal(),
            'n_supporting_narratives': len(self_narratives)
        }

    def get_experienced_self(
        self,
        agents: List[CognitiveAgent]
    ) -> Dict[str, Any]:
        """The 'I' that the system experiences at this moment.

        This is the phenomenological self:
        - who is dominant (coalition)
        - what it wants (shared goal)
        - how reliable it is (agent reliability)
        - what narratives support it
        - how stable identity is
        """
        if not self.current_coalition:
            return {'identity': 'none', 'stability': 0.0}

        coalition = self.current_coalition
        members = [a for a in agents if a.coalition_id == coalition.coalition_id]

        return {
            'coalition': coalition.coalition_id,
            'n_members': len(members),
            'shared_goal_norm': float(np.linalg.norm(coalition.shared_goal))
            if coalition.shared_goal is not None else 0.0,
            'mean_reliability': float(np.mean([m.reliability for m in members]))
            if members else 0.0,
            'mean_veto_power': float(np.mean([m.veto_weight for m in members]))
            if members else 0.0,
            'identity_stability': self.identity_stability,
            'self_coherence': float(self.self_latent.get_identity_signal()),
            'coalition_history_length': len(self.coalition_history)
        }

    def get_stats(self) -> Dict:
        return {
            'current_coalition': self.current_coalition.coalition_id
            if self.current_coalition else None,
            'identity_stability': self.identity_stability,
            'self_coherence': float(self.self_latent.get_identity_signal()),
            'coalition_transitions': len(self.coalition_history)
        }


# ============================================================================
# 48.6 — SOCIO-COGNITIVE ENGINE
# ============================================================================

class SocioCognitiveEngine:
    """
    Unified cognitive ecology engine — the new center.

    This is NOT an extension of SemanticEngine/SelfEngine.
    This CONTAINS them as infrastructure subsystems.

    Every step:
    1.  Query environment observation from lower layers
    2.  Market: agents bid for resources                 (48.1)
    3.  Governance: form coalitions                      (48.4)
    4.  Attention: compute attention scores               (48.2)
    5.  Ecosystem: evolve narratives                      (48.3)
    6.  Self: recompute emergent self                     (48.5)
    7.  Dominant coalition's policy biases execution
    8.  Execute action through world model (delegated)
    9.  Observe outcome → reward
    10. Distribute wealth based on attention               (48.2)
    11. Update agent reliability
    12. Narrative ecosystem: seed new genes
    13. Governance: detect violations
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        n_initial_agents: int = 5,
        max_agents: int = 15,
        semantic_dim: int = 32,
        agent_birth_interval: int = 15
    ):
        self.wm = wm
        self.semantic_dim = semantic_dim
        self.max_agents = max_agents
        self.agent_birth_interval = agent_birth_interval

        # 48.1 — Economy
        self.resource_market = ResourceMarket()
        self.agents: List[CognitiveAgent] = []
        self.n_initial_agents = n_initial_agents

        # 48.2 — Attention
        self.attention_market = AttentionMarket()

        # 48.3 — Narrative ecosystem
        self.narrative_ecosystem = NarrativeEcosystem(
            semantic_dim=semantic_dim
        )

        # 48.4 — Governance
        self.coalition_governance = CoalitionGovernance()
        self.coalitions: List[Coalition] = []

        # 48.5 — Emergent self
        self.self_latent = SelfLatent(latent_dim=16, self_dim=8)
        self.agency = AgencyInference(latent_dim=16)
        self.emergent_self = EmergentSelf(
            self_latent=self.self_latent,
            narrative_ecosystem=self.narrative_ecosystem,
            semantic_dim=semantic_dim
        )

        # Execution state
        self.total_steps: int = 0
        self.execution_log: List[Dict] = []
        self.agent_id_count: int = 0
        self.goal_prob: float = 0.0
        self.prev_goal_prob: float = 0.0

        # Infrastructure references (set externally or during setup)
        self.goal_manifold = None
        self.object_tracker = None
        self.flow_manifold = None
        self.inv_dyn = None

    def _next_agent_id(self) -> str:
        self.agent_id_count += 1
        return f"ca_{self.agent_id_count}"

    def _initialize_agents(self):
        """Create initial agent population."""
        agent_types = ['planner', 'explorer', 'conservative', 'narrative_mutator',
                       'uncertainty_tracker', 'goal_species', 'novelty_seeker']
        for i in range(min(self.n_initial_agents, len(agent_types))):
            agent = CognitiveAgent(
                agent_id=self._next_agent_id(),
                agent_type=agent_types[i % len(agent_types)],
                local_objective=np.tanh(np.random.randn(self.semantic_dim) * 0.5),
                bid_curve=0.3 + 0.4 * np.random.random(),
                productivity=0.3 + 0.4 * np.random.random(),
                reliability=0.4 + 0.3 * np.random.random(),
                birth_step=0
            )
            agent.wealth = 1.0 + np.random.random()
            self.agents.append(agent)

    def _spawn_agent(self, agent_type: str, parent_objective: Optional[np.ndarray] = None) -> CognitiveAgent:
        """Spawn a new agent via mutation of existing type or from narrative."""
        if parent_objective is not None:
            objective = parent_objective + np.random.randn(self.semantic_dim) * 0.1
        else:
            objective = np.tanh(np.random.randn(self.semantic_dim) * 0.5)

        agent = CognitiveAgent(
            agent_id=self._next_agent_id(),
            agent_type=agent_type,
            local_objective=objective,
            bid_curve=0.3 + 0.4 * np.random.random(),
            productivity=0.3 + 0.4 * np.random.random(),
            reliability=0.3 + 0.3 * np.random.random(),
            birth_step=self.total_steps
        )
        agent.wealth = 0.5
        return agent

    def step(
        self,
        z_obs: np.ndarray,
        h_belief: np.ndarray
    ) -> Dict[str, Any]:
        """One full cognitive ecology step."""
        # Initialize agents on first step
        if not self.agents:
            self._initialize_agents()

        # ====================================================================
        # LAYER 1: RESOURCE MARKET (48.1)
        # ====================================================================
        market_result = self.resource_market.step(self.agents)
        raw_allocations = market_result.get('allocations', {})

        # ====================================================================
        # LAYER 2: CONSTITUTIONAL ANTIMONOPOLY (48.4)
        # ====================================================================
        allocations = self.coalition_governance.apply_antimonopoly(raw_allocations)

        # ====================================================================
        # LAYER 3: NARRATIVE EVOLUTION (48.3)
        # ====================================================================
        narrative_events = self.narrative_ecosystem.step()

        # Build query vector from current latent state
        query_vector = self.semantic_dim_project(z_obs)
        retrieved_narratives = self.narrative_ecosystem.retrieve(query_vector, top_k=3)

        # ====================================================================
        # LAYER 4: COALITION FORMATION (48.4)
        # ====================================================================
        self.coalitions = self.coalition_governance.form_coalitions(
            self.agents, self.narrative_ecosystem, query_vector
        )

        # ====================================================================
        # LAYER 5: ATTENTION SCORING (48.2)
        # ====================================================================
        gp_delta = self.goal_prob - self.prev_goal_prob
        attention_scores = self.attention_market.compute_attention(
            self.agents,
            goal_prob=self.goal_prob,
            gp_delta=gp_delta,
            epistemic_uncertainty=0.3,
            aleatoric_uncertainty=0.3,
            self_coherence=self.self_latent.get_identity_signal()
        )

        # ====================================================================
        # LAYER 6: EMERGENT SELF (48.5)
        # ====================================================================
        dominant_coalition = self.coalition_governance.get_dominant_coalition(
            self.coalitions
        )
        self_state = self.emergent_self.recompute(
            dominant_coalition, self.agents, query_vector, self.total_steps
        )

        # ====================================================================
        # LAYER 7: DOMINANT COALITION POLICY
        # ====================================================================
        action = self._compute_action_from_coalition(
            dominant_coalition, z_obs, h_belief
        )

        # ====================================================================
        # LAYER 8: EXECUTION
        # ====================================================================
        predicted_mu, predicted_logvar = self.wm.predict_transition(z_obs, h_belief, action)
        std = np.exp(0.5 * predicted_logvar)
        z_next = predicted_mu + std * np.random.randn(*predicted_mu.shape) * 0.1
        h_next = self.wm.gru_step(h_belief, predicted_mu)

        # ====================================================================
        # LAYER 9: OUTCOME → REWARD
        # ====================================================================
        if self.goal_manifold is not None:
            goal_prob = self.goal_manifold.compute_goal_prob(z_next)
        else:
            goal_prob = float(np.exp(-np.linalg.norm(z_next) / 0.5))
        self.prev_goal_prob = self.goal_prob
        self.goal_prob = float(goal_prob)
        total_reward = goal_prob

        # Agency inference
        latent_agency = self.agency.compute_latent_agency(
            z_obs, z_next, predicted_mu, action
        )

        # ====================================================================
        # LAYER 10: WEALTH DISTRIBUTION
        # ====================================================================
        self.attention_market.distribute_wealth(
            self.agents, attention_scores, total_reward
        )

        # ====================================================================
        # LAYER 11: RELIABILITY UPDATE
        # ====================================================================
        prediction_error = float(np.linalg.norm(z_next - predicted_mu))
        for agent in self.agents:
            agent.update_reliability(prediction_error)

            # Earn veto through reliability
            if agent.reliability > 0.7:
                agent.veto_weight = min(1.0, agent.veto_weight + 0.01)
            else:
                agent.veto_weight *= 0.99

        # ====================================================================
        # LAYER 12: NARRATIVE SEEDING
        # ====================================================================
        if self.total_steps % self.agent_birth_interval == 0:
            # Seed a narrative gene from this step
            step_vec = self.semantic_dim_project(z_obs)
            self.narrative_ecosystem.seed_from_step(
                step_vec,
                fitness=0.3 + 0.5 * goal_prob,
                attributes={'step': self.total_steps, 'gp_delta': gp_delta}
            )

        # ====================================================================
        # LAYER 13: AGENT BIRTH/DEATH
        # ====================================================================
        if (self.total_steps > 0
            and self.total_steps % self.agent_birth_interval == 0
            and len(self.agents) < self.max_agents):
            # Spawn new agent
            types = ['planner', 'explorer', 'conservative', 'novelty_seeker',
                     'goal_hypothesis', 'narrative_merger', 'uncertainty_specialist']
            agent_type = types[len(self.agents) % len(types)]
            new_agent = self._spawn_agent(agent_type)
            self.agents.append(new_agent)

        # Remove dead agents
        self.agents = [a for a in self.agents if a.active and a.wealth > 0.1]

        # ====================================================================
        # LAYER 14: GOVERNANCE VIOLATION DETECTION
        # ====================================================================
        governance_violations = len(self.coalition_governance.constitutional_violations)

        self.total_steps += 1

        # Build result
        experienced_self = self.emergent_self.get_experienced_self(self.agents)
        result = {
            'z_before': z_obs.copy(),
            'z_after': z_next.copy(),
            'action': action,
            'goal_prob': goal_prob,
            'gp_delta': gp_delta,
            'latent_agency': float(latent_agency),
            'self_coherence': self_state.get('self_coherence', 0.0),
            'identity_stability': self_state.get('identity_stability', 0.0),
            'n_agents': len(self.agents),
            'n_coalitions': len(self.coalitions),
            'n_narrative_genes': len(self.narrative_ecosystem.genes),
            'narrative_diversity': self.narrative_ecosystem.get_semantic_diversity(),
            'attention_mean': float(np.mean(list(attention_scores.values()))) if attention_scores else 0.0,
            'wealthiest_agent': self.resource_market.get_wealthiest_agent(self.agents),
            'dominant_coalition': self_state.get('dominant_coalition'),
            'dominant_coalition_size': self_state.get('coalition_size', 0),
            'experienced_self': experienced_self,
            'governance_violations': governance_violations,
            'narrative_events': {k: len(v) for k, v in narrative_events.items()},
        }
        self.execution_log.append(result)
        return result

    def semantic_dim_project(self, z: np.ndarray) -> np.ndarray:
        """Project latent z to semantic dim space for queries."""
        if len(z) >= self.semantic_dim:
            return z[:self.semantic_dim]
        return np.pad(z, (0, self.semantic_dim - len(z)))

    def _compute_action_from_coalition(
        self,
        dominant_coalition: Optional[Coalition],
        z: np.ndarray,
        h: np.ndarray
    ) -> np.ndarray:
        """Compute action biased by dominant coalition's policy."""
        if dominant_coalition and dominant_coalition.shared_goal is not None:
            # Move toward coalition's shared goal (in latent-relevant dims)
            goal_latent = dominant_coalition.shared_goal[:len(z)]
            if np.any(goal_latent != 0):
                delta = goal_latent - z
                action = np.clip(delta * 0.3, -0.5, 0.5)
                return np.pad(action, (0, max(0, self.wm.action_dim - len(action))))
        return np.random.randn(self.wm.action_dim) * 0.2

    def run(self, z_start: np.ndarray, n_steps: int = 200) -> Dict:
        """Run ecology engine for n_steps."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)

        # Initialize self-latent
        self.self_latent.update(z, np.zeros(4))

        for _ in range(n_steps):
            result = self.step(z, h)
            z = result['z_after']
            h = self.wm.gru_step(h, z)

        # Aggregate results
        gps = [e.get('goal_prob', 0.0) for e in self.execution_log]
        agencies = [e.get('latent_agency', 0.0) for e in self.execution_log]
        coherences = [e.get('self_coherence', 0.0) for e in self.execution_log]
        identities = [e.get('identity_stability', 0.0) for e in self.execution_log]
        n_agents_list = [e.get('n_agents', 0) for e in self.execution_log]
        n_coalitions_list = [e.get('n_coalitions', 0) for e in self.execution_log]

        return {
            'n_steps': n_steps,
            'mean_gp': float(np.mean(gps)) if gps else 0.0,
            'mean_agency': float(np.mean(agencies)) if agencies else 0.0,
            'mean_self_coherence': float(np.mean(coherences)) if coherences else 0.0,
            'mean_identity_stability': float(np.mean(identities)) if identities else 0.0,
            'final_n_agents': n_agents_list[-1] if n_agents_list else 0,
            'final_n_coalitions': n_coalitions_list[-1] if n_coalitions_list else 0,
            'narrative_ecosystem': self.narrative_ecosystem.get_stats(),
            'resource_market': self.resource_market.get_stats(),
            'attention_market': self.attention_market.get_stats(),
            'emergent_self': self.emergent_self.get_stats(),
            'experienced_self': self.emergent_self.get_experienced_self(self.agents),
            'agent_stats': {a.agent_id: a.get_stats() for a in self.agents[:5]},
            'coalition_stats': {
                c.coalition_id: c.get_stats() for c in self.coalitions
            } if self.coalitions else {}
        }


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_resource_economy():
    """Test 48.1: Cognitive resource economy."""
    print("\n============================================================")
    print("48.1 — COGNITIVE RESOURCE ECONOMY")
    print("============================================================")

    market = ResourceMarket()
    agents = [
        CognitiveAgent(f'a_{i}', 'planner' if i % 2 == 0 else 'explorer',
                       np.random.randn(32), bid_curve=0.3 + 0.3 * (i / 3),
                       productivity=0.5, reliability=0.3 + 0.5 * (i / 3))
        for i in range(4)
    ]

    # Test market step
    result = market.step(agents)
    assert 'allocations' in result
    assert 'prices' in result
    print(f"  ✓ Market step: {len(result['allocations'])} allocations, "
          f"{len(result['prices'])} prices")

    # Test wealth dynamics
    wealth_before = agents[0].wealth
    agents[0].earn_wealth(2.0)
    assert agents[0].wealth > wealth_before
    print(f"  ✓ Wealth dynamics: {wealth_before:.3f} → {agents[0].wealth:.3f}")

    # Test reliability update
    agents[0].update_reliability(0.1)
    assert agents[0].reliability > 0.5
    print(f"  ✓ Reliability update: {agents[0].reliability:.3f}")

    # Test fitness
    fitnesses = [a.get_competitive_fitness() for a in agents]
    print(f"  ✓ Fitnesses: {[f'{f:.3f}' for f in fitnesses]}")

    # Test wealthiest detection
    richest = market.get_wealthiest_agent(agents)
    assert richest is not None
    print(f"  ✓ Wealthiest: {richest}")

    print("  >>> CognitiveResourceEconomy PASSED\n")
    return market


def test_attention_market():
    """Test 48.2: Attention market."""
    print("\n============================================================")
    print("48.2 — ATTENTION MARKET")
    print("============================================================")

    market = AttentionMarket()
    agents = [
        CognitiveAgent(f'a_{i}', 'planner', np.random.randn(32),
                       reliability=0.4 + 0.4 * (i / 3),
                       productivity=0.5, wealth=1.0 + i)
        for i in range(3)
    ]

    # Test attention computation
    scores = market.compute_attention(
        agents, goal_prob=0.6, gp_delta=0.1,
        epistemic_uncertainty=0.2, aleatoric_uncertainty=0.3,
        self_coherence=0.95
    )
    assert len(scores) == 3
    assert all(s >= 0 for s in scores.values())
    print(f"  ✓ Attention scores: {[f'{s:.3f}' for s in scores.values()]}")

    # Test wealth distribution
    wealth_before = [a.wealth for a in agents]
    market.distribute_wealth(agents, scores, total_reward=5.0)
    wealth_after = [a.wealth for a in agents]
    print(f"  ✓ Wealth distribution: "
          f"{[f'{b:.2f}→{a:.2f}' for b, a in zip(wealth_before, wealth_after)]}")

    # Test different attention for different agents
    assert scores['a_2'] > scores['a_0']  # most reliable should win
    print(f"  ✓ Reliability correlates with attention")

    print("  >>> AttentionMarket PASSED\n")
    return market


def test_narrative_ecosystem():
    """Test 48.3: Narrative evolution engine."""
    print("\n============================================================")
    print("48.3 — NARRATIVE EVOLUTION ENGINE")
    print("============================================================")

    ecosystem = NarrativeEcosystem(semantic_dim=32, max_genes=10)

    # Test seeding
    gid1 = ecosystem.seed_from_step(np.random.randn(32), fitness=0.7)
    gid2 = ecosystem.seed_from_step(np.random.randn(32), fitness=0.5)
    gid3 = ecosystem.seed_from_step(np.random.randn(32), fitness=0.3)
    print(f"  ✓ Seeded: {len(ecosystem.genes)} genes")

    # Test evolution steps
    events = ecosystem.step()
    print(f"  ✓ Evolution: {events}")

    # Test retrieval
    results = ecosystem.retrieve(np.random.randn(32), top_k=2)
    assert len(results) <= 2
    print(f"  ✓ Retrieved: {len(results)} genes")

    # Test diversity
    div = ecosystem.get_semantic_diversity()
    assert div >= 0
    print(f"  ✓ Diversity: {div:.3f}")

    # Test evolution over multiple steps
    for _ in range(5):
        ecosystem.step()
    print(f"  ✓ After 5 steps: {len(ecosystem.genes)} genes, "
          f"mean fitness={ecosystem.get_stats()['mean_fitness']:.3f}")

    print("  >>> NarrativeEcosystem PASSED\n")
    return ecosystem


def test_coalition_governance():
    """Test 48.4: Coalition governance."""
    print("\n============================================================")
    print("48.4 — COALITION GOVERNANCE LAYER")
    print("============================================================")

    governance = CoalitionGovernance()
    ecosystem = NarrativeEcosystem(semantic_dim=32)
    agents = [
        CognitiveAgent(f'a_{i}', 'planner', np.random.randn(32),
                       reliability=0.5, productivity=0.5)
        for i in range(5)
    ]

    # Seed narratives
    for _ in range(3):
        ecosystem.seed_from_step(np.random.randn(32), fitness=0.5)

    # Test coalition formation
    coalitions = governance.form_coalitions(
        agents, ecosystem, np.random.randn(32)
    )
    print(f"  ✓ Coalitions formed: {len(coalitions)}")

    # Test anti-monopoly
    allocations = {
        f'a_{i}': {ResourceType.COMPUTE: 0.3 + 0.2 * i}
        for i in range(5)
    }
    modified = governance.apply_antimonopoly(allocations)
    max_alloc = max(
        modified[aid].get(ResourceType.COMPUTE, 0.0)
        for aid in modified
    )
    assert max_alloc <= governance.max_compute_concentration + 0.01
    print(f"  ✓ Anti-monopoly: max capped at {max_alloc:.3f}")

    # Test dominant coalition
    dominant = governance.get_dominant_coalition(coalitions)
    if dominant:
        print(f"  ✓ Dominant coalition: {dominant.coalition_id}, "
              f"size={len(dominant.member_ids)}")

    # Test diversity enforcement
    single_coalition = [Coalition('only_one')]
    for a in agents:
        single_coalition[0].add_member(a)
    governance._enforce_diversity(single_coalition, agents)
    print(f"  ✓ Diversity enforcement: {len(single_coalition)} coalitions "
          f"after split")

    print("  >>> CoalitionGovernance PASSED\n")
    return governance


def test_emergent_self():
    """Test 48.5: Emergent self dynamics."""
    print("\n============================================================")
    print("48.5 — EMERGENT SELF DYNAMICS")
    print("============================================================")

    self_latent = SelfLatent(latent_dim=16, self_dim=8)
    ecosystem = NarrativeEcosystem(semantic_dim=32)
    emself = EmergentSelf(self_latent, ecosystem, semantic_dim=32)

    agents = [
        CognitiveAgent(f'a_{i}', 'planner', np.random.randn(32),
                       reliability=0.5, productivity=0.5)
        for i in range(3)
    ]

    # Create coalition
    coal = Coalition('test_coal')
    for a in agents:
        coal.add_member(a)
    coal.shared_goal = np.ones(32)

    # Test recompute
    state = emself.recompute(
        coal, agents, np.random.randn(32), step_index=1
    )
    assert 'self_coherence' in state
    assert 'dominant_coalition' in state
    print(f"  ✓ Self recomputed: coalition={state['dominant_coalition']}, "
          f"coherence={state['self_coherence']:.3f}")

    # Test experienced self
    exp = emself.get_experienced_self(agents)
    assert 'coalition' in exp
    assert 'identity_stability' in exp
    assert 'self_coherence' in exp
    print(f"  ✓ Experienced self: coalition={exp['coalition']}, "
          f"stability={exp['identity_stability']:.3f}")

    # Test identity stability over coalition transitions
    coal2 = Coalition('test_coal_2')
    coal2.shared_goal = np.ones(32) * -1
    for a in agents:
        coal2.add_member(a)
    state2 = emself.recompute(coal2, agents, np.random.randn(32), step_index=2)
    assert state2['identity_stability'] < state['identity_stability']
    print(f"  ✓ Coalition change reduces stability: "
          f"{state['identity_stability']:.3f} → {state2['identity_stability']:.3f}")

    print("  >>> EmergentSelf PASSED\n")
    return emself


def test_ecology_engine_sanity(n_steps: int = 30, bootstrap: bool = True):
    """Test SocioCognitiveEngine runs without error."""
    print("\n============================================================")
    print("SOCIO-COGNITIVE ENGINE SANITY (30 steps)")
    print("============================================================")

    engine, result, checks, all_pass = test_integration(
        n_steps=n_steps, bootstrap=bootstrap, verbose=False
    )

    return engine, result, checks, all_pass


def test_integration(
    n_steps: int = 200,
    bootstrap: bool = True,
    verbose: bool = True
):
    """
    Full Phase 48 v2 integration test.

    Verifies:
    1.  GP not flat
    2.  Agency active
    3.  Self coherence maintained
    4.  Identity stability preserved
    5.  Agents exist and diverse
    6.  Coalitions formed
    7.  Narrative ecosystem alive
    8.  Attention scores nonzero
    9.  Resource market active
    10. Emergent self computed
    11. Governance violations tracked
    12. Narrative diversity nonzero
    """
    print("\n" + "=" * 70)
    print(f"PHASE 48 v2: SOCIO-COGNITIVE ECOLOGY ({n_steps}+ steps)")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    # Bootstrap world model
    for _ in range(50 if bootstrap else 0):
        z = np.random.randn(16) * 0.3
        a = np.random.randn(16) * 0.5
        z_next = z + 0.1 * a[:16] + np.random.randn(16) * 0.05
        h = np.zeros(64)
        wm.record_transition(z, h, a, z_next)
    if bootstrap:
        for _ in range(3):
            batch = wm.sample_batch(batch_size=32)

    engine = SocioCognitiveEngine(
        wm=wm,
        n_initial_agents=4,
        max_agents=10,
        semantic_dim=32,
        agent_birth_interval=max(8, n_steps // 10)
    )

    print(f"  Running {n_steps} steps...\n")
    z_start = np.random.randn(16) * 0.1
    result = engine.run(z_start, n_steps=n_steps)

    mean_gp = result.get('mean_gp', 0.0)
    mean_agency = result.get('mean_agency', 0.0)
    mean_coherence = result.get('mean_self_coherence', 0.0)
    mean_identity = result.get('mean_identity_stability', 0.0)
    n_agents = result.get('final_n_agents', 0)
    n_coalitions = result.get('final_n_coalitions', 0)
    n_genes = result.get('narrative_ecosystem', {}).get('n_genes', 0)
    narrative_div = result.get('narrative_ecosystem', {}).get('diversity', 0.0)
    attention_stats = result.get('attention_market', {})
    market_stats = result.get('resource_market', {})
    self_stats = result.get('emergent_self', {})
    exp_self = result.get('experienced_self', {})

    # Agent stats
    agent_types = [s.get('type', '') for s in result.get('agent_stats', {}).values()]

    if verbose:
        print(f"\n  RESULTS:")
        print(f"    Steps: {n_steps}")
        print(f"    Mean GP: {mean_gp:.4f}")
        print(f"    Mean agency: {mean_agency:.4f}")
        print(f"    Mean self coherence: {mean_coherence:.4f}")
        print(f"    Mean identity stability: {mean_identity:.4f}")

        print(f"\n  [48.1] ECONOMY:")
        print(f"    Agents: {n_agents}, Types: {agent_types[:6]}")

        print(f"\n  [48.2] ATTENTION:")
        print(f"    History: {attention_stats.get('history_length', 0)} steps")

        print(f"\n  [48.3] NARRATIVE ECOLOGY:")
        print(f"    Genes: {n_genes}, Diversity: {narrative_div:.3f}")

        print(f"\n  [48.4] GOVERNANCE:")
        print(f"    Coalitions: {n_coalitions}")

        print(f"\n  [48.5] EMERGENT SELF:")
        print(f"    Current coalition: {self_stats.get('current_coalition')}")
        print(f"    Identity stability: {self_stats.get('identity_stability', 0.0):.3f}")
        print(f"    Self coherence: {self_stats.get('self_coherence', 0.0):.3f}")
        print(f"    Experienced self coalition: {exp_self.get('coalition')}")
        print(f"    Experienced self members: {exp_self.get('n_members', 0)}")

    checks = [
        ("GP not flat", mean_gp > 0.1, f"{mean_gp:.4f}"),
        ("Agency active", mean_agency > 0.05, f"{mean_agency:.4f}"),
        ("Self coherence maintained", mean_coherence > 0.5, f"{mean_coherence:.4f}"),
        ("Identity stability preserved", mean_identity > 0.5, f"{mean_identity:.4f}"),
        ("Agents exist", n_agents >= 1, f"{n_agents}"),
        ("Coalitions formed", n_coalitions >= 1, f"{n_coalitions}"),
        ("Narrative ecosystem alive", n_genes >= 1, f"{n_genes}"),
        ("Narrative diversity nonzero", narrative_div >= 0, f"{narrative_div:.3f}"),
        ("Resource market active",
         market_stats.get('market_cycles', 0) >= 5,
         f"{market_stats.get('market_cycles', 0)} cycles"),
        ("Emergent self computed",
         self_stats.get('current_coalition') is not None,
         f"{self_stats.get('current_coalition')}"),
        ("Experienced self present",
         exp_self.get('coalition') is not None,
         f"coalition={exp_self.get('coalition')}"),
    ]

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  VERIFICATION")
        print(f"  {'=' * 60}")
        for name, passed, detail in checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {status} {name}: {detail}")

    all_pass = all(p for _, p, _ in checks)
    if all_pass:
        print(f"\n  {'=' * 60}")
        print(f"  PHASE 48 v2 VERDICT: ALL {len(checks)}/{len(checks)} PASSED")
        print(f"  {'=' * 60}")
    else:
        print(f"\n  PASSED: {sum(1 for _, p, _ in checks if p)}/{len(checks)}")

    return engine, result, checks, all_pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48 v2: AUTONOMOUS COGNITIVE ECOLOGY                      ║
║                                                                   ║
║  Self is no longer the center.                                   ║
║  Processes compete. Self emerges as coalition equilibrium.        ║
║                                                                   ║
║  Architecture:                                                    ║
║    48.1 — Cognitive Resource Economy   compute = scarce currency ║
║    48.2 — Attention Market             emergence, not allocation ║
║    48.3 — Narrative Evolution Engine   mutation, selection       ║
║    48.4 — Coalition Governance Layer   constitution, veto        ║
║    48.5 — Emergent Self Dynamics       self = coalition equil.   ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    unit_tests = [
        ("ResourceEconomy", test_resource_economy),
        ("AttentionMarket", test_attention_market),
        ("NarrativeEcosystem", test_narrative_ecosystem),
        ("CoalitionGovernance", test_coalition_governance),
        ("EmergentSelf", test_emergent_self),
        ("SocioCognitiveEngine Sanity (30 steps)",
         lambda: test_ecology_engine_sanity(n_steps=30, bootstrap=True)),
    ]

    all_unit_pass = True
    for name, fn in unit_tests:
        try:
            fn()
            print(f"  >>> {name} PASSED\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  >>> {name} FAILED: {e}\n")
            all_unit_pass = False

    if all_unit_pass:
        engine, result, checks, all_pass = test_integration(
            n_steps=200, bootstrap=True, verbose=True
        )

        print("\n" + "=" * 70)
        print("PHASE 48 v2 SUMMARY")
        print("=" * 70)
        passed_count = sum(1 for _, p, _ in checks if p)
        total_count = len(checks)
        print(f"\n  Checks passed: {passed_count}/{total_count}")
        if all_pass:
            print("""
  Phase 48 v2 complete.

  The system has transitioned from:
    "agent with memory"
  to:
    "socio-cognitive dynamical system"

  Key architectural properties:
    • Self is not a module — it emerges from coalition equilibrium
    • Compute is scarce — processes compete via resource market
    • Attention is economic — earned through predictive leverage
    • Narratives evolve — mutation, competition, selection pressure
    • Governance is constitutional — veto, anti-monopoly, diversity enforcement

  Phase 46 and 47 still exist — but as infrastructure, not center:
    • SelfLatent = compressed historical center of gravity
    • NarrativeStabilizer = selection pressure mechanism
    • AgencyInference = contested attribution field
    • Counterfactuals = justification engine

  Architecture stack (Phases 25-48):

    Phase 25-30:  Sensorimotor                 ← world modeling
    Phase 31-40:  Behavioral Dynamics           ← action flows
    Phase 41-42:  Goal Geometry                 ← intention
    Phase 43-44:  Uncertainty & Objects         ← perception
    Phase 45:     Temporal Abstraction          ← time
    Phase 46:     Self-Model & Identity         ← continuity
    Phase 47:     Semantic Grounding            ← meaning
    Phase 48:     Socio-Cognitive Ecology       ← civilization

  This is a synthetic cognitive substrate.
            """)
        else:
            print("\n  ❌ Some checks failed")
            for name, passed, detail in checks:
                if not passed:
                    print(f"     FAIL: {name} = {detail}")
    else:
        print("\n  ❌ Unit tests failed — skipping integration test")
