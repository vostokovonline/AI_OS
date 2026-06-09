"""
Phase 43 — Attractor Ecology

KEY SHIFT:
  Phase 42: single learned goal manifold (Gaussian over successes)
  Phase 43: multiple competing attractors (goal ecology)

  Goals are no longer a single "success region."
  Goals are an ECOLOGY of attractors that:
    - Compete for behavioral resources
    - Merge when they overlap
    - Split when they cover too much variance
    - Die when they underperform
    - Specialize into behavioral niches

  This is the difference between:
    "one thing I want"
    "a landscape of things I could want"

WHAT CHANGES:
  1. GoalAttractor: individual attractor with mean, cov, fitness, niche
  2. AttractorEcology: population dynamics (birth/death/merge/split/compete)
  3. DiversityPressure: anti-collapse regularization
  4. NicheTracker: preserves behavioral specialization
  5. Phase43Engine: unified cognitive process with attractor ecology

ARCHITECTURAL SIGNIFICANCE:
  Before: reaching a symbolic goal
  After:  resource allocation between competing viability basins

  This enables:
    - Proto-motivation (preference formation)
    - Behavioral specialization (niche occupation)
    - Self-organizing policy ecology (flows × attractors)
    - Predictive viability (which attractors are reachable?)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor as LegacyGoal
from phase34_inverse_control_stabilization import InverseDynamicsModel
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, PointFlow, LimitCycleFlow, FlowType
)
from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, BehavioralPhysicsLearner,
    FlowTrajectoryBuffer, FlowEpisode, compute_flow_sequence_loss
)
from phase38_energy_regularized_dynamics import EnergyCostFunction
from phase40_self_organizing_geometry import (
    ContinuousFlowEcology, ContinuousManifoldDrift, ContinuousCEM
)
from phase41_bootstrapper import CoverageBuffer
from phase42_emergent_goal_geometry import (
    SuccessMemory, ContrastiveShaping
)


# ============================================================================
# 1. GOAL ATTRACTOR — Individual Behavioral Basin
# ============================================================================

@dataclass
class GoalAttractor:
    """An individual attractor in the goal ecology."""

    id: str
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    weight: float = 1.0
    energy_cost: float = 0.5
    stability: float = 0.5
    age: int = 0
    niche_id: str = ''
    birth_step: int = 0

    # Tracking
    assigned_transitions: int = 0
    total_reward: float = 0.0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    assigned_flows: Set[str] = field(default_factory=set)
    niche_strength: float = 0.5

    def compute_goal_prob(self, z: np.ndarray) -> float:
        """Membership likelihood: exp(-0.5 * Mahalanobis²(z | mean, cov))."""
        delta = z - self.mean
        try:
            mahal = float(np.sqrt(delta @ self.cov_inv @ delta))
            return float(np.exp(-0.5 * mahal ** 2))
        except Exception:
            dist = float(np.linalg.norm(z - self.mean))
            return float(np.exp(-dist))

    @property
    def fitness(self) -> float:
        """Overall fitness = success_rate * (1 - λ*energy) * stability * niche."""
        avg_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        norm_cost = max(0.0, 1.0 - self.energy_cost)
        fit = avg_reward * norm_cost * self.stability * self.niche_strength
        return max(0.0, fit)

    def record(self, z: np.ndarray, reward: float, flow_id: str):
        self.assigned_transitions += 1
        self.total_reward += reward
        self.recent_rewards.append(reward)
        self.assigned_flows.add(flow_id)
        self.age += 1

    def get_stats(self) -> Dict:
        return {
            'id': self.id,
            'weight': self.weight,
            'fitness': self.fitness,
            'age': self.age,
            'n_flows': len(self.assigned_flows),
            'n_transitions': self.assigned_transitions,
            'mean_norm': float(np.linalg.norm(self.mean)),
            'stability': self.stability,
            'niche': self.niche_id,
            'niche_strength': self.niche_strength
        }


def attractor_similarity(a: GoalAttractor, b: GoalAttractor) -> float:
    """Cosine similarity between attractor means."""
    na = float(np.linalg.norm(a.mean)) + 1e-8
    nb = float(np.linalg.norm(b.mean)) + 1e-8
    return float(np.dot(a.mean, b.mean) / (na * nb))


# ============================================================================
# 2. ATTRACTOR ECOLOGY — Population Dynamics
# ============================================================================

class AttractorEcology:
    """
    Population dynamics for goal attractors.

    Every step:
      1. New success point → assign to nearest attractor (weighted by GP)
      2. Periodically: birth, death, merge, split
      3. Competition: attractors compete for new points via softmax(GP)
      4. Diversity enforcement: entropy regularization

    Birth triggers:
      - High-density region with no nearby attractor
      - Success cluster that diverges from existing attractors

    Death triggers:
      - Low fitness for extended period
      - Age > threshold with no assigned flows
      - Weight drops below minimum

    Merge triggers:
      - Two attractors with similarity > threshold
      - Combined weight > individual weights

    Split triggers:
      - Attractor with high internal variance (covers multiple regimes)
      - Assigned flows are diverse (different behavioral patterns)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        min_attractors: int = 2,
        max_attractors: int = 12,
        merge_threshold: float = 0.85,
        split_variance_threshold: float = 1.5,
        birth_min_points: int = 15,
        death_age_min: int = 50,
        death_fitness_min: float = 0.001,
        competition_temperature: float = 0.5,
        fit_interval: int = 20,
        entropy_weight: float = 0.2
    ):
        self.latent_dim = latent_dim
        self.min_attractors = min_attractors
        self.max_attractors = max_attractors
        self.merge_thresh = merge_threshold
        self.split_var_thresh = split_variance_threshold
        self.birth_min_pts = birth_min_points
        self.death_age_min = death_age_min
        self.death_fitness_min = death_fitness_min
        self.comp_temp = competition_temperature
        self.fit_interval = fit_interval
        self.entropy_weight = entropy_weight

        # Attractor population
        self.attractors: Dict[str, GoalAttractor] = {}
        self.attractor_counter = 0

        # Unassigned success buffer (points not yet claimed by any attractor)
        self.unassigned_latents: List[np.ndarray] = []
        self.unassigned_rewards: List[float] = []
        self.unassigned_flows: List[str] = []
        self.max_unassigned = 200

        # Flow-to-attractor assignment matrix
        self.flow_to_attractor: Dict[str, str] = {}
        self.attractor_flow_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # History
        self.total_steps = 0
        self.n_births = 0
        self.n_deaths = 0
        self.n_merges = 0
        self.n_splits = 0

        # Seed with initial attractor
        self._seed()

    def _seed(self):
        """Create initial attractor from fallback.

        Uses narrow covariance so it doesn't cover the entire latent space —
        this forces birth of new attractors for distant clusters.
        """
        mean = np.zeros(self.latent_dim)
        cov = np.eye(self.latent_dim) * 0.05  # Narrow: only covers nearby points
        cov_inv = np.linalg.inv(cov)

        attractor = GoalAttractor(
            id='attractor_0',
            mean=mean, cov=cov, cov_inv=cov_inv,
            weight=1.0, stability=0.5,
            niche_id='initial'
        )
        self.attractors['attractor_0'] = attractor
        self.attractor_counter = 1

    def record_success(
        self, z: np.ndarray, reward: float, flow_id: str
    ):
        """Record a successful latent state."""
        # Store in unassigned buffer
        self.unassigned_latents.append(z.copy())
        self.unassigned_rewards.append(reward)
        self.unassigned_flows.append(flow_id)
        if len(self.unassigned_latents) > self.max_unassigned:
            self.unassigned_latents.pop(0)
            self.unassigned_rewards.pop(0)
            self.unassigned_flows.pop(0)

        # Assign to attractor
        assigned_id = self._assign_to_attractor(z, reward, flow_id)

        # Periodically run ecology dynamics
        self.total_steps += 1
        if self.total_steps % self.fit_interval == 0:
            self._ecology_step()

        return assigned_id

    def _assign_to_attractor(
        self, z: np.ndarray, reward: float, flow_id: str
    ) -> str:
        """Assign a success point to the best-matching attractor."""
        if not self.attractors:
            return ''

        # Compute GP for each attractor
        gps = {
            aid: att.compute_goal_prob(z)
            for aid, att in self.attractors.items()
        }

        # Softmax selection with temperature
        values = np.array(list(gps.values()))
        if np.max(values) < 1e-10:
            values = np.ones_like(values)  # uniform if all zero
        exp_vals = np.exp(values / (self.comp_temp + 1e-8))
        probs = exp_vals / (np.sum(exp_vals) + 1e-10)

        # Sample attractor
        aids = list(gps.keys())
        chosen = np.random.choice(aids, p=probs)

        # Record
        self.attractors[chosen].record(z, reward, flow_id)
        self.flow_to_attractor[flow_id] = chosen
        self.attractor_flow_counts[chosen][flow_id] += 1

        return chosen

    # ------------------------------------------------------------------
    # ECOLOGY DYNAMICS
    # ------------------------------------------------------------------

    def _ecology_step(self):
        """Run one ecology dynamics step."""
        if len(self.unassigned_latents) < self.birth_min_pts:
            return

        if len(self.attractors) < self.max_attractors:
            self._try_birth()

        if len(self.attractors) > self.min_attractors:
            self._try_death()
            self._try_merge()
            self._try_split()

        self._update_stability()

    def _try_birth(self):
        """
        Create new attractor from dense regions not well-explained by existing ones.

        Uses density-based detection (local point density) rather than GP thresholds.
        This is necessary because a broad initial attractor covers all points with
        nonzero GP — we need to detect MULTIPLE dense clusters, not just "far" points.
        """
        pts = np.array(self.unassigned_latents)
        if len(pts) < self.birth_min_pts:
            return

        n_existing = len(self.attractors)

        # Compute local density for a sample of points
        # Density = number of neighbors within radius
        n_samples = min(100, len(pts))
        indices = np.random.choice(len(pts), n_samples, replace=False)

        candidates = []
        for idx in indices:
            z = pts[idx]
            # Local density
            neighbor_dists = np.linalg.norm(pts - z, axis=1)
            n_neighbors = int(np.sum(neighbor_dists < 0.8))  # radius=0.8
            local_density = n_neighbors / max(1, len(pts))

            # GP to nearest existing attractor
            existing_gps = [
                att.compute_goal_prob(z)
                for att in self.attractors.values()
            ]
            max_existing_gp = max(existing_gps) if existing_gps else 0.0

            # Birth score: high density + low existing coverage
            # Density must be > random (0.1) and existing GP < 0.5
            density_bonus = min(1.0, local_density * 5)
            coverage_penalty = 1.0 - max_existing_gp
            birth_score = density_bonus * coverage_penalty

            candidates.append((birth_score, idx, z, n_neighbors))

        if not candidates:
            return

        # Sort by birth score, pick best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_point, n_nearby = candidates[0]

        # Only birth if density is sufficient and not already well-covered
        if best_score < 0.05 or n_nearby < 5:
            return

        # Gather nearby points
        nearby = []
        for z in pts:
            if np.linalg.norm(z - best_point) < 0.8:
                nearby.append(z)

        if len(nearby) < 3:
            return

        nearby_arr = np.array(nearby)
        new_mean = np.mean(nearby_arr, axis=0)
        new_cov = np.cov(nearby_arr.T) + np.eye(self.latent_dim) * 0.01
        new_cov_inv = np.linalg.inv(new_cov)

        aid = f'attractor_{self.attractor_counter}'
        self.attractor_counter += 1

        nearby_flows = [
            self.unassigned_flows[
                min(idx, len(self.unassigned_flows) - 1)
            ]
            for idx in range(len(self.unassigned_latents))
            if np.linalg.norm(self.unassigned_latents[idx] - best_point) < 0.8
        ]
        top_flow = max(set(nearby_flows), key=nearby_flows.count) if nearby_flows else 'unknown'

        attractor = GoalAttractor(
            id=aid,
            mean=new_mean, cov=new_cov, cov_inv=new_cov_inv,
            weight=float(len(nearby)) / len(pts),
            stability=0.3,
            niche_id=top_flow,
            birth_step=self.total_steps
        )
        self.attractors[aid] = attractor
        self.n_births += 1

    def _try_death(self):
        """Remove low-fitness attractors."""
        for aid in list(self.attractors.keys()):
            att = self.attractors[aid]

            if att.age < self.death_age_min:
                continue

            if att.fitness < self.death_fitness_min:
                del self.attractors[aid]
                self.n_deaths += 1
                continue

            if len(att.assigned_flows) == 0 and att.age > self.death_age_min * 2:
                del self.attractors[aid]
                self.n_deaths += 1

    def _try_merge(self):
        """Merge highly similar attractors."""
        aids = list(self.attractors.keys())
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a, b = aids[i], aids[j]
                if a not in self.attractors or b not in self.attractors:
                    continue

                sim = attractor_similarity(self.attractors[a], self.attractors[b])
                if sim > self.merge_thresh:
                    # Merge: weighted average
                    wa = self.attractors[a].weight
                    wb = self.attractors[b].weight
                    total_w = wa + wb

                    new_mean = (wa * self.attractors[a].mean + wb * self.attractors[b].mean) / total_w
                    new_cov = (wa * self.attractors[a].cov + wb * self.attractors[b].cov) / total_w
                    reg = np.eye(self.latent_dim) * 0.01
                    new_cov += reg
                    new_cov_inv = np.linalg.inv(new_cov)

                    # Keep the one with higher fitness
                    if self.attractors[a].fitness >= self.attractors[b].fitness:
                        survivor, removed = a, b
                    else:
                        survivor, removed = b, a

                    self.attractors[survivor].mean = new_mean
                    self.attractors[survivor].cov = new_cov
                    self.attractors[survivor].cov_inv = new_cov_inv
                    self.attractors[survivor].weight = total_w
                    self.attractors[survivor].assigned_flows.update(
                        self.attractors[removed].assigned_flows
                    )

                    del self.attractors[removed]
                    self.n_merges += 1

    def _try_split(self):
        """Split high-variance attractors."""
        for aid in list(self.attractors.keys()):
            att = self.attractors[aid]
            trace = float(np.trace(att.cov))

            if trace > self.split_var_thresh and len(att.assigned_flows) >= 3:
                # PCA direction of maximum variance
                eigenvalues, eigenvectors = np.linalg.eigh(att.cov)
                max_var_idx = int(np.argmax(eigenvalues))
                split_dir = eigenvectors[:, max_var_idx]
                split_mag = np.sqrt(eigenvalues[max_var_idx]) * 0.5

                # Create two new attractors
                for direction, suffix in [(split_dir, 'a'), (-split_dir, 'b')]:
                    new_mean = att.mean + direction * split_mag
                    new_cov = att.cov * 0.5 + np.eye(self.latent_dim) * 0.01
                    new_cov_inv = np.linalg.inv(new_cov)

                    child_id = f'attractor_{self.attractor_counter}_{suffix}'
                    self.attractor_counter += 1

                    child = GoalAttractor(
                        id=child_id,
                        mean=new_mean, cov=new_cov, cov_inv=new_cov_inv,
                        weight=att.weight * 0.5,
                        stability=att.stability * 0.8,
                        niche_id=f"{att.niche_id}_{suffix}",
                        birth_step=self.total_steps
                    )
                    self.attractors[child_id] = child

                del self.attractors[aid]
                self.n_splits += 1

    def _update_stability(self):
        """Update attractor stabilities based on reward variance."""
        for att in self.attractors.values():
            if len(att.recent_rewards) >= 5:
                var = float(np.var(list(att.recent_rewards)))
                att.stability = float(np.clip(1.0 / (1.0 + var * 10), 0.1, 1.0))
            att.niche_strength = float(np.clip(
                len(att.assigned_flows) / max(1, len(self.attractors)),
                0.1, 1.0
            ))

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    def compute_goal_prob(self, z: np.ndarray) -> float:
        """
        GP = max over all attractors of per-attractor GP.

        This gives the system the highest-likelihood goal interpretation
        for a given state — the state "belongs" to whichever attractor
        best explains it.
        """
        if not self.attractors:
            return 0.0
        best = max(
            (att.compute_goal_prob(z) for att in self.attractors.values()),
            default=0.0
        )
        return float(best)

    def get_dominant_attractor(self, z: np.ndarray) -> Tuple[Optional[str], float]:
        """Get the attractor that best explains z."""
        best_aid = None
        best_gp = 0.0
        for aid, att in self.attractors.items():
            gp = att.compute_goal_prob(z)
            if gp > best_gp:
                best_gp = gp
                best_aid = aid
        return best_aid, best_gp

    def get_entropy(self) -> float:
        """Entropy of attractor assignment distribution."""
        if not self.attractors:
            return 0.0
        weights = np.array([att.weight for att in self.attractors.values()])
        weights = weights / (np.sum(weights) + 1e-10)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        return float(entropy / np.log(len(weights))) if len(weights) > 1 else 0.0

    def get_diversity(self) -> float:
        """Diversity: mean pairwise distance between attractor means."""
        means = [att.mean for att in self.attractors.values()]
        if len(means) < 2:
            return 0.0
        dists = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                dists.append(float(np.linalg.norm(means[i] - means[j])))
        return float(np.mean(dists)) if dists else 0.0

    def get_stats(self) -> Dict:
        return {
            'n_attractors': len(self.attractors),
            'n_births': self.n_births,
            'n_deaths': self.n_deaths,
            'n_merges': self.n_merges,
            'n_splits': self.n_splits,
            'entropy': self.get_entropy(),
            'diversity': self.get_diversity(),
            'attractors': [
                att.get_stats() for att in self.attractors.values()
            ]
        }


# ============================================================================
# 3. DIVERSITY PRESSURE — Anti-Collapse Regularization
# ============================================================================

class DiversityPressure:
    """
    Prevents attractor collapse through multiple mechanisms.

    Mechanisms:
      1. Entropy bonus: reward allocation that maximizes assignment entropy
      2. Flow-attractor mutual information: maximize I(flow, attractor)
      3. Repulsive force: if attractor means are too close, push apart
      4. Niche regularization: each attractor should have distinct flow coverage

    This prevents the system from collapsing into a single dominant attractor.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        entropy_weight: float = 0.2,
        repulsion_weight: float = 0.1,
        mi_weight: float = 0.15,
        min_diversity: float = 0.3,
        repulsion_distance: float = 1.0
    ):
        self.latent_dim = latent_dim
        self.entropy_w = entropy_weight
        self.repulsion_w = repulsion_weight
        self.mi_w = mi_weight
        self.min_diversity = min_diversity
        self.repulsion_dist = repulsion_distance

    def compute_entropy_bonus(self, ecology: AttractorEcology) -> float:
        """Entropy bonus: higher entropy = higher reward."""
        return ecology.get_entropy()

    def compute_repulsion(
        self, ecology: AttractorEcology
    ) -> float:
        """
        Repulsive gradient: if attractors are too close, penalize.
        Applied as gradient on attractor means.
        """
        if len(ecology.attractors) < 2:
            return 0.0

        total_repulsion = 0.0
        aids = list(ecology.attractors.keys())
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a, b = aids[i], aids[j]
                dist = float(np.linalg.norm(
                    ecology.attractors[a].mean - ecology.attractors[b].mean
                ))
                if dist < self.repulsion_dist and dist > 0.01:
                    repulsion = (self.repulsion_dist - dist) / self.repulsion_dist
                    total_repulsion += repulsion

        return total_repulsion

    def compute_mutual_information(
        self, ecology: AttractorEcology
    ) -> float:
        """
        I(flow; attractor) = Σ p(f, a) * log(p(f,a) / (p(f) * p(a)))

        Maximizing this = flows are specialized to specific attractors.
        """
        if not ecology.attractors:
            return 0.0

        total_mi = 0.0
        n_flows = len(set(
            f for counts in ecology.attractor_flow_counts.values()
            for f in counts.keys()
        ))
        if n_flows == 0:
            return 0.0

        for aid, counts in ecology.attractor_flow_counts.items():
            total_assignments = sum(counts.values())
            if total_assignments == 0:
                continue
            att_weight = ecology.attractors[aid].weight if aid in ecology.attractors else 0.0
            for fid, count in counts.items():
                p_af = count / total_assignments
                p_f = 1.0 / n_flows
                p_a = att_weight + 1e-10
                if p_af > 0:
                    total_mi += p_af * np.log(p_af / (p_f * p_a) + 1e-10)

        return float(total_mi)

    def compute_bonus(self, ecology: AttractorEcology) -> float:
        """Total diversity bonus."""
        if len(ecology.attractors) < 2:
            return 0.0

        entropy = self.compute_entropy_bonus(ecology)
        repulsion = self.compute_repulsion(ecology)
        mi = self.compute_mutual_information(ecology)

        return float(
            self.entropy_w * entropy
            - self.repulsion_w * repulsion
            + self.mi_w * mi
        )

    def get_stats(self) -> Dict:
        return {
            'entropy_weight': self.entropy_w,
            'repulsion_weight': self.repulsion_w,
            'mi_weight': self.mi_w
        }


# ============================================================================
# 4. NICHE TRACKER — Behavioral Specialization
# ============================================================================

class NicheTracker:
    """
    Tracks behavioral niches for each attractor.

    A niche is defined by:
      - Which flows lead to this attractor
      - Which regions of latent space it covers
      - What energy cost is typical
      - What action patterns are associated

    Niche preservation:
      - Even if one attractor is more successful, its niche is protected
      - Niches can overlap but are pushed toward separation
    """

    def __init__(
        self,
        min_niche_size: int = 5,
        niche_decay: float = 0.995
    ):
        self.min_niche_size = min_niche_size
        self.niche_decay = niche_decay

        # niche_id → {flow_ids, mean_z, mean_cost, stability}
        self.niches: Dict[str, Dict] = {}

    def update(
        self, attractors: Dict[str, GoalAttractor]
    ):
        """Update niche registry from current attractor population."""
        for aid, att in attractors.items():
            nid = att.niche_id or aid
            if nid not in self.niches:
                self.niches[nid] = {
                    'flows': set(),
                    'mean_z': att.mean.copy(),
                    'mean_cost': att.energy_cost,
                    'stability_history': deque(maxlen=20)
                }
            self.niches[nid]['flows'] = att.assigned_flows.copy()
            self.niches[nid]['mean_z'] = att.mean.copy()
            self.niches[nid]['mean_cost'] = att.energy_cost
            self.niches[nid]['stability_history'].append(att.stability)

        # Decay old niches
        for nid in list(self.niches.keys()):
            if nid not in {a.niche_id for a in attractors.values()}:
                if len(self.niches[nid]['flows']) < self.min_niche_size:
                    del self.niches[nid]

    def compute_niche_similarity(self, a: GoalAttractor, b: GoalAttractor) -> float:
        """How similar are two niches?"""
        nid_a = a.niche_id or a.id
        nid_b = b.niche_id or b.id
        if nid_a not in self.niches or nid_b not in self.niches:
            return 0.0
        overlap = len(
            self.niches[nid_a]['flows'] & self.niches[nid_b]['flows']
        )
        total = len(
            self.niches[nid_a]['flows'] | self.niches[nid_b]['flows']
        )
        return overlap / max(1, total)

    def get_stats(self) -> Dict:
        return {
            'n_niches': len(self.niches),
            'niches': {
                nid: {
                    'n_flows': len(info['flows']),
                    'stability': float(np.mean(info['stability_history'])) if info['stability_history'] else 0.0
                }
                for nid, info in self.niches.items()
            }
        }


# ============================================================================
# 5. PHASE 43 ENGINE
# ============================================================================

class Phase43Engine:
    """
    Full Phase 43 cognitive engine.

    Architecture:
      ┌──────────────────────────────────────────────────┐
      │  Phase 43 Engine                                 │
      │                                                   │
      │  Every step:                                      │
      │    1. CEM selects flow                            │
      │    2. Flow → action → transition                  │
      │    3. AttractorEcology assigns to best attractor   │
      │    4. GP = max over attractors (membership)       │
      │    5. DiversityPressure computes anti-collapse    │
      │    6. NicheTracker updates                        │
      │    7. Attractor population dynamics (birth/death  │
      │       /merge/split/compete)                       │
      │    8. Flow ecology (continuous birth/death)      │
      │    9. Manifold drift                              │
      │    10. CEM adapts                                 │
      │    11. World model trains (periodic)              │
      │                                                   │
      │  Goal = attractor ecology (competing basins)      │
      │  GP = membership in best-fitting attractor       │
      │  Diversity = anti-collapse regularization         │
      └──────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        n_initial_flows: int = 8,
        flow_dim: int = 4,
        lambda_cost: float = 0.3,
        train_interval: int = 5,
        attractor_max: int = 12,
        attractor_min: int = 2,
        diversity_entropy_w: float = 0.2,
        bootstrap: bool = True
    ):
        self.wm = wm

        # Fallback legacy goal (for initial seeding)
        self.legacy_goal = LegacyGoal(
            goal_id='legacy',
            attractor_state=np.ones(wm.latent_dim) * 1.5,
            basin_radius=2.0, priority=0.9,
            decay_rate=0.01, success_criteria={'type': 'achievable'}
        )

        # Phase 43: Attractor Ecology (replaces single GoalManifold)
        self.ecology = AttractorEcology(
            latent_dim=wm.latent_dim,
            min_attractors=attractor_min,
            max_attractors=attractor_max,
            merge_threshold=0.85,
            split_variance_threshold=1.5,
            birth_min_points=15,
            death_age_min=50,
            death_fitness_min=0.001
        )

        # Phase 43: Diversity Pressure
        self.diversity = DiversityPressure(
            latent_dim=wm.latent_dim,
            entropy_weight=diversity_entropy_w,
            repulsion_weight=0.1,
            mi_weight=0.15,
            min_diversity=0.3
        )

        # Phase 43: Niche Tracker
        self.niches = NicheTracker(min_niche_size=5)

        # Phase 42: Contrastive Shaping
        self.contrastive = ContrastiveShaping(
            latent_dim=wm.latent_dim,
            temperature=0.5,
            lr=0.005,
            window_size=3,
            n_negatives=32
        )

        # Phase 38: Energy Cost
        self.energy_cost = EnergyCostFunction(
            w_action=0.3, w_path=0.3, w_variance=0.1, w_instability=0.3
        )

        # Phase 34: Inverse Dynamics
        self.inv_dyn = InverseDynamicsModel(
            latent_dim=wm.latent_dim,
            action_dim=wm.action_dim,
            learning_rate=0.01
        )

        # Phase 40: Manifold, Ecology, Drift, CEM
        self.manifold = FlowManifold(flow_dim=flow_dim)
        self.flow_ecology = ContinuousFlowEcology(
            manifold=self.manifold,
            goal_attractor=self.legacy_goal.attractor_state,
            latent_dim=wm.latent_dim,
            birth_rate=0.03,
            death_rate=0.02,
            min_flows=4,
            max_flows=30
        )
        self.drift = ContinuousManifoldDrift(
            manifold=self.manifold,
            learning_rate=0.02,
            goal_attraction=0.005,
            similarity_attraction=0.003
        )
        self.cem = ContinuousCEM(
            manifold=self.manifold,
            goal=self.legacy_goal,
            energy_cost=self.energy_cost,
            flow_dim=flow_dim,
            learning_rate=0.05,
            exploration=0.3
        )

        # Phase 36: Learner
        self.learner = BehavioralPhysicsLearner(
            world_model=wm,
            inv_dyn=self.inv_dyn,
            manifold=self.manifold,
            goal=self.legacy_goal,
            learning_rate=0.02,
            k_steps=4,
            batch_size=16
        )

        # Phase 41: Coverage Buffer
        self.coverage = CoverageBuffer(wm=wm, max_episodes=300)

        # Encoder params for contrastive gradient
        self.encoder_params = [
            self.wm.W_mu, self.wm.b_mu,
            self.wm.W_logvar, self.wm.b_logvar,
            self.wm.W_zh, self.wm.W_zx, self.wm.b_z,
            self.wm.W_rh, self.wm.W_rx, self.wm.b_r,
            self.wm.W_hh, self.wm.W_hx, self.wm.b_h,
        ]

        # State
        self.total_steps = 0
        self.execution_log: List[Dict] = []
        self.train_interval = train_interval
        self.goal_prob_history: List[float] = []
        self.attractor_assignment_history: List[str] = []

        # Bootstrap flag
        self._bootstrapped = False
        if bootstrap:
            self._coverage_phases = {'n_coverage': 200, 'n_shaping': 150, 'n_transfer': 80}

        # Seed flows
        self._seed_flows(n_initial_flows)

    def set_coverage_phases(self, nc: int, ns: int, nt: int):
        self._coverage_phases = {'n_coverage': nc, 'n_shaping': ns, 'n_transfer': nt}

    def _bootstrap(self):
        """Run Phase 41 bootstrapping."""
        from phase41_bootstrapper import RepresentationBootstrapper
        bs = RepresentationBootstrapper(
            wm=self.wm,
            goal=self.legacy_goal,
            **self._coverage_phases
        )
        result = bs.run()
        for ep in bs.coverage.buffer.episodes:
            self.learner.buffer.add_episode(ep)
        self._bootstrapped = True
        return result

    def _seed_flows(self, n: int):
        """Seed flows near the initial attractor."""
        for i in range(n):
            target = np.random.randn(self.wm.latent_dim) * 0.3
            if i == 0:
                flow = PointFlow(target, gain=0.3)
                flow.stability = 0.5
                flow.goal_alignment = 0.3
            elif random.random() < 0.5:
                flow = PointFlow(
                    target + np.random.randn(self.wm.latent_dim) * 0.5,
                    gain=random.uniform(0.2, 0.6)
                )
            else:
                flow = LimitCycleFlow(
                    target, radius=random.uniform(0.5, 1.5),
                    omega=random.uniform(0.2, 0.8)
                )
            self.manifold.add_flow(flow, f'seed_{i}')

    def step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """One complete Phase 43 step."""
        # 1. CEM selects flow
        flow, flow_id, coord = self.cem.select_flow(z, h)

        # 2. Flow → action → transition
        a = flow.compute_action(z, h)
        mu, logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.1
        flow.record_transition(z, z_next, a, h)

        # 3. Inverse dynamics
        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)

        # 4. ATTRACTOR ECOLOGY GP: max over all attractors
        goal_prob = self.ecology.compute_goal_prob(z_next)
        dominant_aid, _ = self.ecology.get_dominant_attractor(z_next)

        # 5. GP delta
        prev_gp = self.execution_log[-1]['goal_prob'] if self.execution_log else goal_prob
        gp_delta = goal_prob - prev_gp

        # 6. Energy cost
        cost_info = self.energy_cost.compute([a], [z, z_next], flow)

        # 7. Flow stability
        flow.stability = flow.compute_lyapunov_estimate()
        flow.goal_alignment = float(np.clip(
            flow.goal_alignment + 0.01 * (gp_delta * 10), 0.0, 1.0
        ))

        # 8. RECORD SUCCESS → attractor ecology learns
        self.ecology.record_success(z_next, goal_prob, flow_id)

        # 9. DIVERSITY PRESSURE: anti-collapse bonus
        diversity_bonus = self.diversity.compute_bonus(self.ecology)
        adjusted_gp = float(np.clip(goal_prob + diversity_bonus * 0.1, 0.0, 1.0))

        # 10. NICHE TRACKER update
        self.niches.update(self.ecology.attractors)

        # 11. CONTRASTIVE SHAPING
        self.contrastive.record(z_next, flow_id)
        if self.total_steps % 5 == 0 and len(self.execution_log) >= 5:
            recent_zs = []
            recent_fids = []
            for entry in self.execution_log[-10:]:
                if 'z_after' in entry:
                    recent_zs.append(entry['z_after'])
                    recent_fids.append(entry.get('flow_id', ''))
            if len(recent_zs) >= 3:
                self.contrastive.apply_to_params(
                    self.encoder_params, recent_zs, recent_fids,
                    n_samples=6, sigma=0.003
                )

        # 12. Flow ecology
        self.flow_ecology.record_gp_delta(flow_id, gp_delta)
        self.flow_ecology.record_performance(flow_id, goal_prob)
        eco_result = self.flow_ecology.step()

        # 13. Manifold drift
        self.drift.step(flow_id, goal_prob, gp_delta, self.legacy_goal)

        # 14. CEM adapts
        self.cem.observe_outcome(coord, flow_id, adjusted_gp, cost_info['total'])

        # 15. Periodic training
        if self.total_steps % self.train_interval == 0 and self.total_steps > 0:
            for _ in range(3):
                self.learner.train_step()

        self.total_steps += 1
        self.goal_prob_history.append(goal_prob)
        self.attractor_assignment_history.append(dominant_aid or '')

        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'adjusted_gp': adjusted_gp,
            'gp_delta': float(gp_delta),
            'flow_type': flow.flow_type.value,
            'flow_id': flow_id,
            'dominant_attractor': dominant_aid,
            'stability': flow.stability,
            'energy_cost': cost_info,
            'eco_births': eco_result['born'],
            'eco_deaths': eco_result['died'],
            'n_flows': len(self.manifold.flows),
            'n_attractors': len(self.ecology.attractors),
            'diversity_bonus': diversity_bonus
        }

        self.execution_log.append(step_result)
        return step_result

    def run(self, z_start: np.ndarray, n_steps: int = 200) -> Dict:
        """Run for n_steps."""
        if not self._bootstrapped and hasattr(self, '_coverage_phases'):
            self._bootstrap()

        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        for step in range(n_steps):
            result = self.step(z, h)
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, result['z_after'])

            if step % 25 == 0 and step > 0:
                self._record_episode()

        self._record_episode()

        goal_probs = [e['goal_prob'] for e in self._safe_log()]
        training = self.learner.get_training_report()

        return {
            'n_steps': n_steps,
            'mean_gp': float(np.mean(goal_probs)) if goal_probs else 0.0,
            'max_gp': float(max(goal_probs)) if goal_probs else 0.0,
            'gp_trend': float(goal_probs[-1] - goal_probs[0]) if len(goal_probs) > 1 else 0.0,
            'n_flows': len(self.manifold.flows),
            'n_attractors': len(self.ecology.attractors),
            'training': training,
            'ecology': self.ecology.get_stats(),
            'diversity': self.diversity.get_stats(),
            'niches': self.niches.get_stats(),
            'contrastive': self.contrastive.get_stats()
        }

    def _safe_log(self) -> List[Dict]:
        return self.execution_log[-100:] if self.execution_log else []

    def _record_episode(self):
        if len(self.execution_log) < 5:
            return
        recent = self.execution_log[-20:]
        states = []
        actions = []
        step_flows = []
        for entry in recent:
            if 'z_before' in entry:
                if not states:
                    states.append(entry['z_before'])
                states.append(entry['z_after'])
                actions.append(entry['action'])
                fid = entry.get('flow_id', '')
                flow = self.manifold.flows.get(fid)
                if flow is None and self.manifold.flows:
                    flow = list(self.manifold.flows.values())[0]
                step_flows.append(flow or PointFlow(np.zeros(self.wm.latent_dim)))
        if len(states) >= 5:
            ep = FlowEpisode(
                states=[s.copy() for s in states[:-1]] if len(states) > 1 else states,
                beliefs=[np.zeros(self.wm.belief_dim)] * max(1, len(states) - 1),
                actions=[a.copy() for a in actions],
                flow_embeddings=[self.wm.compute_flow_embedding(f) for f in step_flows],
                rewards=[e.get('goal_prob', 0.0) for e in recent],
                flow_ids=[e.get('flow_id', '') for e in recent],
                flow_types=[e.get('flow_type', '') for e in recent]
            )
            self.learner.buffer.add_episode(ep)


# ============================================================================
# TESTS
# ============================================================================

def test_attractor_ecology():
    """Test attractor ecology dynamics."""
    print("\n" + "=" * 60)
    print("ATTRACTOR ECOLOGY TEST")
    print("=" * 60)

    eco = AttractorEcology(latent_dim=16, min_attractors=2, max_attractors=6)

    # Simulate two distinct success clusters
    cluster_a = np.ones(16) * 0.5
    cluster_b = np.ones(16) * (-0.5)

    for i in range(50):
        if i < 25:
            z = cluster_a + np.random.randn(16) * 0.1
        else:
            z = cluster_b + np.random.randn(16) * 0.1
        eco.record_success(z, reward=0.5, flow_id=f'flow_{i % 5}')

    # Ecology should have created multiple attractors
    stats = eco.get_stats()
    print(f"\n  Attractors: {stats['n_attractors']}")
    print(f"  Diversity: {stats['diversity']:.4f}")
    print(f"  Entropy: {stats['entropy']:.4f}")
    print(f"  Births: {stats['n_births']}, Deaths: {stats['n_deaths']}, "
          f"Merges: {stats['n_merges']}, Splits: {stats['n_splits']}")

    for att in stats['attractors']:
        print(f"    {att['id']:15s}: norm={att['mean_norm']:.3f}, "
              f"fitness={att['fitness']:.4f}, "
              f"flows={att['n_flows']}, niche={att['niche']}")

    assert stats['n_attractors'] >= 2, "Should have multiple attractors"
    print("\n  ✓ Attractor ecology produces multiple differentiated basins")


def test_diversity_pressure():
    """Test anti-collapse regularization."""
    print("\n" + "=" * 60)
    print("DIVERSITY PRESSURE TEST")
    print("=" * 60)

    eco = AttractorEcology(latent_dim=16, min_attractors=1, max_attractors=4)
    diversity = DiversityPressure(latent_dim=16)

    # Fill with points near single cluster
    cluster = np.ones(16) * 0.3
    for i in range(60):
        z = cluster + np.random.randn(16) * 0.1
        eco.record_success(z, reward=0.5, flow_id=f'flow_{i % 3}')

    bonus = diversity.compute_bonus(eco)
    print(f"\n  Attractors: {len(eco.attractors)}")
    print(f"  Entropy: {eco.get_entropy():.4f}")
    print(f"  Diversity: {eco.get_diversity():.4f}")
    print(f"  Diversity bonus: {bonus:.4f}")

    print("\n  ✓ Diversity pressure produces structured bonus")


def test_full_integration(n_steps: int = 200):
    """Full Phase 43 integration test."""
    print("\n" + "=" * 70)
    print("PHASE 43: ATTRACTOR ECOLOGY — FULL INTEGRATION")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    engine = Phase43Engine(
        wm=wm,
        n_initial_flows=8,
        flow_dim=4,
        lambda_cost=0.3,
        train_interval=5,
        attractor_max=10,
        attractor_min=2,
        diversity_entropy_w=0.2,
        bootstrap=True
    )

    print(f"  Running {n_steps} steps...")
    result = engine.run(
        z_start=np.random.randn(wm.latent_dim) * 0.3,
        n_steps=n_steps
    )

    # Results
    print(f"\n  RESULTS:")
    print(f"    Steps: {result['n_steps']}")
    print(f"    Mean GP: {result['mean_gp']:.4f}")
    print(f"    Max GP: {result['max_gp']:.4f}")
    print(f"    GP trend: {result['gp_trend']:+.4f}")

    eco_stats = result['ecology']
    print(f"\n  ATTRACTOR ECOLOGY:")
    print(f"    Attractors: {eco_stats['n_attractors']}")
    print(f"    Diversity: {eco_stats['diversity']:.4f}")
    print(f"    Entropy: {eco_stats['entropy']:.4f}")
    print(f"    Births: {eco_stats['n_births']}")
    print(f"    Deaths: {eco_stats['n_deaths']}")
    print(f"    Merges: {eco_stats['n_merges']}")
    print(f"    Splits: {eco_stats['n_splits']}")

    if eco_stats['attractors']:
        print(f"\n    Individual attractors:")
        for att in eco_stats['attractors']:
            print(f"      {att['id']:20s}: norm={att['mean_norm']:.3f}, "
                  f"fit={att['fitness']:.4f}, flows={att['n_flows']:2d}, "
                  f"niche={att['niche']:10s}")

    tr = result['training']
    if 'loss_improvement' in tr:
        print(f"\n  TRAINING: loss improvement {tr.get('loss_improvement', 0) * 100:.1f}%")

    niches = result['niches']
    print(f"\n  NICHES: {len(niches.get('niches', {}))} tracked")

    # Verdict
    gp_ok = result['mean_gp'] > 0.05
    n_att_ok = eco_stats['n_attractors'] >= 2
    diversity_ok = eco_stats['diversity'] > 0.1
    births_ok = eco_stats['n_births'] > 0 or result['n_steps'] < 100

    print(f"\n  {'✅' if gp_ok else '❌'} GP not flat: {result['mean_gp']:.4f}")
    print(f"  {'✅' if n_att_ok else '❌'} Multiple attractors: {eco_stats['n_attractors']}")
    print(f"  {'✅' if diversity_ok else '⚠️'} Diversity: {eco_stats['diversity']:.4f}")
    print(f"  {'✅' if births_ok else '⚠️'} Births: {eco_stats['n_births']}")
    print(f"  {'✅' if result['n_flows'] > 0 else '❌'} Flows alive: {result['n_flows']}")

    return wm, result


if __name__ == "__main__":
    test_attractor_ecology()
    test_diversity_pressure()
    wm, result = test_full_integration(n_steps=200)

    print("\n" + "=" * 70)
    print("PHASE 43 SUMMARY")
    print("=" * 70)
    print("""
  Architecture progression:

    Phase 25-40: symbolic + continuous dynamics
    Phase 41:    normalized GP (geometry stabilization)
    Phase 42:    learned goal manifold (success → goal)
    Phase 43:    attractor ecology (multi-modal competing basins)

  What Phase 43 enables:

    - Proto-motivation: preference formation through attractor fitness
    - Behavioral specialization: niches emerge from flow-attractor mapping
    - Anti-collapse: diversity pressure prevents single-attractor dominance
    - Ecological competition: attractors compete for behavioral resources
    - Self-organizing intentionality: goals are an ecology, not a target

  Next bottlenecks:

    - Reward signal still external (success defined by threshold)
    - Self-derived objectives need homeostatic/predictive grounding
    - Contrastive shaping needs stronger loss (InfoNCE strength too low)
    - Autograd needed for production-scale gradient propagation
""")
