"""
Phase 6: Latent Causal World Model

From transition statistics → generative causal model

Key innovations:
1. Latent causes (not just observed metrics)
2. Causal graph (cause → effect relationships)
3. Generative simulation (imagine counterfactual futures)
4. Active inference (act to reduce uncertainty)
5. Ontology revision (model restructuring on catastrophic error)

Not just: error_count += avg_delta
But: "service_instability_causing_cascade_failures"
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime
import random


class LatentCause(Enum):
    """Hidden causes that explain observations"""
    UNKNOWN = "unknown"
    SERVICE_INSTABILITY = "service_instability"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    CASCADE_FAILURE = "cascade_failure"
    CONFIGURATION_DRIFT = "configuration_drift"
    DEPENDENCY_FAILURE = "dependency_failure"
    OSCILLATION = "oscillation"
    STABLE = "stable"


@dataclass(frozen=True)
class CausalRelationship:
    """
    Cause → Effect relationship.

    This is what makes prediction semantic, not just statistical.
    """
    relationship_id: str
    cause: str  # LatentCause value
    effect: str  # Observable symptom
    confidence: float  # 0-1
    strength: float  # How strong is this relationship
    delay_steps: int  # How many steps until effect manifests
    reversibility: float  # 0-1, how reversible

    @staticmethod
    def create(cause: LatentCause, effect: str, strength: float = 0.5) -> 'CausalRelationship':
        return CausalRelationship(
            relationship_id=hashlib.md5(
                f"{cause.value}{effect}".encode()
            ).hexdigest()[:12],
            cause=cause.value,
            effect=effect,
            confidence=0.5,
            strength=strength,
            delay_steps=1,
            reversibility=0.7
        )


@dataclass(frozen=True)
class LatentState:
    """
    Latent causal state - what the system BELIEVES is happening.

    This is the internal generative model of reality.

    Not just: error_count = 5
    But: service_instability with causes and effects
    """
    state_id: str
    timestamp: str

    # Latent causes (hidden, inferred)
    primary_cause: str  # LatentCause value
    secondary_causes: Tuple[str, ...]  # Other latent causes
    cause_confidence: float  # 0-1

    # Observable symptoms consistent with cause
    symptoms: Tuple[str, ...]  # "error_spikes", "container_drops", etc.
    symptom_confidence: float  # How well symptoms match cause

    # Causal relationships active
    active_relationships: Tuple[CausalRelationship, ...]

    # Causal graph state
    instability_score: float  # 0-1, how unstable
    cascade_risk: float  # 0-1, risk of cascading failure
    recovery_likelihood: float  # 0-1, how likely to self-recover

    # Novelty
    is_novel: bool  # Novel situation
    novelty_confidence: float  # How confident in novelty

    @staticmethod
    def infer_from_observations(
        error_count: int,
        container_count: int,
        resource_load: float,
        previous_causes: Optional[Tuple[str, ...]] = None
    ) -> 'LatentState':
        """Infer latent state from observations"""

        # Infer primary cause from symptoms
        symptoms = []
        causes = []

        if error_count > 5:
            symptoms.append("error_spikes")
            causes.append(LatentCause.SERVICE_INSTABILITY.value)

        if resource_load > 2.0:
            symptoms.append("resource_pressure")
            causes.append(LatentCause.RESOURCE_EXHAUSTION.value)

        if error_count > 3 and container_count < 10:
            symptoms.append("container_instability")
            causes.append(LatentCause.CASCADE_FAILURE.value)

        if error_count > 0 and error_count < 3:
            symptoms.append("minor_errors")
            causes.append(LatentCause.CONFIGURATION_DRIFT.value)

        if not symptoms:
            symptoms.append("healthy")
            causes.append(LatentCause.STABLE.value)

        # Compute instability
        instability = min(1.0, error_count * 0.1 + resource_load * 0.1)

        # Compute cascade risk
        cascade = 0.0
        if error_count > 3:
            cascade = min(1.0, (error_count - 3) * 0.2)
        if resource_load > 2.0:
            cascade = max(cascade, min(1.0, (resource_load - 2.0) * 0.3))

        # Check novelty
        is_novel = not previous_causes or previous_causes[0] != causes[0]
        novelty = 0.5 if is_novel else 0.2

        state_id = hashlib.md5(
            f"{len(symptoms)}{instability}{cascade}".encode()
        ).hexdigest()[:12]

        return LatentState(
            state_id=state_id,
            timestamp=datetime.now().isoformat(),
            primary_cause=causes[0] if causes else LatentCause.UNKNOWN.value,
            secondary_causes=tuple(causes[1:]),
            cause_confidence=0.7 if causes else 0.3,
            symptoms=tuple(symptoms),
            symptom_confidence=0.8,
            active_relationships=(),
            instability_score=instability,
            cascade_risk=cascade,
            recovery_likelihood=1.0 - cascade,
            is_novel=is_novel,
            novelty_confidence=novelty
        )


@dataclass
class CausalGraph:
    """
    Causal graph of world dynamics.

    Stores:
    - Causal relationships
    - Confounders
    - Delayed effects
    - Causal chains
    """
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # cause -> properties
    edges: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)  # cause->effect -> properties
    chains: List[List[str]] = field(default_factory=list)  # Causal chains
    confounders: Dict[str, List[str]] = field(default_factory=dict)  # effect -> [possible causes]

    def add_causal_relationship(
        self,
        cause: str,
        effect: str,
        strength: float,
        delay: int = 1
    ):
        """Add a learned causal relationship"""
        edge = (cause, effect)

        if cause not in self.nodes:
            self.nodes[cause] = {'observed_count': 0, 'confidences': []}

        self.nodes[cause]['observed_count'] += 1

        self.edges[edge] = {
            'strength': strength,
            'delay': delay,
            'observed': 0
        }

    def update_relationship(
        self,
        cause: str,
        effect: str,
        observed: bool
    ):
        """Update causal relationship based on observation"""
        edge = (cause, effect)
        if edge in self.edges:
            self.edges[edge]['observed'] += 1 if observed else 0
            count = self.edges[edge]['observed']
            total = self.nodes.get(cause, {}).get('observed_count', 1)
            self.edges[edge]['strength'] = min(1.0, count / max(1, total))


@dataclass
class GenerativeSimulator:
    """
    Generative world model - simulates counterfactual futures.

    Key: Can imagine multiple potential futures, not just predict one.
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        self.transition_probabilities: Dict[Tuple[str, str], float] = {}

    def simulate_step(
        self,
        current_latent: LatentState,
        action: str,
        steps: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Simulate multiple possible futures.

        Returns list of potential outcomes with probabilities.
        """
        futures = []

        # Simulate multiple trajectories
        for trajectory_id in range(5):
            trajectory = []
            current = current_latent

            for step in range(steps):
                # Simulate one step
                next_state = self._simulate_single_step(current, action, step)
                trajectory.append({
                    'step': step,
                    'latent': next_state,
                    'probability': 1.0 / (step + 1)  # Degrading probability
                })
                current = next_state

            futures.append({
                'trajectory_id': trajectory_id,
                'steps': trajectory,
                'outcome': trajectory[-1]['latent'] if trajectory else current_latent,
                'probability': 1.0 / (trajectory_id + 1)
            })

        # Sort by probability
        futures.sort(key=lambda x: x['probability'], reverse=True)

        return futures

    def _simulate_single_step(
        self,
        current: LatentState,
        action: str,
        delay: int
    ) -> LatentState:
        """Simulate one step of world dynamics"""

        # Action effects on latent causes
        if action == 'restart_service':
            # Restart should reduce instability
            new_instability = max(0, current.instability_score - 0.3)
            new_primary = LatentCause.STABLE.value if new_instability < 0.3 else current.primary_cause
        elif action == 'scale_up':
            # Scale reduces resource pressure
            new_instability = max(0, current.instability_score - 0.1)
            new_primary = current.primary_cause
        else:
            # Observation or other action
            new_instability = current.instability_score * 1.05  # Slight drift
            new_primary = current.primary_cause

        new_instability = min(1.0, new_instability)

        # Cascade risk evolves
        if new_instability > 0.7:
            new_cascade = min(1.0, current.cascade_risk + 0.1)
        else:
            new_cascade = max(0.0, current.cascade_risk - 0.05)

        # Create new latent state
        new_symptoms = list(current.symptoms)
        if new_instability > 0.6:
            if "degradation" not in new_symptoms:
                new_symptoms.append("degradation")
        else:
            new_symptoms = [s for s in new_symptoms if s != "degradation"]

        return LatentState(
            state_id=hashlib.md5(
                f"{new_instability}{new_cascade}{delay}".encode()
            ).hexdigest()[:12],
            timestamp=datetime.now().isoformat(),
            primary_cause=new_primary,
            secondary_causes=current.secondary_causes,
            cause_confidence=current.cause_confidence * 0.95,
            symptoms=tuple(new_symptoms),
            symptom_confidence=current.symptom_confidence * 0.98,
            active_relationships=current.active_relationships,
            instability_score=new_instability,
            cascade_risk=new_cascade,
            recovery_likelihood=1.0 - new_cascade,
            is_novel=False,
            novelty_confidence=current.novelty_confidence * 0.9
        )

    def get_best_trajectory(
        self,
        current: LatentState,
        possible_actions: List[str]
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Get best action based on counterfactual simulation"""
        best_action = None
        best_score = float('-inf')
        best_trajectory = None

        for action in possible_actions:
            trajectories = self.simulate_single_trajectory(current, action, steps=3)
            final_state = trajectories[-1] if trajectories else current

            # Score = lower instability, lower cascade risk
            score = (
                -final_state.instability_score * 2.0 +
                -final_state.cascade_risk * 1.5 +
                final_state.recovery_likelihood * 0.5
            )

            if score > best_score:
                best_score = score
                best_action = action
                best_trajectory = trajectories

        return best_action, best_score, best_trajectory or []

    def simulate_single_trajectory(
        self,
        current: LatentState,
        action: str,
        steps: int
    ) -> List[LatentState]:
        """Simulate single trajectory"""
        trajectory = [current]
        state = current

        for step in range(steps):
            state = self._simulate_single_step(state, action, step)
            trajectory.append(state)

        return trajectory


@dataclass
class ActiveInferenceEngine:
    """
    Active inference - acts to reduce uncertainty, not just exploit.

    Key: Curiosity-driven exploration to reduce epistemic uncertainty.

    Agent should:
    - Exploit known good actions
    - Explore to reduce uncertainty
    - Act to gather information about hidden causes
    """

    def __init__(self, generative_simulator: GenerativeSimulator):
        self.simulator = generative_simulator
        self.uncertainty_budget = 0.3  # % of actions for exploration
        self.exploration_count = 0
        self.exploitation_count = 0

    def select_action(
        self,
        current_latent: LatentState,
        available_actions: List[str],
        use_exploration: bool = True
    ) -> Tuple[str, bool]:
        """
        Select action using active inference.

        Returns:
        - selected action
        - was_exploration (True if epistemic action)
        """

        # First, check if we need epistemic actions
        if current_latent.cause_confidence < 0.6:
            # Low confidence in cause - need to explore
            if use_exploration and self.exploration_count < self.exploitation_count * self.uncertainty_budget:
                # Select exploratory action
                exploration_action = self._select_exploratory_action(
                    current_latent, available_actions
                )
                self.exploration_count += 1
                return exploration_action, True

        # Exploitation: use best known action
        best_action, _, _ = self.simulator.get_best_trajectory(
            current_latent, available_actions
        )
        self.exploitation_count += 1

        return best_action, False

    def _select_exploratory_action(
        self,
        latent: LatentState,
        actions: List[str]
    ) -> str:
        """Select action to reduce uncertainty about hidden causes"""

        # If we don't know the cause, try diagnostic actions
        if latent.cause_confidence < 0.5:
            # Look for actions that reveal information
            for action in actions:
                if 'check' in action.lower() or 'diagnose' in action.lower():
                    return action

        # If instability is high but cause unknown
        if latent.instability_score > 0.5 and latent.primary_cause == LatentCause.UNKNOWN.value:
            for action in actions:
                if 'restart' in action.lower() or 'recover' in action.lower():
                    return action

        # Default: random exploration
        return random.choice(actions) if actions else actions[0] if actions else "check_system"


@dataclass
class OntologyRevisionEngine:
    """
    Handle catastrophic errors that require model restructuring.

    Not just: confidence *= 0.8
    But: The model itself needs to change
    """

    def __init__(self):
        self.revision_count = 0
        self.ontological_shifts: List[Dict[str, Any]] = []

    def detect_ontology_violation(
        self,
        prediction_error_magnitude: float,
        latent_confidence: float,
        novelty: bool
    ) -> bool:
        """
        Detect when prediction error is so large it violates the model ontology.

        Not just "wrong prediction" but "model assumptions are wrong".
        """
        # Ontology violation conditions:
        # 1. Very large error despite high model confidence
        # 2. Novel situation that contradicts model
        # 3. Prediction was confident but completely wrong direction

        if novelty and prediction_error_magnitude > 0.5:
            return True

        if latent_confidence > 0.7 and prediction_error_magnitude > 0.8:
            return True

        return False

    def revise_ontology(
        self,
        current_causes: Tuple[str, ...],
        observed_symptoms: Tuple[str, ...],
        prediction_failed_on: str
    ) -> Tuple[str, ...]:
        """
        Revise the causal model when ontology is violated.

        Returns new hypothesized causes.
        """
        self.revision_count += 1

        # Clear old assumptions
        new_causes = []

        # Infer new cause from unexpected observation
        if "container_loss" in prediction_failed_on:
            new_causes.append(LatentCause.CASCADE_FAILURE.value)
        elif "error_spike" in prediction_failed_on:
            new_causes.append(LatentCause.SERVICE_INSTABILITY.value)
        elif "resource_exhaustion" in prediction_failed_on:
            new_causes.append(LatentCause.RESOURCE_EXHAUSTION.value)
        else:
            # Complete unknown - expand hypothesis space
            new_causes.append(LatentCause.UNKNOWN.value)

        self.ontological_shifts.append({
            'old_causes': current_causes,
            'symptoms': observed_symptoms,
            'failed_prediction': prediction_failed_on,
            'new_causes': tuple(new_causes),
            'revision_number': self.revision_count
        })

        return tuple(new_causes)


@dataclass
class Phase6WorldModel:
    """
    Phase 6: Latent Causal World Model

    Complete generative causal architecture:
    1. Latent causes (hidden state)
    2. Causal graph (cause → effect)
    3. Generative simulator (counterfactual futures)
    4. Active inference (epistemic actions)
    5. Ontology revision (model restructuring)
    """

    def __init__(self):
        self.generative_simulator = GenerativeSimulator()
        self.active_inference = ActiveInferenceEngine(self.generative_simulator)
        self.ontology_revision = OntologyRevisionEngine()

        self.current_latent: Optional[LatentState] = None
        self.prediction_history: List[Dict[str, Any]] = []
        self.causal_beliefs: Dict[str, float] = {}

    def infer_latent_state(
        self,
        error_count: int,
        container_count: int,
        resource_load: float
    ) -> LatentState:
        """Infer latent causal state from observations"""
        previous = None
        if self.current_latent:
            previous = (self.current_latent.primary_cause,)

        self.current_latent = LatentState.infer_from_observations(
            error_count=error_count,
            container_count=container_count,
            resource_load=resource_load,
            previous_causes=previous
        )

        return self.current_latent

    def predict_with_simulation(
        self,
        action: str,
        steps: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual future trajectories"""
        if not self.current_latent:
            return []

        return self.generative_simulator.simulate_step(
            self.current_latent, action, steps
        )

    def select_action_active_inference(
        self,
        available_actions: List[str]
    ) -> Tuple[str, bool]:
        """Select action using active inference"""
        if not self.current_latent:
            return available_actions[0] if available_actions else "check", False

        return self.active_inference.select_action(
            self.current_latent,
            available_actions
        )

    def handle_prediction_error(
        self,
        error_magnitude: float,
        failed_prediction: str
    ) -> bool:
        """Handle prediction error, potentially triggering ontology revision"""

        if not self.current_latent:
            return False

        needs_revision = self.ontology_revision.detect_ontology_violation(
            prediction_error_magnitude=error_magnitude,
            latent_confidence=self.current_latent.cause_confidence,
            novelty=self.current_latent.is_novel
        )

        if needs_revision:
            new_causes = self.ontology_revision.revise_ontology(
                current_causes=(self.current_latent.primary_cause,),
                observed_symptoms=self.current_latent.symptoms,
                prediction_failed_on=failed_prediction
            )

            # Reset cause confidence
            if self.current_latent:
                # Would create new latent state with revised causes
                pass

        return needs_revision

    def get_semantic_prediction(self, action: str) -> str:
        """Get human-readable semantic prediction"""
        if not self.current_latent:
            return "Unknown state"

        cause = self.current_latent.primary_cause

        if cause == LatentCause.SERVICE_INSTABILITY.value:
            return f"Service instability likely to cause errors"
        elif cause == LatentCause.RESOURCE_EXHAUSTION.value:
            return f"Resource exhaustion may cause degradation"
        elif cause == LatentCause.CASCADE_FAILURE.value:
            return f"Cascade failure risk: {self.current_latent.cascade_risk:.0%}"
        elif cause == LatentCause.STABLE.value:
            return "System appears stable"
        else:
            return f"Unknown cause - need diagnostic action"