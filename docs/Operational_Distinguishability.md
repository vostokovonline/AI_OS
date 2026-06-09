# Operational Distinguishability

## Purpose

Formalize Level 0: **predictive distinguishability structure D**.

This is the computable foundation of AI_OS.

**Not philosophy. Not metaphor. Operational definition.**

---

## The Core Definition

### What is Distinguishability?

```
Two futures F1 and F2 are DISTINGUISHABLE
iff there exists NO single predictive model M
that can stably jointly predict both F1 and F2.

Two futures F1 and F2 are INDISTINGUISHABLE
iff there exists a predictive model M
that stably predicts both F1 and F2.
```

### Key Insight

Distinguishability is NOT about:
- Distance in state space
- Information content
- Causal differences
- Probability distributions

Distinguishability is about:
**Bifurcation of predictive structure**

If prediction must split → distinguishable.
If prediction can unify → indistinguishable.

---

## The Bifurcation Criterion

### Formal Definition

```
D(F1, F2) = 1 (distinguishable) ⇔
  ∀ predictive models M:
    P(bifurcate(M, F1, F2)) > threshold
    
D(F1, F2) = 0 (indistinguishable) ⇔
  ∃ M such that:
    P(predict_stably(M, {F1, F2})) > threshold
```

### Practical Interpretation

Two futures are distinguishable when:
1. Any unified model attempting to predict both must "split"
2. The model cannot maintain coherent prediction across both
3. Prediction error increases when trying to jointly model them

Two futures are indistinguishable when:
1. A single model structure can predict both
2. The model maintains stable coherent predictions
3. No bifurcation needed in predictive structure

---

## How Geometry Emerges from Distinguishability

### From Bifurcation to Metric

The **metric g** is defined as:

```
g(z1, z2) = sensitivity of bifurcation structure

If small change in z causes bifurcation in predictions → high g
If small change in z does not affect predictions → low g
```

### Geometric Interpretation

| Distinguishability Structure | Geometry |
|-----------------------------|----------|
| Gradual distinguishability change | Low curvature |
| Rapid distinguishability change | High curvature |
| Bifurcation boundary | Curvature singularity |
| Stable indistinguishability | Flat region / attractor |

### From Metric to Manifold

The manifold emerges from distinguishability gradients:

```
Chart at z = local coordinate system of distinguishability classes
Atlas = collection of charts covering distinguishability structure
Geodesic = path minimizing distinguishability bifurcation
```

---

## How State z Emerges from Distinguishability

### State as Equivalence Class

z is NOT a primitive. z is RECONSTRUCTED from distinguishability:

```
z = local coordinateization of equivalence class of indistinguishable futures
```

### Reconstructing z from D

```
Given: D (distinguishability structure)
Construct: equivalence classes under D

z_i = coordinate in equivalence class E_i
iff: ∀ F ∈ E_i: D(F, F_i) < threshold for representative F_i
```

### Properties of Reconstructed z

1. **z is not given. It emerges from D.**
2. **z is local chart, not global coordinate**
3. **z can be reconstructed from any point via distinguishability tests**
4. **z continuity = distinguishability continuity**

---

## The Distinguishability Operator

### D as Primary Structure

D is the fundamental operator:

```
D : (Future, Future) → {0, 1, [0,1]}

D(F1, F2) = degree of distinguishability
```

### Properties of D

1. **Symmetry**: D(F1, F2) = D(F2, F1)
2. **Reflexivity**: D(F, F) = 0
3. **Triangle inequality**: D(F1, F3) ≤ D(F1, F2) + D(F2, F3)
4. **Non-transitive**: NOT F1 ~ F2 and F2 ~ F3 ⇒ F1 ~ F3

### Non-transitivity is Critical

This distinguishes D from equivalence relations:

```
F1 and F2 indistinguishable
F2 and F3 indistinguishable
BUT F1 and F3 may be distinguishable

This creates the topology of distinguishability classes
```

---

## Bifurcation Dynamics

### When Does Bifurcation Occur?

Bifurcation happens when:

```
∂²P/∂F² ≠ 0  (second derivative of prediction wrt future)

Or equivalently:

∇²(distinguishability) ≠ 0
```

### Bifurcation Types

| Type | Distinguishability Effect |
|------|--------------------------|
| **Soft** | Model branches but maintains coherence |
| **Hard** | Model must completely split structure |
| **Phase transition** | Global restructuring of distinguishability |

### Bifurcation Sensitivity = Curvature

```
Curvature R(z) = sensitivity of bifurcation structure to perturbations

High R = small perturbation causes large predictive bifurcation
Low R = perturbation doesn't affect predictive structure
```

---

## Operationalizing Distinguishability

### Learning D from Data

The distinguishability structure must be LEARNED:

```
1. Observe pairs of futures (F1, F2)
2. For each pair, test if unified model can predict both
3. Build distinguishability graph
4. Extract equivalence classes
5. Reconstruct local coordinates z
```

### Implementation Approach

```python
class DistinguishabilityStructure:
    """
    Level 0: Predictive distinguishability structure
    
    D(F1, F2) is learned from trajectory data.
    """
    
    def __init__(self):
        self.futures = []          # Observed futures
        self.models = []           # Predictive models
        self.distinguishability_graph = {}
    
    def observe_pair(self, F1: Future, F2: Future, outcome: str):
        """Observe two futures and test joint predictability."""
        
        # Can a unified model predict both?
        unified_possible = self._test_joint_predictability(F1, F2)
        
        # If not, they are distinguishable
        if not unified_possible:
            self.distinguishability_graph[(F1, F2)] = 1.0
    
    def _test_joint_predictability(self, F1, F2) -> bool:
        """
        Test if exists model that stably predicts both F1 and F2.
        
        This is the key operation.
        """
        # Train unified model on both
        unified_model = train_on_both(F1, F2)
        
        # Test prediction stability
        error_unified = test_joint_error(unified_model, {F1, F2})
        error_separate = test_separate_errors(unified_model, F1, F2)
        
        # If unified error ≪ separate errors → indistinguishable
        # If unified error >> → distinguishable
        return error_unified < threshold
    
    def construct_equivalence_classes(self):
        """Reconstruct z from distinguishability structure."""
        
        # Build distinguishability graph
        # Extract connected components under indistinguishability
        # These ARE the equivalence classes (states z)
        
        classes = extract_connected_components(self.distinguishability_graph)
        
        return classes
    
    def reconstruct_z(self, future: Future) -> List[float]:
        """
        Reconstruct local coordinate z from distinguishability.
        
        z is NOT given. It is computed from equivalence class position.
        """
        
        # Find equivalence class of future
        eq_class = self._find_equivalence_class(future)
        
        # z = position in distinguishability manifold
        # Computed from distances to other equivalence classes
        z = self._compute_distinguishability_coordinates(eq_class)
        
        return z
```

---

## From Distinguishability to Metric Tensor

### The Metric Emerges Naturally

The metric tensor g is defined by bifurcation sensitivity:

```
g_ij(z) = ∂²(bifurcation) / ∂z_i ∂z_j
```

### Practical Computation

```
1. For each z, find its equivalence class E_z
2. Perturb z slightly → z + ε
3. Find equivalence class of perturbed point
4. g_ij = change in equivalence class / ε²
```

### Properties of g

- **Positive definite**: More perturbation → more distinguishability
- **Anisotropic**: Direction matters (not isotropic)
- **Local**: Defined at each z
- **Learned**: From actual bifurcation behavior

---

## Memory as Distinguishability Persistence

### Old View: Memory = Stored Information

### New View: Memory = Distinguishability Constraints

```
Memory = patterns of distinguishability that persist across contexts

Memory(z1, z2) = how consistently F1 and F2 remain distinguishable
                across different conditions
```

### Memory Operations

1. **Persist**: Increase stability of distinguishability between certain futures
2. **Retrieve**: Find futures with similar distinguishability patterns
3. **Forget**: Reduce distinguishability stability (allow merging)

---

## Goals as Distinguishability Asymmetry

### Old View: Goal = Target State

### New View: Goal = Preferred Distinguishability Landscape

```
Goal = target bifurcation structure

Goal is NOT "reach state z*"
Goal is "shape distinguishability such that X and Y become distinguishable"
```

### Goal-Directed Dynamics

```
If current D creates indistinguishability between F_wanted and F_unwanted:
  → shape D to make them distinguishable
  → this IS goal achievement
```

---

## Cognitive Phenomena Through Distinguishability

| Phenomenon | Distinguishability Definition |
|------------|-------------------------------|
| **Attention** | Focus on high-distinguishability-gradient regions |
| **Planning** | Find paths that maintain desired distinguishability |
| **Confusion** | Conflicting distinguishability requirements (can't satisfy both) |
| **Insight** | Discovery of new distinguishability structure |
| **Habit** | Stable distinguishability = low bifurcation |
| **Identity** | Recursive distinguishability invariance |
| **Emotion** | Global distinguishability field modulation |

---

## Attention Allocation

### How Attention Emerges from D

```
Attention(z) = gradient of distinguishability at z

High attention = high distinguishability gradient
  → small changes cause large predictive bifurcation
  → need to track carefully
  
Low attention = flat distinguishability
  → perturbations don't affect predictions much
  → can "look away"
```

### Implementation

```python
def compute_attention(z: List[float], D: DistinguishabilityStructure) -> float:
    """
    Attention based on distinguishability gradient.
    """
    
    # Compute distinguishability around z
    dD = gradient_of_distinguishability(z, D)
    
    # Attention proportional to gradient magnitude
    attention = magnitude(dD)
    
    return attention
```

---

## Confusion and Insight

### Confusion = Distinguishability Conflict

```
Confusion occurs when:
  D(F1, F2) = low (indistinguishable)
  But task requires D(F1, F2) = high (distinguishable)

System cannot satisfy distinguishability constraint.
```

### Insight = Distinguishability Restructuring

```
Insight occurs when:
  New distinguishability structure discovered
  Previously indistinguishable futures become distinguishable
  Or vice versa
  
The manifold topology changes.
```

---

## Critical Implementation Questions

### Q1: How to test joint predictability?

```
Option A: Train neural network, test if it can predict both
Option B: Use information-theoretic criterion
Option C: Check if prediction error is minimized together
Option D: Statistical tests for joint predictability
```

### Q2: How to handle continuous distinguishability?

```
Currently defined as binary: distinguishable or not.

But distinguishability might be continuous:
  D(F1, F2) ∈ [0, 1]
  
How to operationalize continuous distinguishability?
```

### Q3: How to learn D efficiently?

```
Testing all pairs of futures is O(n²).

How to learn D structure efficiently?
- Active learning (test most informative pairs)
- Clustering (find equivalence classes without testing all)
- Transfer (learn D from similar contexts)
```

### Q4: How to handle context-dependence?

```
Distinguishability may depend on context C:
  D(F1, F2 | C)

How to incorporate context into distinguishability structure?
```

---

## The Learning Signal

### What Trains D?

The learning signal for D is **prediction consistency**:

```
If model M predicts futures F1 and F2 with low joint error:
  → D(F1, F2) = low (indistinguishable)
  
If model M cannot predict both without high error:
  → D(F1, F2) = high (distinguishable)
```

### D is learned from prediction failures

```
Distinguishability = inverse of joint predictability

Where joint predictability fails = distinguishability increases
```

---

## Summary: Operational D

```
D is NOT a function between states.

D is the primary structure of distinguishability.

D defines:
  - Which futures are distinguishable
  - Where bifurcation occurs
  - How geometry emerges
  - How state z is reconstructed

D is learned from prediction behavior.

Metric g emerges from bifurcation sensitivity.

Manifold M is local reconstruction of D.

Cognitive phenomena are regulation of distinguishability.
```

---

## Next Steps

1. **Define atomic distinguishability test**
   - How to determine if two futures are distinguishable?
   - Implementation of joint predictability test

2. **Define bifurcation detection**
   - When does predictive model bifurcate?
   - How to measure bifurcation sensitivity?

3. **Define state reconstruction**
   - How to compute z from D structure?
   - How to maintain consistency of reconstruction?

These operational definitions will be the foundation for Phase 18 implementation.

---

## Status

**Level 0**: Distinguishability structure D [OPERATIONALIZED]

**Key insight**: Distinguishability = bifurcation of joint predictability

**Next**: Atomic operations and learning procedures