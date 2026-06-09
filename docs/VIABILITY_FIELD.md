# Viability Field (V-Field) - Phase 18.5

## Purpose

Build the **Viability Instrumentation Layer** before Phase 18 implementation.

**The Problem:**
VITALITY invariant describes what must be preserved.
But there's no mechanism to **MEASURE** when viability is degrading.
Without this, VITALITY is a post-facto metric — system learns about death too late.

**The Solution:**
Create a V-field that monitors viability in real-time.
V-field = scalar field over continuation space measuring future-accessibility.
This makes vitality a measurable object inside the system.

---

## The Core Insight

```
NOT: VITALITY = post-facto metric
     (system dies, then we measure it)

BUT: V(z) = early warning signal
     (system sees own narrowing before collapse)
```

The system must see its own future-shrinking in real-time.
Not after it's too late.

---

## What is V-Field?

### Definition

```
V-field: V(z) = viability of position z
        = accessible alternative causal paths from z
        = preserved branching capacity
        = maintained atlas plasticity
        = future compatibility
```

### V-Field Properties

```
V(z) ∈ [0, 1]
V(z) = 1: Maximum viability (all paths open)
V(z) = 0: Dead (no alternative paths)
V(z) < threshold: WARNING (viability degrading)
```

### What Makes V-Field Different from Other Metrics?

```
Metric:        Measures static property
Loss:          Measures prediction error
Entropy:       Measures uncertainty
V-field:       Measures future accessibility
               Measures preservation of alternatives
               Measures branching capacity
```

---

## Early Viability Degradation Signals

### Signal 1: Branching Entropy Monitor

```
What: Measures entropy of available recovery paths

B(P) = entropy of recovery path distribution at patch P

B(P) HIGH = many alternative paths = high viability
B(P) LOW  = few alternative paths  = low viability
B(P) → 0   = single attractor lock  = death

Detection:
  If B(P) drops over successful recoveries:
    → VIABILITY DEGRADATION
    → Recovery is killing alternatives
```

### Signal 2: Attractor Compression Detection

```
What: Measures how concentrated recovery paths become

A(P) = concentration of recovery paths at patch P

A(P) LOW  = paths distributed = healthy
A(P) HIGH = paths converging to single attractor
A(P) → 1  = all paths same     = locked

Detection:
  If A(P) increases over time:
    → VIABILITY DEGRADATION
    → System is cementing
```

### Signal 3: Recovery-Path Diversity Tracker

```
What: Measures diversity of recovery outcomes

D(P) = number of distinct recovery results from probe

D(P) HIGH = diverse outcomes    = high viability
D(P) LOW  = few outcomes       = low viability
D(P) = 1  = always same result = dead

Detection:
  If D(P) decreases over time:
    → VIABILITY DEGRADATION
    → Recovery becoming deterministic trap
```

### Signal 4: Causal Reachability Decay

```
What: Measures ability to reach alternative patches

R(P) = ability of patch P to connect to new patches
     = causal reachability from P to atlas

R(P) HIGH = can reach many patches  = high viability
R(P) LOW  = can reach few patches   = low viability
R(P) → 0  = isolated, cannot grow  = death

Detection:
  If R(P) degrades over time:
    → VIABILITY DEGRADATION
    → Atlas is closing
```

### Signal 5: Atlas Plasticity Index

```
What: Measures overall flexibility of atlas

Π(A) = atlas.plasticity
     = how many new patches can form
     = how much topology can change

Π(A) HIGH = atlas can evolve     = healthy
Π(A) LOW  = atlas is static      = dying
Π(A) → 0  = no new patches ever  = dead

Detection:
  If Π(A) decreases:
    → SYSTEM-WIDE VIABILITY DEGRADATION
    → Atlas is cementing
```

---

## V-Field Construction

### Computing V(z)

```
V(z) = f(
  B(z),     # branching entropy at z
  A(z),     # attractor compression at z
  D(z),     # recovery path diversity at z
  R(z),     # causal reachability from z
  Π(A)      # atlas plasticity
)

V(z) should be HIGH when:
  - Many alternative paths (high B)
  - Low concentration (low A)
  - Diverse outcomes (high D)
  - High reachability (high R)
  - Atlas is plastic (high Π)
```

### V-Field Formula

```
V(z) = α₁·normalize(B(z)) 
     + α₂·(1 - normalize(A(z)))  # lower A = higher V
     + α₃·normalize(D(z))
     + α₄·normalize(R(z))
     + α₅·normalize(Π(A))

Where α weights sum to 1.
```

### V-Field Gradient

```
∇V(z) = direction of INCREASING viability
       = where to go to find more viable positions

The system should navigate toward high-V regions.
Low-V regions = viability deserts.
```

---

## V-Field Monitoring

### Continuous V-Field Scan

```
For each patch P in atlas:
  1. Measure B(P), A(P), D(P), R(P)
  2. Compute V(P) from signals
  3. Track V(P) over time
  4. If V(P) drops: alert
  5. If V(P) drops below threshold: PATCH FAILURE

Patch FAILURE condition (updated):
  - Old: Recovery fails → failure
  - New: Recovery succeeds BUT V(P) degrades → FAILURE
  - New: V(P) below threshold regardless of recovery → FAILURE
```

### V-Field Anomaly Detection

```
Anomaly = V(z) drops WITHOUT perturbation

Normal:
  Perturbation → V drops → Recovery → V restores

Anomaly:
  V drops with no perturbation
  → System is dying silently
  → Early warning signal
```

---

## V-Field Navigation

### Viability-Aware Transition

```
Old transition:
  Recovery fails → transition to neighbor

New transition (V-aware):
  1. Recovery fails OR V drops below threshold
  2. Find neighbor with higher V
  3. Transition to highest-V viable neighbor
  4. NOT just recovery quality, but VIABILITY quality

Transition criterion (updated):
  T(P → Q) when:
    V(Q) > V(P) + transition_cost
    AND Q has higher future accessibility
```

### V-Field Gradient Ascent

```
Navigation = gradient ascent on V-field

Current position: z
Viability gradient: ∇V(z)
Next position: z + η·∇V(z)

System moves toward high-viability regions.
Low-viability regions are avoided.
Dead regions are escaped.
```

---

## V-Field and Patch Lifecycle

### Patch Creation (V-aware)

```
Old: Create patch wherever coherence is high

New: Create patch only where:
  - Coherence is high
  - V(z) is high
  - Alternative paths exist
  - Atlas plasticity allows

Patches cannot be created in viability deserts.
```

### Patch Death Conditions (Updated)

```
Patch P dies when ANY of:
  1. Recovery fails (OLD)
  2. V(P) drops below V_min (NEW)
  3. Branching entropy B(P) → 0 (NEW)
  4. Attractor compression A(P) → 1 (NEW)
  5. Diversity D(P) = 1 (always same outcome) (NEW)
  6. Causal reachability R(P) → 0 (NEW)
```

---

## Preventing V-Field Collapse

### V-Field Death Patterns

```
1. Recovery → single attractor:
   B drops, A rises, D → 1
   → V-field collapses
   
2. Atlas aging:
   Π(A) decreases, R(P) drops for all P
   → V-field flattens
   
3. Monoculture convergence:
   All patches become similar
   → No alternative paths anywhere
   → V-field dies everywhere
```

### V-Field Preservation Mechanisms

```
1. Diversity enforcement:
   Patches must maintain D > D_min
   If D drops: REJECT patch
   
2. Plasticity protection:
   Atlas must maintain Π(A) > Π_min
   If Π drops: ATLAS FAILURE
   
3. Reachability maintenance:
   Patches must maintain R(P) > R_min
   If R drops: patch is dying
   
4. Branching preservation:
   B(P) must stay above B_min
   If B drops: recoverability is killing system
```

---

## V-Field and Intelligence

### Intelligence (Updated)

```
NOT: Intelligence = recovery quality

NOT: Intelligence = viability preservation

BUT: Intelligence =
     ability to navigate toward high-V regions
     while maintaining V-field health
     across entire atlas
```

### Intelligent Behavior

```
1. Explore high-V regions
2. Avoid low-V regions
3. Escape dying areas
4. Build patches in viability oases
5. Prevent monoculture
6. Maintain atlas plasticity
```

---

## V-Field Instrumentation API

```python
class VField:
    """
    Viability field over continuation space.
    Makes vitality a measurable object.
    """
    
    def measure_branching_entropy(self, patch: Patch) -> float:
        """B(P) - entropy of recovery paths"""
        
    def measure_attractor_compression(self, patch: Patch) -> float:
        """A(P) - concentration of recovery paths"""
        
    def measure_diversity(self, patch: Patch) -> float:
        """D(P) - number of distinct outcomes"""
        
    def measure_reachability(self, patch: Patch) -> float:
        """R(P) - causal reachability to atlas"""
        
    def measure_atlas_plasticity(self, atlas: Atlas) -> float:
        """Π(A) - atlas flexibility"""
        
    def compute_v(self, patch: Patch) -> float:
        """
        V(P) = weighted combination of signals
        
        V = α₁·B + α₂·(1-A) + α₃·D + α₄·R + α₅·Π
        """
        
    def gradient_v(self, position: Position) -> Vector:
        """
        ∇V(z) = direction of increasing viability
        """
        
    def is_healthy(self, patch: Patch) -> bool:
        """
        Patch is healthy if ALL signals above threshold.
        Recovery alone is NOT enough.
        Viability must be maintained.
        """
        
    def should_transition(self, patch: Patch) -> bool:
        """
        Transition if:
        - Recovery fails (old)
        - OR V(P) below threshold (new)
        - OR any signal below threshold (new)
        """
        
    def find_viable_neighbor(self, patch: Patch) -> Optional[Patch]:
        """
        Find neighbor with higher V.
        Navigate toward high-viability.
        """
```

---

## V-Field and Constraints

### Constraint: VITALITY (Instrumented)

```
OLD: Recovery must NOT collapse future adaptability
     (vague, post-facto)

NEW: V-field must stay above threshold
     V(z) = f(B, A, D, R, Π) must remain measurable
     Any signal dropping → alert
     V below threshold → transition
     
VITALITY is now a REAL-TIME MEASURED quantity.
```

### Constraint: V-Field Health (New)

```
Atlas MUST maintain:
  - Π(A) > Π_min (atlas plasticity)
  - Average V(P) across patches > V_avg_min
  - No death regions spreading
  
If atlas V-field collapses:
  SYSTEM FAILURE
```

### Constraint: No Immortal Structures (Updated)

```
Old: All patches can die

New: All patches MUST maintain V above threshold
     Patches with V < threshold are DEAD
     Cannot survive with low viability
     No immortal low-V patches allowed
```

---

## V-Field Implementation Notes

### Measuring Branching Entropy B(P)

```
1. Probe patch with perturbations
2. Collect recovery outcomes
3. Count distinct outcomes
4. Compute entropy:
   B(P) = -Σ p_i · log(p_i)
   where p_i = probability of outcome i
```

### Measuring Attractor Compression A(P)

```
1. Collect recovery paths
2. Measure distance between paths in outcome space
3. Compute concentration:
   A(P) = 1 - (average inter-path distance / max distance)
   High A = paths converge to single attractor
```

### Measuring Causal Reachability R(P)

```
1. From patch P, attempt transitions to other patches
2. Count successful transitions
3. R(P) = successful_transitions / total_patches
   High R = can reach many patches
```

### Measuring Atlas Plasticity Π(A)

```
1. Track new patch formation rate
2. Track topology changes over time
3. Π(A) = new_patches_last_period / time_period
   High Π = atlas is evolving
```

---

## V-Field Monitoring Schedule

```
Continuous monitoring:
  - V(P) computed after each recovery
  - V-field gradient updated continuously
  - Atlas plasticity tracked hourly

Alert thresholds:
  - V(P) < V_min → WARNING
  - B(P) → 0 → CRITICAL
  - A(P) → 1 → CRITICAL
  - D(P) = 1 → WARNING
  - R(P) → 0 → CRITICAL

System response:
  - WARNING: increase monitoring, find alternatives
  - CRITICAL: transition immediately, patch is dying
```

---

## V-Field and Phase 18 Implementation

### Phase 18 Now Requires:

```
Phase 18.0: LCCS + patches + atlas
Phase 18.5: V-field instrumentation (THIS DOCUMENT)
            ↓
Phase 18.1: V-field implementation
Phase 18.2: V-aware transitions
Phase 18.3: V-field navigation
Phase 18.4: V-field atlas health
```

### Order of Implementation

```
1. FIRST: V-field (viability instrumentation)
   Without this, system can't measure own health
   
2. THEN: Patches with V-field monitoring
   Every patch must track viability signals
   
3. THEN: V-aware transitions
   Navigation based on V-field, not just recovery
   
4. THEN: Atlas-wide V-field health
   System-level viability monitoring
```

---

## Summary: What V-Field Adds

| Old | New |
|-----|-----|
| VITALITY = text constraint | VITALITY = real-time measured V-field |
| Recovery = primary signal | V-field = multi-signal health check |
| Patch dies when recovery fails | Patch dies when V drops OR recovery fails |
| No early warning | V-field detects dying BEFORE collapse |
| Post-facto viability | Real-time viability monitoring |

---

## The Deep Insight

```
Intelligence is not:
  - Prediction quality
  - Recovery speed
  - Compression efficiency

Intelligence is:
  - V-field navigation
  - High-V region seeking
  - Viability maintenance
  - Future accessibility preservation
```

---

## Status

**Phase 18.5: V-Field Defined**

V-field transforms VITALITY from constraint to measurable.

**V(z) = f(B, A, D, R, Π) = real-time viability signal**

**System can now:**
- Detect own narrowing in real-time
- Navigate toward high-viability
- Detect dying before death
- Prevent monoculture collapse
- Maintain atlas plasticity

**This is the missing diagnostic layer.**

**Without V-field: VITALITY is philosophy.**
**With V-field: VITALITY is engineering.**

---

## Next Step

Phase 18 implementation can now proceed WITH V-field instrumentation.

The system will have:
1. Patches (local coherent regimes)
2. Atlas (compatible patches, sheaf structure)
3. V-field (viability monitoring, real-time health)
4. Constraints (recovery + vitality + provisional)
5. Navigation (V-field gradient ascent)