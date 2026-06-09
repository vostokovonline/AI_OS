# Operational Coherence

## Purpose

**Answer the last unresolved question before Phase 18:**

```
How does the system locally determine
that continuation remains coherent?

Not philosophically.
Not through "meaning" or "consistency".

As a computable criterion.
```

---

## The Self-Reference Problem

### Current State

```
LCCS = structure enabling coherence survival
       under reparameterization
       
BUT:
  Coherence defined through compatibility
  Compatibility defined through coherent stitching
  
= Self-referential loop
```

### Why This is Dangerous

Without operational closure, implementation will silently drift to:

```
❌ prediction error
❌ reconstruction loss
❌ KL divergence
❌ latent smoothness
❌ Jacobian stability

All = hidden handcrafted proxies
     that break the ontology
```

---

## Engineering Cut: The Primitive Observable

### Requirement

A **primitive operational observable** that:
- Is locally computable
- Requires no global reference
- Does not smuggle in state-space assumptions
- Can be measured without interpretation

### Candidate Criterion

**Local coherence = stable compressibility under adaptive continuation**

```
C(z) = coherence at z
     = ability of local description
       to preserve low description growth
       under perturbative continuation

NOT just "compressible"
BUT "compressible AND adaptive to continuation pressure"
```

### The Key Word: Adaptive

```
Compression alone is insufficient.

Reason:
  - Dead structure (noise, loops, frozen attractors)
    can be extremely compressible
  - But not adaptive to continuation
  - No meaningful continuation under perturbation

Dead structure:
  - compressible ✓
  - NOT adaptive ✗
  - NOT coherent

Coherent structure:
  - compressible ✓
  - adaptive to continuation ✓
  - survives perturbative probing

C(z) = stable compressibility
     = compression survives adaptive continuation
```

### Why This Works

```
Continuation actively probes the patch.
Coherence survives perturbation.
Compression remains stable.
Local law does not collapse.

If local law collapses under perturbation:
  → coherence breaks
  → patch transition

If local law survives perturbation:
  → coherence maintained
  → patch stable
```

This is NOT:
- Prediction error (requires model)
- Reconstruction loss (requires encoder)
- Statistical regularity (too loose)

This IS:
- Description complexity (information-theoretic)
- Local (no global reference)
- Computable (can be measured)
- Removes hidden primitives

---

## The Compression Criterion

### Definition

```
A local description D at z is coherent
iff D remains compressible under continuation

Specifically:
  Adding future context does not increase
  description complexity of D

Equivalently:
  Local law can extend without information expansion
```

### Formalization

```
C(z) = ability to preserve low description growth
     under perturbative continuation

Adaptive coherence test:
  1. Take local description D_z
  2. Apply perturbative continuation P (probes patch)
  3. Measure: does compression remain stable?
  4. C(z) = compression_stability(D_z | perturbation)

Coherent ⇔ compression survives adaptive probing
Incoherent ⇔ compression breaks under perturbation

This prevents:
  - Dead structure traps (compressible but not adaptive)
  - Overcompressed symbolic stasis
  - Frozen attractors
  - Pathological loops
```

### Intuition

```
If local law survives perturbation
→ compressible AND adaptive
→ coherent

If local law collapses under perturbation
→ compression fails
→ coherence breaks

Dead structure can be compressed
BUT:
  Does NOT survive perturbative continuation
  NOT coherent
```

---

## Patch Stability Criterion

### What is a Patch?

```
Patch P at z = adaptively stable local compression regime

P is stable at z if:
  - D_z is compressible
  - D_z survives perturbative continuation
  - Compression remains stable under probing

P becomes unstable when:
  - Perturbative continuation breaks compression
  - Local law collapses under probing
  - Adaptive coherence fails
```

### Patch Boundary

```
Patch boundary = where adaptive coherence first fails

Where perturbative continuation breaks compression
despite local description being statistically compressible.

Critical distinction:
  - Compression stability (static)
  - Adaptive coherence (under perturbation)

Boundary detects where BOTH fail,
not just compressibility.
```

---

## Transition Criterion

### When Does Chart Transition Occur?

```
Transition occurs when:
  Patch P at z_1 becomes unstable
  New patch Q at z_2 is more coherent
  ΔC(P → Q) > transition cost

Transition condition:
  C_Q(z_2) - C_P(z_1) > τ
  where τ = transition cost threshold
```

### What Triggers Transition?

```
Chart change = compression restructuring

NOT:
  - Prediction error
  - Loss function
  - Gradient
  - Reward

BUT:
  - Compression complexity increase
  - Local law invalidation
  - Description failure
```

---

## Metric Emergence from Compression

### How Metric Arises

```
Local metric g(z) = resistance of C(z) to perturbation

g(z) = ∂²C/∂z² (second derivative of coherence)

High g = small perturbation → coherence drops sharply
Low g  = large perturbation → coherence stable
```

### Why This is Different from Previous Metrics

```
Old: g = bifurcation sensitivity
     g = prediction error rate
     g = Hessian(energy)

New: g = coherence curvature
     g = compressibility stability
     g = how description complexity changes with position
```

### Curvature = Compression Instability

```
R(z) = ∂g/∂z = rate of coherence resistance change

High R = rapidly changing coherence landscape
Low R = flat coherence landscape
R → ∞ = coherence singularity (no stable patch)
```

---

## Distinguishability from Compression

### When Are Futures Distinguishable?

```
F1 and F2 distinguishable
iff they require different local compression structures

Equivalently:
  F1 and F2 belong to different patches
  Compression law differs
  Description complexity behaves differently
```

### Distinguishability = Compression Structure Difference

```
D(F1, F2) = difference in coherence requirements

D = 0  ⇔ same patch (indistinguishable)
D > 0  ⇔ different patches (distinguishable)
```

---

## Local vs Global Computation

### Challenge

```
Compression criterion might require global analysis
(looking at all futures to determine compressibility)

This would violate locality requirement.
```

### Solution: Local Approximation

```
Use LOCAL compression test:

Instead of: K(D_z | all futures)
Use: K(D_z | local neighborhood futures)

With sufficient local sampling,
local compression ≈ global compression
```

### Implementation

```
At each z:
  1. Sample local continuation paths
  2. Build local description D_z
  3. Measure K(D_z | sample futures)
  4. If K stable → patch coherent
  5. If K increases → patch unstable
```

---

## Operational Procedure

### Step 1: Initialize Local Description

```
At position z:
  Build minimal local description D_z
  Using local context C_z
  
D_z = shortest description of C_z
```

### Step 2: Test Compression Stability

```
For each future step t = 1, 2, 3, ...:
  Extend description: D_z → D_z_extended
  
  Measure: K(D_z_extended)
  
  If K increases significantly:
    → Patch unstable
    → Transition needed
  Else:
    → Continue
```

### Step 3: Determine Patch Boundary

```
Patch boundary at z* when:
  K(D_z*_extended) - K(D_z*) > threshold
  
Boundary = first point where compression fails
```

### Step 4: Manage Transitions

```
When patch unstable at z:
  1. Find neighboring positions
  2. Test local coherence at each
  3. Select position with highest C(z')
  4. Transition to new patch
  5. Update atlas
```

---

## Compression as Primitive Observable

### Why Compression is the Right Primitive

| Alternative | Problem |
|-------------|---------|
| Prediction error | Requires model (not primitive) |
| Reconstruction loss | Requires encoder (not primitive) |
| KL divergence | Requires distribution (not primitive) |
| Latent smoothness | Circular (smoothness defined how?) |
| Jacobian stability | Requires parameterization (not primitive) |

| Compression | Works |
|-------------|-------|
| Primitive | No model, no encoder required |
| Local | Can be computed from local context |
| Computable | Algorithmic complexity is well-defined |
| Orthogonal | Not smuggling in state-space |
| Universal | Not domain-specific |

---

## Information-Theoretic Foundation

### Kolmogorov Complexity Connection

```
K(s) = length of shortest program that produces s

K is:
  - Uncomputable in general
  - But local approximations exist
  - And relative complexity is computable
  
We use relative compression:
  K(D | F) = K(D+F) - K(D)
  
This measures: how much does F add to description of D?
```

### Practical Approximation

```
Full Kolmogorov complexity is uncomputable.
But we can use practical approximations:

- gzip compression ratio
- Neural compression (autoencoder loss)
- Minimum description length (MDL)
- Entropy rate

All approximate K locally.
```

### Which Approximation to Use?

```
For implementation, we need computable approximation.

Options:
1. Entropy rate of local context
   - Computable
   - Information-theoretic
   - Requires stationarity assumption
   
2. Neural compression loss
   - Computable
   - Learned
   - Approximates K
   
3. MDL principle
   - Computable
   - Principled
   - Requires discretization

Recommendation: Start with entropy rate approximation
Refine to neural compression if needed
```

---

## Patch Construction Algorithm

### Input

```
Local context at position z:
  Recent continuation samples
  Local pattern statistics
  Neighbor relationships
```

### Output

```
Patch P = {
  center: z
  radius: r (max distance before coherence breaks)
  description: D_z (local compression structure)
  stability: C(z)
  boundary: set of transition points
}
```

### Algorithm

```
1. Start at z
2. Build local description D_z from context
3. Set radius r = 0
4. For increasing radius:
   a. Sample continuation at z + Δ
   b. Measure C(z + Δ) = compression stability
   c. If C drops below threshold:
      - Set boundary at r
      - Stop
   d. Else:
      - Extend radius
5. Return patch P
```

---

## Atlas Construction

### How Patches Form Atlas

```
Atlas A = collection of patches {P_i}
         where:
           - Patches overlap at boundaries
           - Compatibility at overlaps = coherent stitching
           - A covers continuation space
```

### Patch Compatibility

```
Two patches P_i and P_j are compatible
iff at their overlap region:
  - Compression structures agree
  - Local descriptions stitch without conflict
  - Transition is smooth (ΔC is small)
```

### Atlas Consistency

```
Atlas is consistent if:
  All patch pairs are compatible in overlaps
  Global structure emerges from local patches
  
This is the sheaf condition:
  Locally coherent sections
  stitch to globally consistent structure
```

---

## Transition Between Patches

### When Transition Occurs

```
At position z in patch P:
  If C(z) drops below threshold:
    → Find neighboring patch Q with higher C(z')
    → Transition to Q
    → Update current position
```

### Transition Cost

```
Transition cost T(P → Q) = structural reorganization required

High cost:
  Different compression structures
  Need new local description
  Information must be re-encoded
  
Low cost:
  Similar compression structures
  Description transfers
  Minimal reorganization
```

### Optimal Path

```
Path P_0 → P_1 → ... → P_n is optimal
iff total transition cost is minimized

This is the geodesic in compression space
(not in state space)
```

---

## Metric from Patch Structure

### Local Metric

```
g_ij(z) = resistance of patch at z to coordinate change

g = second derivative of compression stability
g = how quickly does C(z) change with position
```

### Curvature from Patch Boundaries

```
Curvature R at z = incompatibility of neighboring patches

High R:
  Neighboring patches have high transition cost
  Atlas has sharp boundaries
  Incoherent regime overlap
  
Low R:
  Neighboring patches compatible
  Smooth atlas structure
  Coherent transitions
```

---

## Cognitive Phenomena Through Adaptive Coherence

| Phenomenon | Adaptive Coherence Meaning |
|------------|---------------------------|
| **Attention** | Monitoring patch stability under perturbation |
| **Confusion** | Cannot find stable adaptive compression |
| **Insight** | New adaptive compression discovered |
| **Habit** | Stable, adaptive, low-cost compression regime |
| **Fatigue** | Adaptive compression cost rises |
| **Emotion** | Global adaptive coherence modulation |
| **Identity** | Recursive adaptive compression pattern |
| **Planning** | Path through adaptive compression space |
| **Memory** | Adaptive pattern persistence |

### What This Prevents

```
Crystallized intelligence traps:
  - Compressible structure that doesn't adapt
  - Dead attractors
  - Overcompressed symbolic stasis
  - Frozen local minima

These are compressible but NOT adaptive.
NOT coherent by corrected definition.
```

---

## Implementation Summary

### Layer 0: Local Compression Test

```python
def coherence(z: Position, context: Context) -> float:
    """
    Compute local coherence C(z)
    
    C = compression stability
    C = how much description extends without expansion
    """
    
    # Build local description
    D_z = build_description(context)
    
    # Sample continuation
    futures = sample_local_continuations(z, context)
    
    # Measure compression stability
    K_D = complexity(D_z)
    K_D_ext = complexity(D_z + futures)
    
    # Coherence = stability of compression
    C = (K_D - K_D_ext) / K_D  # or similar
    
    return C
```

### Layer 1: Patch Construction

```python
def build_patch(z: Position) -> Patch:
    """
    Construct patch around z
    
    Returns patch with center, radius, stability
    """
    
    C = coherence(z)
    radius = 0
    
    while C > threshold:
        radius += step
        z_test = z + radius * direction
        C_test = coherence(z_test)
        
        if C_test < threshold:
            break
    
    return Patch(center=z, radius=radius, stability=C)
```

### Layer 2: Atlas Formation

```python
def build_atlas(coverage: float) -> Atlas:
    """
    Build atlas covering continuation space
    """
    
    patches = []
    
    while coverage < target:
        z = next_uncovered_position()
        patch = build_patch(z)
        patches.append(patch)
        coverage += patch.area
    
    return Atlas(patches=patches)
```

### Layer 3: Transition

```python
def transition(z: Position, current_patch: Patch) -> Position:
    """
    Find next position with stable coherence
    """
    
    C = coherence(z)
    
    if C > threshold:
        return z  # Stay in current patch
    
    # Find neighboring patch with higher coherence
    for neighbor in current_patch.neighbors():
        C_n = coherence(neighbor.center)
        if C_n > C:
            return neighbor.center
    
    # No stable patch found
    return None  # Singularity
```

---

## The Bridge to Phase 18

```
LCCS (formalism)
  ↓
Adaptive coherence (stable compressibility under continuation)
  ↓
Patch stability (adaptively stable local compression regime)
  ↓
Transition criterion (collapse of adaptive compression stability)
  ↓
Metric emergence (resistance to perturbative continuation)
  ↓
Atlas construction (adaptive patch dynamics)
  ↓
Cognitive dynamics (regulation of adaptive coherence)
```

**Ontology operationally closed with adaptive correction.**

**Dead structures (compressible but not adaptive) excluded.**

**Phase 18 can proceed.**

---

## Status

**Primitive operational observable:** Stable compressibility under adaptive continuation

**Definition:** 
- NOT just "compressible"
- BUT "compressible AND survives perturbative continuation"

**What this prevents:**
- Dead structure traps (compressible but not adaptive)
- Frozen attractors
- Overcompressed symbolic stasis
- Pathological loops

**Computation:** 
- Perturbative probing of local compression
- Measure stability under continuation pressure
- Reject dead structures that compress but don't adapt

**Metric emergence:** Second derivative of adaptive coherence

---

## Open Questions (Implementation)

1. **Which compression approximation?** (entropy, neural, MDL)
2. **How many local samples needed?** (locality vs accuracy trade-off)
3. **How to handle non-stationary regimes?** (adaptive threshold)
4. **How to initialize atlas?** (bootstrapping problem)

These are engineering questions, not ontological ones.

The ontology is stable. Implementation can proceed.