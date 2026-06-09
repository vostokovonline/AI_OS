# Phase 18 - Architectural Constraints

## Purpose

Protect the ontology from regression during implementation.

Without these constraints, the system will silently drift back to:
- Latent embeddings
- Transition predictors
- Memory buffers
- World model transformers
- RL with prettier words

This document defines HARD constraints that implementation MUST preserve.

---

## The Ontology Stack (Summary)

```
Level 0: LCCS (Local Coherence Compatibility Structure) [PRIMARY]
    Enables coherence survival under reparameterization
    
Level 1: Recoverable Coherence [DEEPEST PRIMITIVE]
    Compressible + survives perturbation + RECOVERS
    
Level 2: Patches (Recovery Dynamic Regimes)
    Regions with stable recovery dynamics
    NOT states, NOT embeddings
    
Level 3: Atlas (Recovery Compatibility Structure)
    Collection of compatible patches
    No global coordinate system
    
Level 4: Transition (Recovery Transfer)
    Collapse of recovery → transfer to compatible patch
    
Level 5: Metric (Recovery Resistance Field)
    Resistance to perturbation = recovery capacity
    
Level 6: Cognition (Recovery Maintenance)
    Regulation of recoverable coherence
```

---

## Part 1: Forbidden Primitives

### No Global Latent State

FORBIDDEN:
- z_t = encoder(x_t)
- state = embedding(observation)
- latent = model.encode(input)

WHY FORBIDDEN:
- Global coordinate collapse
- Hidden state ontology
- Chart-independent representation
- Destroys locality
- Chart is primary, not state

### No Universal Predictor

FORBIDDEN:
- model.predict(any_future)
- predictor(z, any_context)
- global_world_model(futures)

WHY FORBIDDEN:
- Global predictive law
- Negation of regime ontology
- Forces single chart
- Destroys atlas structure

### No Static Memory Storage

FORBIDDEN:
- memory.append(experience)
- store(state, memory)
- memory_buffer.add(observation)

WHY FORBIDDEN:
- Memory is NOT storage
- Memory = persistence of adaptive coherence structure
- Memory = recovery pattern persistence
- Storage = state ontology

### No Trajectory-First Planning

FORBIDDEN:
- search in state graph
- rollout in latent space
- policy over state transitions
- trajectory optimization

WHY FORBIDDEN:
- State space as primary
- Trajectory = sequence of states
- Planning = search in state graph
- Should be: coherent patch traversal

### No Single Chart Coverage

FORBIDDEN:
- model covers all of state space
- single representation everywhere
- global coordinate system

WHY FORBIDDEN:
- Atlas requires many local charts
- No global description exists
- Chart validity is LOCAL

### No Prediction as Primary

FORBIDDEN:
- loss = prediction_error
- objective = predict_next_state
- reward = prediction_accuracy

WHY FORBIDDEN:
- Prediction = global model
- Should be: coherence maintenance
- Recovery > prediction

### No Compression-Only Criterion

FORBIDDEN:
- good_patch = high_compression
- C(z) = description_length(D_z)

WHY FORBIDDEN:
- Dead structures compress perfectly
- Frozen attractors = high compression
- Must be: recoverable coherence
- Compression + survival + recovery

### No Perfect Optimization

FORBIDDEN:
- minimize loss completely
- compress maximally
- find optimal representation

WHY FORBIDDEN:
- Perfect compression kills recovery
- Optimal = death
- Living requires redundancy
- Over-optimization = frozen

### No State Transition Functions

FORBIDDEN:
- f(s_t) → s_t+1
- transition(state, action) → next_state
- dynamics(z_t, a_t) → z_t+1

WHY FORBIDDEN:
- States as primitives
- Transitions between states
- Should be: patch transition
- Patch collapse → recovery transfer

---

## Part 2: Permitted Primitives

### Local Adaptive Coherence Probes

PERMITTED:
- probe(local_patch, perturbation) → coherence_response
- test_recovery(patch, P) → RecoveryResult
- measure_adaptive_compression(D_z, P)

NOT PERMITTED:
- predict(next_state)

### Patch-Local Descriptions Only

PERMITTED:
- patch.description
- patch.boundary
- patch.compatibility
- patch.recovery_dynamics

NOT PERMITTED:
- global_world_representation
- universal_encoding

### Transition by Recovery Collapse

PERMITTED:
- current_patch.recovery_fails()
- find_neighboring_patch.with_higher_recovery()
- transfer_recovery_capacity()
- atlas_restructure()

NOT PERMITTED:
- state_t → state_t+1
- transition_function(state, action)

### Metric as Recovery Resistance

PERMITTED:
- g(z) = recovery_capacity(z)
- g(z) = resistance to irreversible_damage
- curvature = incompatibility_of_recovery_regimes

NOT PERMITTED:
- g = Hessian(E)
- g = Fisher_metric
- g = distance_in_latent_space

### Atlas as Compatibility Structure

PERMITTED:
- atlas = collection_of_compatible_patches
- atlas.coverage = region_covered
- atlas.compatibility = local_stitching_quality

NOT PERMITTED:
- manifold = embedding_space
- atlas = coordinate_system

### Cognitive Dynamics as Recovery Regulation

PERMITTED:
- attention = monitoring_recovery_capacity
- planning = finding_recovery_friendly_path
- confusion = failed_recovery
- insight = discovered_new_recovery_path

NOT PERMITTED:
- attention = feature_importance
- planning = state_graph_search

---

## Part 3: The Recovery Invariant

### The Hardest Constraint

ALL patches MUST maintain recoverable coherence.

Every implementation decision must satisfy:
- For every patch P: P can recover from perturbations
- P's recovery quality is measurable
- Failed recovery leads to transition
- No frozen/unrecoverable structures allowed

### What This Enforces

1. No over-compression: P must have redundancy for recovery
2. No attractor locking: P must have multiple recovery pathways
3. No damage accumulation: P must track degradation over time
4. No rigidity: P must remain flexible

---

## Part 3b: The VITALITY Invariant (Critical)

### The Problem with Recovery Alone

Recovery can be pathological:

```
1. System always recovers to same attractor
   → Recovery is working, but viability is dying
   
2. Each recovery narrows possibility space
   → Recovery works, but branching capacity dies
   
3. Recovery becomes more rigid over time
   → Compresses better, but adaptability dies
   
4. Recovery destroys alternative pathways
   → Survives perfectly, but futures collapse
   
5. Recovery becomes immune overreaction
   → Fast recovery, but diversity dies
```

This is:
- PTSD (recovery narrows response)
- Scar tissue (heals but reduces function)
- Chronic rigidity (stable but fragile)
- Overfitted systems (robust but dead)

### The VITALITY Criterion

```
NOT: Can recover?
BUT: Can recover WITHOUT collapsing future adaptability?
```

### VITALITY = Preserved Possibility Space

```
V(P) = ability to recover
     + preserved branching capacity
     + maintained alternative recovery paths
     + supported atlas plasticity
     + preserved future compatibility
```

### VITALITY Invariant

```
For every patch P:
  Recovery MUST NOT:
    - Decrease future adaptive capacity
    - Reduce multiplicity of continuation
    - Collapse atlas flexibility
    - Narrow future compatibility
    - Destroy alternative recovery paths
    
  Recovery MUST:
    - Restore coherence
    - Preserve branching capacity
    - Maintain atlas plasticity
    - Support future alternatives
    - Keep possibility space open
```

### What This Prevents

```
Recovery trap: recovery reinforces same pathway
  → VIOLATION: Patch becomes attractor
  → VIOLATION: Alternatives die
  
Degradation trap: recovery succeeds but narrows futures
  → VIOLATION: System survives but viability dies
  
Rigidity trap: each recovery increases rigidity
  → VIOLATION: More stable but less adaptive
  
Monoculture trap: all patches converge to same pattern
  → VIOLATION: Diversity dies
```

### VITALITY Metrics

For each patch, measure:
- Branching capacity: number of viable recovery paths
- Atlas plasticity: how much flexibility atlas has
- Future compatibility: ability to form new patches
- Alternative viability: preservation of counterfactual paths

If any metric degrades over successful recoveries → VIOLATION

### Recovery Quality (Corrected)

```
Old: C(z) = compressible + survives + recovers

New: C(z) = compressible
            + survives
            + recovers
            + PRESERVES FUTURE VIABILITY
```

The last criterion is the hardest and most important.

### This Connects AI_OS to

- Viability theory
- Autopoiesis
- Active inference
- Metastability
- Morphogenetic fields

### Intelligence (Corrected)

```
NOT: Intelligence = recovery quality

BUT: Intelligence =
     maintenance of viable future continuation structure
     under perturbation
```

---

## Part 3c: The Provisional Structure Invariant

### The Trap of Over-Rigid Prohibition

If we prohibit ALL global structure too strictly, we accidentally forbid:

- Emergent meso-scale structure
- Temporary coordinatization
- Transient global organization
- Compressive abstraction layers
- High-level cognition
- Symbolic emergence
- Temporary global synchronization

Living systems HAVE these.
The difference is:

they are PROVISIONAL
not ONTOLOGICALLY PRIMARY

### Corrected Invariant

NOT:
```
FORBIDDEN: global structure
```

BUT:
```
FORBIDDEN: globally privileged permanent coordinate ontology

PERMITTED: provisional emergent global structures
          that are temporary, revisable, non-privileged
```

### What is Allowed (Provisional)

```
✓ Emergent meso-scale patterns (temporary coordination)
✓ Abstraction layers (compressive, revisable)
✓ Temporary global synchronization (transient)
✓ Symbolic structures (emergent, not primitive)
✓ High-level coordinatization (contextual, non-permanent)
✓ Transient organization (not static)
```

### What is Forbidden (Permanent)

```
❌ Global coordinate system that is ontologically primary
❌ Latent state that exists independently of patches
❌ Universal representation that persists forever
❌ Encoder-decoder with fixed state space
❌ Single privileged chart that covers everything
```

### The Key Distinction

```
Provisional global structure:
  - Emerges from local patches
  - Is revisable
  - Can be discarded
  - Is not privileged
  - Is secondary to locality

Permanent global state:
  - Is primitive
  - Cannot be revised
  - Is fixed
  - Is privileged
  - Is primary over locality
```

### What This Means for Implementation

Implementation CAN have:
- Temporary abstractions
- Emergent coordination
- Transitive global patterns
- Symbolic layers

Implementation CANNOT have:
- Fixed global coordinates
- Permanent latent state
- Universal encoder
- Single privileged representation

The test: Is the structure provisional and revisable? → ALLOWED
The test: Is the structure permanent and privileged? → FORBIDDEN

---

## Part 4: Forbidden Architectural Patterns

### Pattern 1: Encoder-Decoder

FORBIDDEN:
```
observation → encoder → latent → decoder → prediction
This is latent state space architecture.
```

REQUIRED:
```
observation → local_coherence_probe → patch → atlas
No global encoding. No latent state.
```

### Pattern 2: World Model

FORBIDDEN:
```
world_model.predict(future_states)
model.fit(observations)
```
This is universal predictor.

REQUIRED:
```
patch.probe(perturbation) → coherence_response
atlas.stitch(local_patches) → compatibility
transition based on recovery failure
```

### Pattern 3: Memory Buffer

FORBIDDEN:
```
memory_buffer.add(experience)
memory.retrieve(query)
```
This is storage-based memory.

REQUIRED:
```
memory = recovery_pattern_persistence
memory = persistence of adaptive_coherence
```

### Pattern 4: State Graph

FORBIDDEN:
```
states = nodes in graph
transitions = edges
planning = search in state graph
```
This is state-first architecture.

REQUIRED:
```
patches = regions with recovery dynamics
transitions = recovery collapse and transfer
planning = recovery-friendly path through atlas
```

### Pattern 5: Policy

FORBIDDEN:
```
policy(state) → action
RL agent with policy
```
Policy over states.

REQUIRED:
```
recovery_regulation(patch, perturbation) → transition
coherent_path_finding(atlas, constraints) → next_patch
```

---

## Part 5: The Sheaf Invariant

### The Core Invariant

AI_OS is a sheaf, not a space.

INVARIANT:
- Global structure = consistent collection of local sections
- No global section exists
- Only local coherent patches

This is NOT:
- Manifold with global coordinates
- Latent space with embeddings
- State space with transitions

This IS:
- Sheaf of compatible patches
- Atlas without global coordinate
- Local coherence compatibility structure

---

## Part 6: Testing for Regression

### How to Detect Ontology Regression

If your implementation has:
- encoder() function
- latent state variable
- global predictor
- memory buffer
- state graph
- trajectory rollout

Then regression has occurred.

### Test Questions

1. Does system have global latent state? → REGRESSION
2. Does system have universal predictor? → REGRESSION
3. Does memory act as storage? → REGRESSION
4. Does planning search in state graph? → REGRESSION
5. Can patches die (lose recovery)? → If no, REGRESSION
6. Does over-compression kill recovery? → REGRESSION

---

## Part 7: The Adaptive Recovery Contract

### The Contract

Every implementation MUST satisfy:

For every patch P:
1. P is created with sufficient redundancy for recovery
2. P is regularly probed with perturbations
3. P's recovery quality is measured continuously
4. P is rejected/failed if recovery degrades
5. P is transitioned if recovery fails
6. No immortal patches allowed

For the atlas:
1. All patches are mutually compatible (sheaf condition)
2. Transitions are governed by recovery transfer
3. Atlas reorganizes around successful recovery
4. No global coordinate system

### The Death of Immortal Structures

In classical ML:
- Parameters can live forever
- Weights stabilize
- Model converges to optimum

In AI_OS:
- All patches can die
- Patches with failed recovery must transition
- No structure is immortal
- Continual vitality required

---

## Summary

### Forbidden Primitives

- No global latent state
- No universal predictor
- No static memory storage
- No trajectory-first planning
- No single chart coverage
- No prediction as primary
- No compression-only criterion
- No perfect optimization
- No state transition functions

### Permitted Primitives

- Local adaptive coherence probes
- Patch-local descriptions
- Transition by recovery collapse
- Metric as recovery resistance
- Atlas as compatibility structure
- Cognitive dynamics as recovery regulation

### Invariants

1. Recoverable Coherence Invariant: Every patch must maintain recoverable coherence.
2. Sheaf Invariant: Atlas = consistent collection of local sections. No global section exists.
3. No Immortal Structures Invariant: All patches can die. Failed recovery → transition.
4. VITALITY Invariant: Recovery must NOT collapse future adaptability.
5. Provisional Structure Invariant: Global structures allowed if PROVISIONAL.

---

## Status

Phase 18 Architectural Constraints: DEFINED

These constraints protect the ontology during implementation.

**Implementation MUST preserve:**
- No global state (permanent, privileged)
- No universal predictor
- Recovery as primary
- Atlas as compatibility structure
- All structures mortal
- VITALITY (future adaptability preserved)
- Provisional structures allowed (temporary, revisable)

**Implementation WILL be tested against these constraints.**

**Any regression → implementation failure.**