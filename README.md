# RUGC: Rust Unified Geometric Cognition

**The first deterministic geometric reasoning engine.**

RUGC is a formal, independent Rust implementation of a semantic compilation stack grounded in geometric invariants. It sits between the **UGC geometric representational layer** and **CPU hardware execution** — the missing middle of deterministic AI.

## Philosophy

RUGC rejects the statistical/probabilistic paradigm that dominates modern AI. Instead, it proposes:

- **Determinism**: Same input + state → identical output (byte-stable reasoning frames)
- **Rigor**: All operations derive from formal geometric and semantic invariants
- **Auditability**: Every conclusion is traceable; every derivation is reproducible
- **Exactness**: CPU-native arithmetic without approximation where it matters

This is not pattern-matching. This is cognition.

## Design Axiom: Geometry Emerges From Stability

RUGC adopts a developmental architecture where geometry is not assumed at initialization.

Naive paradigm:

Predefined geometry -> learning inside it -> reasoning on top of it.

RUGC paradigm:

Stable identity -> exploratory interaction -> relational regularities -> endogenous geometry -> understanding -> intelligence.

### Core Axiom

**GeometricState is a state capable of generating geometry.**

### Container vs Generative Geometry

- Container geometry: geometry is predefined and cognition is inserted into it
- Generative geometry: stability and continuity are established first, then geometry is discovered from repeated interaction

RUGC explicitly follows the generative model.

### Developmental Sequence in RUGC

- Invariant -> state -> attractor -> play -> relationship discovery -> geometry -> understanding -> intelligence
- `GeometricState` provides the persistent substrate and reference stability
- Concept anchors serve as attractors that preserve identity and continuity
- Multi-frame loops provide deterministic exploratory interaction across perspectives
- Emergent concepts encode newly stabilized relational structure

### Why the Current Order Is Intentional

RUGC is not missing geometry. RUGC is implementing the preconditions that make geometry meaningful and auditable.

- Without persistent state, there is no stable reference frame
- Without a stable reference frame, relational terms (near/far, same/different, before/after) are operationally undefined
- Therefore, geometry is a developmental consequence, not a bootstrap primitive

This is why geometric geometry is treated as a later phase milestone, not a starting data type.

For the formal architecture statement, see `ARCHITECTURE_AXIOMS.md`.

## Core Architecture

RUGC is organized into three conceptual layers:

### 1. **Geometric Primitives** (`src/geom/`)

Defines the mathematical foundations:

- `invariants.rs` — **The cognitive kernel**: Core invariants, `GeometricState` trait, `ConstraintEvaluator` trait
- `space.rs` — Coordinate systems, transformations, spatial relationships
- `field.rs` — Semantic field definitions encoding meaning and relationships
- `mode.rs` — Resonance modes for semantic propagation

### 2. **Semantic Reasoning** (`src/cognition/`)

Implements constraint-driven reasoning:

- `node.rs` — Individual reasoning entities
- `constraint.rs` — Semantic constraint definitions
- `evaluator.rs` — Deterministic constraint satisfaction engine
- `scheduler.rs` — Task orchestration and parallel reasoning

### 3. **Deterministic Runtime** (`src/runtime/`)

Ensures reproducible execution:

- `parallel.rs` — Safe, reproducible parallelism without data races
- `logging.rs` — Audit trail recording for full traceability
- `determinism.rs` — Verification that all operations are deterministic

## The Cognitive Kernel

The foundation of RUGC is a three-piece cognitive kernel:

### 1. Core Invariants

All reasoning must satisfy four non-negotiable invariants:

```rust
pub enum CoreInvariant {
    /// Same input + state → identical output
    Determinism,
    
    /// No contradictory conclusions in final state
    Consistency,
    
    /// All frames reach deterministic final states
    Closure,
    
    /// All derivations traceable and reproducible
    Auditability,
}
```

### 2. Geometric State Trait

Defines what a reasoning frame must support:

```rust
pub trait GeometricState: Send + Sync {
    fn frame_id(&self) -> String;              // Deterministic identifier
    fn closure_status(&self) -> ClosureStatus;  // Current state
    fn validate(&self) -> Result<(), InvariantViolation>;
    fn attempt_closure(&self) -> (Self, Option<ClosureTransition>);
    fn record_derivation(&mut self, step: String);
    fn audit_trail(&self) -> Vec<String>;
}
```

### 3. Constraint Evaluator Trait

Defines how constraints are evaluated deterministically:

```rust
pub trait ConstraintEvaluator: Send + Sync {
    type Constraint;
    type EvaluationResult;
    
    fn evaluate(&self, constraint: &Self::Constraint) -> Result<Self::EvaluationResult, InvariantViolation>;
    fn detect_conflicts(&self, constraints: &[Self::Constraint]) -> Vec<(usize, usize, String)>;
    fn resolve_contradictions(&self, constraints: &[Self::Constraint], audit_trail: &mut Vec<String>) -> Result<Vec<Self::Constraint>, InvariantViolation>;
}
```

These three pieces form the foundation. Everything else grows from them.

## Getting Started

### Build

```bash
cargo build --release
```

### Run Examples

Basic geometric reasoning with frame determinism:

```bash
cargo run --example basic_reasoning
```

Constraint satisfaction and conflict detection:

```bash
cargo run --example geometric_constraint
```

Phase 3 multi-frame deterministic cognition demo:

```bash
cargo run --example phase3_multiframe
```

Phase 4 emergent stable-structure demo:

```bash
cargo run --example phase4_emergent
```

Phase 4.5 concept-anchor attractor demo:

```bash
cargo run --example phase45_concept_anchor
```

Phase 4.6 anchor-weighted interpretation demo:

```bash
cargo run --example phase46_anchor_weighted
```

Phase 4.7 anchor-driven emergent concept formation demo:

```bash
cargo run --example phase47_emergent_concepts
```

Phase 5.0 anchor-derived relational distance demo:

```bash
cargo run --example phase50_anchor_distance
```

### Run Tests

```bash
cargo test
```

## Design Principles

### 1. Determinism by Default

- No floating-point unless necessary; prefer exact arithmetic
- No unordered collections; use sorted structures with deterministic ordering
- All state transitions produce audit trails for replay
- Frame IDs are SHA256 hashes of semantic content (stable across runs)

### 2. Invariants Enforced at the Type Level

- `GeometricState` trait enforces frame closure semantics
- `ConstraintEvaluator` trait enforces consistent resolution
- `CoreInvariant` enum documents the four non-negotiable properties

### 3. Auditability as First-Class

- Every semantic step is recorded in an audit trail
- Reasoning is reproducible: given same input + audit trail, replay produces same output
- Contradictions are explicit, not hidden

### 4. Mechanical Verification Ready

- JSON schema validation for semantic tensors
- Trait-based architecture enables mock implementations for testing
- Determinism enables property-based testing across reasoning chains

## Historical Context

RUGC represents a fundamental departure from the GPU-driven statistical era:

| Aspect | Statistical Era | RUGC |
|--------|-----------------|------|
| **Paradigm** | Probabilistic pattern-fitting | Deterministic geometric reasoning |
| **Output** | High-confidence approximations | Formally justified conclusions |
| **Auditability** | Black-box (millions of parameters) | Transparent (traced derivations) |
| **Reproducibility** | Non-deterministic (even with fixed seeds) | Byte-stable (exact replay) |
| **Hardware** | GPU massively parallel | CPU deterministic parallelism |

RUGC is the first system designed for **actual cognition**, not statistical fitting.

## Roadmap

### Phase 1: Cognitive Kernel (Current)
✅ Core invariants and traits defined  
✅ Basic geometric state implementation  
✅ Constraint evaluation framework  

### Phase 2: Semantic Field Implementation (In Progress)
✅ Encode semantic meaning as geometric fields
✅ Implement resonance modes for semantic propagation
✅ Multi-sense disambiguation through field interference

### Phase 3: Deterministic Parallelism (In Progress)
✅ Deterministic scheduler/evaluator contradiction-resolution handoff
✅ Reproducible task scheduling
- Lock-free constraint resolution
- Parallel reasoning without non-determinism

### Phase 3 Acceptance Gates (Enforced in Tests/CI)
- Gate A: Full constraint-to-closure pipeline hash is identical across worker counts
- Gate B: Full pipeline hash is identical across repeated runs
- Gate C: Canonicalized audit traces are byte-stable for replay
- Gate D: Parallel contradiction resolution outputs deterministic resolved constraints

These gates are asserted in `tests/phase3_acceptance.rs` and run in CI.

### Phase 4: Multi-Frame Cognition (In Progress)
✅ Multi-frame deterministic loop: evaluate -> transform -> resolve -> stabilize -> repeat
✅ Cross-frame field sharing and deterministic constraint propagation
✅ Resonance-driven inference in iterative frame updates
✅ Semantic field normalization, compression, and concept clustering primitives
✅ Convergence detection and memory consolidation artifact hashing
- Multi-frame contradiction negotiation policies (advanced)
- Long-horizon iterative reasoning with bounded convergence proofs

### Phase 4 Acceptance Gates (Enforced in Tests/CI)
- Gate E: Multi-frame loop converges in K iterations under configured thresholds
- Gate F: Consolidated memory artifact hash is identical across worker counts
- Gate G: Consolidated memory artifact hash is stable across repeated replays

These gates are asserted in `tests/phase4_convergence.rs` and run in CI.

### Phase 4.5: Concept Anchors (In Progress)
✅ Stable low-energy attractor detection in shared semantic fields
✅ Anchor persistence under deterministic perturbation tests
✅ Anchor registry integrated into multi-frame interpretation loop
✅ Anchor-guided stabilization and self-basis continuity checks
- Internal/external perturbation classifiers driven by anchor drift metrics

### Phase 4.5 Acceptance Gates (Enforced in Tests/CI)
- Gate H: Concept anchors are registered only after persistence threshold is reached
- Gate I: Anchor registry hash is invariant across worker counts
- Gate J: External perturbation changes consolidated memory while preserving at least one anchor basis

These gates are asserted in `tests/phase45_anchors.rs` and run in CI.

### Phase 4.6: Anchor-Weighted Interpretation (In Progress)
✅ Anchor-weighted resonance (amplify near-anchor, dampen far-anchor, highlight contradictions)
✅ Anchor-guided fusion with anchor/field alignment bias
✅ Self-continuity metrics (overlap, drift, stability, anchor-field coherence)
✅ Anchor-driven consolidation scores for internal continuity vs external change
- Anchor-driven adaptive policy learning for long-horizon interpretation

### Phase 4.6 Acceptance Gates (Enforced in Tests/CI)
- Gate K: Anchor-field coherence improves across iterative runs
- Gate L: Anchor-weighted consolidation outputs are invariant across worker counts
- Gate M: Anchor-guided interpretation reduces highlighted contradictions over iterations

These gates are asserted in `tests/phase46_anchor_weighted.rs` and run in CI.

### Phase 4.7: Anchor-Driven Emergent Concept Formation (In Progress)
✅ Emergent concept candidates detected from anchor-aligned stable clusters
✅ Persistent candidate promotion into deterministic emergent concept registry
✅ Emergent constraint synthesis expands internal ontology deterministically
✅ Consolidated memory captures emergent concepts and ontology expansion score
- Adaptive emergent concept pruning/merging over long-horizon memory windows

### Phase 4.7 Acceptance Gates (Enforced in Tests/CI)
- Gate N: Emergent concepts form only after anchor-aligned persistence thresholds
- Gate O: Emergent concept registry and consolidation outputs are worker-invariant
- Gate P: Emergent constraint synthesis expands ontology with replay-stable signatures

These gates are asserted in `tests/phase47_emergent_concepts.rs` and run in CI.

### Phase 4: Cross-Lingual Auditing (Planned)
- Canonical token normalization across languages
- Polarity conflict detection (affirm vs. negate)
- Contradiction counting and marking

### Phase 5: Formal Verification (Future)
- Mechanical proof of invariant satisfaction
- Safety properties formalized in Coq/Lean
- Determinism certified at proof level

### Phase 5.0: Anchor-Derived Relational Distance (In Progress)
✅ First endogenous relational distance derived from anchor basis overlap and emergent concept overlap
✅ Internal continuity and external change deltas integrated into a replay-stable distance score
✅ Deterministic near/far ordering across baseline replay vs perturbed runs
✅ Worker-invariant relational distance computation across parallel schedules
- Extend scalar relational distance into higher-order endogenous manifold construction

### Phase 5.0 Acceptance Gates (Enforced in Tests/CI)
- Gate Q: Relational distance orders near/far (baseline replay is nearer than external perturbation)
- Gate R: Relational distance is invariant across worker counts
- Gate S: Relational distance detects externally injected change signals

These gates are asserted in `tests/phase50_anchor_distance.rs` and run in CI.

### Phase 5.1: Emergent Cognitive Geometry (Planned)
- Promote stable relational regularities into explicit geometric coordinate structures
- Learn geometry from anchor-preserving developmental interaction traces
- Establish endogenous distance, topology, and transformation semantics from replay-stable histories

## Integration with UGC-Model

RUGC depends on the geometric representations defined in UGC-Model (`CSIF`, `RWIF`, resonance geometry) but takes them forward into CPU-native implementation:

- **UGC-Model** defines the *theory* and *representational formats*
- **RUGC** provides the *runtime*, *compiler*, and *cognitive kernel*

Together they form a complete stack from geometric theory to deterministic execution.

## License

Licensed under the Apache License, Version 2.0.

See [LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0 for the full text.

## Contributing

RUGC is foundational work toward open-sourcing after validation on a 31B brain on this system. See the main workspace for context.

---

**RUGC is the counter-proposal to the statistical GPU-driven era.**  
**It is deterministic. It is auditable. It is cognition.**
