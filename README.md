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
- Encode semantic meaning as geometric fields
- Implement resonance modes for semantic propagation
- Multi-sense disambiguation through field interference

### Phase 3: Deterministic Parallelism (Planned)
- Lock-free constraint resolution
- Reproducible task scheduling
- Parallel reasoning without non-determinism

### Phase 4: Cross-Lingual Auditing (Planned)
- Canonical token normalization across languages
- Polarity conflict detection (affirm vs. negate)
- Contradiction counting and marking

### Phase 5: Formal Verification (Future)
- Mechanical proof of invariant satisfaction
- Safety properties formalized in Coq/Lean
- Determinism certified at proof level

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
