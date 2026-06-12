//! RUGC: Rust Unified Geometric Cognition
//!
//! A deterministic geometric reasoning engine built on Rust invariants,
//! grounded in mathematical rigor, and designed for formal verification.
//!
//! ## Architecture
//!
//! RUGC forms the missing middle between the UGC geometric representational layer
//! and CPU hardware execution. It provides:
//!
//! - **Geometric Primitives** (`geom/`): Space, fields, and resonance modes
//! - **Semantic Reasoning** (`cognition/`): Nodes, constraints, and evaluators
//! - **Deterministic Runtime** (`runtime/`): Parallel execution with reproducibility
//!
//! ## Core Invariants
//!
//! All reasoning operations must satisfy four non-negotiable invariants:
//!
//! 1. **Determinism**: Same input + state produces identical output (byte-stable)
//! 2. **Consistency**: No contradictory conclusions in final state
//! 3. **Closure**: All frames reach deterministic final states
//! 4. **Auditability**: All derivations traceable and reproducible
//!
//! These invariants are enforced by the cognitive kernel in `geom::invariants`.

pub mod geom {
    pub mod invariants;
    pub mod space;
    pub mod field;
    pub mod mode;

    pub use invariants::{
        CoreInvariant, ClosureStatus, ClosureTransition, InvariantViolation,
        GeometricState, ConstraintEvaluator, ConstraintSystem,
    };
}

pub mod cognition {
    pub mod node;
    pub mod constraint;
    pub mod evaluator;
    pub mod scheduler;

    pub use constraint::SemanticConstraint;
    pub use evaluator::ConstraintEvalEngine;
    pub use node::SemanticNode;
    pub use scheduler::TaskScheduler;
}

pub mod runtime {
    pub mod parallel;
    pub mod logging;
    pub mod determinism;

    pub use parallel::DeterministicRuntime;
    pub use logging::AuditLogger;
    pub use determinism::DeterminismVerifier;
}

pub use geom::{
    CoreInvariant, ClosureStatus, ClosureTransition, InvariantViolation,
    GeometricState, ConstraintEvaluator, ConstraintSystem,
};

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn core_invariants_accessible() {
        let _determinism = CoreInvariant::Determinism;
        let _consistency = CoreInvariant::Consistency;
        let _closure = CoreInvariant::Closure;
        let _auditability = CoreInvariant::Auditability;
    }

    #[test]
    fn closure_status_transitions() {
        use ClosureStatus::*;
        assert!(!Open.is_final());
        assert!(Closed.is_final());
        assert!(Contradictory.is_final());
        assert!(!Partial.is_final());
    }
}
