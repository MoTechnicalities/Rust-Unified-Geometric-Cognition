/// # RUGC Invariants: The Cognitive Kernel
///
/// This module defines the core invariants, traits, and contracts that form
/// the foundation of deterministic geometric reasoning in RUGC.
///
/// ## Three Core Pieces of the Cognitive Kernel:
///
/// 1. **Core Invariants** (CoreInvariant enum)
///    - Determinism: Same input + state produces identical output
///    - Consistency: No contradictory semantic conclusions
///    - Closure: All frames must reach deterministic final state
///    - Auditability: All derivations are traceable and reproducible
///
/// 2. **Geometric State Trait** (GeometricState trait)
///    - Defines what a semantic/geometric state must support
///    - State representation and deterministic transitions
///    - Validation, closure checking, hash-based identity
///
/// 3. **Constraint Evaluation Trait** (ConstraintEvaluator trait)
///    - Defines how constraints are satisfied deterministically
///    - Resolution with audit trail
///    - Mechanical enforcement of invariants

use serde::{Deserialize, Serialize};
use std::fmt;

/// Core invariants that must hold for all geometric reasoning operations.
/// These are the non-negotiable properties of deterministic cognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreInvariant {
    /// **Determinism**: Same input + state must produce identical output.
    /// Ensures that replaying a reasoning sequence produces byte-stable frames.
    /// Violations: floating-point arithmetic, unordered collections, external randomness.
    Determinism,

    /// **Consistency**: No contradictory semantic conclusions can coexist in final state.
    /// Ensures that polarity conflicts are detected and marked explicitly.
    /// Violations: affirm and negate same concept without contradiction_count > 0.
    Consistency,

    /// **Closure**: All frames must reach a deterministic final state (closed | contradictory | partial).
    /// Ensures no frame remains in ambiguous/open state without explicit clarification needs.
    /// Violations: closure_status not in {open, closed, contradictory, partial}.
    Closure,

    /// **Auditability**: All derivations must be traceable and reproducible.
    /// Ensures that any conclusion can be reconstructed from inputs + audit trail.
    /// Violations: non-deterministic state transitions, lossy intermediate representations.
    Auditability,
}

impl fmt::Display for CoreInvariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreInvariant::Determinism => write!(f, "Determinism"),
            CoreInvariant::Consistency => write!(f, "Consistency"),
            CoreInvariant::Closure => write!(f, "Closure"),
            CoreInvariant::Auditability => write!(f, "Auditability"),
        }
    }
}

/// Result type for constraint violations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantViolation {
    Determinism { message: String },
    Consistency { message: String, contradicting_terms: Vec<String> },
    Closure { message: String, unresolved_frame_id: String },
    Auditability { message: String, missing_audit_step: String },
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvariantViolation::Determinism { message } => {
                write!(f, "Determinism violation: {}", message)
            }
            InvariantViolation::Consistency { message, contradicting_terms } => {
                write!(f, "Consistency violation: {} (terms: {:?})", message, contradicting_terms)
            }
            InvariantViolation::Closure { message, unresolved_frame_id } => {
                write!(f, "Closure violation: {} (frame: {})", message, unresolved_frame_id)
            }
            InvariantViolation::Auditability { message, missing_audit_step } => {
                write!(f, "Auditability violation: {} (missing: {})", message, missing_audit_step)
            }
        }
    }
}

/// Closure status of a geometric frame.
/// Indicates whether a semantic reasoning task has resolved to a final state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClosureStatus {
    /// Frame is still accumulating semantic data; no closure attempt yet.
    Open,

    /// Frame has resolved; all ambiguities resolved, all constraints satisfied.
    Closed,

    /// Frame has contradictions that require user intervention or clarification.
    Contradictory,

    /// Frame is partially closed; needs one or more clarifications to reach Closed.
    Partial,
}

impl fmt::Display for ClosureStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClosureStatus::Open => write!(f, "open"),
            ClosureStatus::Closed => write!(f, "closed"),
            ClosureStatus::Contradictory => write!(f, "contradictory"),
            ClosureStatus::Partial => write!(f, "partial"),
        }
    }
}

impl ClosureStatus {
    pub fn is_final(&self) -> bool {
        matches!(self, ClosureStatus::Closed | ClosureStatus::Contradictory)
    }
}

/// Deterministic transition record for closure state changes.
/// Enables auditability: proves that a state transitioned and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosureTransition {
    pub from_status: ClosureStatus,
    pub to_status: ClosureStatus,
    pub resolved_by_last_user_turn: bool,
    pub reasoning_summary: String,
}

/// TRAIT 1: GeometricState
///
/// The trait that defines what a semantic/geometric state must support.
/// Any implementor represents a reasoning frame that can be validated, checked for closure,
/// and transitioned deterministically.
pub trait GeometricState: Send + Sync + Serialize {
    /// Unique deterministic identifier for this geometric state.
    /// Must be stable: same semantic content → same frame_id.
    fn frame_id(&self) -> String;

    /// Current closure status of this frame.
    fn closure_status(&self) -> ClosureStatus;

    /// Validate internal consistency of this state.
    /// Returns InvariantViolation if any invariant is violated.
    fn validate(&self) -> Result<(), InvariantViolation>;

    /// Attempt to close this frame by resolving all ambiguities/constraints.
    /// Returns a new state with updated closure_status and transition record.
    fn attempt_closure(&self) -> (Self, Option<ClosureTransition>)
    where
        Self: Sized;

    /// Record an audit trail entry for reproducibility.
    /// Implementors track derivation history to satisfy Auditability invariant.
    fn record_derivation(&mut self, step: String);

    /// Get audit trail (all derivation steps) for this frame.
    fn audit_trail(&self) -> Vec<String>;
}

/// TRAIT 2: ConstraintEvaluator
///
/// The trait that defines how constraints are evaluated deterministically.
/// Evaluates satisfaction, detects conflicts, and produces audit trails.
pub trait ConstraintEvaluator: Send + Sync {
    /// Type of constraint this evaluator handles.
    type Constraint;

    /// Type of evaluation result (e.g., satisfied, violated, needs_clarification).
    type EvaluationResult;

    /// Deterministically evaluate whether a constraint is satisfied.
    /// Must produce identical output for identical inputs.
    fn evaluate(&self, constraint: &Self::Constraint) -> Result<Self::EvaluationResult, InvariantViolation>;

    /// Detect conflicts between constraints.
    /// Returns list of contradicting constraint pairs and their conflict description.
    fn detect_conflicts(&self, constraints: &[Self::Constraint]) -> Vec<(usize, usize, String)>;

    /// Resolve contradictions deterministically.
    /// May mark as Contradictory if resolution is impossible without user input.
    /// Must include reasoning in audit trail.
    fn resolve_contradictions(
        &self,
        constraints: &[Self::Constraint],
        audit_trail: &mut Vec<String>,
    ) -> Result<Vec<Self::Constraint>, InvariantViolation>;
}

/// Deterministic constraint system for managing geometric constraints.
/// Implements core invariants enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSystem<C> {
    constraints: Vec<C>,
    audit_trail: Vec<String>,
    consistency_check_count: usize,
}

impl<C> ConstraintSystem<C> {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            audit_trail: Vec::new(),
            consistency_check_count: 0,
        }
    }

    pub fn add_constraint(&mut self, constraint: C) {
        self.constraints.push(constraint);
        self.audit_trail.push(format!("Added constraint #{}", self.constraints.len() - 1));
    }

    pub fn constraints(&self) -> &[C] {
        &self.constraints
    }

    pub fn audit_trail(&self) -> &[String] {
        &self.audit_trail
    }

    pub fn consistency_check_count(&self) -> usize {
        self.consistency_check_count
    }

    pub fn record_step(&mut self, step: String) {
        self.audit_trail.push(step);
    }

    pub fn increment_consistency_checks(&mut self) {
        self.consistency_check_count += 1;
    }
}

impl<C> Default for ConstraintSystem<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closure_status_is_final_works() {
        assert!(!ClosureStatus::Open.is_final());
        assert!(ClosureStatus::Closed.is_final());
        assert!(ClosureStatus::Contradictory.is_final());
        assert!(!ClosureStatus::Partial.is_final());
    }

    #[test]
    fn core_invariants_display() {
        assert_eq!(CoreInvariant::Determinism.to_string(), "Determinism");
        assert_eq!(CoreInvariant::Consistency.to_string(), "Consistency");
        assert_eq!(CoreInvariant::Closure.to_string(), "Closure");
        assert_eq!(CoreInvariant::Auditability.to_string(), "Auditability");
    }

    #[test]
    fn constraint_system_audit_trail() {
        let mut system: ConstraintSystem<String> = ConstraintSystem::new();
        system.record_step("Step 1: Initialize".to_string());
        system.record_step("Step 2: Evaluate".to_string());

        assert_eq!(system.audit_trail().len(), 2);
        assert_eq!(system.audit_trail()[0], "Step 1: Initialize");
    }
}
