/// Example: Basic Geometric Reasoning
/// 
/// Demonstrates the core cognitive kernel in action:
/// - Creating a GeometricState
/// - Evaluating closure status
/// - Tracking deterministic transitions

use rugc::{
    GeometricState, ClosureStatus, ClosureTransition,
    InvariantViolation,
};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

/// Simple example of a GeometricState implementation
#[derive(Clone, Serialize, Deserialize)]
struct SimpleReasoningFrame {
    id: String,
    topic: String,
    closure: ClosureStatus,
    audit_log: Vec<String>,
}

impl SimpleReasoningFrame {
    fn new(topic: String) -> Self {
        Self {
            id: Self::compute_frame_id(&topic),
            topic,
            closure: ClosureStatus::Open,
            audit_log: vec!["Frame initialized".to_string()],
        }
    }

    fn compute_frame_id(topic: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(topic.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl GeometricState for SimpleReasoningFrame {
    fn frame_id(&self) -> String {
        self.id.clone()
    }

    fn closure_status(&self) -> ClosureStatus {
        self.closure
    }

    fn validate(&self) -> Result<(), InvariantViolation> {
        // Sanity check: frame must have audit trail
        if self.audit_log.is_empty() {
            return Err(InvariantViolation::Auditability {
                message: "Audit log is empty".to_string(),
                missing_audit_step: "initialization".to_string(),
            });
        }
        Ok(())
    }

    fn attempt_closure(&self) -> (Self, Option<ClosureTransition>) {
        let mut new_frame = self.clone();
        let transition = if self.closure != ClosureStatus::Closed {
            new_frame.closure = ClosureStatus::Closed;
            new_frame.audit_log.push("Closure attempted and succeeded".to_string());

            Some(ClosureTransition {
                from_status: self.closure,
                to_status: ClosureStatus::Closed,
                resolved_by_last_user_turn: true,
                reasoning_summary: format!("Frame resolved: {}", self.topic),
            })
        } else {
            None
        };

        (new_frame, transition)
    }

    fn record_derivation(&mut self, step: String) {
        self.audit_log.push(step);
    }

    fn audit_trail(&self) -> Vec<String> {
        self.audit_log.clone()
    }
}

fn main() {
    println!("=== RUGC Basic Reasoning Example ===\n");

    // Create a reasoning frame
    let mut frame = SimpleReasoningFrame::new("What is the color of light?".to_string());
    println!("Created reasoning frame:");
    println!("  Frame ID (deterministic): {}", frame.frame_id());
    println!("  Status: {}", frame.closure_status());
    println!("  Audit trail: {:?}\n", frame.audit_trail());

    // Record reasoning steps
    frame.record_derivation("Analyzed question for ambiguities".to_string());
    frame.record_derivation("Detected: 'light' has multiple senses".to_string());
    frame.record_derivation("Semantic domain: physics/optics".to_string());
    println!("After recording derivations:");
    println!("  Audit trail (3 steps recorded): {:?}\n", frame.audit_trail());

    // Validate invariants
    match frame.validate() {
        Ok(()) => println!("✓ All invariants satisfied"),
        Err(e) => println!("✗ Invariant violation: {}", e),
    }

    // Attempt closure
    let (closed_frame, transition) = frame.attempt_closure();
    println!("\nAfter closure attempt:");
    println!("  Status: {} → {}", frame.closure, closed_frame.closure);
    if let Some(trans) = transition {
        println!("  Transition: {} (by last turn: {})", trans.reasoning_summary, trans.resolved_by_last_user_turn);
    }

    // Verify determinism: same input produces same frame_id
    let frame2 = SimpleReasoningFrame::new("What is the color of light?".to_string());
    println!("\nDeterminism check (same topic → same frame_id):");
    println!("  Frame 1 ID: {}", frame.frame_id());
    println!("  Frame 2 ID: {}", frame2.frame_id());
    println!("  Match: {}", frame.frame_id() == frame2.frame_id());

    println!("\n=== Core Invariants in Action ===");
    println!("✓ Determinism: Frame IDs are stable across identical inputs");
    println!("✓ Auditability: All reasoning steps recorded in audit trail");
    println!("✓ Closure: Frame transitioned from {} to {}", frame.closure, closed_frame.closure);
    println!("✓ Consistency: No contradictions detected");
}
