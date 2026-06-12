/// Example: Geometric Constraint Evaluation
///
/// Demonstrates the constraint evaluation system:
/// - Defining constraints on geometric reasoning
/// - Evaluating constraint satisfaction
/// - Detecting conflicts between constraints
/// - Maintaining audit trails for all operations

use rugc::{ConstraintEvaluator, ConstraintSystem, InvariantViolation};

/// Example constraint: a semantic assertion with polarity
#[derive(Clone, Debug)]
struct SemanticAssertion {
    subject: String,
    predicate: String,
    polarity: bool, // true = affirmed, false = negated
}

/// Example constraint evaluator
struct SemanticAssertionEvaluator;

impl ConstraintEvaluator for SemanticAssertionEvaluator {
    type Constraint = SemanticAssertion;
    type EvaluationResult = bool; // true = satisfied, false = violated

    fn evaluate(&self, constraint: &Self::Constraint) -> Result<Self::EvaluationResult, InvariantViolation> {
        // Satisfaction check: constraint is satisfied if it's well-formed
        let is_well_formed = !constraint.subject.is_empty() && !constraint.predicate.is_empty();
        Ok(is_well_formed)
    }

    fn detect_conflicts(&self, constraints: &[Self::Constraint]) -> Vec<(usize, usize, String)> {
        let mut conflicts = Vec::new();

        // Check for contradictions: same (subject, predicate) with opposite polarity
        for i in 0..constraints.len() {
            for j in (i + 1)..constraints.len() {
                if constraints[i].subject == constraints[j].subject
                    && constraints[i].predicate == constraints[j].predicate
                    && constraints[i].polarity != constraints[j].polarity
                {
                    conflicts.push((
                        i,
                        j,
                        format!(
                            "Contradiction: '{}' is {} '{}' vs {} '{}'",
                            constraints[i].subject,
                            if constraints[i].polarity { "affirmed" } else { "negated" },
                            constraints[i].predicate,
                            if constraints[j].polarity { "affirmed" } else { "negated" },
                            constraints[j].predicate
                        ),
                    ));
                }
            }
        }

        conflicts
    }

    fn resolve_contradictions(
        &self,
        constraints: &[Self::Constraint],
        audit_trail: &mut Vec<String>,
    ) -> Result<Vec<Self::Constraint>, InvariantViolation> {
        let conflicts = self.detect_conflicts(constraints);

        if !conflicts.is_empty() {
            for (i, j, desc) in &conflicts {
                audit_trail.push(format!(
                    "Contradiction detected between constraint {} and {}: {}",
                    i, j, desc
                ));
            }

            return Err(InvariantViolation::Consistency {
                message: format!("{} contradiction(s) detected", conflicts.len()),
                contradicting_terms: constraints.iter().map(|c| c.predicate.clone()).collect(),
            });
        }

        audit_trail.push("All constraints are consistent".to_string());
        Ok(constraints.to_vec())
    }
}

fn main() {
    println!("=== RUGC Geometric Constraint Example ===\n");

    // Create a constraint system
    let mut system: ConstraintSystem<SemanticAssertion> = ConstraintSystem::new();
    let evaluator = SemanticAssertionEvaluator;

    // Add consistent constraints
    println!("Adding consistent constraints:");
    system.add_constraint(SemanticAssertion {
        subject: "light".to_string(),
        predicate: "has_wave_properties".to_string(),
        polarity: true,
    });
    println!("  ✓ Light has wave properties (affirmed)");

    system.add_constraint(SemanticAssertion {
        subject: "light".to_string(),
        predicate: "has_particle_properties".to_string(),
        polarity: true,
    });
    println!("  ✓ Light has particle properties (affirmed)");

    system.record_step("Added wave and particle assertions".to_string());

    // Evaluate constraints
    println!("\nEvaluating constraint satisfaction:");
    let mut eval_results = Vec::new();
    for (i, constraint) in system.constraints().iter().enumerate() {
        match evaluator.evaluate(constraint) {
            Ok(satisfied) => {
                println!("  Constraint {}: {}", i, if satisfied { "SATISFIED" } else { "VIOLATED" });
                eval_results.push((i, satisfied));
            }
            Err(e) => println!("  Constraint {}: ERROR - {}", i, e),
        }
    }
    
    for (i, satisfied) in eval_results {
        system.record_step(format!("Constraint {} evaluated: {}", i, if satisfied { "satisfied" } else { "violated" }));
    }

    // Check for conflicts (none expected yet)
    println!("\nChecking for conflicts (consistent scenario):");
    let conflicts = evaluator.detect_conflicts(system.constraints());
    if conflicts.is_empty() {
        println!("  ✓ No conflicts detected");
        system.record_step("Consistency check passed: no conflicts".to_string());
    } else {
        for (i, j, desc) in &conflicts {
            println!("  ✗ Conflict between {} and {}: {}", i, j, desc);
        }
    }

    // Resolve constraints
    println!("\nResolving constraints:");
    let mut audit_trail = system.audit_trail().to_vec();
    match evaluator.resolve_contradictions(system.constraints(), &mut audit_trail) {
        Ok(resolved) => {
            println!("  ✓ All {} constraints resolved successfully", resolved.len());
        }
        Err(e) => println!("  ✗ Resolution failed: {}", e),
    }

    // Show audit trail
    println!("\nAudit trail ({} steps):", audit_trail.len());
    for (i, step) in audit_trail.iter().enumerate() {
        println!("  {}. {}", i + 1, step);
    }

    // Demonstrate contradiction detection
    println!("\n=== Detecting Contradictions ===\n");

    let mut system2: ConstraintSystem<SemanticAssertion> = ConstraintSystem::new();
    
    // Add contradictory constraints
    system2.add_constraint(SemanticAssertion {
        subject: "sky".to_string(),
        predicate: "color".to_string(),
        polarity: true,
    });
    println!("Added: Sky is blue (color = affirmed)");

    system2.add_constraint(SemanticAssertion {
        subject: "sky".to_string(),
        predicate: "color".to_string(),
        polarity: false,
    });
    println!("Added: Sky is not blue (color = negated)");

    println!("\nChecking for conflicts (contradictory scenario):");
    let conflicts = evaluator.detect_conflicts(system2.constraints());
    for (i, _j, desc) in &conflicts {
        println!("  ✗ Conflict {}: {}", i, desc);
    }

    println!("\nResolving contradictory constraints:");
    let mut audit_trail2 = vec![];
    match evaluator.resolve_contradictions(system2.constraints(), &mut audit_trail2) {
        Ok(_) => println!("  ✓ Resolved"),
        Err(e) => println!("  ✗ Cannot resolve: {}", e),
    }

    println!("\n=== Core Invariants Enforced ===");
    println!("✓ Determinism: Constraint evaluation is repeatable");
    println!("✓ Consistency: Contradictions are detected and tracked");
    println!("✓ Auditability: All operations recorded in audit trail");
    println!("✓ Closure: Constraints reach resolved state (resolved or contradiction)");
}
