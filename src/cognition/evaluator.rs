/// Semantic constraint evaluation engine.
/// Implements deterministic constraint satisfaction and conflict detection.

use crate::cognition::constraint::SemanticConstraint;
use crate::cognition::node::SemanticNode;
use crate::geom::field::{FieldPoint, SemanticField};
use crate::geom::invariants::{ConstraintEvaluator, InvariantViolation};
use crate::geom::mode::{ArithmeticMode, ResonanceMode, ResonanceTransform};
use crate::geom::space::Coordinate3;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintStatus {
    Satisfied,
    Violated,
}

#[derive(Debug, Default)]
pub struct ConstraintEvalEngine;

impl ConstraintEvalEngine {
    pub fn new() -> Self {
        Self
    }

    /// Transform constraints into semantic nodes.
    pub fn constraints_to_nodes(&self, constraints: &[SemanticConstraint]) -> Vec<SemanticNode> {
        constraints
            .iter()
            .map(|c| {
                let concept = format!("{}:{}", c.subject, c.predicate);
                SemanticNode::new(concept, c.affirmed, c.weight)
            })
            .collect()
    }

    /// Transform nodes into a geometric semantic field with deterministic coordinate assignment.
    pub fn project_nodes_to_field(&self, nodes: &[SemanticNode]) -> SemanticField {
        let mut field = SemanticField::new();
        for (idx, node) in nodes.iter().enumerate() {
            let x = idx as i64;
            let y = if node.polarity { 1 } else { -1 };
            let z = node.confidence as i64;

            field.upsert_concept(
                node.concept.clone(),
                FieldPoint {
                    position: Coordinate3::new(x, y, z),
                    intensity: if node.polarity {
                        node.confidence as i64
                    } else {
                        -(node.confidence as i64)
                    },
                },
            );
        }
        field
    }

    /// First real cognitive transform: apply resonance by aggregate confidence.
    pub fn apply_resonance_transform(&self, field: &mut SemanticField, nodes: &[SemanticNode]) {
        self.apply_resonance_transform_with_mode(field, nodes, ArithmeticMode::Exact);
    }

    pub fn apply_resonance_transform_with_mode(
        &self,
        field: &mut SemanticField,
        nodes: &[SemanticNode],
        arithmetic: ArithmeticMode,
    ) {
        let signed_energy: i64 = nodes
            .iter()
            .map(|n| if n.polarity { n.confidence as i64 } else { -(n.confidence as i64) })
            .sum();

        let (mode, magnitude) = if signed_energy > 0 {
            (ResonanceMode::Amplify, (signed_energy / 20).max(1))
        } else if signed_energy < 0 {
            (ResonanceMode::Dampen, ((-signed_energy) / 20).max(1))
        } else {
            (ResonanceMode::Balance, 0)
        };

        ResonanceTransform::new(mode, magnitude, arithmetic).apply(field);
    }
}

impl ConstraintEvaluator for ConstraintEvalEngine {
    type Constraint = SemanticConstraint;
    type EvaluationResult = ConstraintStatus;

    fn evaluate(&self, constraint: &Self::Constraint) -> Result<Self::EvaluationResult, InvariantViolation> {
        if constraint.subject.trim().is_empty() || constraint.predicate.trim().is_empty() {
            return Err(InvariantViolation::Consistency {
                message: "subject/predicate cannot be empty".to_string(),
                contradicting_terms: vec![constraint.subject.clone(), constraint.predicate.clone()],
            });
        }

        if constraint.weight == 0 {
            return Ok(ConstraintStatus::Violated);
        }

        Ok(ConstraintStatus::Satisfied)
    }

    fn detect_conflicts(&self, constraints: &[Self::Constraint]) -> Vec<(usize, usize, String)> {
        let mut seen: BTreeMap<(String, String, Option<String>), (usize, bool)> = BTreeMap::new();
        let mut conflicts = Vec::new();

        for (idx, c) in constraints.iter().enumerate() {
            let key = c.key();
            if let Some((prior_idx, prior_polarity)) = seen.get(&key) {
                if *prior_polarity != c.affirmed {
                    conflicts.push((
                        *prior_idx,
                        idx,
                        format!(
                            "Conflict on {}:{} polarity mismatch",
                            c.subject, c.predicate
                        ),
                    ));
                }
            } else {
                seen.insert(key, (idx, c.affirmed));
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
            audit_trail.push(format!("{} conflict(s) detected", conflicts.len()));
            return Err(InvariantViolation::Consistency {
                message: "unable to auto-resolve contradictory constraints".to_string(),
                contradicting_terms: conflicts.into_iter().map(|(_, _, msg)| msg).collect(),
            });
        }

        let mut resolved = constraints.to_vec();
        resolved.sort_by(|a, b| {
            a.subject
                .cmp(&b.subject)
                .then(a.predicate.cmp(&b.predicate))
                .then(a.object.cmp(&b.object))
                .then(a.affirmed.cmp(&b.affirmed))
                .then(a.weight.cmp(&b.weight))
        });
        audit_trail.push(format!("resolved {} constraints", resolved.len()));
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::constraint::SemanticConstraint;

    #[test]
    fn detects_polarity_conflicts() {
        let engine = ConstraintEvalEngine::new();
        let constraints = vec![
            SemanticConstraint::assertion("light", "wave", true, 100),
            SemanticConstraint::assertion("light", "wave", false, 90),
        ];
        let conflicts = engine.detect_conflicts(&constraints);
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn cognitive_transforms_generate_field() {
        let engine = ConstraintEvalEngine::new();
        let constraints = vec![SemanticConstraint::assertion("light", "wave", true, 90)];
        let nodes = engine.constraints_to_nodes(&constraints);
        let mut field = engine.project_nodes_to_field(&nodes);
        engine.apply_resonance_transform(&mut field, &nodes);
        assert!(field.concept_count() > 0);
    }

    #[test]
    fn bounded_mode_produces_quantized_field() {
        let engine = ConstraintEvalEngine::new();
        let constraints = vec![SemanticConstraint::assertion("light", "wave", true, 91)];
        let nodes = engine.constraints_to_nodes(&constraints);
        let mut field = engine.project_nodes_to_field(&nodes);

        engine.apply_resonance_transform_with_mode(
            &mut field,
            &nodes,
            ArithmeticMode::BoundedApproximate { max_error: 3 },
        );

        let intensity = field.concept("light:wave").map(|p| p.intensity).unwrap_or_default();
        assert_eq!(intensity % 4, 0);
    }
}
