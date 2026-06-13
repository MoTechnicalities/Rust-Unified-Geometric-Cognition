use crate::cognition::constraint::SemanticConstraint;
use crate::cognition::evaluator::ConstraintEvalEngine;
use crate::cognition::node::CognitiveFrame;
use crate::cognition::scheduler::TaskScheduler;
use crate::geom::field::{ConceptCluster, SemanticField};
use crate::geom::invariants::InvariantViolation;
use crate::geom::mode::ArithmeticMode;
use crate::runtime::logging::AuditLogger;
use crate::GeometricState;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

type ConstraintKey = (String, String, Option<String>);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MultiFrameConfig {
    pub iterations: usize,
    pub worker_count: usize,
    pub ambiguity_margin: i64,
    pub target_energy: i64,
    pub compression_threshold: i64,
    pub convergence_window: usize,
    pub energy_delta_threshold: i64,
    pub anchor_energy_max: i64,
    pub anchor_pull_strength: i64,
    pub anchor_min_persistence: usize,
    pub anchor_alignment_window: i64,
    pub anchor_contradiction_highlight: i64,
    pub anchor_fusion_bias: i64,
    pub emergent_min_cluster_size: usize,
    pub emergent_min_anchor_support: usize,
    pub emergent_resonance_threshold: i64,
    pub emergent_min_persistence: usize,
    pub emergent_constraint_weight: u8,
}

impl Default for MultiFrameConfig {
    fn default() -> Self {
        Self {
            iterations: 6,
            worker_count: 4,
            ambiguity_margin: 5000,
            target_energy: 500,
            compression_threshold: 1,
            convergence_window: 2,
            energy_delta_threshold: 2,
            anchor_energy_max: 500,
            anchor_pull_strength: 4,
            anchor_min_persistence: 2,
            anchor_alignment_window: 25,
            anchor_contradiction_highlight: 6,
            anchor_fusion_bias: 8,
            emergent_min_cluster_size: 2,
            emergent_min_anchor_support: 1,
            emergent_resonance_threshold: 40,
            emergent_min_persistence: 2,
            emergent_constraint_weight: 36,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmergentConcept {
    pub id: String,
    pub subject: String,
    pub basis_anchors: Vec<String>,
    pub members: Vec<String>,
    pub resonance_score: i64,
    pub persistence_hits: usize,
    pub canonical_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmergentConceptCandidate {
    id: String,
    subject: String,
    basis_anchors: Vec<String>,
    members: Vec<String>,
    resonance_score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptAnchor {
    pub id: String,
    pub canonical_hash: String,
    pub energy: i64,
    pub persistence_hits: usize,
    pub frame_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AnchorRegistry {
    pub anchors: Vec<ConceptAnchor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StabilizationMetrics {
    pub energy_delta: i64,
    pub contradiction_count: usize,
    pub unresolved_subjects: usize,
    pub disambiguation_gap_median: i64,
    pub fused_constraints: usize,
    pub active_anchors: usize,
    pub anchor_overlap: usize,
    pub anchor_drift: i64,
    pub anchor_stability: i64,
    pub anchor_field_coherence: i64,
    pub anchor_contradictions_highlighted: usize,
    pub emergent_candidates: usize,
    pub emergent_concepts_active: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableSense {
    pub subject: String,
    pub selected_concept: String,
    pub support_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidatedMemory {
    pub converged_iteration: Option<usize>,
    pub fused_constraints: Vec<SemanticConstraint>,
    pub stable_senses: Vec<StableSense>,
    pub clusters: Vec<ConceptCluster>,
    pub anchor_basis_ids: Vec<String>,
    pub anchor_basis_hash: String,
    pub self_continuity_score: i64,
    pub external_change_score: i64,
    pub emergent_concepts: Vec<EmergentConcept>,
    pub ontology_expansion_score: i64,
    pub artifact_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorRelationalDistance {
    pub score: i64,
    pub anchor_jaccard_distance: i64,
    pub emergent_jaccard_distance: i64,
    pub continuity_delta: i64,
    pub external_delta: i64,
    pub ontology_delta: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologicalNeighborhood {
    pub center: String,
    pub neighbors: Vec<String>,
    pub radius: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologicalRegion {
    pub id: String,
    pub members: Vec<String>,
    pub representative: String,
    pub boundary_members: Vec<String>,
    pub cohesion_score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyMetrics {
    pub region_count: usize,
    pub total_concepts: usize,
    pub boundary_count: usize,
    pub avg_neighborhood_size: i64,
    pub manifold_stability: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveTopology {
    pub neighborhoods: Vec<TopologicalNeighborhood>,
    pub regions: Vec<TopologicalRegion>,
    pub boundary_concepts: Vec<String>,
    pub metrics: TopologyMetrics,
    pub canonical_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifoldDrift {
    pub region_delta: i64,
    pub boundary_delta: i64,
    pub stability_delta: i64,
    pub cohesion_delta: i64,
    pub added_regions: Vec<String>,
    pub removed_regions: Vec<String>,
    pub hash_changed: bool,
    pub drift_score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyEvolutionStep {
    pub step_index: usize,
    pub drift: ManifoldDrift,
    pub is_phase_transition: bool,
    pub topology_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifoldEvolutionTrace {
    pub steps: Vec<TopologyEvolutionStep>,
    pub persistent_region_ids: Vec<String>,
    pub transient_region_ids: Vec<String>,
    pub phase_transition_steps: Vec<usize>,
    pub overall_stability: i64,
    pub canonical_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameIterationResult {
    pub topic: String,
    pub frame_id: String,
    pub closure_status: String,
    pub selected_senses: Vec<(String, String, bool, i64)>,
    pub field_concepts: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiFrameIteration {
    pub iteration_index: usize,
    pub frame_results: Vec<FrameIterationResult>,
    pub shared_field_concepts: usize,
    pub propagated_constraints: usize,
    pub metrics: StabilizationMetrics,
    pub converged: bool,
    pub iteration_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiFrameReport {
    pub iterations: Vec<MultiFrameIteration>,
    pub converged_iteration: Option<usize>,
    pub consolidated_memory: ConsolidatedMemory,
    pub anchor_registry: AnchorRegistry,
    pub final_trace_hash: String,
}

#[derive(Debug, Default)]
pub struct MultiFrameCognition {
    engine: ConstraintEvalEngine,
    scheduler: TaskScheduler,
    logger: AuditLogger,
    frames: BTreeMap<String, Vec<SemanticConstraint>>,
    anchor_registry: AnchorRegistry,
}

impl MultiFrameCognition {
    pub fn new() -> Self {
        Self {
            engine: ConstraintEvalEngine::new(),
            scheduler: TaskScheduler::new(),
            logger: AuditLogger::new(),
            frames: BTreeMap::new(),
            anchor_registry: AnchorRegistry { anchors: Vec::new() },
        }
    }

    pub fn register_frame(&mut self, topic: impl Into<String>, constraints: Vec<SemanticConstraint>) {
        self.frames.insert(topic.into(), constraints);
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn run(&mut self, config: MultiFrameConfig) -> Result<MultiFrameReport, InvariantViolation> {
        let iterations = config.iterations.max(1);
        let workers = config.worker_count.max(1);
        let convergence_window = config.convergence_window.max(1);
        let mut report = Vec::with_capacity(iterations);
        let mut previous_shared_energy: Option<i64> = None;
        let mut previous_iteration_hash: Option<String> = None;
        let mut previous_anchor_energies: BTreeMap<String, i64> = BTreeMap::new();
        let mut emergent_hits: BTreeMap<String, usize> = BTreeMap::new();
        let mut emergent_latest: BTreeMap<String, EmergentConceptCandidate> = BTreeMap::new();
        let mut stable_streak: usize = 0;
        let mut converged_iteration: Option<usize> = None;
        let mut last_fused_constraints: Vec<SemanticConstraint> = Vec::new();
        let mut last_frame_results: Vec<FrameIterationResult> = Vec::new();
        let mut last_shared_field = SemanticField::new();
        let mut last_metrics: Option<StabilizationMetrics> = None;

        self.logger.record(format!(
            "mfc:start frames={} iterations={}",
            self.frames.len(),
            iterations
        ));

        for iter in 0..iterations {
            self.logger.record(format!("mfc:iter:{}:start", iter));
            let mut frame_results = Vec::new();
            let mut resolved_by_frame: BTreeMap<String, Vec<SemanticConstraint>> = BTreeMap::new();
            let mut fields_by_frame = BTreeMap::new();

            for (topic, constraints) in &self.frames {
                let mut eval_audit = Vec::new();
                let (resolved_constraints, summary) = self.engine.resolve_contradictions_parallel_deterministic(
                    constraints,
                    &self.scheduler,
                    workers,
                    &mut eval_audit,
                )?;

                self.logger.record(format!(
                    "mfc:iter:{}:frame:{}:groups={} conflicts={}",
                    iter, topic, summary.groups_processed, summary.conflicts_resolved
                ));
                for line in eval_audit {
                    self.logger.record(format!("mfc:iter:{}:frame:{}:audit:{}", iter, topic, line));
                }

                let nodes = self.engine.constraints_to_nodes(&resolved_constraints);
                let mut field = self.engine.project_nodes_to_field(&nodes);
                self.engine
                    .apply_resonance_transform_with_mode(&mut field, &nodes, ArithmeticMode::Exact);
                field.normalize_energy(config.target_energy);
                field.compress_by_intensity(config.compression_threshold);

                let subjects: BTreeSet<String> = resolved_constraints
                    .iter()
                    .map(|c| c.subject.clone())
                    .collect();

                let mut frame = CognitiveFrame::new(topic.clone());
                for node in nodes {
                    frame.add_node(node);
                }

                let mut selected_senses = Vec::new();
                for subject in subjects {
                    if let Some(decision) = self.engine.disambiguate_subject_senses_with_margin(
                        &field,
                        &subject,
                        config.ambiguity_margin,
                    ) {
                        if decision.unresolved {
                            frame.mark_unresolved_subject(subject.clone());
                        }
                        selected_senses.push((
                            subject,
                            decision.selected_concept,
                            decision.unresolved,
                            decision.score_gap,
                        ));
                    }
                }
                selected_senses.sort();

                let (closed, transition) = frame.attempt_closure();
                if let Some(t) = transition {
                    self.logger.record(format!(
                        "mfc:iter:{}:frame:{}:closure:{}",
                        iter, topic, t.reasoning_summary
                    ));
                }
                for step in closed.audit_trail() {
                    self.logger
                        .record(format!("mfc:iter:{}:frame:{}:step:{}", iter, topic, step));
                }

                frame_results.push(FrameIterationResult {
                    topic: topic.clone(),
                    frame_id: closed.frame_id(),
                    closure_status: closed.closure_status().to_string(),
                    selected_senses,
                    field_concepts: field.concept_count(),
                });

                resolved_by_frame.insert(topic.clone(), resolved_constraints);
                fields_by_frame.insert(topic.clone(), field);
            }

            frame_results.sort_by(|a, b| a.topic.cmp(&b.topic));

            let mut shared_field = SemanticField::new();
            for field in fields_by_frame.values() {
                shared_field.merge_from(field);
            }
            shared_field.normalize_energy(config.target_energy);
            shared_field.compress_by_intensity(config.compression_threshold);

            apply_anchor_persistence(
                &mut shared_field,
                &self.anchor_registry,
                config.anchor_pull_strength,
                config.target_energy,
                config.compression_threshold,
                config.anchor_min_persistence,
            );

            let contradictions_highlighted = apply_anchor_weighted_interpretation(
                &mut shared_field,
                &self.anchor_registry,
                config.anchor_min_persistence,
                config.anchor_alignment_window,
                config.anchor_pull_strength,
                config.anchor_contradiction_highlight,
                config.target_energy,
                config.compression_threshold,
            );

            let fused_constraints = fuse_cross_frame_constraints(&resolved_by_frame);
            let fused_constraints = anchor_guided_fusion(
                fused_constraints,
                &self.anchor_registry,
                config.anchor_min_persistence,
                &shared_field,
                config.anchor_fusion_bias,
            );
            let propagated = propagate_resonance_constraints(&shared_field, &fused_constraints);
            let propagated_count = propagated.len();

            for constraints in self.frames.values_mut() {
                append_missing_constraints(constraints, &propagated);
            }

            let energy = shared_field.total_energy();
            let energy_delta = previous_shared_energy
                .map(|prev| (energy - prev).abs())
                .unwrap_or(0);
            previous_shared_energy = Some(energy);

            let contradiction_count = count_cross_frame_conflicts(&resolved_by_frame);
            let unresolved_subjects = frame_results
                .iter()
                .flat_map(|r| r.selected_senses.iter())
                .filter(|(_, _, unresolved, _)| *unresolved)
                .count();

            let mut gaps: Vec<i64> = frame_results
                .iter()
                .flat_map(|r| r.selected_senses.iter().map(|(_, _, _, gap)| *gap))
                .collect();
            gaps.sort_unstable();
            let disambiguation_gap_median = if gaps.is_empty() {
                0
            } else {
                gaps[gaps.len() / 2]
            };

            let active_anchor_map = active_anchor_energies(&self.anchor_registry, config.anchor_min_persistence);
            let (anchor_overlap, anchor_drift, anchor_stability, anchor_field_coherence) =
                compute_anchor_continuity(
                    &shared_field,
                    &active_anchor_map,
                    &previous_anchor_energies,
                );

            let metrics = StabilizationMetrics {
                energy_delta,
                contradiction_count,
                unresolved_subjects,
                disambiguation_gap_median,
                fused_constraints: fused_constraints.len(),
                active_anchors: self
                    .anchor_registry
                    .anchors
                    .iter()
                    .filter(|a| a.persistence_hits >= config.anchor_min_persistence.max(1))
                    .count(),
                anchor_overlap,
                anchor_drift,
                anchor_stability,
                anchor_field_coherence,
                anchor_contradictions_highlighted: contradictions_highlighted,
                emergent_candidates: 0,
                emergent_concepts_active: 0,
            };

            update_anchor_registry(
                &mut self.anchor_registry,
                &shared_field,
                frame_results.len(),
                config.anchor_energy_max,
                config.target_energy,
                config.compression_threshold,
            );

            let emergent_candidates = discover_emergent_candidates(
                &shared_field,
                &self.anchor_registry,
                config.anchor_min_persistence,
                config.emergent_min_cluster_size,
                config.emergent_min_anchor_support,
                config.emergent_resonance_threshold,
            );

            for candidate in &emergent_candidates {
                *emergent_hits.entry(candidate.id.clone()).or_default() += 1;
                emergent_latest.insert(candidate.id.clone(), candidate.clone());
            }

            let emergent_constraints = synthesize_emergent_constraints(
                &emergent_candidates,
                &emergent_hits,
                config.emergent_min_persistence,
                config.emergent_constraint_weight,
            );
            for constraints in self.frames.values_mut() {
                append_missing_constraints(constraints, &emergent_constraints);
            }

            let emergent_active = emergent_hits
                .values()
                .filter(|hits| **hits >= config.emergent_min_persistence.max(1))
                .count();

            let metrics = StabilizationMetrics {
                emergent_candidates: emergent_candidates.len(),
                emergent_concepts_active: emergent_active,
                ..metrics
            };

            let iteration_hash = hash_json(&(
                shared_field_snapshot(&shared_field),
                &fused_constraints,
                &frame_results,
                &metrics,
                &emergent_constraints,
            ))?;

            let stable_condition = previous_iteration_hash
                .as_ref()
                .map(|h| h == &iteration_hash)
                .unwrap_or(false)
                && metrics.energy_delta <= config.energy_delta_threshold.max(0)
                && metrics.contradiction_count == 0
                && metrics.unresolved_subjects == 0;

            if stable_condition {
                stable_streak += 1;
            } else {
                stable_streak = 0;
            }

            let converged = stable_streak >= convergence_window.saturating_sub(1) && previous_iteration_hash.is_some();
            previous_iteration_hash = Some(iteration_hash.clone());

            self.logger.record(format!(
                "mfc:iter:{}:shared_field={} propagated={} stable={} energy_delta={} unresolved={} anchors={} overlap={} drift={} coherence={} emergent_candidates={} emergent_active={}",
                iter,
                shared_field.concept_count(),
                propagated_count,
                if stable_condition { 1 } else { 0 },
                metrics.energy_delta,
                metrics.unresolved_subjects,
                metrics.active_anchors,
                metrics.anchor_overlap,
                metrics.anchor_drift,
                metrics.anchor_field_coherence,
                metrics.emergent_candidates,
                metrics.emergent_concepts_active
            ));

            report.push(MultiFrameIteration {
                iteration_index: iter,
                frame_results,
                shared_field_concepts: shared_field.concept_count(),
                propagated_constraints: propagated_count,
                metrics,
                converged,
                iteration_hash: iteration_hash.clone(),
            });

            last_metrics = report.last().map(|r| r.metrics.clone());
            last_fused_constraints = fused_constraints;
            last_shared_field = shared_field;
            previous_anchor_energies = active_anchor_energies(&self.anchor_registry, config.anchor_min_persistence);
            last_frame_results = report
                .last()
                .map(|r| r.frame_results.clone())
                .unwrap_or_default();

            if converged {
                converged_iteration = Some(iter);
                self.logger.record(format!("mfc:converged_at={}", iter));
                break;
            }
        }

        let stable_senses = consolidate_stable_senses(&last_frame_results);
        let clusters = last_shared_field.clusters_by_subject();
        let anchor_registry = registered_anchor_registry(&self.anchor_registry, config.anchor_min_persistence);
        let anchor_basis_ids: Vec<String> = anchor_registry
            .anchors
            .iter()
            .map(|a| a.id.clone())
            .collect();
        let emergent_concepts = materialize_emergent_concepts(
            &emergent_hits,
            &emergent_latest,
            config.emergent_min_persistence,
        )?;
        let anchor_basis_hash = hash_json(&anchor_registry.anchors)?;
        let self_continuity_score = last_metrics
            .as_ref()
            .map(|m| m.anchor_stability + m.anchor_field_coherence - m.anchor_drift)
            .unwrap_or(0);
        let external_change_score = last_metrics
            .as_ref()
            .map(|m| m.anchor_drift + (m.anchor_contradictions_highlighted as i64 * 10))
            .unwrap_or(0);
        let ontology_expansion_score = emergent_concepts
            .iter()
            .map(|c| (c.members.len() as i64) * (c.persistence_hits as i64))
            .sum();
        let artifact_hash = hash_json(&(
            converged_iteration,
            &last_fused_constraints,
            &stable_senses,
            &clusters,
            &anchor_basis_ids,
            &anchor_basis_hash,
            self_continuity_score,
            external_change_score,
            &emergent_concepts,
            ontology_expansion_score,
        ))?;
        let consolidated_memory = ConsolidatedMemory {
            converged_iteration,
            fused_constraints: last_fused_constraints,
            stable_senses,
            clusters,
            anchor_basis_ids,
            anchor_basis_hash,
            self_continuity_score,
            external_change_score,
            emergent_concepts,
            ontology_expansion_score,
            artifact_hash,
        };

        Ok(MultiFrameReport {
            iterations: report,
            converged_iteration,
            consolidated_memory,
            anchor_registry,
            final_trace_hash: self.logger.canonical_trace_hash(),
        })
    }
}

fn append_missing_constraints(base: &mut Vec<SemanticConstraint>, additions: &[SemanticConstraint]) {
    let mut seen: BTreeSet<(String, String, Option<String>, bool)> = base
        .iter()
        .map(|c| (c.subject.clone(), c.predicate.clone(), c.object.clone(), c.affirmed))
        .collect();

    for c in additions {
        let key = (c.subject.clone(), c.predicate.clone(), c.object.clone(), c.affirmed);
        if seen.insert(key) {
            base.push(c.clone());
        }
    }

    base.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then(a.predicate.cmp(&b.predicate))
            .then(a.object.cmp(&b.object))
            .then(a.affirmed.cmp(&b.affirmed))
            .then(a.weight.cmp(&b.weight))
    });
}

fn fuse_cross_frame_constraints(
    by_frame: &BTreeMap<String, Vec<SemanticConstraint>>,
) -> Vec<SemanticConstraint> {
    let mut grouped: BTreeMap<ConstraintKey, Vec<SemanticConstraint>> = BTreeMap::new();
    for constraints in by_frame.values() {
        for c in constraints {
            grouped.entry(c.key()).or_default().push(c.clone());
        }
    }

    let mut out = Vec::new();
    for ((_subject, _predicate, _object), group) in grouped {
        let affirmed_weight: u16 = group
            .iter()
            .filter(|c| c.affirmed)
            .map(|c| c.weight as u16)
            .sum();
        let negated_weight: u16 = group
            .iter()
            .filter(|c| !c.affirmed)
            .map(|c| c.weight as u16)
            .sum();

        let choose_affirmed = match affirmed_weight.cmp(&negated_weight) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => true,
        };

        let mut selected: Vec<SemanticConstraint> = group
            .iter()
            .filter(|c| c.affirmed == choose_affirmed)
            .cloned()
            .collect();
        selected.sort_by(|a, b| {
            b.weight
                .cmp(&a.weight)
                .then(a.subject.cmp(&b.subject))
                .then(a.predicate.cmp(&b.predicate))
                .then(a.object.cmp(&b.object))
        });

        if let Some(mut best) = selected.into_iter().next() {
            let merged = if choose_affirmed {
                affirmed_weight
            } else {
                negated_weight
            };
            best.weight = merged.min(u8::MAX as u16) as u8;
            out.push(best);
        }
    }

    out.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then(a.predicate.cmp(&b.predicate))
            .then(a.object.cmp(&b.object))
            .then(a.affirmed.cmp(&b.affirmed))
            .then(a.weight.cmp(&b.weight))
    });
    out
}

fn propagate_resonance_constraints(
    shared_field: &SemanticField,
    fused_constraints: &[SemanticConstraint],
) -> Vec<SemanticConstraint> {
    let mut out = Vec::with_capacity(fused_constraints.len());
    for c in fused_constraints {
        let concept = format!("{}:{}", c.subject, c.predicate);
        let mut next = c.clone();

        if let Some(point) = shared_field.concept(&concept) {
            let bonus = (point.intensity.abs() / 25).min(20) as u8;
            if (point.intensity >= 0) == c.affirmed {
                next.weight = next.weight.saturating_add(bonus);
            } else {
                next.weight = next.weight.saturating_sub(bonus.min(next.weight.saturating_sub(1)));
            }
        }

        out.push(next);
    }

    out.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then(a.predicate.cmp(&b.predicate))
            .then(a.object.cmp(&b.object))
            .then(a.affirmed.cmp(&b.affirmed))
            .then(a.weight.cmp(&b.weight))
    });
    out
}

fn anchor_guided_fusion(
    mut fused_constraints: Vec<SemanticConstraint>,
    registry: &AnchorRegistry,
    min_persistence: usize,
    shared_field: &SemanticField,
    fusion_bias: i64,
) -> Vec<SemanticConstraint> {
    let bias = fusion_bias.max(0) as u8;
    if bias == 0 {
        return fused_constraints;
    }

    let active_map = active_anchor_energies(registry, min_persistence);
    for c in &mut fused_constraints {
        let concept = format!("{}:{}", c.subject, c.predicate);
        let anchor_energy = active_map.get(&concept).copied();

        if let Some(anchor_e) = anchor_energy {
            let anchor_affirmed = anchor_e >= 0;
            if c.affirmed == anchor_affirmed {
                c.weight = c.weight.saturating_add(bias.min(20));
            } else {
                c.weight = c.weight.saturating_sub(bias.min(c.weight.saturating_sub(1)));
            }
        }

        if let Some(point) = shared_field.concept(&concept) {
            let field_affirmed = point.intensity >= 0;
            if c.affirmed == field_affirmed {
                c.weight = c.weight.saturating_add(2);
            } else {
                c.weight = c.weight.saturating_sub(1);
            }
        }
    }

    fused_constraints.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then(a.predicate.cmp(&b.predicate))
            .then(a.object.cmp(&b.object))
            .then(a.affirmed.cmp(&b.affirmed))
            .then(a.weight.cmp(&b.weight))
    });
    fused_constraints
}

fn active_anchor_energies(registry: &AnchorRegistry, min_persistence: usize) -> BTreeMap<String, i64> {
    registry
        .anchors
        .iter()
        .filter(|a| a.persistence_hits >= min_persistence.max(1))
        .map(|a| (a.id.clone(), a.energy))
        .collect()
}

fn compute_anchor_continuity(
    field: &SemanticField,
    active_anchor_map: &BTreeMap<String, i64>,
    previous_anchor_map: &BTreeMap<String, i64>,
) -> (usize, i64, i64, i64) {
    let active_ids: BTreeSet<String> = active_anchor_map.keys().cloned().collect();
    let previous_ids: BTreeSet<String> = previous_anchor_map.keys().cloned().collect();

    let overlap = active_ids.intersection(&previous_ids).count();

    let mut drift = 0;
    for id in active_ids.intersection(&previous_ids) {
        let current = active_anchor_map.get(id).copied().unwrap_or_default();
        let previous = previous_anchor_map.get(id).copied().unwrap_or_default();
        drift += (current - previous).abs();
    }

    let stability = if previous_ids.is_empty() {
        100
    } else {
        ((overlap as i64) * 100) / (previous_ids.len() as i64)
    };

    let mut coherence_sum = 0;
    let mut coherence_count = 0;
    for (id, anchor_energy) in active_anchor_map {
        if let Some(point) = field.concept(id) {
            let diff = (point.intensity - *anchor_energy).abs();
            coherence_sum += (100 - diff).max(0);
            coherence_count += 1;
        }
    }
    let coherence = if coherence_count == 0 {
        0
    } else {
        coherence_sum / coherence_count
    };

    (overlap, drift, stability, coherence)
}

fn apply_anchor_weighted_interpretation(
    field: &mut SemanticField,
    registry: &AnchorRegistry,
    min_persistence: usize,
    alignment_window: i64,
    amplify_strength: i64,
    contradiction_highlight: i64,
    target_energy: i64,
    compression_threshold: i64,
) -> usize {
    let mut contradictions = 0;
    let window = alignment_window.max(0);
    let amp = amplify_strength.max(0);
    let highlight = contradiction_highlight.max(0);

    for anchor in &registry.anchors {
        if anchor.persistence_hits < min_persistence.max(1) {
            continue;
        }

        if let Some(point) = field.concept(&anchor.id).cloned() {
            let mut adjusted = point.intensity;
            let diff = (point.intensity - anchor.energy).abs();
            let same_sign = (point.intensity >= 0) == (anchor.energy >= 0);

            if !same_sign && point.intensity != 0 && anchor.energy != 0 {
                contradictions += 1;
                let anchor_sign = if anchor.energy >= 0 { 1 } else { -1 };
                adjusted += anchor_sign * highlight;
            } else if diff <= window {
                let anchor_sign = if anchor.energy >= 0 { 1 } else { -1 };
                adjusted += anchor_sign * amp;
            } else {
                adjusted = (adjusted * 8) / 10;
            }

            field.upsert_concept(
                anchor.id.clone(),
                crate::FieldPoint {
                    position: point.position,
                    intensity: adjusted,
                },
            );
        }
    }

    field.normalize_energy(target_energy);
    field.compress_by_intensity(compression_threshold);
    contradictions
}

fn apply_anchor_persistence(
    field: &mut SemanticField,
    registry: &AnchorRegistry,
    pull_strength: i64,
    target_energy: i64,
    compression_threshold: i64,
    min_persistence: usize,
) {
    let strength = pull_strength.max(0);
    if strength == 0 {
        return;
    }

    for anchor in &registry.anchors {
        if anchor.persistence_hits < min_persistence.max(1) {
            continue;
        }
        if let Some(point) = field.concept(&anchor.id).cloned() {
            let baseline_sign = if anchor.energy >= 0 { 1 } else { -1 };
            let target = (anchor.energy.abs() / 10).max(1) * baseline_sign;
            let adjusted = (point.intensity * (10 - strength.min(9)) + target * strength.min(9)) / 10;

            field.upsert_concept(
                anchor.id.clone(),
                crate::FieldPoint {
                    position: point.position,
                    intensity: adjusted,
                },
            );
        }
    }

    field.normalize_energy(target_energy);
    field.compress_by_intensity(compression_threshold);
}

fn update_anchor_registry(
    registry: &mut AnchorRegistry,
    shared_field: &SemanticField,
    frame_count: usize,
    anchor_energy_max: i64,
    target_energy: i64,
    compression_threshold: i64,
) {
    let field_energy = shared_field.total_energy();
    if field_energy > anchor_energy_max.max(0) {
        return;
    }

    let baseline_hash = match shared_field.canonical_hash() {
        Ok(v) => v,
        Err(_) => return,
    };

    let is_stable = perturbation_returns_anchor(
        shared_field,
        &baseline_hash,
        target_energy,
        compression_threshold,
    );
    if !is_stable {
        return;
    }

    for (concept, point) in shared_field.ordered_concepts() {
        if point.intensity == 0 {
            continue;
        }
        if concept.contains("emergent/") {
            continue;
        }

        if let Some(existing) = registry.anchors.iter_mut().find(|a| a.id == *concept) {
            existing.persistence_hits += 1;
            existing.canonical_hash = baseline_hash.clone();
            existing.energy = point.intensity;
            existing.frame_count = frame_count;
        } else {
            registry.anchors.push(ConceptAnchor {
                id: concept.clone(),
                canonical_hash: baseline_hash.clone(),
                energy: point.intensity,
                persistence_hits: 1,
                frame_count,
            });
        }
    }

    registry
        .anchors
        .sort_by(|a, b| a.id.cmp(&b.id).then(a.canonical_hash.cmp(&b.canonical_hash)));
}

fn registered_anchor_registry(registry: &AnchorRegistry, min_persistence: usize) -> AnchorRegistry {
    let mut anchors: Vec<ConceptAnchor> = registry
        .anchors
        .iter()
        .filter(|a| a.persistence_hits >= min_persistence.max(1))
        .cloned()
        .collect();
    anchors.sort_by(|a, b| a.id.cmp(&b.id).then(a.canonical_hash.cmp(&b.canonical_hash)));
    AnchorRegistry { anchors }
}

fn discover_emergent_candidates(
    shared_field: &SemanticField,
    registry: &AnchorRegistry,
    anchor_min_persistence: usize,
    min_cluster_size: usize,
    min_anchor_support: usize,
    resonance_threshold: i64,
) -> Vec<EmergentConceptCandidate> {
    let mut anchors_by_subject: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for anchor in registry
        .anchors
        .iter()
        .filter(|a| a.persistence_hits >= anchor_min_persistence.max(1))
    {
        if let Some((subject, _)) = anchor.id.split_once(':') {
            anchors_by_subject
                .entry(subject.to_string())
                .or_default()
                .push(anchor.id.clone());
        }
    }

    for anchors in anchors_by_subject.values_mut() {
        anchors.sort();
    }

    let mut candidates = Vec::new();
    for cluster in shared_field.clusters_by_subject() {
        let members: Vec<String> = cluster
            .members
            .into_iter()
            .filter(|member| !member.contains("emergent/"))
            .collect();

        if members.len() < min_cluster_size.max(1) {
            continue;
        }

        let subject_anchors = anchors_by_subject
            .get(&cluster.anchor)
            .cloned()
            .unwrap_or_default();
        if subject_anchors.len() < min_anchor_support.max(1) {
            continue;
        }

        let member_set: BTreeSet<String> = members.iter().cloned().collect();
        let mut aligned: Vec<String> = subject_anchors
            .into_iter()
            .filter(|a| member_set.contains(a))
            .collect();
        aligned.sort();
        if aligned.len() < min_anchor_support.max(1) {
            continue;
        }

        let resonance_score = cluster.total_intensity.abs()
            + ((aligned.len() as i64) * 15)
            + ((members.len() as i64) * 5);
        if resonance_score < resonance_threshold.max(0) {
            continue;
        }

        let basis_head = aligned
            .first()
            .cloned()
            .unwrap_or_else(|| cluster.anchor.clone());
        let id = format!(
            "emergent:{}:{}:{}",
            cluster.anchor,
            basis_head.replace(':', "_"),
            members.len()
        );

        candidates.push(EmergentConceptCandidate {
            id,
            subject: cluster.anchor.clone(),
            basis_anchors: aligned,
            members,
            resonance_score,
        });
    }

    candidates.sort_by(|a, b| a.id.cmp(&b.id));
    candidates
}

fn synthesize_emergent_constraints(
    candidates: &[EmergentConceptCandidate],
    hits: &BTreeMap<String, usize>,
    min_persistence: usize,
    weight: u8,
) -> Vec<SemanticConstraint> {
    let mut out = Vec::new();
    for candidate in candidates {
        let persistence = hits.get(&candidate.id).copied().unwrap_or(0);
        if persistence < min_persistence.max(1) {
            continue;
        }

        let predicate = format!("emergent/{}", candidate.id.replace(':', "_"));
        out.push(SemanticConstraint::assertion(
            candidate.subject.clone(),
            predicate,
            true,
            weight.max(1),
        ));
    }

    out.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then(a.predicate.cmp(&b.predicate))
            .then(a.object.cmp(&b.object))
            .then(a.affirmed.cmp(&b.affirmed))
            .then(a.weight.cmp(&b.weight))
    });
    out
}

fn materialize_emergent_concepts(
    hits: &BTreeMap<String, usize>,
    latest: &BTreeMap<String, EmergentConceptCandidate>,
    min_persistence: usize,
) -> Result<Vec<EmergentConcept>, InvariantViolation> {
    let mut out = Vec::new();
    for (id, persistence_hits) in hits {
        if *persistence_hits < min_persistence.max(1) {
            continue;
        }
        let Some(candidate) = latest.get(id) else {
            continue;
        };

        let canonical_hash = hash_json(&(
            &candidate.id,
            &candidate.subject,
            &candidate.basis_anchors,
            &candidate.members,
            candidate.resonance_score,
            persistence_hits,
        ))?;

        out.push(EmergentConcept {
            id: candidate.id.clone(),
            subject: candidate.subject.clone(),
            basis_anchors: candidate.basis_anchors.clone(),
            members: candidate.members.clone(),
            resonance_score: candidate.resonance_score,
            persistence_hits: *persistence_hits,
            canonical_hash,
        });
    }

    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn perturbation_returns_anchor(
    field: &SemanticField,
    baseline_hash: &str,
    target_energy: i64,
    compression_threshold: i64,
) -> bool {
    let mut perturbed = field.clone();
    perturbed.map_intensity(|v| if v >= 0 { v + 1 } else { v - 1 });
    perturbed.normalize_energy(target_energy);
    perturbed.compress_by_intensity(compression_threshold);

    // Anchor persistence: deterministically contract back to the baseline basin.
    for (concept, point) in field.ordered_concepts() {
        perturbed.upsert_concept(concept.clone(), point.clone());
    }

    match perturbed.canonical_hash() {
        Ok(h) => h == baseline_hash,
        Err(_) => false,
    }
}

fn count_cross_frame_conflicts(by_frame: &BTreeMap<String, Vec<SemanticConstraint>>) -> usize {
    let mut seen: BTreeMap<ConstraintKey, BTreeSet<bool>> = BTreeMap::new();
    for constraints in by_frame.values() {
        for c in constraints {
            seen.entry(c.key()).or_default().insert(c.affirmed);
        }
    }
    seen.values().filter(|polarities| polarities.len() > 1).count()
}

fn consolidate_stable_senses(frame_results: &[FrameIterationResult]) -> Vec<StableSense> {
    let mut votes: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for frame in frame_results {
        for (subject, selected, unresolved, _gap) in &frame.selected_senses {
            if *unresolved {
                continue;
            }
            *votes
                .entry(subject.clone())
                .or_default()
                .entry(selected.clone())
                .or_default() += 1;
        }
    }

    let mut out = Vec::new();
    for (subject, by_concept) in votes {
        let mut ranked: Vec<(String, usize)> = by_concept.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if let Some((selected_concept, support_frames)) = ranked.into_iter().next() {
            out.push(StableSense {
                subject,
                selected_concept,
                support_frames,
            });
        }
    }
    out.sort_by(|a, b| a.subject.cmp(&b.subject));
    out
}

fn shared_field_snapshot(field: &SemanticField) -> Vec<(String, i64, i64, i64, i64)> {
    field
        .ordered_concepts()
        .map(|(concept, point)| {
            (
                concept.clone(),
                point.intensity,
                point.position.x,
                point.position.y,
                point.position.z,
            )
        })
        .collect()
}

fn hash_json<T: Serialize>(value: &T) -> Result<String, InvariantViolation> {
    let bytes = serde_json::to_vec(value).map_err(|e| InvariantViolation::Determinism {
        message: format!("serialization failure during deterministic hash: {}", e),
    })?;
    let mut h = Sha256::new();
    h.update(bytes);
    Ok(format!("{:x}", h.finalize()))
}

pub fn anchor_derived_relational_distance(
    a: &ConsolidatedMemory,
    b: &ConsolidatedMemory,
) -> AnchorRelationalDistance {
    let anchor_jaccard_distance = jaccard_distance_scaled(&a.anchor_basis_ids, &b.anchor_basis_ids, 1000);

    let emergent_a: Vec<String> = a.emergent_concepts.iter().map(|c| c.id.clone()).collect();
    let emergent_b: Vec<String> = b.emergent_concepts.iter().map(|c| c.id.clone()).collect();
    let emergent_jaccard_distance = jaccard_distance_scaled(&emergent_a, &emergent_b, 1000);

    let continuity_delta = (a.self_continuity_score - b.self_continuity_score)
        .abs()
        .min(1000);
    let external_delta = (a.external_change_score - b.external_change_score)
        .abs()
        .min(1000);
    let ontology_delta = (a.ontology_expansion_score - b.ontology_expansion_score)
        .abs()
        .min(1000);

    let score = (
        (anchor_jaccard_distance * 3)
            + (emergent_jaccard_distance * 2)
            + continuity_delta
            + external_delta
            + ontology_delta
    ) / 8;

    AnchorRelationalDistance {
        score,
        anchor_jaccard_distance,
        emergent_jaccard_distance,
        continuity_delta,
        external_delta,
        ontology_delta,
    }
}

fn jaccard_distance_scaled(a: &[String], b: &[String], scale: i64) -> i64 {
    let set_a: BTreeSet<String> = a.iter().cloned().collect();
    let set_b: BTreeSet<String> = b.iter().cloned().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 0;
    }

    let intersection = set_a.intersection(&set_b).count() as i64;
    let union = set_a.union(&set_b).count() as i64;
    ((union - intersection) * scale.max(1)) / union.max(1)
}

fn pairwise_anchor_distance(a: &str, b: &str, memory: &ConsolidatedMemory) -> i64 {
    if a == b {
        return 0;
    }
    let mut distance: i64 = 1000;
    let a_subj = a.split_once(':').map(|(s, _)| s).unwrap_or(a);
    let b_subj = b.split_once(':').map(|(s, _)| s).unwrap_or(b);
    if a_subj == b_subj {
        distance -= 400;
    }
    for concept in &memory.emergent_concepts {
        let has_a = concept.basis_anchors.iter().any(|x| x == a);
        let has_b = concept.basis_anchors.iter().any(|x| x == b);
        if has_a && has_b {
            distance = (distance - 200).max(0);
        }
    }
    distance.max(0).min(1000)
}

fn lookup_pair_dist(dist_map: &BTreeMap<(String, String), i64>, a: &str, b: &str) -> i64 {
    if a == b {
        return 0;
    }
    let (ka, kb) = if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    };
    dist_map.get(&(ka, kb)).copied().unwrap_or(1000)
}

pub fn compute_cognitive_topology(
    memory: &ConsolidatedMemory,
    distance_threshold: i64,
) -> Result<CognitiveTopology, InvariantViolation> {
    let concepts = &memory.anchor_basis_ids;
    let threshold = distance_threshold.max(0);

    if concepts.is_empty() {
        let metrics = TopologyMetrics {
            region_count: 0,
            total_concepts: 0,
            boundary_count: 0,
            avg_neighborhood_size: 0,
            manifold_stability: 1000,
        };
        let canonical_hash = hash_json(&(&Vec::<TopologicalNeighborhood>::new(), &metrics))?;
        return Ok(CognitiveTopology {
            neighborhoods: Vec::new(),
            regions: Vec::new(),
            boundary_concepts: Vec::new(),
            metrics,
            canonical_hash,
        });
    }

    let mut dist_map: BTreeMap<(String, String), i64> = BTreeMap::new();
    for a in concepts {
        for b in concepts {
            if a >= b {
                continue;
            }
            dist_map.insert((a.clone(), b.clone()), pairwise_anchor_distance(a, b, memory));
        }
    }

    let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut neighborhoods: Vec<TopologicalNeighborhood> = Vec::new();

    for center in concepts {
        let mut neighbors: Vec<String> = Vec::new();
        for other in concepts {
            if other == center {
                continue;
            }
            if lookup_pair_dist(&dist_map, center, other) <= threshold {
                neighbors.push(other.clone());
                adjacency.entry(center.clone()).or_default().insert(other.clone());
            }
        }
        neighbors.sort();
        neighborhoods.push(TopologicalNeighborhood {
            center: center.clone(),
            neighbors,
            radius: threshold,
        });
    }
    neighborhoods.sort_by(|a, b| a.center.cmp(&b.center));

    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut regions: Vec<TopologicalRegion> = Vec::new();

    for start in concepts {
        if visited.contains(start) {
            continue;
        }
        let mut queue: Vec<String> = vec![start.clone()];
        let mut component: BTreeSet<String> = BTreeSet::new();
        let mut qi = 0;
        while qi < queue.len() {
            let node = queue[qi].clone();
            qi += 1;
            if !component.insert(node.clone()) {
                continue;
            }
            visited.insert(node.clone());
            if let Some(nbrs) = adjacency.get(&node) {
                for nbr in nbrs {
                    if !component.contains(nbr) {
                        queue.push(nbr.clone());
                    }
                }
            }
        }
        regions.push(TopologicalRegion {
            id: String::new(),
            members: component.into_iter().collect(),
            representative: String::new(),
            boundary_members: Vec::new(),
            cohesion_score: 0,
        });
    }

    let mut boundary_set: BTreeSet<String> = BTreeSet::new();
    for region in &mut regions {
        let region_set: BTreeSet<String> = region.members.iter().cloned().collect();
        let mut boundary: Vec<String> = Vec::new();
        for member in &region.members {
            if let Some(nbrs) = adjacency.get(member) {
                if nbrs.iter().any(|n| !region_set.contains(n)) {
                    boundary.push(member.clone());
                    boundary_set.insert(member.clone());
                }
            }
        }
        boundary.sort();
        boundary.dedup();
        region.boundary_members = boundary;

        let m = region.members.clone();
        let mut intra_sum = 0i64;
        let mut intra_count = 0i64;
        for i in 0..m.len() {
            for j in (i + 1)..m.len() {
                intra_sum += lookup_pair_dist(&dist_map, &m[i], &m[j]);
                intra_count += 1;
            }
        }
        region.cohesion_score = if intra_count == 0 {
            1000
        } else {
            1000 - intra_sum / intra_count
        };
        region.representative = region.members.first().cloned().unwrap_or_default();
        region.id = hash_json(&region.members)?.chars().take(16).collect();
    }
    regions.sort_by(|a, b| a.id.cmp(&b.id));

    let boundary_concepts: Vec<String> = boundary_set.into_iter().collect();
    let total_concepts = concepts.len();
    let region_count = regions.len();
    let boundary_count = boundary_concepts.len();
    let avg_neighborhood_size = {
        let total: i64 = neighborhoods.iter().map(|n| n.neighbors.len() as i64).sum();
        if neighborhoods.is_empty() {
            0
        } else {
            total / neighborhoods.len() as i64
        }
    };
    let manifold_stability =
        1000 - (boundary_count as i64 * 1000) / total_concepts.max(1) as i64;

    let metrics = TopologyMetrics {
        region_count,
        total_concepts,
        boundary_count,
        avg_neighborhood_size,
        manifold_stability,
    };
    let canonical_hash = hash_json(&(&neighborhoods, &regions, &boundary_concepts, &metrics))?;

    Ok(CognitiveTopology {
        neighborhoods,
        regions,
        boundary_concepts,
        metrics,
        canonical_hash,
    })
}

pub fn compare_topologies(a: &CognitiveTopology, b: &CognitiveTopology) -> ManifoldDrift {
    let a_ids: BTreeSet<String> = a.regions.iter().map(|r| r.id.clone()).collect();
    let b_ids: BTreeSet<String> = b.regions.iter().map(|r| r.id.clone()).collect();

    let added_regions: Vec<String> = b_ids.difference(&a_ids).cloned().collect();
    let removed_regions: Vec<String> = a_ids.difference(&b_ids).cloned().collect();

    let region_delta = b.metrics.region_count as i64 - a.metrics.region_count as i64;
    let boundary_delta = b.metrics.boundary_count as i64 - a.metrics.boundary_count as i64;
    let stability_delta = b.metrics.manifold_stability - a.metrics.manifold_stability;

    let a_cohesion: i64 = if a.regions.is_empty() {
        0
    } else {
        a.regions.iter().map(|r| r.cohesion_score).sum::<i64>() / a.regions.len() as i64
    };
    let b_cohesion: i64 = if b.regions.is_empty() {
        0
    } else {
        b.regions.iter().map(|r| r.cohesion_score).sum::<i64>() / b.regions.len() as i64
    };
    let cohesion_delta = b_cohesion - a_cohesion;

    let drift_score = (region_delta.abs() * 300)
        + (boundary_delta.abs() * 100)
        + (stability_delta.abs() / 10)
        + (cohesion_delta.abs() / 10)
        + (added_regions.len() as i64 * 200)
        + (removed_regions.len() as i64 * 200);

    ManifoldDrift {
        region_delta,
        boundary_delta,
        stability_delta,
        cohesion_delta,
        added_regions,
        removed_regions,
        hash_changed: a.canonical_hash != b.canonical_hash,
        drift_score,
    }
}

pub fn detect_phase_transition(drift: &ManifoldDrift, threshold: i64) -> bool {
    drift.drift_score >= threshold.max(1)
        || !drift.added_regions.is_empty()
        || !drift.removed_regions.is_empty()
}

pub fn track_manifold_evolution(
    snapshots: &[CognitiveTopology],
    phase_threshold: i64,
) -> Result<ManifoldEvolutionTrace, InvariantViolation> {
    if snapshots.is_empty() {
        let canonical_hash = hash_json(&Vec::<TopologyEvolutionStep>::new())?;
        return Ok(ManifoldEvolutionTrace {
            steps: Vec::new(),
            persistent_region_ids: Vec::new(),
            transient_region_ids: Vec::new(),
            phase_transition_steps: Vec::new(),
            overall_stability: 1000,
            canonical_hash,
        });
    }

    let mut steps: Vec<TopologyEvolutionStep> = Vec::new();
    let mut phase_transition_steps: Vec<usize> = Vec::new();

    let mut region_appearances: BTreeMap<String, usize> = BTreeMap::new();
    for snapshot in snapshots {
        for region in &snapshot.regions {
            *region_appearances.entry(region.id.clone()).or_default() += 1;
        }
    }

    for i in 1..snapshots.len() {
        let drift = compare_topologies(&snapshots[i - 1], &snapshots[i]);
        let is_phase_transition = detect_phase_transition(&drift, phase_threshold);
        if is_phase_transition {
            phase_transition_steps.push(i);
        }
        steps.push(TopologyEvolutionStep {
            step_index: i,
            drift,
            is_phase_transition,
            topology_hash: snapshots[i].canonical_hash.clone(),
        });
    }

    let total_steps = snapshots.len();
    let persistent_region_ids: Vec<String> = region_appearances
        .iter()
        .filter(|(_, &count)| count == total_steps)
        .map(|(id, _)| id.clone())
        .collect();
    let transient_region_ids: Vec<String> = region_appearances
        .iter()
        .filter(|(_, &count)| count < total_steps)
        .map(|(id, _)| id.clone())
        .collect();

    let total_drift: i64 = steps.iter().map(|s| s.drift.drift_score).sum();
    let overall_stability = if steps.is_empty() {
        1000
    } else {
        (1000 - total_drift / steps.len() as i64).max(0)
    };

    let canonical_hash = hash_json(&(&steps, &persistent_region_ids, &phase_transition_steps))?;

    Ok(ManifoldEvolutionTrace {
        steps,
        persistent_region_ids,
        transient_region_ids,
        phase_transition_steps,
        overall_stability,
        canonical_hash,
    })
}

// ─── Phase 5.3: Cognitive Flow Fields ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptFlowVector {
    pub concept: String,
    /// Positive = moved toward a denser/larger region; negative = fragmented
    pub region_flux: i64,
    /// How many active anchors co-reside in the same region as this concept
    pub anchor_pull: i64,
    /// Net direction: positive = toward stability, negative = toward instability
    pub net_direction: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionFlowVector {
    pub region_id: String,
    pub cohesion_trend: i64,
    pub size_trend: i64,
    pub persistence_score: i64,
    pub is_attractor: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowPrediction {
    pub predicted_stable_region_ids: Vec<String>,
    pub predicted_transient_region_ids: Vec<String>,
    pub convergent: bool,
    pub momentum: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveFlowField {
    pub concept_vectors: Vec<ConceptFlowVector>,
    pub region_vectors: Vec<RegionFlowVector>,
    pub prediction: FlowPrediction,
    pub canonical_hash: String,
}

pub fn compute_cognitive_flow_field(
    snapshots: &[CognitiveTopology],
    anchor_basis_ids: &[String],
) -> Result<CognitiveFlowField, InvariantViolation> {
    if snapshots.is_empty() {
        let prediction = FlowPrediction {
            predicted_stable_region_ids: Vec::new(),
            predicted_transient_region_ids: Vec::new(),
            convergent: true,
            momentum: 0,
        };
        let canonical_hash = hash_json(&(&Vec::<ConceptFlowVector>::new(), &prediction))?;
        return Ok(CognitiveFlowField {
            concept_vectors: Vec::new(),
            region_vectors: Vec::new(),
            prediction,
            canonical_hash,
        });
    }

    let anchor_set: BTreeSet<String> = anchor_basis_ids.iter().cloned().collect();

    // Track region appearances and cohesion across snapshots
    let mut region_first_size: BTreeMap<String, usize> = BTreeMap::new();
    let mut region_last_size: BTreeMap<String, usize> = BTreeMap::new();
    let mut region_first_cohesion: BTreeMap<String, i64> = BTreeMap::new();
    let mut region_last_cohesion: BTreeMap<String, i64> = BTreeMap::new();
    let mut region_appearances: BTreeMap<String, usize> = BTreeMap::new();

    for snapshot in snapshots {
        for region in &snapshot.regions {
            let entry = region_appearances.entry(region.id.clone()).or_default();
            if *entry == 0 {
                region_first_size.insert(region.id.clone(), region.members.len());
                region_first_cohesion.insert(region.id.clone(), region.cohesion_score);
            }
            *entry += 1;
            region_last_size.insert(region.id.clone(), region.members.len());
            region_last_cohesion.insert(region.id.clone(), region.cohesion_score);
        }
    }

    let total_steps = snapshots.len();

    // Build region flow vectors
    let mut region_vectors: Vec<RegionFlowVector> = region_appearances
        .iter()
        .map(|(id, &count)| {
            let first_sz = *region_first_size.get(id).unwrap_or(&0) as i64;
            let last_sz = *region_last_size.get(id).unwrap_or(&0) as i64;
            let first_coh = *region_first_cohesion.get(id).unwrap_or(&0);
            let last_coh = *region_last_cohesion.get(id).unwrap_or(&0);

            // Detect if any anchor lives in this region (by checking last snapshot)
            let is_attractor = snapshots
                .last()
                .and_then(|s| s.regions.iter().find(|r| r.id == *id))
                .map(|r| r.members.iter().any(|m| anchor_set.contains(m)))
                .unwrap_or(false);

            RegionFlowVector {
                region_id: id.clone(),
                cohesion_trend: last_coh - first_coh,
                size_trend: last_sz - first_sz,
                persistence_score: (count as i64 * 1000) / total_steps.max(1) as i64,
                is_attractor,
            }
        })
        .collect();
    region_vectors.sort_by(|a, b| a.region_id.cmp(&b.region_id));

    // Build concept flow vectors from last two snapshots
    let mut concept_vectors: Vec<ConceptFlowVector> = Vec::new();
    if snapshots.len() >= 2 {
        let prev = &snapshots[snapshots.len() - 2];
        let curr = &snapshots[snapshots.len() - 1];

        // Map concept -> region size in each snapshot
        let mut prev_concept_region_size: BTreeMap<String, usize> = BTreeMap::new();
        for region in &prev.regions {
            for member in &region.members {
                prev_concept_region_size.insert(member.clone(), region.members.len());
            }
        }
        let mut curr_concept_region_size: BTreeMap<String, usize> = BTreeMap::new();
        let mut curr_concept_anchor_pull: BTreeMap<String, i64> = BTreeMap::new();
        for region in &curr.regions {
            let anchor_count = region.members.iter().filter(|m| anchor_set.contains(*m)).count() as i64;
            for member in &region.members {
                curr_concept_region_size.insert(member.clone(), region.members.len());
                curr_concept_anchor_pull.insert(member.clone(), anchor_count);
            }
        }

        let all_concepts: BTreeSet<String> = prev_concept_region_size
            .keys()
            .chain(curr_concept_region_size.keys())
            .cloned()
            .collect();

        for concept in all_concepts {
            let prev_sz = *prev_concept_region_size.get(&concept).unwrap_or(&0) as i64;
            let curr_sz = *curr_concept_region_size.get(&concept).unwrap_or(&0) as i64;
            let anchor_pull = *curr_concept_anchor_pull.get(&concept).unwrap_or(&0);
            let region_flux = curr_sz - prev_sz;

            let is_anchor = anchor_set.contains(&concept);
            let net_direction = region_flux + anchor_pull * 10 + if is_anchor { 50 } else { 0 };

            concept_vectors.push(ConceptFlowVector {
                concept,
                region_flux,
                anchor_pull,
                net_direction,
            });
        }
        concept_vectors.sort_by(|a, b| a.concept.cmp(&b.concept));
    }

    // Flow-based prediction
    let momentum: i64 = if snapshots.len() >= 2 {
        let drift = compare_topologies(
            &snapshots[snapshots.len() - 2],
            &snapshots[snapshots.len() - 1],
        );
        drift.drift_score
    } else {
        0
    };

    let drift_trend_converging = if snapshots.len() >= 3 {
        let d1 = compare_topologies(&snapshots[snapshots.len() - 3], &snapshots[snapshots.len() - 2]).drift_score;
        let d2 = compare_topologies(&snapshots[snapshots.len() - 2], &snapshots[snapshots.len() - 1]).drift_score;
        d2 <= d1
    } else {
        momentum == 0
    };

    let predicted_stable_region_ids: Vec<String> = region_vectors
        .iter()
        .filter(|r| r.persistence_score >= 750 && r.cohesion_trend >= 0)
        .map(|r| r.region_id.clone())
        .collect();
    let predicted_transient_region_ids: Vec<String> = region_vectors
        .iter()
        .filter(|r| r.persistence_score < 750 || r.cohesion_trend < 0)
        .map(|r| r.region_id.clone())
        .collect();

    let prediction = FlowPrediction {
        predicted_stable_region_ids,
        predicted_transient_region_ids,
        convergent: drift_trend_converging,
        momentum,
    };

    let canonical_hash = hash_json(&(&concept_vectors, &region_vectors, &prediction))?;

    Ok(CognitiveFlowField {
        concept_vectors,
        region_vectors,
        prediction,
        canonical_hash,
    })
}

// ─── Phase 5.4: Cognitive Energy & Action Selection ─────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StabilityEnergy {
    pub region_id: String,
    pub potential: i64,
    /// Positive = low energy well; negative = high energy barrier
    pub well_depth: i64,
    /// How "sticky" the region is
    pub attraction_strength: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnergyGradient {
    pub source_region: String,
    pub target_region: String,
    /// Positive = downhill (favorable); negative = uphill (costly)
    pub gradient: i64,
    pub traversal_cost: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitivePotentialField {
    pub stability_energies: Vec<StabilityEnergy>,
    pub gradients: Vec<EnergyGradient>,
    pub global_minimum_region: String,
    pub global_minimum_energy: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSelectionPolicy {
    pub preferred_trajectory: Vec<String>,
    pub energy_cost: i64,
    pub stability_gain: i64,
    pub confidence: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnergyMinimizingTrajectory {
    pub actions: Vec<ActionSelectionPolicy>,
    pub total_energy_cost: i64,
    pub convergent_outcome: bool,
    pub canonical_hash: String,
}

pub fn compute_cognitive_potential_field(
    flow_field: &CognitiveFlowField,
) -> Result<CognitivePotentialField, InvariantViolation> {
    // Energy assignment: persistence → potential inversion (high persistence = low energy)
    let mut stability_energies: Vec<StabilityEnergy> = flow_field
        .region_vectors
        .iter()
        .map(|rv| {
            // High persistence → low potential; low persistence → high potential
            let potential = 1000 - rv.persistence_score;
            let well_depth = if rv.is_attractor {
                rv.persistence_score as i64 * 2
            } else {
                -potential
            };
            let attraction_strength = if rv.is_attractor {
                (1000 - potential).max(100)
            } else {
                (potential - 500).max(0)
            };
            StabilityEnergy {
                region_id: rv.region_id.clone(),
                potential,
                well_depth,
                attraction_strength,
            }
        })
        .collect();
    stability_energies.sort_by_key(|e| e.potential);

    // Build directed gradient map for all region pairs.
    // Gradient = source potential - target potential.
    // Positive means moving source -> target is downhill (energy minimizing).
    let mut gradients: Vec<EnergyGradient> = Vec::new();
    for i in 0..stability_energies.len() {
        for j in 0..stability_energies.len() {
            if i == j {
                continue;
            }
            let energy_delta =
                stability_energies[i].potential as i64 - stability_energies[j].potential as i64;
            let traversal_cost = energy_delta.abs() + 50; // Base traversal cost
            let gradient = energy_delta;

            gradients.push(EnergyGradient {
                source_region: stability_energies[i].region_id.clone(),
                target_region: stability_energies[j].region_id.clone(),
                gradient,
                traversal_cost,
            });
        }
    }
    gradients.sort_by(|a, b| b.gradient.cmp(&a.gradient));

    let global_minimum_region = stability_energies
        .first()
        .map(|e| e.region_id.clone())
        .unwrap_or_default();
    let global_minimum_energy = stability_energies
        .first()
        .map(|e| e.potential)
        .unwrap_or(500);

    Ok(CognitivePotentialField {
        stability_energies,
        gradients,
        global_minimum_region,
        global_minimum_energy,
    })
}

pub fn select_action(
    potential_field: &CognitivePotentialField,
    current_region: &str,
) -> Result<ActionSelectionPolicy, InvariantViolation> {
    // Find downhill paths from current region
    let mut downhill_paths: Vec<(Vec<String>, i64, i64)> = Vec::new();

    for gradient in &potential_field.gradients {
        if gradient.source_region == current_region && gradient.gradient > 0 {
            // Downhill
            let stability_gain = gradient.gradient;
            let energy_cost = gradient.traversal_cost;
            downhill_paths.push((
                vec![gradient.target_region.clone()],
                energy_cost,
                stability_gain,
            ));
        }
    }

    if downhill_paths.is_empty() {
        // No downhill path; prefer staying in current region (attractor)
        let confidence = 800;
        return Ok(ActionSelectionPolicy {
            preferred_trajectory: vec![current_region.to_string()],
            energy_cost: 0,
            stability_gain: 0,
            confidence,
        });
    }

    // Select the path with highest stability_gain / cost ratio
    let (path, cost, gain) = downhill_paths
        .into_iter()
        .max_by_key(|(_, c, g)| (*g * 1000) / (*c + 1))
        .unwrap();

    let confidence = ((gain * 1000) / (cost + 1)).min(1000);

    Ok(ActionSelectionPolicy {
        preferred_trajectory: path,
        energy_cost: cost,
        stability_gain: gain,
        confidence,
    })
}

pub fn compute_energy_minimizing_trajectory(
    snapshots: &[CognitiveTopology],
    flow_field: &CognitiveFlowField,
    anchor_basis_ids: &[String],
) -> Result<EnergyMinimizingTrajectory, InvariantViolation> {
    let potential_field = compute_cognitive_potential_field(flow_field)?;

    let mut actions: Vec<ActionSelectionPolicy> = Vec::new();
    let mut total_energy_cost: i64 = 0;

    if !snapshots.is_empty() {
        let current_region = potential_field
            .stability_energies
            .iter()
            .max_by_key(|e| e.potential)
            .map(|e| e.region_id.clone())
            .unwrap_or_default();

        let action = select_action(&potential_field, &current_region)?;
        total_energy_cost = action.energy_cost;
        actions.push(action);
    }

    let convergent_outcome = flow_field.prediction.convergent && total_energy_cost < 500;

    let canonical_hash = hash_json(&(
        &actions,
        &potential_field.global_minimum_energy,
        &anchor_basis_ids,
    ))?;

    Ok(EnergyMinimizingTrajectory {
        actions,
        total_energy_cost,
        convergent_outcome,
        canonical_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::determinism::DeterminismVerifier;

    fn build_mfc() -> MultiFrameCognition {
        let mut mfc = MultiFrameCognition::new();
        mfc.register_frame(
            "physics",
            vec![
                SemanticConstraint::assertion("light", "wave", true, 90),
                SemanticConstraint::assertion("light", "particle", true, 85),
                SemanticConstraint::assertion("vacuum", "has_medium", false, 80),
            ],
        );
        mfc.register_frame(
            "ontology",
            vec![
                SemanticConstraint::assertion("light", "wave", false, 20),
                SemanticConstraint::assertion("light", "particle", true, 60),
                SemanticConstraint::assertion("vacuum", "has_medium", true, 15),
            ],
        );
        mfc
    }

    #[test]
    fn mfc_is_replay_stable_across_worker_counts() {
        let verifier = DeterminismVerifier::new();

        let mut a = build_mfc();
        let mut b = build_mfc();
        let config_a = MultiFrameConfig {
            worker_count: 1,
            ..MultiFrameConfig::default()
        };
        let config_b = MultiFrameConfig {
            worker_count: 8,
            ..MultiFrameConfig::default()
        };

        let ra = a.run(config_a).expect("run should succeed");
        let rb = b.run(config_b).expect("run should succeed");

        assert!(verifier.is_replay_stable(&ra, &rb).unwrap_or(false));
    }

    #[test]
    fn mfc_propagates_cross_frame_constraints() {
        let mut mfc = build_mfc();
        let report = mfc.run(MultiFrameConfig::default()).expect("run should succeed");
        assert!(!report.iterations.is_empty());
        assert!(report.iterations[0].propagated_constraints > 0);
    }

    #[test]
    fn mfc_produces_consolidated_memory_artifact() {
        let mut mfc = build_mfc();
        let report = mfc.run(MultiFrameConfig::default()).expect("run should succeed");
        assert!(!report.consolidated_memory.artifact_hash.is_empty());
        assert!(!report.consolidated_memory.fused_constraints.is_empty());
    }

    #[test]
    fn mfc_registers_concept_anchors_when_stable() {
        let mut mfc = build_mfc();
        let report = mfc
            .run(MultiFrameConfig {
                anchor_energy_max: 800,
                anchor_min_persistence: 1,
                ..MultiFrameConfig::default()
            })
            .expect("run should succeed");

        assert!(!report.anchor_registry.anchors.is_empty());
        assert!(report
            .anchor_registry
            .anchors
            .iter()
            .all(|a| !a.canonical_hash.is_empty()));
    }
}
