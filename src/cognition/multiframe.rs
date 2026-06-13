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
