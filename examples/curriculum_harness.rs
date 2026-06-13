use rugc::{
    arbitrate_intent_field, build_meta_intent_field, compute_cognitive_flow_field,
    compute_cognitive_potential_field, compute_cognitive_topology, compute_intent_field,
    MultiFrameCognition, MultiFrameConfig, SemanticConstraint,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum Domain {
    Locomotion,
    Manipulation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum EpisodeKind {
    Held,
    SupportedPlay,
    Perturbation,
    Recovery,
    Holdout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum RunMode {
    FullStack,
    NoMeta,
}

impl RunMode {
    fn as_str(self) -> &'static str {
        match self {
            RunMode::FullStack => "full_stack",
            RunMode::NoMeta => "no_meta",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum TuningPass {
    Canonical,
    ConvergenceGate,
    FlowEnergyDescent,
    AnchorStabilization,
}

impl TuningPass {
    fn as_str(self) -> &'static str {
        match self {
            TuningPass::Canonical => "canonical",
            TuningPass::ConvergenceGate => "convergence_gate_tuning",
            TuningPass::FlowEnergyDescent => "flow_energy_descent_sharpening",
            TuningPass::AnchorStabilization => "anchor_stabilization_acceleration",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct EpisodeSpec {
    id: &'static str,
    label: &'static str,
    domain: Domain,
    kind: EpisodeKind,
    support_strength: i64,
    wobble_strength: i64,
    contradiction_strength: i64,
    recovery_bias: i64,
    novelty_tag: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct EpisodeMetrics {
    mode: RunMode,
    id: String,
    label: String,
    domain: Domain,
    kind: EpisodeKind,
    novelty_tag: String,
    support_strength: i64,
    wobble_strength: i64,
    contradiction_strength: i64,
    recovery_bias: i64,
    converged_iteration: Option<usize>,
    active_anchors: usize,
    emergent_active: usize,
    self_continuity_score: i64,
    external_change_score: i64,
    topology_regions: usize,
    manifold_stability: i64,
    momentum: i64,
    minimum_energy: i64,
    intent_goal_count: usize,
    arbitration_confidence: i64,
    self_consistency: i64,
    meta_revision_count: usize,
    final_trace_hash: String,
}

#[derive(Debug, Clone)]
struct PassFailRubric {
    min_holdout_self_consistency: i64,
    min_holdout_arbitration_confidence: i64,
    min_anchor_advantage: isize,
    min_region_advantage: isize,
    min_goal_advantage: isize,
    min_holdout_count: usize,
    min_domain_count: usize,
    max_average_external_change_delta: i64,
    max_average_recovery_converged_iteration: usize,
    min_average_recovery_consistency_advantage: i64,
}

#[derive(Debug, Clone, Serialize)]
struct VerificationCheck {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerificationOutcome {
    passed: bool,
    checks: Vec<VerificationCheck>,
}

#[derive(Debug, Clone, Serialize)]
struct HoldoutPairResult {
    holdout_id: String,
    domain: Domain,
    trained_holdout: EpisodeMetrics,
    fresh_holdout: EpisodeMetrics,
    trained_recovery: EpisodeMetrics,
    fresh_recovery: EpisodeMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticBaseline {
    mode: RunMode,
    stage_d_recovery_median_iteration: usize,
    derived_recovery_budget_2x_median: usize,
    canonical_recovery_budget: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ModeRun {
    mode: RunMode,
    diagnostic_baseline: DiagnosticBaseline,
    training: Vec<EpisodeMetrics>,
    holdouts: Vec<HoldoutPairResult>,
    verification: VerificationOutcome,
}

#[derive(Debug, Clone, Serialize)]
struct ModeComparison {
    full_stack_avg_recovery_iteration: usize,
    no_meta_avg_recovery_iteration: usize,
    full_stack_avg_recovery_self_consistency: i64,
    no_meta_avg_recovery_self_consistency: i64,
    interpretation: String,
}

#[derive(Debug, Clone, Serialize)]
struct ExportRubric {
    min_holdout_self_consistency: i64,
    min_holdout_arbitration_confidence: i64,
    min_anchor_advantage: isize,
    min_region_advantage: isize,
    min_goal_advantage: isize,
    min_holdout_count: usize,
    min_domain_count: usize,
    max_average_external_change_delta: i64,
    max_average_recovery_converged_iteration: usize,
    min_average_recovery_consistency_advantage: i64,
}

#[derive(Debug, Clone, Serialize)]
struct ExportBundle {
    rubric: ExportRubric,
    mode_runs: Vec<ModeRun>,
    comparison: ModeComparison,
}

#[derive(Debug, Clone, Serialize)]
struct MicroExperimentGate {
    name: String,
    passed: bool,
    required_max_recovery_iteration: usize,
    observed_full_stack_recovery_iteration: usize,
    non_speed_checks_passed: bool,
    anti_shortcut_quality_passed: bool,
    required_min_recovery_continuity: i64,
    observed_full_stack_recovery_continuity: i64,
    required_min_recovery_regions: usize,
    observed_full_stack_recovery_regions: usize,
    required_min_recovery_anchors: usize,
    observed_full_stack_recovery_anchors: usize,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct MicroExperimentResult {
    pass: TuningPass,
    mode_runs: Vec<ModeRun>,
    comparison: ModeComparison,
    gate: MicroExperimentGate,
    promoted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Phase6TuningSequence {
    canonical_full_stack_recovery_iteration: usize,
    experiments: Vec<MicroExperimentResult>,
    all_gates_passed: bool,
}

#[derive(Debug, Clone, Copy)]
struct FullStackQualitySnapshot {
    recovery_iteration: usize,
    recovery_continuity: i64,
    recovery_regions: usize,
    recovery_anchors: usize,
}

fn cfg() -> MultiFrameConfig {
    MultiFrameConfig {
        iterations: 12,
        worker_count: 4,
        ambiguity_margin: 5000,
        target_energy: 500,
        compression_threshold: 1,
        convergence_window: 2,
        energy_delta_threshold: 2,
        anchor_energy_max: 2000,
        anchor_pull_strength: 4,
        anchor_min_persistence: 1,
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

fn cfg_for_pass(pass: TuningPass) -> MultiFrameConfig {
    let mut cfg = cfg();
    match pass {
        TuningPass::Canonical => {}
        TuningPass::ConvergenceGate => {
            cfg.convergence_window = 1;
            cfg.energy_delta_threshold = 1;
            cfg.ambiguity_margin = 3000;
        }
        TuningPass::FlowEnergyDescent => {
            cfg.convergence_window = 1;
            cfg.energy_delta_threshold = 1;
            cfg.ambiguity_margin = 3000;
            cfg.target_energy = 350;
            cfg.anchor_pull_strength = 6;
            cfg.anchor_fusion_bias = 12;
        }
        TuningPass::AnchorStabilization => {
            cfg.convergence_window = 1;
            cfg.energy_delta_threshold = 1;
            cfg.ambiguity_margin = 3000;
            cfg.target_energy = 350;
            cfg.anchor_pull_strength = 8;
            cfg.anchor_fusion_bias = 12;
            cfg.anchor_alignment_window = 16;
            cfg.emergent_resonance_threshold = 30;
            cfg.emergent_min_persistence = 1;
        }
    }
    cfg
}

fn curriculum() -> Vec<EpisodeSpec> {
    vec![
        EpisodeSpec {
            id: "held_01",
            label: "Held baseline support",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Held,
            support_strength: 96,
            wobble_strength: 5,
            contradiction_strength: 4,
            recovery_bias: 92,
            novelty_tag: "held",
        },
        EpisodeSpec {
            id: "held_02",
            label: "Held replay support",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Held,
            support_strength: 94,
            wobble_strength: 6,
            contradiction_strength: 6,
            recovery_bias: 90,
            novelty_tag: "held",
        },
        EpisodeSpec {
            id: "play_01",
            label: "Supported play mild wobble",
            domain: Domain::Locomotion,
            kind: EpisodeKind::SupportedPlay,
            support_strength: 82,
            wobble_strength: 18,
            contradiction_strength: 10,
            recovery_bias: 84,
            novelty_tag: "play_a",
        },
        EpisodeSpec {
            id: "play_02",
            label: "Supported play lateral wobble",
            domain: Domain::Locomotion,
            kind: EpisodeKind::SupportedPlay,
            support_strength: 78,
            wobble_strength: 22,
            contradiction_strength: 14,
            recovery_bias: 82,
            novelty_tag: "play_b",
        },
        EpisodeSpec {
            id: "perturb_01",
            label: "External perturbation forward fall risk",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Perturbation,
            support_strength: 60,
            wobble_strength: 34,
            contradiction_strength: 24,
            recovery_bias: 70,
            novelty_tag: "perturb_a",
        },
        EpisodeSpec {
            id: "recover_01",
            label: "Recovery to upright gait",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Recovery,
            support_strength: 90,
            wobble_strength: 8,
            contradiction_strength: 4,
            recovery_bias: 94,
            novelty_tag: "recover",
        },
        EpisodeSpec {
            id: "stack_held_01",
            label: "Held grasp baseline alignment",
            domain: Domain::Manipulation,
            kind: EpisodeKind::Held,
            support_strength: 95,
            wobble_strength: 6,
            contradiction_strength: 4,
            recovery_bias: 90,
            novelty_tag: "stack_held",
        },
        EpisodeSpec {
            id: "stack_play_01",
            label: "Supported block stacking play",
            domain: Domain::Manipulation,
            kind: EpisodeKind::SupportedPlay,
            support_strength: 80,
            wobble_strength: 20,
            contradiction_strength: 12,
            recovery_bias: 82,
            novelty_tag: "stack_play",
        },
        EpisodeSpec {
            id: "stack_perturb_01",
            label: "Slip perturbation during stack placement",
            domain: Domain::Manipulation,
            kind: EpisodeKind::Perturbation,
            support_strength: 58,
            wobble_strength: 30,
            contradiction_strength: 26,
            recovery_bias: 68,
            novelty_tag: "stack_slip",
        },
        EpisodeSpec {
            id: "stack_recover_01",
            label: "Regrasp recovery after slip",
            domain: Domain::Manipulation,
            kind: EpisodeKind::Recovery,
            support_strength: 88,
            wobble_strength: 10,
            contradiction_strength: 5,
            recovery_bias: 92,
            novelty_tag: "stack_recover",
        },
        EpisodeSpec {
            id: "holdout_01",
            label: "Holdout unsupported diagonal step",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Holdout,
            support_strength: 48,
            wobble_strength: 28,
            contradiction_strength: 18,
            recovery_bias: 76,
            novelty_tag: "holdout_diagonal",
        },
        EpisodeSpec {
            id: "holdout_02",
            label: "Holdout noisy staggered step",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Holdout,
            support_strength: 36,
            wobble_strength: 46,
            contradiction_strength: 38,
            recovery_bias: 64,
            novelty_tag: "spiral_lurch_terrain_shear",
        },
        EpisodeSpec {
            id: "holdout_03",
            label: "Holdout cross-body recovery step",
            domain: Domain::Locomotion,
            kind: EpisodeKind::Holdout,
            support_strength: 32,
            wobble_strength: 52,
            contradiction_strength: 44,
            recovery_bias: 60,
            novelty_tag: "counterweight_spiral_trip",
        },
        EpisodeSpec {
            id: "holdout_04",
            label: "Holdout blind regrasp under load",
            domain: Domain::Manipulation,
            kind: EpisodeKind::Holdout,
            support_strength: 34,
            wobble_strength: 44,
            contradiction_strength: 36,
            recovery_bias: 62,
            novelty_tag: "blind_regrasp_load_shift",
        },
        EpisodeSpec {
            id: "holdout_05",
            label: "Holdout offset stack with torsion",
            domain: Domain::Manipulation,
            kind: EpisodeKind::Holdout,
            support_strength: 30,
            wobble_strength: 50,
            contradiction_strength: 42,
            recovery_bias: 58,
            novelty_tag: "offset_stack_torsion_swap",
        },
    ]
}

fn recovery_spec_from_holdout(spec: &EpisodeSpec) -> EpisodeSpec {
    let id = format!("{}_recovery", spec.id);
    let label = format!("Recovery after {}", spec.label);
    let novelty = format!("{}_recovery", spec.novelty_tag);
    EpisodeSpec {
        id: Box::leak(id.into_boxed_str()),
        label: Box::leak(label.into_boxed_str()),
        domain: spec.domain,
        kind: EpisodeKind::Recovery,
        support_strength: (spec.support_strength + 30).min(96),
        wobble_strength: (spec.wobble_strength / 3).max(8),
        contradiction_strength: (spec.contradiction_strength / 4).max(4),
        recovery_bias: (spec.recovery_bias + 26).min(96),
        novelty_tag: Box::leak(novelty.into_boxed_str()),
    }
}

fn weight(value: i64) -> u8 {
    value.clamp(0, 100) as u8
}

fn register_locomotion_episode(mfc: &mut MultiFrameCognition, spec: &EpisodeSpec) {
    let support = spec.support_strength;
    let wobble = spec.wobble_strength;
    let contradiction = spec.contradiction_strength;
    let recovery = spec.recovery_bias;
    let unsupported = matches!(spec.kind, EpisodeKind::Holdout | EpisodeKind::Perturbation);
    let parent_support = !unsupported;

    mfc.register_frame(
        "body_dynamics",
        vec![
            SemanticConstraint::assertion("torso", "is_upright", true, weight(support)),
            SemanticConstraint::assertion("center_of_mass", "inside_base", true, weight(recovery)),
            SemanticConstraint::assertion(
                "step_cycle",
                "is_balanced",
                true,
                weight(support - wobble / 2),
            ),
            SemanticConstraint::assertion("wobble", "is_present", wobble > 12, weight(wobble.max(8))),
            SemanticConstraint::assertion(
                "fall_risk",
                "is_high",
                wobble > 26,
                weight((wobble + contradiction).max(8)),
            ),
        ],
    );

    mfc.register_frame(
        "support_context",
        vec![
            SemanticConstraint::assertion(
                "parent",
                "provides_support",
                parent_support,
                weight(support.max(10)),
            ),
            SemanticConstraint::assertion(
                "child",
                "self_stabilizes",
                unsupported,
                weight((recovery + wobble / 2).max(8)),
            ),
            SemanticConstraint::assertion(
                "constraint_loop",
                "restores_balance",
                true,
                weight(recovery.max(10)),
            ),
            SemanticConstraint::assertion("support_state", spec.novelty_tag, true, weight(50)),
        ],
    );

    mfc.register_frame(
        "interpretation",
        vec![
            SemanticConstraint::assertion(
                "walking",
                "is_learned",
                unsupported,
                weight((support - contradiction).max(8)),
            ),
            SemanticConstraint::assertion(
                "walking",
                "requires_support",
                parent_support,
                weight((support + contradiction / 2).max(8)),
            ),
            SemanticConstraint::assertion(
                "balance",
                "recovers_after_perturbation",
                true,
                weight(recovery.max(8)),
            ),
            SemanticConstraint::assertion(
                "geometry",
                "stabilizes_motion",
                true,
                weight((recovery + support / 4).max(8)),
            ),
            SemanticConstraint::assertion("novelty", spec.novelty_tag, true, weight(42)),
        ],
    );
}

fn register_manipulation_episode(mfc: &mut MultiFrameCognition, spec: &EpisodeSpec) {
    let support = spec.support_strength;
    let wobble = spec.wobble_strength;
    let contradiction = spec.contradiction_strength;
    let recovery = spec.recovery_bias;
    let unsupported = matches!(spec.kind, EpisodeKind::Holdout | EpisodeKind::Perturbation);
    let parent_support = !unsupported;

    mfc.register_frame(
        "body_dynamics",
        vec![
            SemanticConstraint::assertion("grip", "is_stable", true, weight(support)),
            SemanticConstraint::assertion("block_stack", "is_aligned", true, weight(recovery)),
            SemanticConstraint::assertion(
                "contact_patch",
                "is_centered",
                true,
                weight(support - wobble / 2),
            ),
            SemanticConstraint::assertion("slip", "is_present", wobble > 12, weight(wobble.max(8))),
            SemanticConstraint::assertion(
                "collapse_risk",
                "is_high",
                wobble > 28,
                weight((wobble + contradiction).max(8)),
            ),
        ],
    );

    mfc.register_frame(
        "support_context",
        vec![
            SemanticConstraint::assertion(
                "parent",
                "guides_grasp",
                parent_support,
                weight(support.max(10)),
            ),
            SemanticConstraint::assertion(
                "child",
                "self_regrasps",
                unsupported,
                weight((recovery + wobble / 2).max(8)),
            ),
            SemanticConstraint::assertion(
                "constraint_loop",
                "restores_stack",
                true,
                weight(recovery.max(10)),
            ),
            SemanticConstraint::assertion("support_state", spec.novelty_tag, true, weight(50)),
        ],
    );

    mfc.register_frame(
        "interpretation",
        vec![
            SemanticConstraint::assertion(
                "stacking",
                "is_learned",
                unsupported,
                weight((support - contradiction).max(8)),
            ),
            SemanticConstraint::assertion(
                "stacking",
                "requires_support",
                parent_support,
                weight((support + contradiction / 2).max(8)),
            ),
            SemanticConstraint::assertion(
                "grasp_geometry",
                "recovers_after_slip",
                true,
                weight(recovery.max(8)),
            ),
            SemanticConstraint::assertion(
                "geometry",
                "stabilizes_stack",
                true,
                weight((recovery + support / 4).max(8)),
            ),
            SemanticConstraint::assertion("novelty", spec.novelty_tag, true, weight(42)),
        ],
    );
}

fn register_episode(mfc: &mut MultiFrameCognition, spec: &EpisodeSpec) {
    match spec.domain {
        Domain::Locomotion => register_locomotion_episode(mfc, spec),
        Domain::Manipulation => register_manipulation_episode(mfc, spec),
    }

    let contradiction = spec.contradiction_strength;
    if contradiction > 0 {
        let contradiction_frames = match spec.domain {
            Domain::Locomotion => vec![
                SemanticConstraint::assertion("torso", "is_upright", false, weight(contradiction)),
                SemanticConstraint::assertion("fall_risk", "is_high", true, weight(contradiction + 8)),
                SemanticConstraint::assertion("walking", "requires_support", true, weight(contradiction)),
            ],
            Domain::Manipulation => vec![
                SemanticConstraint::assertion("grip", "is_stable", false, weight(contradiction)),
                SemanticConstraint::assertion("collapse_risk", "is_high", true, weight(contradiction + 8)),
                SemanticConstraint::assertion("stacking", "requires_support", true, weight(contradiction)),
            ],
        };
        mfc.register_frame("contradiction_probe", contradiction_frames);
    } else {
        mfc.register_frame(
            "contradiction_probe",
            vec![SemanticConstraint::assertion("balance", "is_stable", true, weight(16))],
        );
    }
}

fn run_episode(
    mfc: &mut MultiFrameCognition,
    spec: &EpisodeSpec,
    mode: RunMode,
    config: &MultiFrameConfig,
) -> EpisodeMetrics {
    register_episode(mfc, spec);
    let report = mfc
        .run(config.clone())
        .expect("episode run should succeed");

    let topo_a = compute_cognitive_topology(&report.consolidated_memory, 500)
        .expect("topology A should compute");
    let topo_b = compute_cognitive_topology(&report.consolidated_memory, 500)
        .expect("topology B should compute");
    let anchors = report.consolidated_memory.anchor_basis_ids.clone();

    let flow = compute_cognitive_flow_field(&[topo_a.clone(), topo_b], &anchors)
        .expect("flow should compute");
    let potential = compute_cognitive_potential_field(&flow).expect("potential should compute");
    let intent = compute_intent_field(&potential, &anchors).expect("intent should compute");

    let base_weights: BTreeMap<String, i64> = potential
        .stability_energies
        .iter()
        .map(|e| (e.region_id.clone(), e.attraction_strength))
        .collect();
    let arbitrated = arbitrate_intent_field(&[intent.clone(), intent], &potential, &base_weights)
        .expect("arbitration should compute");

    let (self_consistency, meta_revision_count) = match mode {
        RunMode::FullStack => {
            let meta = build_meta_intent_field(&arbitrated, &[]).expect("meta intent should compute");
            (
                meta.self_coherence.self_consistency,
                meta.revision_candidates.len(),
            )
        }
        RunMode::NoMeta => {
            // Pass-through mode: skip meta-intent synthesis and use arbitration confidence as top-level signal.
            (arbitrated.arbitration_confidence, 0)
        }
    };

    EpisodeMetrics {
        mode,
        id: spec.id.to_string(),
        label: spec.label.to_string(),
        domain: spec.domain,
        kind: spec.kind,
        novelty_tag: spec.novelty_tag.to_string(),
        support_strength: spec.support_strength,
        wobble_strength: spec.wobble_strength,
        contradiction_strength: spec.contradiction_strength,
        recovery_bias: spec.recovery_bias,
        converged_iteration: report.converged_iteration,
        active_anchors: report.anchor_registry.anchors.len(),
        emergent_active: report.consolidated_memory.emergent_concepts.len(),
        self_continuity_score: report.consolidated_memory.self_continuity_score,
        external_change_score: report.consolidated_memory.external_change_score,
        topology_regions: topo_a.metrics.region_count,
        manifold_stability: topo_a.metrics.manifold_stability,
        momentum: flow.prediction.momentum,
        minimum_energy: potential.global_minimum_energy,
        intent_goal_count: arbitrated.goal_set.goals.len(),
        arbitration_confidence: arbitrated.arbitration_confidence,
        self_consistency,
        meta_revision_count,
        final_trace_hash: report.final_trace_hash,
    }
}

fn average_i64(values: &[i64]) -> i64 {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<i64>() / values.len() as i64
    }
}

fn average_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<usize>() / values.len()
    }
}

fn average_isize(values: &[isize]) -> isize {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<isize>() / values.len() as isize
    }
}

fn holdout_domain_count(holdout_results: &[HoldoutPairResult]) -> usize {
    holdout_results
        .iter()
        .map(|r| r.domain)
        .collect::<BTreeSet<_>>()
        .len()
}

fn verify_learning(
    first_training: &EpisodeMetrics,
    holdout_results: &[HoldoutPairResult],
    rubric: &PassFailRubric,
    max_iterations: usize,
) -> VerificationOutcome {
    let anchor_advantages: Vec<isize> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.active_anchors as isize - r.fresh_holdout.active_anchors as isize)
        .collect();
    let region_advantages: Vec<isize> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.topology_regions as isize - r.fresh_holdout.topology_regions as isize)
        .collect();
    let goal_advantages: Vec<isize> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.intent_goal_count as isize - r.fresh_holdout.intent_goal_count as isize)
        .collect();
    let holdout_consistency: Vec<i64> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.self_consistency)
        .collect();
    let holdout_confidence: Vec<i64> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.arbitration_confidence)
        .collect();
    let external_deltas: Vec<i64> = holdout_results
        .iter()
        .map(|r| r.trained_holdout.external_change_score - r.fresh_holdout.external_change_score)
        .collect();
    let trained_recovery_converged: Vec<usize> = holdout_results
        .iter()
        .map(|r| r.trained_recovery.converged_iteration.unwrap_or(max_iterations + 1))
        .collect();
    let recovery_consistency_advantages: Vec<i64> = holdout_results
        .iter()
        .map(|r| r.trained_recovery.self_consistency - r.fresh_recovery.self_consistency)
        .collect();

    let avg_anchor_adv = average_isize(&anchor_advantages);
    let avg_region_adv = average_isize(&region_advantages);
    let avg_goal_adv = average_isize(&goal_advantages);
    let avg_consistency = average_i64(&holdout_consistency);
    let avg_confidence = average_i64(&holdout_confidence);
    let avg_external_delta = average_i64(&external_deltas);
    let avg_recovery_converged = average_usize(&trained_recovery_converged);
    let avg_recovery_consistency_adv = average_i64(&recovery_consistency_advantages);

    let checks = vec![
        VerificationCheck {
            name: "holdout battery size meets minimum".to_string(),
            passed: holdout_results.len() >= rubric.min_holdout_count,
            detail: format!("holdouts={} required>={}", holdout_results.len(), rubric.min_holdout_count),
        },
        VerificationCheck {
            name: "holdout battery spans multiple domains".to_string(),
            passed: holdout_domain_count(holdout_results) >= rubric.min_domain_count,
            detail: format!("domains={} required>={}", holdout_domain_count(holdout_results), rubric.min_domain_count),
        },
        VerificationCheck {
            name: "average holdout self consistency meets threshold".to_string(),
            passed: avg_consistency >= rubric.min_holdout_self_consistency,
            detail: format!("avg_self_consistency={} threshold>={}", avg_consistency, rubric.min_holdout_self_consistency),
        },
        VerificationCheck {
            name: "average holdout arbitration confidence meets threshold".to_string(),
            passed: avg_confidence >= rubric.min_holdout_arbitration_confidence,
            detail: format!("avg_arbitration_confidence={} threshold>={}", avg_confidence, rubric.min_holdout_arbitration_confidence),
        },
        VerificationCheck {
            name: "training grows anchor memory before holdouts".to_string(),
            passed: holdout_results
                .first()
                .map(|r| {
                    r.trained_holdout.active_anchors as isize - first_training.active_anchors as isize
                        >= rubric.min_anchor_advantage
                })
                .unwrap_or(false),
            detail: holdout_results
                .first()
                .map(|r| format!(
                    "first_training.active_anchors={} first_trained_holdout.active_anchors={} required_growth>={}",
                    first_training.active_anchors, r.trained_holdout.active_anchors, rubric.min_anchor_advantage
                ))
                .unwrap_or_else(|| "no holdouts".to_string()),
        },
        VerificationCheck {
            name: "trained learner has more anchors across holdouts".to_string(),
            passed: avg_anchor_adv >= rubric.min_anchor_advantage,
            detail: format!("avg_anchor_advantage={} required>={}", avg_anchor_adv, rubric.min_anchor_advantage),
        },
        VerificationCheck {
            name: "trained learner builds richer topology across holdouts".to_string(),
            passed: avg_region_adv >= rubric.min_region_advantage,
            detail: format!("avg_region_advantage={} required>={}", avg_region_adv, rubric.min_region_advantage),
        },
        VerificationCheck {
            name: "trained learner builds richer goal geometry across holdouts".to_string(),
            passed: avg_goal_adv >= rubric.min_goal_advantage,
            detail: format!("avg_goal_advantage={} required>={}", avg_goal_adv, rubric.min_goal_advantage),
        },
        VerificationCheck {
            name: "trained learner transfers under harder noisy perturbations".to_string(),
            passed: avg_external_delta <= rubric.max_average_external_change_delta,
            detail: format!("avg_external_change_delta={} required<={}", avg_external_delta, rubric.max_average_external_change_delta),
        },
        VerificationCheck {
            name: "trained learner recovers within speed budget".to_string(),
            passed: avg_recovery_converged <= rubric.max_average_recovery_converged_iteration,
            detail: format!(
                "avg_recovery_converged_iteration={} required<={}",
                avg_recovery_converged, rubric.max_average_recovery_converged_iteration
            ),
        },
        VerificationCheck {
            name: "trained learner has stronger recovery consistency".to_string(),
            passed: avg_recovery_consistency_adv >= rubric.min_average_recovery_consistency_advantage,
            detail: format!(
                "avg_recovery_consistency_advantage={} required>={}",
                avg_recovery_consistency_adv, rubric.min_average_recovery_consistency_advantage
            ),
        },
    ];

    VerificationOutcome {
        passed: checks.iter().all(|c| c.passed),
        checks,
    }
}

fn median_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2
    }
}

fn derive_recovery_baseline(
    mode: RunMode,
    training_results: &[EpisodeMetrics],
    canonical_recovery_budget: usize,
    max_iterations: usize,
) -> DiagnosticBaseline {
    let stage_d_recovery_iterations: Vec<usize> = training_results
        .iter()
        .filter(|m| m.kind == EpisodeKind::Recovery)
        .map(|m| m.converged_iteration.unwrap_or(max_iterations + 1))
        .collect();

    let stage_d_recovery_median_iteration = median_usize(&stage_d_recovery_iterations);
    let derived_recovery_budget_2x_median = stage_d_recovery_median_iteration * 2;

    DiagnosticBaseline {
        mode,
        stage_d_recovery_median_iteration,
        derived_recovery_budget_2x_median,
        canonical_recovery_budget,
    }
}

fn export_rubric(rubric: &PassFailRubric) -> ExportRubric {
    ExportRubric {
        min_holdout_self_consistency: rubric.min_holdout_self_consistency,
        min_holdout_arbitration_confidence: rubric.min_holdout_arbitration_confidence,
        min_anchor_advantage: rubric.min_anchor_advantage,
        min_region_advantage: rubric.min_region_advantage,
        min_goal_advantage: rubric.min_goal_advantage,
        min_holdout_count: rubric.min_holdout_count,
        min_domain_count: rubric.min_domain_count,
        max_average_external_change_delta: rubric.max_average_external_change_delta,
        max_average_recovery_converged_iteration: rubric.max_average_recovery_converged_iteration,
        min_average_recovery_consistency_advantage: rubric.min_average_recovery_consistency_advantage,
    }
}

fn print_episode(metrics: &EpisodeMetrics) {
    println!(
        "{} [{}] mode={} domain={:?} kind={:?} conv={:?} anchors={} emergent={} continuity={} external={} regions={} stability={} momentum={} min_energy={} goals={} arb_conf={} self_consistency={} meta_revisions={}",
        metrics.id,
        metrics.label,
        metrics.mode.as_str(),
        metrics.domain,
        metrics.kind,
        metrics.converged_iteration,
        metrics.active_anchors,
        metrics.emergent_active,
        metrics.self_continuity_score,
        metrics.external_change_score,
        metrics.topology_regions,
        metrics.manifold_stability,
        metrics.momentum,
        metrics.minimum_energy,
        metrics.intent_goal_count,
        metrics.arbitration_confidence,
        metrics.self_consistency,
        metrics.meta_revision_count,
    );
    println!("  trace_hash={}... novelty={}", &metrics.final_trace_hash[..16], metrics.novelty_tag);
}

fn average_trained_recovery_iteration(holdouts: &[HoldoutPairResult], max_iterations: usize) -> usize {
    let values: Vec<usize> = holdouts
        .iter()
        .map(|r| r.trained_recovery.converged_iteration.unwrap_or(max_iterations + 1))
        .collect();
    average_usize(&values)
}

fn average_trained_recovery_consistency(holdouts: &[HoldoutPairResult]) -> i64 {
    let values: Vec<i64> = holdouts
        .iter()
        .map(|r| r.trained_recovery.self_consistency)
        .collect();
    average_i64(&values)
}

fn average_trained_recovery_continuity(holdouts: &[HoldoutPairResult]) -> i64 {
    let values: Vec<i64> = holdouts
        .iter()
        .map(|r| r.trained_recovery.self_continuity_score)
        .collect();
    average_i64(&values)
}

fn average_trained_recovery_regions(holdouts: &[HoldoutPairResult]) -> usize {
    let values: Vec<usize> = holdouts
        .iter()
        .map(|r| r.trained_recovery.topology_regions)
        .collect();
    average_usize(&values)
}

fn average_trained_recovery_anchors(holdouts: &[HoldoutPairResult]) -> usize {
    let values: Vec<usize> = holdouts
        .iter()
        .map(|r| r.trained_recovery.active_anchors)
        .collect();
    average_usize(&values)
}

fn build_mode_comparison(mode_runs: &[ModeRun], max_iterations: usize) -> ModeComparison {
    let full = mode_runs
        .iter()
        .find(|r| r.mode == RunMode::FullStack)
        .expect("full_stack run missing");
    let no_meta = mode_runs
        .iter()
        .find(|r| r.mode == RunMode::NoMeta)
        .expect("no_meta run missing");

    let full_avg_recovery = average_trained_recovery_iteration(&full.holdouts, max_iterations);
    let no_meta_avg_recovery = average_trained_recovery_iteration(&no_meta.holdouts, max_iterations);
    let full_avg_consistency = average_trained_recovery_consistency(&full.holdouts);
    let no_meta_avg_consistency = average_trained_recovery_consistency(&no_meta.holdouts);

    let interpretation = if no_meta_avg_recovery < full_avg_recovery {
        "no_meta recovers faster; meta-intent likely overcorrecting".to_string()
    } else if full_avg_recovery > 10 && no_meta_avg_recovery > 10 {
        "both modes recover slowly; drag likely upstream (anchors/topology/flow/energy)".to_string()
    } else {
        "full_stack recovery is not slower than no_meta under this battery".to_string()
    };

    ModeComparison {
        full_stack_avg_recovery_iteration: full_avg_recovery,
        no_meta_avg_recovery_iteration: no_meta_avg_recovery,
        full_stack_avg_recovery_self_consistency: full_avg_consistency,
        no_meta_avg_recovery_self_consistency: no_meta_avg_consistency,
        interpretation,
    }
}

fn export_results(
    mode_runs: &[ModeRun],
    rubric: &PassFailRubric,
    max_iterations: usize,
) -> Result<PathBuf, std::io::Error> {
    let export_dir = PathBuf::from("target/curriculum_harness");
    fs::create_dir_all(&export_dir)?;

    let json_path = export_dir.join("learning_metrics.json");
    let csv_path = export_dir.join("learning_metrics.csv");

    let bundle = ExportBundle {
        rubric: export_rubric(rubric),
        mode_runs: mode_runs.to_vec(),
        comparison: build_mode_comparison(mode_runs, max_iterations),
    };

    let json = serde_json::to_string_pretty(&bundle).expect("bundle should serialize");
    fs::write(&json_path, json)?;

    let mut csv = String::from(
        "split,mode,sequence_index,pair_index,domain,id,label,kind,novelty_tag,support_strength,wobble_strength,contradiction_strength,recovery_bias,converged_iteration,active_anchors,emergent_active,self_continuity_score,external_change_score,topology_regions,manifold_stability,momentum,minimum_energy,intent_goal_count,arbitration_confidence,self_consistency,meta_revision_count,diagnostic_stage_d_recovery_median_iteration,diagnostic_derived_recovery_budget_2x_median,canonical_recovery_budget,final_trace_hash\n",
    );

    fn csv_row(fields: &[String]) -> String {
        let mut row = fields.join(",");
        row.push('\n');
        row
    }

    let mut sequence_index = 0usize;
    for mode_run in mode_runs {
        sequence_index += 1;
        let baseline = &mode_run.diagnostic_baseline;
        csv.push_str(&csv_row(&[
            "diagnostic_baseline".to_string(),
            mode_run.mode.as_str().to_string(),
            sequence_index.to_string(),
            "0".to_string(),
            "NA".to_string(),
            format!("{}_stage_d_baseline", mode_run.mode.as_str()),
            "Stage D recovery median baseline".to_string(),
            "NA".to_string(),
            "diagnostic_baseline".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            baseline.stage_d_recovery_median_iteration.to_string(),
            baseline.derived_recovery_budget_2x_median.to_string(),
            baseline.canonical_recovery_budget.to_string(),
            "NA".to_string(),
        ]));

        for metrics in &mode_run.training {
            sequence_index += 1;
            let fields = vec![
                "training".to_string(),
                mode_run.mode.as_str().to_string(),
                sequence_index.to_string(),
                "0".to_string(),
                format!("{:?}", metrics.domain),
                metrics.id.clone(),
                metrics.label.replace(',', ";"),
                format!("{:?}", metrics.kind),
                metrics.novelty_tag.clone(),
                metrics.support_strength.to_string(),
                metrics.wobble_strength.to_string(),
                metrics.contradiction_strength.to_string(),
                metrics.recovery_bias.to_string(),
                metrics
                    .converged_iteration
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "NA".to_string()),
                metrics.active_anchors.to_string(),
                metrics.emergent_active.to_string(),
                metrics.self_continuity_score.to_string(),
                metrics.external_change_score.to_string(),
                metrics.topology_regions.to_string(),
                metrics.manifold_stability.to_string(),
                metrics.momentum.to_string(),
                metrics.minimum_energy.to_string(),
                metrics.intent_goal_count.to_string(),
                metrics.arbitration_confidence.to_string(),
                metrics.self_consistency.to_string(),
                metrics.meta_revision_count.to_string(),
                "NA".to_string(),
                "NA".to_string(),
                mode_run
                    .diagnostic_baseline
                    .canonical_recovery_budget
                    .to_string(),
                metrics.final_trace_hash.clone(),
            ];
            csv.push_str(&csv_row(&fields));
        }

        for (pair_index, pair) in mode_run.holdouts.iter().enumerate() {
            for (split, metrics) in [
                ("trained_holdout", &pair.trained_holdout),
                ("fresh_holdout", &pair.fresh_holdout),
                ("trained_recovery", &pair.trained_recovery),
                ("fresh_recovery", &pair.fresh_recovery),
            ] {
                sequence_index += 1;
                let fields = vec![
                    split.to_string(),
                    mode_run.mode.as_str().to_string(),
                    sequence_index.to_string(),
                    (pair_index + 1).to_string(),
                    format!("{:?}", metrics.domain),
                    metrics.id.clone(),
                    metrics.label.replace(',', ";"),
                    format!("{:?}", metrics.kind),
                    metrics.novelty_tag.clone(),
                    metrics.support_strength.to_string(),
                    metrics.wobble_strength.to_string(),
                    metrics.contradiction_strength.to_string(),
                    metrics.recovery_bias.to_string(),
                    metrics
                        .converged_iteration
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "NA".to_string()),
                    metrics.active_anchors.to_string(),
                    metrics.emergent_active.to_string(),
                    metrics.self_continuity_score.to_string(),
                    metrics.external_change_score.to_string(),
                    metrics.topology_regions.to_string(),
                    metrics.manifold_stability.to_string(),
                    metrics.momentum.to_string(),
                    metrics.minimum_energy.to_string(),
                    metrics.intent_goal_count.to_string(),
                    metrics.arbitration_confidence.to_string(),
                    metrics.self_consistency.to_string(),
                    metrics.meta_revision_count.to_string(),
                    "NA".to_string(),
                    "NA".to_string(),
                    mode_run
                        .diagnostic_baseline
                        .canonical_recovery_budget
                        .to_string(),
                    metrics.final_trace_hash.clone(),
                ];
                csv.push_str(&csv_row(&fields));
            }
        }
    }

    fs::write(&csv_path, csv)?;
    Ok(export_dir)
}

fn run_mode(
    mode: RunMode,
    episodes: &[EpisodeSpec],
    rubric: &PassFailRubric,
    config: &MultiFrameConfig,
    pass: TuningPass,
) -> ModeRun {
    println!("\n=== Running mode: {} | pass={} ===", mode.as_str(), pass.as_str());

    let mut learner = MultiFrameCognition::new();
    let mut training_results: Vec<EpisodeMetrics> = Vec::new();

    for spec in episodes.iter().filter(|e| e.kind != EpisodeKind::Holdout) {
        let metrics = run_episode(&mut learner, spec, mode, config);
        print_episode(&metrics);
        training_results.push(metrics);
    }

    let holdout_specs: Vec<&EpisodeSpec> = episodes
        .iter()
        .filter(|e| e.kind == EpisodeKind::Holdout)
        .collect();
    let mut holdout_results: Vec<HoldoutPairResult> = Vec::new();

    println!("\n--- Holdout battery ({}, pass={}) ---", mode.as_str(), pass.as_str());
    for holdout_spec in &holdout_specs {
        let trained_holdout = run_episode(&mut learner, holdout_spec, mode, config);
        let recovery_spec = recovery_spec_from_holdout(holdout_spec);
        let trained_recovery = run_episode(&mut learner, &recovery_spec, mode, config);

        let mut fresh = MultiFrameCognition::new();
        let fresh_holdout = run_episode(&mut fresh, holdout_spec, mode, config);
        let fresh_recovery = run_episode(&mut fresh, &recovery_spec, mode, config);

        println!("trained holdout:");
        print_episode(&trained_holdout);
        println!("fresh holdout:");
        print_episode(&fresh_holdout);
        println!("trained recovery:");
        print_episode(&trained_recovery);
        println!("fresh recovery:");
        print_episode(&fresh_recovery);
        println!();

        holdout_results.push(HoldoutPairResult {
            holdout_id: holdout_spec.id.to_string(),
            domain: holdout_spec.domain,
            trained_holdout,
            fresh_holdout,
            trained_recovery,
            fresh_recovery,
        });
    }

    let first_training = training_results.first().expect("at least one training episode");
    let verification = verify_learning(first_training, &holdout_results, rubric, config.iterations);
    let diagnostic_baseline = derive_recovery_baseline(
        mode,
        &training_results,
        rubric.max_average_recovery_converged_iteration,
        config.iterations,
    );

    println!("\n--- Verification ({}) ---", mode.as_str());
    for check in &verification.checks {
        println!(
            "{} {} :: {}",
            if check.passed { "PASS" } else { "FAIL" },
            check.name,
            check.detail
        );
    }

    println!(
        "diagnostic baseline ({}) => stage_d_median={} derived_2x={} canonical_budget={}",
        mode.as_str(),
        diagnostic_baseline.stage_d_recovery_median_iteration,
        diagnostic_baseline.derived_recovery_budget_2x_median,
        diagnostic_baseline.canonical_recovery_budget,
    );

    ModeRun {
        mode,
        diagnostic_baseline,
        training: training_results,
        holdouts: holdout_results,
        verification,
    }
}

fn non_speed_checks_pass(verification: &VerificationOutcome) -> bool {
    verification
        .checks
        .iter()
        .filter(|c| c.name != "trained learner recovers within speed budget")
        .all(|c| c.passed)
}

fn full_stack_recovery_average(mode_runs: &[ModeRun], max_iterations: usize) -> usize {
    let full = mode_runs
        .iter()
        .find(|r| r.mode == RunMode::FullStack)
        .expect("full_stack run missing");
    average_trained_recovery_iteration(&full.holdouts, max_iterations)
}

fn full_stack_quality_snapshot(
    mode_runs: &[ModeRun],
    max_iterations: usize,
) -> FullStackQualitySnapshot {
    let full = mode_runs
        .iter()
        .find(|r| r.mode == RunMode::FullStack)
        .expect("full_stack run missing");

    FullStackQualitySnapshot {
        recovery_iteration: average_trained_recovery_iteration(&full.holdouts, max_iterations),
        recovery_continuity: average_trained_recovery_continuity(&full.holdouts),
        recovery_regions: average_trained_recovery_regions(&full.holdouts),
        recovery_anchors: average_trained_recovery_anchors(&full.holdouts),
    }
}

fn run_mode_pair(
    episodes: &[EpisodeSpec],
    rubric: &PassFailRubric,
    config: &MultiFrameConfig,
    pass: TuningPass,
) -> Vec<ModeRun> {
    let full_stack = run_mode(RunMode::FullStack, episodes, rubric, config, pass);
    let no_meta = run_mode(RunMode::NoMeta, episodes, rubric, config, pass);
    vec![full_stack, no_meta]
}

fn export_phase6_tuning(sequence: &Phase6TuningSequence) -> Result<PathBuf, std::io::Error> {
    let export_dir = PathBuf::from("target/curriculum_harness");
    fs::create_dir_all(&export_dir)?;

    let json_path = export_dir.join("phase6_tuning_summary.json");
    let csv_path = export_dir.join("phase6_tuning_summary.csv");

    let json = serde_json::to_string_pretty(sequence).expect("phase6 summary should serialize");
    fs::write(&json_path, json)?;

    let mut csv = String::from(
        "pass,mode,avg_trained_recovery_iteration,avg_trained_recovery_self_consistency,gate_passed,promoted,required_max_recovery_iteration,observed_full_stack_recovery_iteration,non_speed_checks_passed,anti_shortcut_quality_passed,required_min_recovery_continuity,observed_full_stack_recovery_continuity,required_min_recovery_regions,observed_full_stack_recovery_regions,required_min_recovery_anchors,observed_full_stack_recovery_anchors\n",
    );

    for experiment in &sequence.experiments {
        for mode in [RunMode::FullStack, RunMode::NoMeta] {
            let (avg_recovery, avg_consistency) = match mode {
                RunMode::FullStack => (
                    experiment.comparison.full_stack_avg_recovery_iteration,
                    experiment.comparison.full_stack_avg_recovery_self_consistency,
                ),
                RunMode::NoMeta => (
                    experiment.comparison.no_meta_avg_recovery_iteration,
                    experiment.comparison.no_meta_avg_recovery_self_consistency,
                ),
            };

            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                experiment.pass.as_str(),
                mode.as_str(),
                avg_recovery,
                avg_consistency,
                experiment.gate.passed,
                experiment.promoted,
                experiment.gate.required_max_recovery_iteration,
                experiment.gate.observed_full_stack_recovery_iteration,
                experiment.gate.non_speed_checks_passed,
                experiment.gate.anti_shortcut_quality_passed,
                experiment.gate.required_min_recovery_continuity,
                experiment.gate.observed_full_stack_recovery_continuity,
                experiment.gate.required_min_recovery_regions,
                experiment.gate.observed_full_stack_recovery_regions,
                experiment.gate.required_min_recovery_anchors,
                experiment.gate.observed_full_stack_recovery_anchors,
            ));
        }
    }

    fs::write(&csv_path, csv)?;
    Ok(export_dir)
}

fn run_phase6_tuning_sequence(
    episodes: &[EpisodeSpec],
    rubric: &PassFailRubric,
    canonical_mode_runs: &[ModeRun],
    canonical_iterations: usize,
) -> Phase6TuningSequence {
    let canonical_quality = full_stack_quality_snapshot(canonical_mode_runs, canonical_iterations);
    let canonical_recovery = canonical_quality.recovery_iteration;
    let mut promoted_chain = true;
    let mut previous_best = canonical_recovery;
    let mut experiments: Vec<MicroExperimentResult> = Vec::new();

    for pass in [
        TuningPass::ConvergenceGate,
        TuningPass::FlowEnergyDescent,
        TuningPass::AnchorStabilization,
    ] {
        let config = cfg_for_pass(pass);
        let mode_runs = run_mode_pair(episodes, rubric, &config, pass);
        let comparison = build_mode_comparison(&mode_runs, config.iterations);
        let observed_quality = full_stack_quality_snapshot(&mode_runs, config.iterations);

        let observed = observed_quality.recovery_iteration;
        let required_max = if previous_best > rubric.max_average_recovery_converged_iteration {
            previous_best.saturating_sub(1)
        } else {
            previous_best
        };

        let mut required_min_continuity = canonical_quality.recovery_continuity.saturating_sub(3);
        let mut required_min_regions = canonical_quality.recovery_regions.saturating_sub(1);
        let mut required_min_anchors = canonical_quality.recovery_anchors.saturating_sub(1);

        // Guard against trivial "fast closure" wins that degrade structural recovery quality.
        if observed <= 2 {
            required_min_continuity = canonical_quality.recovery_continuity.saturating_sub(1);
            required_min_regions = canonical_quality.recovery_regions;
            required_min_anchors = canonical_quality.recovery_anchors;
        }

        let non_speed_ok = mode_runs
            .iter()
            .all(|m| non_speed_checks_pass(&m.verification));

        let anti_shortcut_quality_ok = observed_quality.recovery_continuity >= required_min_continuity
            && observed_quality.recovery_regions >= required_min_regions
            && observed_quality.recovery_anchors >= required_min_anchors;

        let gate_passed = observed <= required_max && non_speed_ok && anti_shortcut_quality_ok;
        let promoted = promoted_chain && gate_passed;
        promoted_chain = promoted;
        if promoted {
            previous_best = observed;
        }

        let gate = MicroExperimentGate {
            name: format!("{} gate", pass.as_str()),
            passed: gate_passed,
            required_max_recovery_iteration: required_max,
            observed_full_stack_recovery_iteration: observed,
            non_speed_checks_passed: non_speed_ok,
            anti_shortcut_quality_passed: anti_shortcut_quality_ok,
            required_min_recovery_continuity: required_min_continuity,
            observed_full_stack_recovery_continuity: observed_quality.recovery_continuity,
            required_min_recovery_regions: required_min_regions,
            observed_full_stack_recovery_regions: observed_quality.recovery_regions,
            required_min_recovery_anchors: required_min_anchors,
            observed_full_stack_recovery_anchors: observed_quality.recovery_anchors,
            detail: format!(
                "observed={} required<= {} non_speed_checks_passed={} anti_shortcut_quality_passed={} continuity={}/{} regions={}/{} anchors={}/{}",
                observed,
                required_max,
                non_speed_ok,
                anti_shortcut_quality_ok,
                observed_quality.recovery_continuity,
                required_min_continuity,
                observed_quality.recovery_regions,
                required_min_regions,
                observed_quality.recovery_anchors,
                required_min_anchors,
            ),
        };

        experiments.push(MicroExperimentResult {
            pass,
            mode_runs,
            comparison,
            gate,
            promoted,
        });
    }

    Phase6TuningSequence {
        canonical_full_stack_recovery_iteration: canonical_recovery,
        all_gates_passed: experiments.iter().all(|e| e.gate.passed && e.promoted),
        experiments,
    }
}

fn main() {
    println!("=== RUGC Curriculum Harness: Teach + Verify Learning ===");
    println!();
    println!("Episode schema:");
    println!("  id, label, domain, kind, support_strength, wobble_strength, contradiction_strength, recovery_bias, novelty_tag");
    println!("Metrics schema:");
    println!("  mode, converged_iteration, active_anchors, emergent_active, self_continuity_score, external_change_score,");
    println!("  topology_regions, manifold_stability, momentum, minimum_energy, intent_goal_count,");
    println!("  arbitration_confidence, self_consistency, meta_revision_count, final_trace_hash");
    println!("Rubric:");
    println!("  compare trained vs fresh across a harder multi-domain holdout battery, plus post-holdout recovery quality");
    println!();

    let episodes = curriculum();
    let rubric = PassFailRubric {
        min_holdout_self_consistency: 600,
        min_holdout_arbitration_confidence: 700,
        min_anchor_advantage: 1,
        min_region_advantage: 1,
        min_goal_advantage: 1,
        min_holdout_count: 5,
        min_domain_count: 2,
        max_average_external_change_delta: 15,
        max_average_recovery_converged_iteration: 10,
        min_average_recovery_consistency_advantage: 0,
    };

    let canonical_config = cfg_for_pass(TuningPass::Canonical);
    let full_stack = run_mode(
        RunMode::FullStack,
        &episodes,
        &rubric,
        &canonical_config,
        TuningPass::Canonical,
    );
    let no_meta = run_mode(
        RunMode::NoMeta,
        &episodes,
        &rubric,
        &canonical_config,
        TuningPass::Canonical,
    );
    let mode_runs = vec![full_stack, no_meta];

    let comparison = build_mode_comparison(&mode_runs, canonical_config.iterations);
    println!("\n--- Mode Comparison ---");
    println!(
        "full_stack avg trained_recovery conv={} self_consistency={}",
        comparison.full_stack_avg_recovery_iteration,
        comparison.full_stack_avg_recovery_self_consistency,
    );
    println!(
        "no_meta avg trained_recovery conv={} self_consistency={}",
        comparison.no_meta_avg_recovery_iteration,
        comparison.no_meta_avg_recovery_self_consistency,
    );
    println!("interpretation: {}", comparison.interpretation);

    let export_dir = export_results(&mode_runs, &rubric, canonical_config.iterations)
        .expect("should export JSON/CSV results");
    println!("\nExports written to {}", export_dir.display());

    let sequence = run_phase6_tuning_sequence(
        &episodes,
        &rubric,
        &mode_runs,
        canonical_config.iterations,
    );

    println!("\n--- Phase 6 Tuning Sequence ---");
    println!(
        "baseline full_stack recovery iteration={}",
        sequence.canonical_full_stack_recovery_iteration
    );
    for experiment in &sequence.experiments {
        println!(
            "pass={} gate={} promoted={} detail={}",
            experiment.pass.as_str(),
            if experiment.gate.passed { "PASS" } else { "FAIL" },
            experiment.promoted,
            experiment.gate.detail,
        );
        println!(
            "  full_stack avg recovery={} | no_meta avg recovery={}",
            experiment.comparison.full_stack_avg_recovery_iteration,
            experiment.comparison.no_meta_avg_recovery_iteration,
        );
    }

    let phase6_export_dir =
        export_phase6_tuning(&sequence).expect("should export phase6 tuning summary");
    println!("Phase 6 tuning summary written to {}", phase6_export_dir.display());
    println!();

    let all_passed = mode_runs.iter().all(|m| m.verification.passed) && sequence.all_gates_passed;
    if all_passed {
        println!("LEARNING_VERIFIED");
    } else {
        println!("LEARNING_NOT_VERIFIED");
        std::process::exit(1);
    }
}
