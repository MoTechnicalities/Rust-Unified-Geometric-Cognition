use rugc::{
    compute_cognitive_flow_field, compute_cognitive_potential_field,
    compute_energy_minimizing_trajectory, select_action, DeterminismVerifier,
    MultiFrameCognition, MultiFrameConfig, SemanticConstraint,
};
use std::collections::BTreeMap;

fn cfg(workers: usize) -> MultiFrameConfig {
    MultiFrameConfig {
        iterations: 12,
        worker_count: workers,
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

fn build(external: bool) -> MultiFrameCognition {
    let mut mfc = MultiFrameCognition::new();
    let mut physics = vec![
        SemanticConstraint::assertion("light", "wave", true, 92),
        SemanticConstraint::assertion("light", "particle", true, 88),
        SemanticConstraint::assertion("vacuum", "has_medium", false, 74),
        SemanticConstraint::assertion("photon", "is_quantized", true, 64),
    ];
    if external {
        physics.push(SemanticConstraint::assertion("observer", "injects_noise", true, 70));
    }
    mfc.register_frame("physics", physics);
    mfc.register_frame(
        "ontology",
        vec![
            SemanticConstraint::assertion("light", "wave", false, 30),
            SemanticConstraint::assertion("light", "particle", true, 65),
            SemanticConstraint::assertion("vacuum", "has_medium", true, 20),
            SemanticConstraint::assertion("photon", "is_quantized", true, 58),
        ],
    );
    mfc.register_frame(
        "observation",
        vec![
            SemanticConstraint::assertion("light", "wave", true, 50),
            SemanticConstraint::assertion("light", "particle", true, 52),
            SemanticConstraint::assertion("vacuum", "has_medium", false, 40),
            SemanticConstraint::assertion("photon", "is_quantized", true, 60),
        ],
    );
    mfc
}

fn run_topo(external: bool, workers: usize) -> (rugc::CognitiveTopology, Vec<String>) {
    let report = build(external).run(cfg(workers)).expect("run should succeed");
    let topo = rugc::compute_cognitive_topology(&report.consolidated_memory, 500)
        .expect("topology should compute");
    let anchor_ids = report.consolidated_memory.anchor_basis_ids.clone();
    (topo, anchor_ids)
}

#[test]
fn gate_ae_energy_wells_form_at_attractor_regions() {
    let (stable_topo, anchor_ids) = run_topo(false, 4);
    let flow = compute_cognitive_flow_field(&[stable_topo.clone(), stable_topo], &anchor_ids)
        .expect("flow should compute");
    let potential = compute_cognitive_potential_field(&flow)
        .expect("potential field should compute");

    assert!(!potential.stability_energies.is_empty());
    assert!(!potential.global_minimum_region.is_empty());
    assert!(potential.global_minimum_energy <= 200);

    let well_count = potential
        .stability_energies
        .iter()
        .filter(|e| e.well_depth > 0)
        .count();
    assert!(well_count >= 1, "expected at least one positive-depth well");
}

#[test]
fn gate_af_gradient_descent_follows_flow_field_predictions() {
    let (stable_topo, anchor_ids) = run_topo(false, 4);
    let (perturbed_topo, _) = run_topo(true, 4);

    let flow = compute_cognitive_flow_field(&[stable_topo, perturbed_topo], &anchor_ids)
        .expect("flow should compute");
    let potential = compute_cognitive_potential_field(&flow)
        .expect("potential field should compute");

    assert!(!potential.gradients.is_empty());

    let potential_by_region: BTreeMap<String, i64> = potential
        .stability_energies
        .iter()
        .map(|e| (e.region_id.clone(), e.potential))
        .collect();

    for g in potential.gradients.iter().filter(|g| g.gradient > 0) {
        let src = potential_by_region
            .get(&g.source_region)
            .expect("source potential must exist");
        let dst = potential_by_region
            .get(&g.target_region)
            .expect("target potential must exist");
        assert!(src > dst, "positive gradient must move downhill in energy");
    }
}

#[test]
fn gate_ag_action_selection_minimizes_energy() {
    let (stable_topo, anchor_ids) = run_topo(false, 4);
    let (perturbed_topo, _) = run_topo(true, 4);

    let flow = compute_cognitive_flow_field(&[stable_topo, perturbed_topo], &anchor_ids)
        .expect("flow should compute");
    let potential = compute_cognitive_potential_field(&flow)
        .expect("potential field should compute");

    let current_region = potential
        .stability_energies
        .iter()
        .max_by_key(|e| e.potential)
        .map(|e| e.region_id.clone())
        .expect("must have at least one region");

    let action = select_action(&potential, &current_region)
        .expect("action selection should compute");

    assert!(!action.preferred_trajectory.is_empty());

    let potential_by_region: BTreeMap<String, i64> = potential
        .stability_energies
        .iter()
        .map(|e| (e.region_id.clone(), e.potential))
        .collect();

    if action.preferred_trajectory[0] != current_region {
        let curr = potential_by_region[&current_region];
        let next = potential_by_region[&action.preferred_trajectory[0]];
        assert!(next <= curr, "selected action should not increase energy");
        assert!(action.stability_gain > 0);
    } else {
        assert_eq!(action.energy_cost, 0);
        assert_eq!(action.stability_gain, 0);
    }
}

#[test]
fn gate_ah_external_perturbation_creates_energy_spike_then_recovers() {
    let (stable_topo, anchor_ids) = run_topo(false, 4);
    let (perturbed_topo, _) = run_topo(true, 4);
    let (stable_recovery_topo, _) = run_topo(false, 4);

    let flow_baseline = compute_cognitive_flow_field(
        &[stable_topo.clone(), stable_topo.clone()],
        &anchor_ids,
    )
    .expect("baseline flow should compute");
    let flow_perturbed = compute_cognitive_flow_field(
        &[stable_topo.clone(), perturbed_topo.clone()],
        &anchor_ids,
    )
    .expect("perturbed flow should compute");
    let flow_recovery = compute_cognitive_flow_field(
        &[stable_recovery_topo.clone(), stable_recovery_topo],
        &anchor_ids,
    )
    .expect("recovery flow should compute");

    let baseline = compute_cognitive_potential_field(&flow_baseline)
        .expect("baseline potential should compute");
    let perturbed = compute_cognitive_potential_field(&flow_perturbed)
        .expect("perturbed potential should compute");
    let recovery = compute_cognitive_potential_field(&flow_recovery)
        .expect("recovery potential should compute");

    let baseline_total: i64 = baseline
        .stability_energies
        .iter()
        .map(|e| e.potential)
        .sum();
    let perturbed_total: i64 = perturbed
        .stability_energies
        .iter()
        .map(|e| e.potential)
        .sum();
    let recovery_total: i64 = recovery
        .stability_energies
        .iter()
        .map(|e| e.potential)
        .sum();

    assert!(
        perturbed_total > baseline_total,
        "expected perturbation spike: baseline_total={baseline_total}, perturbed_total={perturbed_total}"
    );
    assert!(
        recovery_total <= perturbed_total,
        "expected recovery after perturbation: perturbed_total={perturbed_total}, recovery_total={recovery_total}"
    );
}

#[test]
fn gate_ai_trajectory_canonical_hash_worker_invariant() {
    let verifier = DeterminismVerifier::new();
    let (t1, a1) = run_topo(false, 1);
    let (t8, a8) = run_topo(false, 8);

    let flow1 = compute_cognitive_flow_field(&[t1.clone(), t1.clone()], &a1)
        .expect("flow1 should compute");
    let flow8 = compute_cognitive_flow_field(&[t8.clone(), t8.clone()], &a8)
        .expect("flow8 should compute");

    assert!(verifier
        .is_replay_stable(&flow1, &flow8)
        .unwrap_or(false));

    let tr1 = compute_energy_minimizing_trajectory(&[t1], &flow1, &a1)
        .expect("trajectory 1 should compute");
    let tr8 = compute_energy_minimizing_trajectory(&[t8], &flow8, &a8)
        .expect("trajectory 8 should compute");

    assert_eq!(tr1.canonical_hash, tr8.canonical_hash);
}
