use rugc::{
    anchor_derived_relational_distance, DeterminismVerifier, MultiFrameCognition, MultiFrameConfig,
    SemanticConstraint,
};

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

#[test]
fn anchor_distance_orders_near_and_far() {
    let mut baseline_a = build(false);
    let mut baseline_b = build(false);
    let mut perturbed = build(true);

    let a = baseline_a.run(cfg(4)).expect("baseline A should run");
    let b = baseline_b.run(cfg(4)).expect("baseline B should run");
    let p = perturbed.run(cfg(4)).expect("perturbed should run");

    let near = anchor_derived_relational_distance(&a.consolidated_memory, &b.consolidated_memory);
    let far = anchor_derived_relational_distance(&a.consolidated_memory, &p.consolidated_memory);

    assert_eq!(near.score, 0);
    assert!(far.score > near.score);
    assert!(far.external_delta > 0 || far.anchor_jaccard_distance > 0);
}

#[test]
fn anchor_distance_is_worker_invariant() {
    let verifier = DeterminismVerifier::new();
    let mut a = build(false);
    let mut b = build(false);

    let r1 = a.run(cfg(1)).expect("worker=1 run should succeed");
    let r8 = b.run(cfg(8)).expect("worker=8 run should succeed");

    let d = anchor_derived_relational_distance(&r1.consolidated_memory, &r8.consolidated_memory);
    assert_eq!(d.score, 0);
    assert!(
        verifier
            .is_replay_stable(&r1.consolidated_memory, &r8.consolidated_memory)
            .unwrap_or(false)
    );
}

#[test]
fn anchor_distance_detects_external_change() {
    let mut baseline = build(false);
    let mut perturbed = build(true);

    let b = baseline.run(cfg(4)).expect("baseline should run");
    let p = perturbed.run(cfg(4)).expect("perturbed should run");

    let d = anchor_derived_relational_distance(&b.consolidated_memory, &p.consolidated_memory);
    assert!(d.score > 0);
    assert!(
        d.external_delta > 0 || d.emergent_jaccard_distance > 0 || d.anchor_jaccard_distance > 0
    );
}
