use rugc::{anchor_derived_relational_distance, MultiFrameCognition, MultiFrameConfig, SemanticConstraint};

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

fn main() {
    println!("=== RUGC Phase 5.0 Anchor-Derived Relational Distance Demo ===\n");

    let mut baseline_a = build(false);
    let mut baseline_b = build(false);
    let mut external = build(true);

    let a = baseline_a.run(cfg(4)).expect("baseline A should run");
    let b = baseline_b.run(cfg(4)).expect("baseline B should run");
    let x = external.run(cfg(4)).expect("external should run");

    let near = anchor_derived_relational_distance(&a.consolidated_memory, &b.consolidated_memory);
    let far = anchor_derived_relational_distance(&a.consolidated_memory, &x.consolidated_memory);

    println!("near(score) baseline->baseline: {}", near.score);
    println!("far(score) baseline->external:  {}", far.score);
    println!();
    println!("near components:");
    println!("  anchor_jaccard_distance={}", near.anchor_jaccard_distance);
    println!("  emergent_jaccard_distance={}", near.emergent_jaccard_distance);
    println!("  continuity_delta={}", near.continuity_delta);
    println!("  external_delta={}", near.external_delta);
    println!("  ontology_delta={}", near.ontology_delta);
    println!();
    println!("far components:");
    println!("  anchor_jaccard_distance={}", far.anchor_jaccard_distance);
    println!("  emergent_jaccard_distance={}", far.emergent_jaccard_distance);
    println!("  continuity_delta={}", far.continuity_delta);
    println!("  external_delta={}", far.external_delta);
    println!("  ontology_delta={}", far.ontology_delta);
}
