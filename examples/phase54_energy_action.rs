use rugc::{
    compute_cognitive_flow_field, compute_cognitive_potential_field,
    compute_cognitive_topology, compute_energy_minimizing_trajectory, select_action,
    MultiFrameCognition, MultiFrameConfig, SemanticConstraint,
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

fn run_topo(external: bool) -> (rugc::CognitiveTopology, Vec<String>) {
    let report = build(external).run(cfg(4)).expect("run should succeed");
    let topo = compute_cognitive_topology(&report.consolidated_memory, 500)
        .expect("topology should compute");
    let anchor_ids = report.consolidated_memory.anchor_basis_ids.clone();
    (topo, anchor_ids)
}

fn print_step(label: &str, snapshots: &[rugc::CognitiveTopology], anchor_ids: &[String]) {
    let flow = compute_cognitive_flow_field(snapshots, anchor_ids)
        .expect("flow should compute");
    let potential = compute_cognitive_potential_field(&flow)
        .expect("potential should compute");

    println!("{label}");
    println!(
        "  Global minimum: {} @ {}",
        potential.global_minimum_region, potential.global_minimum_energy
    );
    println!("  Wells: {}", potential.stability_energies.len());

    for e in &potential.stability_energies {
        println!(
            "    {} -> potential={}, well_depth={}, attraction={}",
            e.region_id, e.potential, e.well_depth, e.attraction_strength
        );
    }

    if let Some(current) = potential
        .stability_energies
        .iter()
        .max_by_key(|e| e.potential)
        .map(|e| e.region_id.clone())
    {
        let action = select_action(&potential, &current)
            .expect("action should compute");
        println!("  Action from {}:", current);
        println!("    trajectory={:?}", action.preferred_trajectory);
        println!(
            "    energy_cost={}, stability_gain={}, confidence={}",
            action.energy_cost, action.stability_gain, action.confidence
        );
    }

    let traj = compute_energy_minimizing_trajectory(snapshots, &flow, anchor_ids)
        .expect("trajectory should compute");
    println!(
        "  Trajectory: convergent={}, total_cost={}, hash={}...",
        traj.convergent_outcome,
        traj.total_energy_cost,
        &traj.canonical_hash[..16]
    );
    println!();
}

fn main() {
    let (stable_topo, anchor_ids) = run_topo(false);
    let (perturbed_topo, _) = run_topo(true);
    let (stable_topo_recovery, _) = run_topo(false);

    println!("=== Phase 5.4: Cognitive Energy & Action Selection ===");
    println!();

    print_step(
        "Iteration 1 (stable):",
        &[stable_topo.clone(), stable_topo.clone()],
        &anchor_ids,
    );
    print_step(
        "Iteration 2 (stable replay):",
        &[stable_topo.clone(), stable_topo.clone()],
        &anchor_ids,
    );
    print_step(
        "Iteration 3 (perturbed):",
        &[stable_topo.clone(), perturbed_topo.clone()],
        &anchor_ids,
    );
    print_step(
        "Iteration 4 (recovery):",
        &[stable_topo, perturbed_topo, stable_topo_recovery],
        &anchor_ids,
    );
}
