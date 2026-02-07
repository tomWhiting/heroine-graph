// Community-Modulated Repulsion with Degree-Weighted Mass & Gravity
//
// Based on ForceAtlas2 research (Jacomy et al., 2014):
// - Degree-weighted mass: F = kr * (deg_i+1) * (deg_j+1) / d^2
// - Community modulation: intra-community = reduced, inter = amplified
// - Distance-independent gravity: F = kg * (deg+1) toward origin
//
// Degree-weighting is key: high-degree nodes (community hubs) repel
// more strongly, which naturally separates dense subgraphs (communities).
// Distance-independent gravity prevents drift without compression.

struct RepulsionUniforms {
    node_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    intra_factor: f32,     // Same-community repulsion multiplier (e.g. 0.5)
    inter_factor: f32,     // Different-community repulsion multiplier (e.g. 1.5)
    gravity_strength: f32, // Distance-independent gravity toward origin
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: RepulsionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> community_ids: array<u32>;
@group(0) @binding(4) var<storage, read> degrees: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = positions[node_idx];
    let my_comm = community_ids[node_idx];
    let my_degree = degrees[node_idx];
    // sqrt(degree+1) dampens mass growth so high-degree parents don't explode
    // outward. A node with degree 50 gets mass ~7 instead of 51.
    let mass_i = sqrt(f32(my_degree + 1u));
    var force = vec2<f32>(0.0, 0.0);

    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = positions[i];
        let delta = node_pos - other_pos;

        let dist_sq = dot(delta, delta);
        let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
        let safe_dist_sq = max(dist_sq, min_dist_sq);
        let dist = sqrt(safe_dist_sq);

        // Coulomb 1/d^2 repulsion (stronger short-range than FA2's 1/d)
        // F = kr * mass_i * mass_j / d^2, mass = sqrt(degree + 1)
        let other_degree = degrees[i];
        let mass_j = sqrt(f32(other_degree + 1u));
        var force_magnitude = uniforms.repulsion_strength * mass_i * mass_j / safe_dist_sq;

        // Modulate by community membership
        let other_comm = community_ids[i];
        if (my_comm == other_comm) {
            force_magnitude *= uniforms.intra_factor;
        } else {
            force_magnitude *= uniforms.inter_factor;
        }

        force += delta * (force_magnitude / dist);
    }

    // FA2-style distance-independent gravity: F = kg * mass * direction_to_center
    // Constant magnitude regardless of distance — prevents drift without compression.
    // Unlike Hooke's law gravity (F proportional to distance), this doesn't crush
    // nodes near the center or create differential compression.
    let gravity_dist = length(node_pos);
    if (gravity_dist > uniforms.min_distance) {
        let gravity_dir = -node_pos / gravity_dist;
        force += gravity_dir * uniforms.gravity_strength * mass_i;
    }

    forces[node_idx] += force;
}
