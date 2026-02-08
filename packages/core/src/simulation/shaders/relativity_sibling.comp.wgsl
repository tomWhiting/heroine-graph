// Relativity Atlas: Sibling + Cousin Repulsion Shader
// Computes repulsion forces between siblings (nodes sharing a parent)
// and optionally between cousins (nodes sharing a grandparent).
//
// This is the key innovation of Relativity Atlas:
// - O(N + E) instead of O(N^2) for repulsion
// - Natural hierarchical structure preservation
// - Siblings spread evenly around their shared parent
// - Cousin repulsion prevents subtree overlap
// - Phantom zones create mass-proportional collision boundaries
//
// Uses vec2<f32> layout for consolidated position/force data.

struct SiblingUniforms {
    node_count: u32,
    edge_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    max_siblings: u32,              // Cap on siblings to check (perf limit for high-degree nodes)
    parent_child_multiplier: f32,   // Weaker repulsion for connected pairs (default: 0.15)
    // -- Cousin repulsion --
    cousin_enabled: u32,            // 0 = off, 1 = on
    cousin_strength: f32,           // Multiplier on cousin repulsion (0..1)
    // -- Phantom zones --
    phantom_enabled: u32,           // 0 = off, 1 = on
    phantom_multiplier: f32,        // How much mass affects zone radius
    // -- Orbital layout --
    orbit_strength: f32,            // Radial spring pulling children to target orbit radius
    tangential_multiplier: f32,     // Amplify tangential repulsion (>1 = more angular spreading)
    orbit_radius_base: f32,         // Base orbit distance from parent
    bubble_mode: u32,               // 0 = off, 1 = use wellRadius for phantom zones + orbit
    orbit_scale: f32,               // Bubble mode: orbit radius = parent_wellRadius * scale
}

@group(0) @binding(0) var<uniform> uniforms: SiblingUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Inverse CSR: incoming edges (parents)
// For node i, csr_inverse_offsets[i]..csr_inverse_offsets[i+1] are parent indices
@group(0) @binding(3) var<storage, read> csr_inverse_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> csr_inverse_sources: array<u32>;

// Forward CSR: outgoing edges (children)
// For parent p, csr_offsets[p]..csr_offsets[p+1] are child indices (siblings of each other)
@group(0) @binding(5) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> csr_targets: array<u32>;

// Node masses (for mass-weighted repulsion)
@group(0) @binding(7) var<storage, read> node_mass: array<f32>;

// Well radii (bubble mode: subtree-based collision boundaries)
@group(0) @binding(8) var<storage, read> well_radius: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;
// Cap on cousin iterations to prevent runaway loops in wide hierarchies
const MAX_COUSIN_ITERATIONS: u32 = 64u;

// Compute repulsive force between two nodes.
// Linear (1/r) repulsion like ForceAtlas2 — maintains force at medium distance
// so siblings spread evenly rather than bunching up.
fn compute_repulsion(delta: vec2<f32>, mass_i: f32, mass_j: f32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);
    let dist = sqrt(safe_dist_sq);

    let force_magnitude = uniforms.repulsion_strength * mass_i * mass_j / dist;

    return (delta / dist) * force_magnitude;
}

// Phantom zone repulsion: extra force when collision zones overlap.
// Normal mode: zone radius = phantom_multiplier * sqrt(mass).
// Bubble mode: zone radius = wellRadius (computed from subtree).
// When zones overlap, apply a soft push proportional to the overlap depth.
fn compute_phantom_force(delta: vec2<f32>, dist: f32, mass_i: f32, mass_j: f32, idx_i: u32, idx_j: u32) -> vec2<f32> {
    var zone_i: f32;
    var zone_j: f32;
    if (uniforms.bubble_mode != 0u) {
        zone_i = well_radius[idx_i];
        zone_j = well_radius[idx_j];
    } else {
        zone_i = uniforms.phantom_multiplier * sqrt(mass_i);
        zone_j = uniforms.phantom_multiplier * sqrt(mass_j);
    }
    let combined_radius = zone_i + zone_j;

    // No overlap — no phantom force
    if (dist >= combined_radius) {
        return vec2<f32>(0.0, 0.0);
    }

    // Overlap depth (0 at boundary, combined_radius at zero distance)
    let overlap = combined_radius - dist;
    // Normalized overlap (0..1)
    let overlap_ratio = overlap / combined_radius;

    // Soft push: quadratic ramp for smooth force onset
    // Force scales with repulsion_strength so it's tunable from the same slider
    let force_magnitude = uniforms.repulsion_strength * mass_i * mass_j * overlap_ratio * overlap_ratio;

    let dir = delta / max(dist, uniforms.min_distance);
    return dir * force_magnitude;
}

// Apply repulsion between this node and another, including phantom zone check.
fn apply_repulsion(pos: vec2<f32>, other_pos: vec2<f32>, mass_i: f32, mass_j: f32, strength_mult: f32, idx_i: u32, idx_j: u32) -> vec2<f32> {
    let delta = pos - other_pos;
    let dist_sq = dot(delta, delta);

    if (dist_sq < EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    var f = compute_repulsion(delta, mass_i, mass_j) * strength_mult;

    // Phantom zone overlay — extra push when collision zones overlap
    if (uniforms.phantom_enabled != 0u) {
        let dist = sqrt(dist_sq);
        f += compute_phantom_force(delta, dist, mass_i, mass_j, idx_i, idx_j) * strength_mult;
    }

    return f;
}

// Compute tangential-amplified sibling repulsion.
// Decomposes force into radial (toward/away from parent) and tangential
// (around parent orbit) components, amplifying the tangential part.
// This makes siblings spread angularly around their parent instead of
// just pushing away linearly.
fn apply_tangential_repulsion(
    pos: vec2<f32>, sib_pos: vec2<f32>,
    parent_pos: vec2<f32>, parent_dist: f32,
    mass_i: f32, mass_j: f32, strength_mult: f32,
    idx_i: u32, idx_j: u32
) -> vec2<f32> {
    let delta = pos - sib_pos;
    let dist_sq = dot(delta, delta);

    if (dist_sq < EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }

    let base_force = compute_repulsion(delta, mass_i, mass_j) * strength_mult;

    // Phantom zone overlay
    var phantom_force = vec2<f32>(0.0, 0.0);
    if (uniforms.phantom_enabled != 0u) {
        let dist = sqrt(dist_sq);
        phantom_force = compute_phantom_force(delta, dist, mass_i, mass_j, idx_i, idx_j) * strength_mult;
    }

    // If tangential amplification is active and we have a valid parent direction
    if (uniforms.tangential_multiplier > 1.0 && parent_dist > EPSILON) {
        let radial_dir = (pos - parent_pos) / parent_dist;

        // Decompose base repulsion into radial and tangential
        let radial_mag = dot(base_force, radial_dir);
        let radial_component = radial_dir * radial_mag;
        let tangential_component = base_force - radial_component;

        // Amplify tangential — this is what creates circular arrangements
        return radial_component + tangential_component * uniforms.tangential_multiplier + phantom_force;
    }

    return base_force + phantom_force;
}

// Main sibling + cousin repulsion computation
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    let mass_i = max(node_mass[node_idx], 1.0);

    var force = vec2<f32>(0.0, 0.0);

    // Get this node's parents (incoming edges)
    let parent_start = csr_inverse_offsets[node_idx];
    let parent_end = csr_inverse_offsets[node_idx + 1u];

    // ================================================================
    // PHASE 1: Sibling repulsion + orbital forces (1-hop: same parent)
    // For each parent:
    //   A. Orbit force: radial spring pulling child to target orbit radius
    //   B. Tangential-amplified sibling repulsion: spread children angularly
    // ================================================================
    for (var p = parent_start; p < parent_end; p++) {
        let parent_idx = csr_inverse_sources[p];

        if (parent_idx >= uniforms.node_count) {
            continue;
        }

        let parent_pos = positions[parent_idx];
        let to_parent = parent_pos - pos;
        let parent_dist = length(to_parent);

        // Get siblings (children of this parent)
        let sibling_start = csr_offsets[parent_idx];
        let sibling_end = csr_offsets[parent_idx + 1u];
        let num_siblings = sibling_end - sibling_start;

        // -- A: Orbit force --
        // Push child toward a target orbit distance from parent.
        // Normal: scales with sqrt(sibling count). Bubble: scales with parent's wellRadius.
        if (uniforms.orbit_strength > 0.0 && parent_dist > EPSILON) {
            var target_radius: f32;
            if (uniforms.bubble_mode != 0u) {
                target_radius = well_radius[parent_idx] * uniforms.orbit_scale;
            } else {
                target_radius = uniforms.orbit_radius_base * sqrt(max(f32(num_siblings), 1.0));
            }
            let radial_dir = to_parent / parent_dist;
            let orbit_error = parent_dist - target_radius;

            // Positive orbit_error = too far from parent, pull inward
            // Negative orbit_error = too close, push outward
            force += radial_dir * orbit_error * uniforms.orbit_strength;
        }

        // -- B: Sibling repulsion with tangential amplification --
        // CRITICAL: Limit loop iterations to prevent infinite loops
        let max_iterations = min(num_siblings, uniforms.max_siblings);

        for (var s = sibling_start; s < sibling_start + max_iterations; s++) {
            let sibling_idx = csr_targets[s];

            // Skip self and invalid indices
            if (sibling_idx == node_idx || sibling_idx >= uniforms.node_count) {
                continue;
            }

            let sib_pos = positions[sibling_idx];
            let mass_j = max(node_mass[sibling_idx], 1.0);

            force += apply_tangential_repulsion(
                pos, sib_pos, parent_pos, parent_dist,
                mass_i, mass_j, 1.0,
                node_idx, sibling_idx
            );
        }

        // ================================================================
        // PHASE 2: Cousin repulsion (2-hop: same grandparent)
        // For each parent, find that parent's parents (grandparents),
        // then each grandparent's other children (uncles/aunts),
        // then each uncle/aunt's children (cousins).
        // ================================================================
        if (uniforms.cousin_enabled != 0u) {
            let gp_start = csr_inverse_offsets[parent_idx];
            let gp_end = csr_inverse_offsets[parent_idx + 1u];

            var cousin_count = 0u;

            for (var g = gp_start; g < gp_end; g++) {
                let grandparent_idx = csr_inverse_sources[g];

                if (grandparent_idx >= uniforms.node_count) {
                    continue;
                }

                // Uncle/aunt = other children of grandparent (excluding our parent)
                let uncle_start = csr_offsets[grandparent_idx];
                let uncle_end = csr_offsets[grandparent_idx + 1u];
                let max_uncles = min(uncle_end - uncle_start, uniforms.max_siblings);

                for (var u = uncle_start; u < uncle_start + max_uncles; u++) {
                    let uncle_idx = csr_targets[u];

                    // Skip our own parent (we already handled siblings above)
                    if (uncle_idx == parent_idx || uncle_idx >= uniforms.node_count) {
                        continue;
                    }

                    // Cousins = children of uncle/aunt
                    let cousin_start_idx = csr_offsets[uncle_idx];
                    let cousin_end_idx = csr_offsets[uncle_idx + 1u];
                    let max_cousins = min(cousin_end_idx - cousin_start_idx, uniforms.max_siblings);

                    for (var c = cousin_start_idx; c < cousin_start_idx + max_cousins; c++) {
                        if (cousin_count >= MAX_COUSIN_ITERATIONS) {
                            break;
                        }

                        let cousin_idx = csr_targets[c];

                        if (cousin_idx == node_idx || cousin_idx >= uniforms.node_count) {
                            continue;
                        }

                        let cousin_pos = positions[cousin_idx];
                        let mass_j = max(node_mass[cousin_idx], 1.0);

                        // Cousin repulsion — uses standard (non-tangential) repulsion
                        force += apply_repulsion(pos, cousin_pos, mass_i, mass_j, uniforms.cousin_strength, node_idx, cousin_idx);
                        cousin_count++;
                    }

                    if (cousin_count >= MAX_COUSIN_ITERATIONS) {
                        break;
                    }
                }

                if (cousin_count >= MAX_COUSIN_ITERATIONS) {
                    break;
                }
            }
        }
    }

    // ================================================================
    // PHASE 3: Parent-child repulsion (direct edges)
    // Weaker repulsion for connected pairs (they're also connected by spring).
    // ================================================================
    let child_start = csr_offsets[node_idx];
    let child_end = csr_offsets[node_idx + 1u];

    let max_children = min(child_end - child_start, uniforms.max_siblings);
    for (var c = child_start; c < child_start + max_children; c++) {
        let child_idx = csr_targets[c];

        if (child_idx >= uniforms.node_count) {
            continue;
        }

        let child_pos = positions[child_idx];
        let mass_j = max(node_mass[child_idx], 1.0);

        force += apply_repulsion(pos, child_pos, mass_i, mass_j, uniforms.parent_child_multiplier, node_idx, child_idx);
    }

    // Accumulate forces
    forces[node_idx] += force;
}
