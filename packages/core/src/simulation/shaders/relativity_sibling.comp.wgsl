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
    // First element of csr_inverse's source region; see the region map below.
    // Lands in what was the struct's implicit tail padding, so this stays 64
    // bytes — the size relativity-atlas.ts allocates.
    csr_inverse_sources_base: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SiblingUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Inverse CSR: incoming edges (parents).
//
// INVERSE CSR REGION MAP (S = uniforms.csr_inverse_sources_base, the ALLOCATED
// offset-row length, not node_count + 1 — the regions must not move when the
// live node count does), in u32 elements:
//   [0, S)      offset row: node i's parents are the source entries
//               [csr_inverse[i], csr_inverse[i+1])
//   [S, ...)    source row: the slot each incoming edge comes from
//
// One binding rather than two: this is the widest pass in the algorithm, and
// with the two rows apart it binds nine storage buffers against a WebGPU
// default of eight, which makes the layout invalid and discards every frame
// recorded against it — see
// ForceAlgorithmInfo.minStorageBuffersPerShaderStage. The host sizes, fills
// and states the same map at csrInverseSourcesBase in
// simulation/algorithms/relativity-atlas.ts.
@group(0) @binding(3) var<storage, read> csr_inverse: array<u32>;

// Forward CSR: outgoing edges (children)
// For parent p, csr_offsets[p]..csr_offsets[p+1] are child indices (siblings of each other)
@group(0) @binding(4) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> csr_targets: array<u32>;

// Node masses (for mass-weighted repulsion)
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;

// Well radii (bubble mode: subtree-based collision boundaries)
@group(0) @binding(7) var<storage, read> well_radius: array<f32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(8) var<storage, read> node_flags: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

fn csr_inverse_offset(node_idx: u32) -> u32 {
    return csr_inverse[node_idx];
}

fn csr_inverse_source(entry_idx: u32) -> u32 {
    return csr_inverse[uniforms.csr_inverse_sources_base + entry_idx];
}

const EPSILON: f32 = 0.0001;
const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;
// Budget on cousin force computations to prevent runaway loops in wide
// hierarchies. Group selection (see sibling_group_count) keeps the expected
// number of selected cousins at or below this, so the cap rarely binds.
const MAX_COUSIN_ITERATIONS: u32 = 64u;

// Number of modulo-groups for a truncated interaction list.
//
// When a list exceeds the interaction cap, nodes interact only with members
// of their own modulo-group (slot_index % group_count). Because
// `i % G == j % G` is symmetric in (i, j) and both endpoints derive the same
// G from the same shared list, pairwise visibility is MUTUAL — Newton's
// third law holds for every computed pair and truncation injects no net
// momentum. The old head-prefix cap pushed late-listed children without any
// recoil on the early-listed ones, driving persistent directional drift in
// large sibling groups.
fn sibling_group_count(list_len: u32, cap: u32) -> u32 {
    return max((list_len + cap - 1u) / max(cap, 1u), 1u);
}

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

    // Inert slots — dead or LOD-hidden — neither receive nor exert forces.
    // CSR data should never reference dead slots, but the receiver check
    // also skips all their list scans.
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let pos = positions[node_idx];
    let mass_i = max(node_mass[node_idx], 1.0);

    var force = vec2<f32>(0.0, 0.0);

    // Get this node's parents (incoming edges)
    let parent_start = csr_inverse_offset(node_idx);
    let parent_end = csr_inverse_offset(node_idx + 1u);

    // ================================================================
    // PHASE 1: Sibling repulsion + orbital forces (1-hop: same parent)
    // For each parent:
    //   A. Orbit force: radial spring pulling child to target orbit radius
    //   B. Tangential-amplified sibling repulsion: spread children angularly
    // ================================================================
    for (var p = parent_start; p < parent_end; p++) {
        let parent_idx = csr_inverse_source(p);

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
        // Symmetric truncation: when the list exceeds max_siblings, interact
        // only within this node's modulo-group (mutual by construction — see
        // sibling_group_count). Expected force computations ~= max_siblings;
        // the scan itself is bounded by the sibling list length.
        let sib_groups = sibling_group_count(num_siblings, uniforms.max_siblings);

        for (var s = sibling_start; s < sibling_end; s++) {
            let sibling_idx = csr_targets[s];

            // Skip self, invalid indices, and inert slots
            if (sibling_idx == node_idx || sibling_idx >= uniforms.node_count) {
                continue;
            }
            if ((node_flags[sibling_idx] & NODE_FLAG_INERT) != 0u) {
                continue;
            }
            // Modulo-group membership (always true when sib_groups == 1)
            if (sib_groups > 1u && (sibling_idx % sib_groups) != (node_idx % sib_groups)) {
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
            let gp_start = csr_inverse_offset(parent_idx);
            let gp_end = csr_inverse_offset(parent_idx + 1u);

            var cousin_count = 0u;

            for (var g = gp_start; g < gp_end; g++) {
                let grandparent_idx = csr_inverse_source(g);

                if (grandparent_idx >= uniforms.node_count) {
                    continue;
                }

                // Uncle/aunt = other children of grandparent (excluding our parent)
                let uncle_start = csr_offsets[grandparent_idx];
                let uncle_end = csr_offsets[grandparent_idx + 1u];

                // Symmetric cousin truncation: both members of a cousin pair
                // share this grandparent, so both derive the SAME group count
                // from its total grandchild count — modulo-group membership
                // is mutual exactly as for siblings. (Offsets-only pre-scan.)
                var grandchild_count = 0u;
                for (var u = uncle_start; u < uncle_end; u++) {
                    let uncle_idx = csr_targets[u];
                    if (uncle_idx >= uniforms.node_count) {
                        continue;
                    }
                    grandchild_count += csr_offsets[uncle_idx + 1u] - csr_offsets[uncle_idx];
                }
                let cousin_groups = sibling_group_count(grandchild_count, MAX_COUSIN_ITERATIONS);

                for (var u = uncle_start; u < uncle_end; u++) {
                    let uncle_idx = csr_targets[u];

                    // Skip our own parent (we already handled siblings above)
                    if (uncle_idx == parent_idx || uncle_idx >= uniforms.node_count) {
                        continue;
                    }

                    // Cousins = children of uncle/aunt
                    let cousin_start_idx = csr_offsets[uncle_idx];
                    let cousin_end_idx = csr_offsets[uncle_idx + 1u];

                    for (var c = cousin_start_idx; c < cousin_end_idx; c++) {
                        // Hard budget backstop; group selection keeps the
                        // expected selections at or below it, so exhaustion
                        // (and its residual asymmetry) is rare.
                        if (cousin_count >= MAX_COUSIN_ITERATIONS) {
                            break;
                        }

                        let cousin_idx = csr_targets[c];

                        if (cousin_idx == node_idx || cousin_idx >= uniforms.node_count) {
                            continue;
                        }
                        if ((node_flags[cousin_idx] & NODE_FLAG_INERT) != 0u) {
                            continue;
                        }
                        if (cousin_groups > 1u &&
                            (cousin_idx % cousin_groups) != (node_idx % cousin_groups)) {
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

    // Gather-only pass (children never reciprocate directly), so no mutuality
    // requirement — but the sample must be unbiased: a head-prefix cap pushed
    // the parent away from its first-listed children only. Modulo-group
    // selection spreads the sampled children across the whole list.
    let child_groups = sibling_group_count(child_end - child_start, uniforms.max_siblings);
    for (var c = child_start; c < child_end; c++) {
        let child_idx = csr_targets[c];

        if (child_idx >= uniforms.node_count) {
            continue;
        }
        if ((node_flags[child_idx] & NODE_FLAG_INERT) != 0u) {
            continue;
        }
        if (child_groups > 1u && (child_idx % child_groups) != (node_idx % child_groups)) {
            continue;
        }

        let child_pos = positions[child_idx];
        let mass_j = max(node_mass[child_idx], 1.0);

        force += apply_repulsion(pos, child_pos, mass_i, mass_j, uniforms.parent_child_multiplier, node_idx, child_idx);
    }

    // Accumulate forces
    forces[node_idx] += force;
}
