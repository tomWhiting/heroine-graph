// ForceAtlas2 Compute Shader
// A force-directed layout algorithm designed for network visualization
//
// Key differences from standard force-directed:
// - Linear attraction (not quadratic spring)
// - Degree-weighted repulsion
// - Optional LinLog mode for better cluster separation
// - Strong gravity option for disconnected components
//
// Uses vec2<f32> layout for consolidated position/force data.

struct ForceAtlas2Uniforms {
    node_count: u32,
    scaling: f32,              // Overall force scaling (kr)
    gravity: f32,              // Gravity strength (kg)
    edge_weight_influence: f32, // How much edge weights affect attraction
    flags: u32,                // Bit flags: 0=linlog, 1=strong_gravity, 2=prevent_overlap
    // Length of the active-index list. The dispatch and both loop bounds come
    // from this, never from node_count.
    active_count: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ForceAtlas2Uniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Output forces (accumulated) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node degrees (for degree-weighted repulsion)
@group(0) @binding(3) var<storage, read> degrees: array<u32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(4) var<storage, read> node_flags: array<u32>;

// Active-index list: the first active_count entries are the slots taking part
// in this tick, ascending. `active` is a reserved WGSL identifier, hence the
// name.
@group(0) @binding(5) var<storage, read> live_idx: array<u32>;

// Per-node simulation mass (f32 per slot, 1.0 = one body). A collapsed LOD
// subtree rolls its members' mass into the visible proxy, and without this the
// proxy repels like the single node it is drawn as: the visible layout would
// contract on collapse and re-inflate on expand, which is the one thing
// semantic LOD must not do. FA2's own degree weighting is orthogonal and
// multiplies on top.
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;

const MIN_DISTANCE: f32 = 0.01;
const FLAG_LINLOG: u32 = 1u;
const FLAG_STRONG_GRAVITY: u32 = 2u;
const FLAG_PREVENT_OVERLAP: u32 = 4u;
const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// ForceAtlas2 repulsion: F = kr * (degree(i) + 1) * (degree(j) + 1) / distance
// This is different from Coulomb repulsion which uses distance^2
//
// `bodies_j` is how many bodies the SOURCE slot stands for — 1.0 for an
// ordinary node, the subtree's mass for a collapsed proxy. Only the source is
// scaled: the receiver then accelerates the way the centre of mass of the
// bodies it stands for does, because the integrator has no mass term
// (repulsion_n2's convention, and for the same reason).
//
// The scale belongs in the magnitude rather than on the accumulated vector: at
// unit mass `mass_j * 1.0` is the exact IEEE identity inside a pure product
// chain, so the un-collapsed path computes the bits it computed before mass
// reached this shader.
fn fa2_repulsion(delta: vec2<f32>, degree_i: u32, degree_j: u32, bodies_j: f32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // Degree-weighted mass
    let mass_i = f32(degree_i + 1u);
    let mass_j = f32(degree_j + 1u) * bodies_j;

    // ForceAtlas2 uses linear distance in denominator (not squared)
    let force_magnitude = uniforms.scaling * mass_i * mass_j / dist;

    // Direction: pointing away from the other node
    let dir = delta / dist;

    return dir * force_magnitude;
}

// Main repulsion kernel — one thread per ENTRY of the active-index list, with
// the inner all-pairs sum over the same list, so the cost is
// O(active_count^2) rather than O(node_count^2): hiding a subtree becomes work
// that is not done rather than threads that return early.
//
// The flag mask is kept on top of the gather. A stale list can only
// over-include, and masking makes that inert, so a host whose list was never
// refreshed (the identity list the fallback writes) stays correct, just slower.
// Ascending list order is load-bearing: it keeps the summed set and the
// addition order identical to slot-order dispatch, which is what makes the
// no-cut path bit-identical to the pre-list shader.
@compute @workgroup_size(256)
fn repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry = global_id.x;

    if (entry >= uniforms.active_count) {
        return;
    }

    let node_idx = live_idx[entry];
    // Inert slots — holes from removals (zeroed to the origin) and
    // LOD-hidden nodes — neither receive nor exert forces
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let node_pos = positions[node_idx];
    let node_degree = degrees[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    // Repulsion from all other live nodes
    for (var k = 0u; k < uniforms.active_count; k++) {
        let i = live_idx[k];
        if (i == node_idx) {
            continue;
        }
        if ((node_flags[i] & NODE_FLAG_INERT) != 0u) {
            continue;
        }

        let other_pos = positions[i];
        let other_degree = degrees[i];

        let delta = node_pos - other_pos;

        total_force += fa2_repulsion(delta, node_degree, other_degree, node_mass[i]);
    }

    // Gravity toward center (degree-weighted per FA2 paper, Equations 4 & 5)
    // Mass = (degree + 1), same as repulsion mass model.
    let mass_i = f32(node_degree + 1u);
    let gravity_dir = -node_pos;
    let gravity_dist = length(gravity_dir);

    if (gravity_dist > MIN_DISTANCE) {
        let gravity_unit = gravity_dir / gravity_dist;

        var gravity_force: vec2<f32>;
        if ((uniforms.flags & FLAG_STRONG_GRAVITY) != 0u) {
            // Strong gravity (Eq 5): Fg = kg * mass * distance * direction
            // Force increases linearly with distance — pulls distant nodes hard.
            gravity_force = gravity_unit * uniforms.gravity * mass_i * gravity_dist;
        } else {
            // Normal gravity (Eq 4): Fg = kg * mass * direction
            // Constant magnitude pull — gentle, distance-independent.
            gravity_force = gravity_unit * uniforms.gravity * mass_i;
        }

        total_force += gravity_force;
    }

    // Add to output forces
    forces[node_idx] += total_force;
}

