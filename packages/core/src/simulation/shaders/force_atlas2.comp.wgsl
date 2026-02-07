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
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: ForceAtlas2Uniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Output forces (accumulated) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node degrees (for degree-weighted repulsion)
@group(0) @binding(3) var<storage, read> degrees: array<u32>;

const MIN_DISTANCE: f32 = 0.01;
const FLAG_LINLOG: u32 = 1u;
const FLAG_STRONG_GRAVITY: u32 = 2u;
const FLAG_PREVENT_OVERLAP: u32 = 4u;

// ForceAtlas2 repulsion: F = kr * (degree(i) + 1) * (degree(j) + 1) / distance
// This is different from Coulomb repulsion which uses distance^2
fn fa2_repulsion(delta: vec2<f32>, degree_i: u32, degree_j: u32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // Degree-weighted mass
    let mass_i = f32(degree_i + 1u);
    let mass_j = f32(degree_j + 1u);

    // ForceAtlas2 uses linear distance in denominator (not squared)
    let force_magnitude = uniforms.scaling * mass_i * mass_j / dist;

    // Direction: pointing away from the other node
    let dir = delta / dist;

    return dir * force_magnitude;
}

// Main repulsion kernel - O(n^2) but with FA2 force model
@compute @workgroup_size(256)
fn repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = positions[node_idx];
    let node_degree = degrees[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    // Repulsion from all other nodes
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = positions[i];
        let other_degree = degrees[i];

        let delta = node_pos - other_pos;

        total_force += fa2_repulsion(delta, node_degree, other_degree);
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

