// LinLog Repulsion + Gravity Compute Shader
//
// Implements the repulsion and gravity components of the LinLog energy model
// (Noack 2009) as described in the ForceAtlas2 paper (Jacomy et al. 2014).
//
// LinLog achieves the best cluster separation of any force-directed energy
// model by using logarithmic attraction (in the companion shader) with
// degree-weighted repulsion (in this shader).
//
// Energy model pair: (attraction=0, repulsion=-1)
//   - FR:     (2, -1) — quadratic attraction, poor cluster separation
//   - FA2:    (1, -1) — linear attraction, moderate separation
//   - LinLog: (0, -1) — logarithmic attraction, best separation
//
// Uses vec2<f32> layout for consolidated position/force data.

struct LinLogUniforms {
    node_count: u32,
    edge_count: u32,
    kr: f32,                    // Repulsion scaling
    kg: f32,                    // Gravity strength
    edge_weight_influence: f32, // δ: exponent on edge weights
    flags: u32,                 // bit 0 = strong_gravity
    _padding: vec2<u32>,
}

const MIN_DISTANCE: f32 = 0.01;
const FLAG_STRONG_GRAVITY: u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms: LinLogUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> degrees: array<u32>;

// ForceAtlas2 repulsion: F = kr * (deg(i)+1) * (deg(j)+1) / distance
// Linear falloff (1/r), degree-weighted mass.
// Gravity is computed inline (same pass, saves a dispatch).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = positions[node_idx];
    let node_degree = degrees[node_idx];
    let mass_i = f32(node_degree + 1u);
    var total_force = vec2<f32>(0.0, 0.0);

    // N² repulsion
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = positions[i];
        let other_degree = degrees[i];
        let mass_j = f32(other_degree + 1u);

        let delta = node_pos - other_pos;
        let dist_sq = dot(delta, delta);
        let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

        // FA2 degree-weighted repulsion: kr * mass_i * mass_j / dist
        let force_magnitude = uniforms.kr * mass_i * mass_j / dist;
        let dir = delta / dist;

        total_force += dir * force_magnitude;
    }

    // Gravity toward center (degree-weighted)
    let gravity_dir = -node_pos;
    let gravity_dist = length(gravity_dir);

    if (gravity_dist > MIN_DISTANCE) {
        let gravity_unit = gravity_dir / gravity_dist;

        var gravity_force: vec2<f32>;
        if ((uniforms.flags & FLAG_STRONG_GRAVITY) != 0u) {
            // Strong gravity: F = kg * mass * d (Eq. 5 in FA2 paper)
            // Distance-linear — pulls distant nodes harder, producing compact layouts.
            gravity_force = gravity_unit * uniforms.kg * mass_i * gravity_dist;
        } else {
            // Normal gravity: F = kg * mass (Eq. 4 in FA2 paper)
            // Distance-independent, degree-weighted. Prevents component drift
            // while allowing natural cluster spacing.
            gravity_force = gravity_unit * uniforms.kg * mass_i;
        }

        total_force += gravity_force;
    }

    forces[node_idx] += total_force;
}
