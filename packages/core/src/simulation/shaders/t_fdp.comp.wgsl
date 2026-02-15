// t-Distribution Force-Directed Placement (t-FDP) Repulsion Compute Shader
//
// Implements bounded repulsion based on Student's t-distribution kernel.
// From: "Force-directed graph layouts revisited: a new force based on the
// t-Distribution" (Zhong, Xue, Zhang, Zhang, Ban, Deussen, Wang)
//
// Key property: Repulsion is BOUNDED at short range, unlike Coulomb (1/r²)
// or linear (1/r) which explode when d→0. This preserves local neighborhoods:
// connected nodes stay close instead of being blasted apart by repulsion.
//
// The force: F = kr × d / (1 + d²)^γ
//   - As d→0: F→0 (bounded, unlike 1/r)
//   - As d→∞: F→1/d^(2γ-1) (standard long-range push)
//   - γ controls the crossover: higher γ = narrower short-range zone
//   - kr = 1/alpha per paper (default: 10.0 when alpha=0.1)
//
// Attraction is handled by the separate t_fdp_attraction shader.
// Uses vec2<f32> layout for consolidated position/force data.

struct TFdpUniforms {
    node_count: u32,
    gamma: f32,              // >= 1.0, controls force shape
    repulsion_scale: f32,    // kr: global repulsion multiplier
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: TFdpUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = positions[i];
        let delta = pos - other_pos;
        let dist_sq = dot(delta, delta);

        if (dist_sq < EPSILON) {
            continue;
        }

        let dist = sqrt(dist_sq);
        let dir = delta / dist;

        // t-distribution repulsive force:
        // F = kr × d / (1 + d²)^γ
        //
        // Original formula is bounded at d→0 (force approaches 0, not infinity)
        // and decays like 1/d^(2γ-1) at large distances.
        // We add a minimum force floor (30% of kr) to prevent node collapse
        // when d is very small — without this, overlapping nodes cannot separate.
        let denominator = pow(1.0 + dist_sq, uniforms.gamma);
        let raw_force = uniforms.repulsion_scale * dist / denominator;
        let min_force = uniforms.repulsion_scale * 0.3;
        let force_magnitude = max(raw_force, min_force);

        total_force += dir * force_magnitude;
    }

    forces[node_idx] += total_force;
}
