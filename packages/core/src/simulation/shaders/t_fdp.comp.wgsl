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
// The force, on the normalized distance d = dist / dist_scale:
//   F = kr × d / (1 + d²)^γ
//   - As d→0: F→0 (bounded, unlike 1/r)
//   - As d→∞: F→1/d^(2γ-1) (standard long-range push)
//   - γ controls the crossover: higher γ = narrower short-range zone
//   - kr = 1/alpha per paper (default: 10.0 when alpha=0.1)
//
// dist_scale maps world units onto the paper's unit scale: the kernel does
// all its work at d ≈ O(1) (it peaks at d = 1/sqrt(2γ-1)), so without the
// normalization it is effectively zero at typical world spacing (tens of
// units) and the layout collapses. The t_fdp_attraction shader uses the
// same dist_scale so the model's P1-P3 force balance holds.
//
// Two entry points share the pair-force math (repulsion_n2 pattern):
// - main:        bindings 0-2, no slot masking
// - main_masked: bindings 0-3, skips dead slots (holes from removals)
// Auto pipeline layouts only require bindings statically reachable from the
// chosen entry point, so main's layout stays 0-2.
//
// Uses vec2<f32> layout for consolidated position/force data.

struct TFdpUniforms {
    node_count: u32,
    gamma: f32,              // >= 1.0, controls force shape
    repulsion_scale: f32,    // kr: global repulsion multiplier
    dist_scale: f32,         // world units per t-kernel unit
}

@group(0) @binding(0) var<uniform> uniforms: TFdpUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node state flags (bit 0 = dead slot from removal) - main_masked only
@group(0) @binding(3) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;
const EPSILON: f32 = 0.0001;

// Anti-collapse floor: fraction of kr applied to overlapping nodes, and the
// fraction of dist_scale over which it fades to zero. The floor exists ONLY
// to separate (near-)coincident nodes, where the t-kernel vanishes — it must
// not touch the long-range regime. (An unconditional max(raw, 0.3*kr) here
// previously made every pair repel at constant magnitude at ALL distances,
// nullifying the t-kernel's decay and inflating the layout without bound.)
const FLOOR_FRACTION: f32 = 0.3;
const FLOOR_ZONE: f32 = 0.1;

fn pair_force(node_idx: u32, pos: vec2<f32>, i: u32) -> vec2<f32> {
    let delta = pos - positions[i];
    let dist_sq = dot(delta, delta);

    // Coincident nodes give no direction to repel along — push in a
    // deterministic golden-angle direction so stacked nodes separate
    // (same tiebreaker as collision.comp.wgsl).
    if (dist_sq < EPSILON) {
        let angle = f32(node_idx) * 0.618033988749895 * 6.28318530718;
        return vec2<f32>(cos(angle), sin(angle)) *
            (uniforms.repulsion_scale * FLOOR_FRACTION);
    }

    let dist = sqrt(dist_sq);
    let dir = delta / dist;

    // t-distribution repulsive force on the normalized distance
    let d = dist / uniforms.dist_scale;
    let t_force = uniforms.repulsion_scale * d / pow(1.0 + d * d, uniforms.gamma);

    // Near-zero separation boost, faded out by FLOOR_ZONE * dist_scale
    let floor_boost = uniforms.repulsion_scale * FLOOR_FRACTION *
        (1.0 - smoothstep(0.0, FLOOR_ZONE * uniforms.dist_scale, dist));

    return dir * (t_force + floor_boost);
}

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

        total_force += pair_force(node_idx, pos, i);
    }

    forces[node_idx] += total_force;
}

@compute @workgroup_size(256)
fn main_masked(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Dead slots (holes from removals, zeroed to the origin) neither
    // receive nor exert repulsion
    if ((node_flags[node_idx] & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let pos = positions[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }
        if ((node_flags[i] & NODE_FLAG_DEAD) != 0u) {
            continue;
        }

        total_force += pair_force(node_idx, pos, i);
    }

    forces[node_idx] += total_force;
}
