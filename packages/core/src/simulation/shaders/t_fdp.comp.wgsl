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
// One entry point, main_masked (bindings 0-4, repulsion_n2 pattern): one
// thread per entry of the LOD active-index list, with the inner all-pairs sum
// over the same list and inert slots (dead or LOD-hidden) masked on top. The
// unmasked `main` it used to also expose was a hole — a removed node's slot
// stays in range with its position zeroed, so it repelled as a phantom body at
// the origin — and there is no configuration in which running it is correct.
//
// Uses vec2<f32> layout for consolidated position/force data.

struct TFdpUniforms {
    node_count: u32,
    gamma: f32,              // >= 1.0, controls force shape
    repulsion_scale: f32,    // kr: global repulsion multiplier
    dist_scale: f32,         // world units per t-kernel unit
    // Length of the active-index list. The dispatch and both loop bounds come
    // from this, never from node_count.
    active_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: TFdpUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(3) var<storage, read> node_flags: array<u32>;

// Active-index list: the first active_count entries are the slots taking part
// in this tick, ascending. `active` is a reserved WGSL identifier, hence the
// name.
@group(0) @binding(4) var<storage, read> live_idx: array<u32>;

// Per-node simulation mass (f32 per slot, 1.0 = one body). A collapsed LOD
// subtree rolls its members' mass into the visible proxy; without this the
// proxy repels like the single node it is drawn as and the visible layout
// contracts on collapse.
@group(0) @binding(5) var<storage, read> node_mass: array<f32>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;
const EPSILON: f32 = 0.0001;

// Anti-collapse floor: fraction of kr applied to overlapping nodes, and the
// fraction of dist_scale over which it fades to zero. The floor exists ONLY
// to separate (near-)coincident nodes, where the t-kernel vanishes — it must
// not touch the long-range regime. (An unconditional max(raw, 0.3*kr) here
// previously made every pair repel at constant magnitude at ALL distances,
// nullifying the t-kernel's decay and inflating the layout without bound.)
const FLOOR_FRACTION: f32 = 0.3;
const FLOOR_ZONE: f32 = 0.1;

// Repulsion exerted on `node_idx` by slot `i`.
//
// `bodies` is how many bodies the SOURCE slot `i` stands for — 1.0 for an
// ordinary node, the subtree's mass for a collapsed proxy. Only the source is
// scaled: the receiver then accelerates the way the centre of mass of the
// bodies it stands for does, because the integrator has no mass term. At unit
// mass the multiply is the exact IEEE identity, so an un-collapsed graph
// computes the bits it computed before mass reached this shader.
fn pair_force(node_idx: u32, pos: vec2<f32>, i: u32, bodies: f32) -> vec2<f32> {
    let delta = pos - positions[i];
    let dist_sq = dot(delta, delta);

    // Coincident nodes give no direction to repel along — push in a
    // deterministic golden-angle direction so stacked nodes separate
    // (same tiebreaker as collision.comp.wgsl).
    if (dist_sq < EPSILON) {
        let angle = f32(node_idx) * 0.618033988749895 * 6.28318530718;
        return vec2<f32>(cos(angle), sin(angle)) *
            (uniforms.repulsion_scale * FLOOR_FRACTION * bodies);
    }

    let dist = sqrt(dist_sq);
    let dir = delta / dist;

    // t-distribution repulsive force on the normalized distance
    let d = dist / uniforms.dist_scale;
    let t_force = uniforms.repulsion_scale * bodies * d / pow(1.0 + d * d, uniforms.gamma);

    // Near-zero separation boost, faded out by FLOOR_ZONE * dist_scale
    let floor_boost = uniforms.repulsion_scale * bodies * FLOOR_FRACTION *
        (1.0 - smoothstep(0.0, FLOOR_ZONE * uniforms.dist_scale, dist));

    return dir * (t_force + floor_boost);
}

// One thread per ENTRY of live_idx, and the inner sum runs over the same list,
// so the cost is O(active_count^2) rather than O(node_count^2).
//
// The flag mask is kept on top of the gather. A stale list can only
// over-include, and masking makes that inert rather than a phantom force, so a
// host whose list was never refreshed (the identity list the fallback writes)
// stays correct, just slower. Masking also keeps the summed set and its
// addition order identical between the two cases, which is what makes the
// no-cut path bit-identical to slot-order dispatch.
@compute @workgroup_size(256)
fn main_masked(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry = global_id.x;

    if (entry >= uniforms.active_count) {
        return;
    }

    let node_idx = live_idx[entry];
    // Inert slots — dead or LOD-hidden — neither
    // receive nor exert repulsion
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let pos = positions[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    for (var k = 0u; k < uniforms.active_count; k++) {
        let i = live_idx[k];
        if (i == node_idx) {
            continue;
        }
        if ((node_flags[i] & NODE_FLAG_INERT) != 0u) {
            continue;
        }

        total_force += pair_force(node_idx, pos, i, node_mass[i]);
    }

    forces[node_idx] += total_force;
}
