// N^2 Repulsion Force Compute Shader
// Simple all-pairs repulsion calculation for small graphs
//
// Uses vec2<f32> layout for consolidated position/force data.
//
// One entry point, main_masked (bindings 0-5): dispatched over the
// active-index list and masking inert slots. The core simulation pipeline and
// the N² algorithm plugin both run it.
//
// There used to be a second, unmasked `main` for the plugin, whose bind group
// carried no node_flags. That was a hole, not an optimisation: a removed node's
// slot stays in range with its position zeroed, so under setForceAlgorithm("n2")
// every hole repelled as a phantom body at the origin, and an LOD-hidden node
// went on pushing the nodes that replaced it on screen. The plugin now binds
// the same six bindings; when its host supplies no list it binds the identity
// one, which is the same slot-order sum the unmasked entry point computed.

struct RepulsionUniforms {
    node_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    max_distance: f32,
    // Length of the active-index list. main_masked's dispatch and both of its
    // loop bounds come from this, never from node_count.
    active_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: RepulsionUniforms;

// Node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators (read-write) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(3) var<storage, read> node_flags: array<u32>;

// Per-node simulation mass (f32 per slot, 1.0 = one body). A collapsed LOD
// subtree rolls its members' mass into the visible proxy so the proxy repels
// like the subtree it stands for. uniforms.repulsion_strength stays the global
// multiplier; this is per-node on top of it.
@group(0) @binding(4) var<storage, read> node_mass: array<f32>;

// Active-index list: the first active_count entries are the slots taking part
// in this tick, ascending. `active` is a reserved WGSL identifier, hence the
// name.
@group(0) @binding(5) var<storage, read> live_idx: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Coulomb-like repulsion between one pair: F = k * m / r^2, where m is the
// SOURCE node's mass. The integrator has no mass term, so scaling the source
// alone makes a collapsed proxy of mass M accelerate like the centre of mass
// of the M bodies it replaces.
//
// Mass belongs in the magnitude rather than on the accumulated vector: at unit
// mass `k * 1.0 / r^2` rounds to exactly `k / r^2`, whereas a trailing
// `pair * mass` next to `force +=` is an fma-contraction candidate and drifts
// by ULPs from the mass-free shader (SC-005 is checked bit-exactly).
fn pair_force(node_pos: vec2<f32>, other_pos: vec2<f32>, other_mass: f32) -> vec2<f32> {
    let delta = node_pos - other_pos;
    let dist_sq = dot(delta, delta);

    // Skip nodes beyond max_distance (0 = no limit)
    if (uniforms.max_distance > 0.0 && dist_sq > uniforms.max_distance * uniforms.max_distance) {
        return vec2<f32>(0.0, 0.0);
    }

    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);
    let dist = sqrt(safe_dist_sq);

    let force_magnitude = uniforms.repulsion_strength * other_mass / safe_dist_sq;

    return delta * (force_magnitude / dist);
}

// One thread per ENTRY of live_idx, and the inner sum
// runs over the same list, so the cost is O(active_count^2) rather than
// O(node_count^2) — the reason the list exists.
//
// The flag mask is kept on top of the gather. The list is derived from
// nodeFlagsShadow (the single CPU-side authority) and is meant to hold exactly
// the non-inert slots, but a stale list can only over-include: masking makes
// that inert rather than a phantom force, so a buffer set whose list was never
// refreshed (the identity list createSimulationBuffers writes) stays correct,
// just slower. Masking also keeps the summed set and its addition order
// identical between the two cases, which is what makes SC-005 bit-exact.
@compute @workgroup_size(256)
fn main_masked(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry = global_id.x;

    if (entry >= uniforms.active_count) {
        return;
    }

    let node_idx = live_idx[entry];
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let node_pos = positions[node_idx];
    var force = vec2<f32>(0.0, 0.0);

    // Direct summation over the active set
    for (var k = 0u; k < uniforms.active_count; k++) {
        let i = live_idx[k];
        if (i == node_idx) {
            continue;
        }
        if ((node_flags[i] & NODE_FLAG_INERT) != 0u) {
            continue;
        }

        force += pair_force(node_pos, positions[i], node_mass[i]);
    }

    forces[node_idx] += force;
}
