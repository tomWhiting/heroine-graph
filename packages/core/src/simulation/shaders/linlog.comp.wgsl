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
    // LOD dispatch counts. One uniform buffer serves both LinLog passes, so
    // this struct is declared identically in linlog.comp.wgsl and
    // linlog_attraction.comp.wgsl and every field keeps its offset in both.
    //
    // active_count: entries of the node-side active-index list the repulsion
    // pass covers. active_edge_count / bundle_count: entries of the LOD
    // active-edge list and of the bundle table the attraction pass's bundled
    // entry point covers; both zero unless a cut is live.
    active_count: u32,
    active_edge_count: u32,
    bundle_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

const MIN_DISTANCE: f32 = 0.01;
const FLAG_STRONG_GRAVITY: u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms: LinLogUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
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
// contract on collapse and re-inflate on expand. LinLog's degree weighting is
// orthogonal and multiplies on top.
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// ForceAtlas2 repulsion: F = kr * (deg(i)+1) * (deg(j)+1) / distance
// Linear falloff (1/r), degree-weighted mass.
// Gravity is computed inline (same pass, saves a dispatch).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    let mass_i = f32(node_degree + 1u);
    var total_force = vec2<f32>(0.0, 0.0);

    // N² repulsion over the active set: one iteration per ENTRY of the list,
    // not per slot. Ascending order keeps the summed set and the addition order
    // identical to slot-order dispatch, so the no-cut path is bit-identical.
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
        // Only the SOURCE is scaled by how many bodies it stands for: the
        // receiver then accelerates the way the centre of mass of those bodies
        // does, because the integrator has no mass term. At unit mass the
        // multiply is the exact IEEE identity inside a pure product chain.
        let mass_j = f32(other_degree + 1u) * node_mass[i];

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
