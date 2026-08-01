// Simplified Spring Force Compute Shader
// Computes attractive forces along edges using Hooke's law
//
// Force accumulation: WGSL has no native f32 atomics, and a plain
// `forces[idx] += force` read-modify-write loses updates catastrophically for
// hub nodes — a node's edges are contiguous in the edge buffer, so the whole
// SIMD wave reads the same stale value and only one lane's write survives
// (~1/k of a k-child hub's attraction). Instead the shared vec2<f32> force
// buffer is bound here as array<atomic<u32>> over the raw f32 bit patterns
// (node i -> components [2i]=x, [2i+1]=y) and accumulated with a
// compare-exchange loop. Same buffer, same bytes — other passes keep their
// array<vec2<f32>> view.

struct SpringUniforms {
    edge_count: u32,
    spring_strength: f32,
    rest_length: f32,
    // Entries of live_edge_idx the bundled entry point gathers, and the number
    // of LOD bundles after them. Both zero unless an LOD cut is live; `main`
    // never reads either, which is what keeps the un-cut path unchanged.
    active_edge_count: u32,
    bundle_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SpringUniforms;

// Node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators — the shared vec2<f32> buffer viewed as f32 bit patterns
// behind atomic<u32> for race-free accumulation (see header comment)
@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;

// Edge data (source, target pairs)
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(5) var<storage, read> node_flags: array<u32>;

// --- LOD edge aggregation (main_bundled only; `main` binds none of these) ---

// Indices of the source edges whose endpoints are both in the visible cut.
// The twin of live_idx on the node side: springs dispatch over this list
// instead of over every edge, so hiding a subtree is work that is not done
// rather than threads that return early.
@group(0) @binding(6) var<storage, read> live_edge_idx: array<u32>;

// Aggregated cross-boundary edges, three u32 per bundle:
// [visible_source, visible_target, weight]. `weight` is how many source edges
// the bundle stands for and scales the spring by exactly that factor — the
// proxy pair then sits where the two subtrees' centres of attraction sat when
// they were expanded.
@group(0) @binding(7) var<storage, read> bundles: array<u32>;

// Per-node simulation mass (1.0 = one body). A proxy carries the mass of the
// subtree it replaces, and the integrator has no mass term, so a spring
// arriving at a proxy is divided by it: the proxy then accelerates like the
// centre of mass of the bodies it stands for rather than like one of them.
@group(0) @binding(8) var<storage, read> node_mass: array<f32>;

const BUNDLE_STRIDE: u32 = 3u;

// Reciprocal of the mass an arriving spring is shared over.
//
// A hidden node's mass is zero — its mass now lives in its proxy — and a mass
// below one body would amplify a spring rather than share it, so both are
// clamped away. Hidden nodes never reach here anyway: `apply_spring` masks
// them first.
//
// One body returns exactly 1.0 rather than computing 1.0 / 1.0, because a
// GPU's f32 divide is permitted to be approximate: an ULP of error there would
// make an identity aggregation differ from no aggregation at all, which is the
// SC-005 guarantee.
fn inverse_mass(node_idx: u32) -> f32 {
    let mass = node_mass[node_idx];
    if (mass <= 1.0) {
        return 1.0;
    }
    return 1.0 / mass;
}

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Race-free float accumulation: CAS loop on the f32 bit pattern.
fn atomic_add_f32(index: u32, value: f32) {
    var old = atomicLoad(&forces[index]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old) + value);
        let result = atomicCompareExchangeWeak(&forces[index], old, new_bits);
        if (result.exchanged) { return; }
        old = result.old_value;
    }
}

// Add a vec2 force to node `node_idx`'s accumulator.
fn accumulate_force(node_idx: u32, force: vec2<f32>) {
    atomic_add_f32(node_idx * 2u, force.x);
    atomic_add_f32(node_idx * 2u + 1u, force.y);
}

// One Hooke spring between two slots, scaled by `weight` and shared over each
// endpoint's own mass.
//
// `weight` is 1.0 for a source edge and both inverse masses are 1.0 with no
// LOD cut — exact IEEE identities, so the un-bundled path computes the bits it
// computed before either existed.
fn apply_spring(
    source_idx: u32,
    target_idx: u32,
    weight: f32,
    source_inv_mass: f32,
    target_inv_mass: f32,
) {
    // Edges are cascade-removed with their nodes, so live edges should never
    // reference dead slots — skip defensively if buffers are mid-update. An
    // edge with an LOD-hidden endpoint is skipped for a stronger reason: a
    // hidden node exerts no force, and a spring would otherwise drag its
    // visible neighbour toward a node that is not on screen.
    if (((node_flags[source_idx] | node_flags[target_idx]) & NODE_FLAG_INERT) != 0u) {
        return;
    }

    // Get positions
    let source_pos = positions[source_idx];
    let target_pos = positions[target_idx];

    let delta = target_pos - source_pos;
    let dist = length(delta);

    // Avoid division by zero
    if (dist < 0.0001) {
        return;
    }

    // Direction from source to target
    let dir = delta / dist;

    // Hooke's law: F = k * (length - rest_length)
    let displacement = dist - uniforms.rest_length;
    let force_magnitude = uniforms.spring_strength * displacement * weight;

    let force = dir * force_magnitude;

    // Add force to source (attractive toward target)
    // Subtract from target (attractive toward source)
    accumulate_force(source_idx, force * source_inv_mass);
    accumulate_force(target_idx, -force * target_inv_mass);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    apply_spring(edge_sources[edge_idx], edge_targets[edge_idx], 1.0, 1.0, 1.0);
}

// Spring pass under a live LOD cut.
//
// The dispatch domain is `active_edge_count + bundle_count`: the source edges
// with both endpoints on screen, then the aggregated bundles standing in for
// every edge that crosses a collapse boundary. Edges buried inside a collapsed
// subtree appear in neither, which is the point — springs cost what is visible.
//
// The gather is defence in depth, not the safety mechanism: `apply_spring`
// still masks on the node flags, so an edge list that has not caught up with a
// transition costs a wasted thread and never a wrong force.
@compute @workgroup_size(256)
fn main_bundled(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx < uniforms.active_edge_count) {
        let edge_idx = live_edge_idx[idx];
        let source_idx = edge_sources[edge_idx];
        let target_idx = edge_targets[edge_idx];
        apply_spring(
            source_idx,
            target_idx,
            1.0,
            inverse_mass(source_idx),
            inverse_mass(target_idx),
        );
        return;
    }

    let bundle_idx = idx - uniforms.active_edge_count;
    if (bundle_idx >= uniforms.bundle_count) {
        return;
    }

    // A bundle pulls with the full strength of the edges it replaces, uncapped.
    // Uncapped because the two ends also repel with the mass of the subtrees
    // they stand for: capping attraction alone would leave repulsion unopposed
    // and move the equilibrium the expanded layout had, which is exactly what
    // SC-002 forbids. What the two ends do NOT share is the *acceleration* —
    // each divides by its own mass, so a five-node module and a five-thousand
    // node one respond to the same bundle as their centres of mass would.
    let base = bundle_idx * BUNDLE_STRIDE;
    let source_idx = bundles[base];
    let target_idx = bundles[base + 1u];
    apply_spring(
        source_idx,
        target_idx,
        f32(bundles[base + 2u]),
        inverse_mass(source_idx),
        inverse_mass(target_idx),
    );
}
