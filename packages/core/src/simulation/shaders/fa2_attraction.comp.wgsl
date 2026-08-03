// ForceAtlas2 Attraction Compute Shader
//
// Implements the linear attraction from the ForceAtlas2 paper (Jacomy et al. 2014).
//
// Standard springs use Hooke's law: F = k * (d - rest_length), creating an
// equilibrium distance that produces grid/lattice patterns. FA2 attraction is
// fundamentally different: F = d (always pulling, no rest length, no equilibrium).
//
// Connected nodes ALWAYS attract. Only repulsion stops them from collapsing.
// This creates organic, non-lattice layouts where cluster structure emerges
// naturally from the balance of attraction and degree-weighted repulsion.
//
// Optional LinLog mode: F = log(1 + d) instead of F = d.
// Caps long-range attraction so inter-cluster edges can't overpower intra-cluster
// repulsion, improving cluster separation.
//
// This shader operates per-edge: each thread processes one edge and applies
// equal-and-opposite forces to the source and target nodes.
//
// Two entry points share the edge math:
// - main:         one thread per source edge, bindings 0-6. The un-cut path,
//                 unchanged: it passes compile-time 1.0s for every LOD term.
// - main_bundled: one thread per entry of the LOD active-edge list, then one
//                 per aggregated bundle; bindings 0-8.
// Relativity Atlas runs this module too (its "linear attraction" pass), so both
// entry points serve both algorithms.

struct FA2AttractionUniforms {
    edge_count: u32,
    edge_weight_influence: f32, // delta: exponent on edge weights (0 = ignore, 1 = linear)
    flags: u32,                 // bit 0 = linlog mode
    // Entries of lod_edge_set's live-edge region the bundled entry point
    // gathers, and the number of bundles after them. The first also bases the
    // bundle region (see bundle_base). Both zero unless an LOD cut is live;
    // `main` never reads either, which keeps the un-cut path unchanged.
    active_edge_count: u32,
    bundle_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

const MIN_DISTANCE: f32 = 0.01;
const FLAG_LINLOG: u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms: FA2AttractionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
// Force accumulators — the shared vec2<f32> buffer viewed as f32 bit patterns
// behind atomic<u32> (node i -> [2i]=x, [2i+1]=y) so per-edge threads can
// accumulate without lost updates on hub nodes. Same buffer, same bytes —
// other passes keep their array<vec2<f32>> view.
@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;
@group(0) @binding(5) var<storage, read> edge_weights: array<f32>;
// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(6) var<storage, read> node_flags: array<u32>;

// --- LOD edge aggregation (main_bundled only; `main` binds none of these) ---

// LOD EDGE SET REGION MAP (A = uniforms.active_edge_count,
// B = uniforms.bundle_count, S = BUNDLE_STRIDE), in u32 elements:
//   [0, A)         indices of the source edges whose endpoints are both in the
//                  visible cut, ascending. Dispatching over this list rather
//                  than over every edge is what makes hiding a subtree work
//                  that is not done rather than threads that return early.
//   [A, A + B*S)   aggregated cross-boundary edges, S words each:
//                  [visible_source, visible_target, weight].
//
// The same buffer, map and contents springs_simple.comp.wgsl reads. The two
// regions share one binding so that this pass binds eight storage buffers
// rather than nine; nine exceeds the WebGPU default, which makes the layout
// invalid and discards every frame recorded against it — see
// ForceAlgorithmInfo.minStorageBuffersPerShaderStage. The host sizes and fills
// the buffer in simulation/pipeline.ts (uploadEdgeBundles,
// LOD_EDGE_SET_WORDS_PER_EDGE), where the same map is stated.
@group(0) @binding(7) var<storage, read> lod_edge_set: array<u32>;

// Per-node simulation mass (1.0 = one body); a proxy carries the mass of the
// subtree it replaces. See inverse_mass.
@group(0) @binding(8) var<storage, read> node_mass: array<f32>;

const BUNDLE_STRIDE: u32 = 3u;

// First element of the bundle region. The live-edge region is exactly
// [0, active_edge_count), so the count that bounds the first loop is the
// offset that addresses the second — no separate layout uniform can drift
// out of step with it.
fn bundle_base() -> u32 {
    return uniforms.active_edge_count;
}

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Reciprocal of the mass an arriving attraction is shared over.
//
// A hidden node's mass is zero — its mass now lives in its proxy — and a mass
// below one body would amplify the pull rather than share it, so both are
// clamped away. Hidden nodes never reach here anyway: apply_attraction masks
// them first.
//
// One body returns exactly 1.0 rather than computing 1.0 / 1.0, because a GPU's
// f32 divide is permitted to be approximate: an ULP of error there would make
// an identity aggregation differ from no aggregation at all.
fn inverse_mass(node_idx: u32) -> f32 {
    let mass = node_mass[node_idx];
    if (mass <= 1.0) {
        return 1.0;
    }
    return 1.0 / mass;
}

// Race-free float accumulation: CAS loop on the f32 bit pattern. WGSL has no
// native f32 atomics; plain `forces[i] += f` dropped a degree-proportional
// fraction of hub attraction (a hub's contiguous edges execute in one SIMD
// wave, all reading the same stale value; only one lane's write survived).
fn atomic_add_f32(index: u32, value: f32) {
    var old = atomicLoad(&forces[index]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old) + value);
        let result = atomicCompareExchangeWeak(&forces[index], old, new_bits);
        if (result.exchanged) { return; }
        old = result.old_value;
    }
}

fn accumulate_force(node_idx: u32, force: vec2<f32>) {
    atomic_add_f32(node_idx * 2u, force.x);
    atomic_add_f32(node_idx * 2u + 1u, force.y);
}

// Edge weight influence: w^delta (1.0 when delta=0, w when delta=1).
//
// A bundle has no source edge weight and passes 1.0, which this returns
// unchanged down every branch — pow(1.0, delta) is 1.0 for all delta — so a
// bundle's whole weighting is its edge count.
fn weight_factor(w: f32) -> f32 {
    if (uniforms.edge_weight_influence == 0.0) {
        return 1.0;
    }
    if (uniforms.edge_weight_influence == 1.0) {
        return w;
    }
    return pow(max(w, MIN_DISTANCE), uniforms.edge_weight_influence);
}

// FA2 attraction between two slots: F = w^delta * d * direction (standard)
//                                or: F = w^delta * log(1 + d) * direction (linlog)
//
// `count` is how many source edges this call stands for — 1.0 for a real edge,
// the bundle weight for an aggregated one. It multiplies the magnitude, and
// that is exactly superposition: the force law is a function of the separation
// alone, so `count` coincident edges between the same pair pull `count` times
// as hard whatever shape the law has. Aggregation replaces the edges of a
// collapsed subtree with edges to the proxy standing at its centre of
// attraction, so the law is evaluated once at that separation instead of
// `count` times at nearly the same one. Neither the linear nor the logarithmic
// branch is linear IN THE DISTANCE, and neither has to be: what must be linear
// for this to hold is the summation of forces, which it is.
//
// `source_inv_mass` / `target_inv_mass` divide the received force by each
// endpoint's aggregate mass. The integrator has no mass term, so a proxy
// receiving the full bundle would accelerate M-times faster than the centre of
// mass it stands for.
//
// `count` is 1.0 and both inverse masses are 1.0 with no LOD cut — exact IEEE
// identities, so `main` computes the bits it computed before any of this
// existed.
fn apply_attraction(
    src: u32,
    tgt: u32,
    factor: f32,
    count: f32,
    source_inv_mass: f32,
    target_inv_mass: f32,
) {
    // Edges touching an inert slot — dead or LOD-hidden — exert no force
    if (((node_flags[src] | node_flags[tgt]) & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let src_pos = positions[src];
    let tgt_pos = positions[tgt];

    let delta = tgt_pos - src_pos;
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // FA2 attraction: always pulling, no rest length, no equilibrium distance
    var force_magnitude: f32;
    if ((uniforms.flags & FLAG_LINLOG) != 0u) {
        // LinLog mode: F = w^delta * log(1 + d)
        // Caps long-range attraction for better cluster separation
        force_magnitude = factor * log(1.0 + dist) * count;
    } else {
        // Standard mode: F = w^delta * d
        // Linear — simple, effective, produces good general layouts
        force_magnitude = factor * dist * count;
    }

    let dir = delta / dist;
    let force = dir * force_magnitude;

    // Apply equal-and-opposite forces (race-free, see atomic_add_f32)
    accumulate_force(src, force * source_inv_mass);
    accumulate_force(tgt, -force * target_inv_mass);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    apply_attraction(
        edge_sources[edge_idx],
        edge_targets[edge_idx],
        weight_factor(edge_weights[edge_idx]),
        1.0,
        1.0,
        1.0,
    );
}

// Attraction under a live LOD cut.
//
// The dispatch domain is `active_edge_count + bundle_count`: the source edges
// with both endpoints on screen, then the aggregated bundles standing in for
// every edge that crosses a collapse boundary. Edges buried inside a collapsed
// subtree appear in neither, which is the point.
//
// Without this, collapsing a module DELETED its cross-cutting imports rather
// than transferring them: every such edge has a hidden endpoint, `main` masks
// it, and the collapsed layout stops resembling the expanded one.
//
// The gather is defence in depth, not the safety mechanism: apply_attraction
// still masks on the node flags, so an edge list that has not caught up with a
// transition costs a wasted thread and never a wrong force.
@compute @workgroup_size(256)
fn main_bundled(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx < uniforms.active_edge_count) {
        let edge_idx = lod_edge_set[idx];
        let src = edge_sources[edge_idx];
        let tgt = edge_targets[edge_idx];
        apply_attraction(
            src,
            tgt,
            weight_factor(edge_weights[edge_idx]),
            1.0,
            inverse_mass(src),
            inverse_mass(tgt),
        );
        return;
    }

    let bundle_idx = idx - uniforms.active_edge_count;
    if (bundle_idx >= uniforms.bundle_count) {
        return;
    }

    // A bundle pulls with the full strength of the edges it replaces, uncapped,
    // because the two ends also repel with the mass of the subtrees they stand
    // for: capping attraction alone would leave repulsion unopposed and move
    // the equilibrium the expanded layout had.
    let base = bundle_base() + bundle_idx * BUNDLE_STRIDE;
    let src = lod_edge_set[base];
    let tgt = lod_edge_set[base + 1u];
    apply_attraction(
        src,
        tgt,
        weight_factor(1.0),
        f32(lod_edge_set[base + 2u]),
        inverse_mass(src),
        inverse_mass(tgt),
    );
}
