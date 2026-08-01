// LinLog Attraction Compute Shader
//
// Implements the logarithmic attraction component of the LinLog energy model
// (Noack 2009) as described in the ForceAtlas2 paper (Jacomy et al. 2014).
//
// Standard springs use Hooke's law: F ~ d (linear attraction).
// LinLog uses: F ~ log(1 + d) (logarithmic attraction).
//
// Why logarithmic? Linear attraction pulls distant nodes proportionally harder,
// dragging everything into a central mass. Logarithmic attraction caps the pull
// at long range — once nodes are "far enough," attraction barely increases.
// Clusters stay separated because inter-cluster edges can't overpower intra-cluster
// repulsion.
//
// This shader operates per-edge: each thread processes one edge and applies
// equal-and-opposite forces to the source and target nodes via race-free
// CAS accumulation on the shared force buffer (see atomic_add_f32).
//
// Two entry points share the edge math:
// - main:         one thread per source edge, bindings 0-6. The un-cut path,
//                 unchanged: it passes compile-time 1.0s for every LOD term.
// - main_bundled: one thread per entry of the LOD active-edge list, then one
//                 per aggregated bundle; bindings 0-9.

struct LinLogUniforms {
    node_count: u32,
    edge_count: u32,
    kr: f32,                    // Repulsion scaling (unused here)
    kg: f32,                    // Gravity strength (unused here)
    edge_weight_influence: f32, // δ: exponent on edge weights
    flags: u32,                 // bit 0 = strong_gravity (unused here)
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

@group(0) @binding(0) var<uniform> uniforms: LinLogUniforms;
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

// Indices of the source edges whose endpoints are both in the visible cut.
// Dispatching over this list rather than over every edge is what makes hiding a
// subtree work that is not done rather than threads that return early.
@group(0) @binding(7) var<storage, read> live_edge_idx: array<u32>;

// Aggregated cross-boundary edges, three u32 per bundle:
// [visible_source, visible_target, weight]. Same table springs_simple reads.
@group(0) @binding(8) var<storage, read> bundles: array<u32>;

// Per-node simulation mass (1.0 = one body); a proxy carries the mass of the
// subtree it replaces. See inverse_mass.
@group(0) @binding(9) var<storage, read> node_mass: array<f32>;

const BUNDLE_STRIDE: u32 = 3u;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Reciprocal of the mass an arriving attraction is shared over. Clamped at one
// body so a hidden node's zero mass cannot divide, and returned as exactly 1.0
// there rather than computed, because a GPU's f32 divide may be approximate and
// the identity case has to be exact.
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

// Weight influence: w^δ (1.0 when δ=0, w when δ=1).
//
// A bundle has no source edge weight and passes 1.0, which this returns
// unchanged down every branch — pow(1.0, δ) is 1.0 for all δ — so a bundle's
// whole weighting is its edge count.
fn weight_factor(w: f32) -> f32 {
    if (uniforms.edge_weight_influence == 0.0) {
        return 1.0;
    }
    if (uniforms.edge_weight_influence == 1.0) {
        return w;
    }
    return pow(max(w, MIN_DISTANCE), uniforms.edge_weight_influence);
}

// LinLog attraction between two slots: F = w^δ * log(1 + d) * direction.
//
// `count` is how many source edges this call stands for — 1.0 for a real edge,
// the bundle weight for an aggregated one. It multiplies the magnitude, and
// that is exactly superposition: `count` coincident edges between the same pair
// pull `count` times as hard however the law bends with distance. What the
// logarithm does change is the approximation: the law is evaluated once at the
// proxy separation rather than `count` times at the separations of the edges it
// replaces, and log is concave, so a bundle standing for a spread-out subtree
// pulls slightly harder than the edges did. That is the same trade the linear
// model makes — a proxy is a point and a subtree is not — and it is bounded by
// the subtree's radius, which is exactly what the collapse decided was too
// small to be worth drawing.
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

    // Logarithmic attraction: F = w^δ * log(1 + d)
    let force_magnitude = factor * log(1.0 + dist) * count;
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
        let edge_idx = live_edge_idx[idx];
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
    let base = bundle_idx * BUNDLE_STRIDE;
    let src = bundles[base];
    let tgt = bundles[base + 1u];
    apply_attraction(
        src,
        tgt,
        weight_factor(1.0),
        f32(bundles[base + 2u]),
        inverse_mass(src),
        inverse_mass(tgt),
    );
}
