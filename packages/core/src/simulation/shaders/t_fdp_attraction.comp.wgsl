// t-FDP Attraction Compute Shader
//
// Implements the full attractive force from the t-FDP model (Zhong et al.):
//   F_a(i,j) = [ alpha * d + beta * d / (1 + d^2) ] * direction
//
// Two components combined:
//   1. Linear spring: alpha * d * dir (standard Hooke's law with rest length 0)
//   2. Attractive t-force: beta * d / (1 + d^2) * dir (short-range boost)
//
// The attractive t-force (component 2) is the key innovation: it adds a bounded
// short-range pull between connected nodes that decays at long range. This makes
// connected nodes cluster together more tightly, satisfying the paper's principle
// P3: connected nodes should be closer than unconnected nodes.
//
// The paper recommends alpha=0.1, beta=8.0, and the constraint alpha*(1+beta) < 1
// must hold for proper force balance (repulsion dominates at zero distance).
//
// Distances are normalized by dist_scale — the SAME normalization the t_fdp
// repulsion shader applies — so attraction and repulsion balance at the
// normalized equilibrium the paper derives (d ≈ O(1)) instead of at raw
// world-unit distances.
//
// This shader operates per-edge: each thread processes one edge and applies
// equal-and-opposite forces to the source and target nodes.
//
// Two entry points share the edge math:
// - main:         one thread per source edge, bindings 0-5. The un-cut path,
//                 unchanged: it passes compile-time 1.0s for every LOD term.
// - main_bundled: one thread per entry of the LOD active-edge list, then one
//                 per aggregated bundle; bindings 0-7.
//
// Uses vec2<f32> layout for consolidated position/force data.

struct TFdpAttractionUniforms {
    edge_count: u32,
    alpha: f32,          // Linear spring weight (paper default: 0.1)
    beta: f32,           // Attractive t-force weight (paper default: 8.0)
    dist_scale: f32,     // world units per t-kernel unit (matches t_fdp)
    // Entries of lod_edge_set's live-edge region the bundled entry point
    // gathers, and the number of bundles after them. The first also bases the
    // bundle region (see bundle_base). Both zero unless an LOD cut is live;
    // `main` never reads either, which keeps the un-cut path unchanged.
    active_edge_count: u32,
    bundle_count: u32,
    _padding0: u32,
    _padding1: u32,
}

const MIN_DISTANCE: f32 = 0.0001;

@group(0) @binding(0) var<uniform> uniforms: TFdpAttractionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
// Force accumulators — the shared vec2<f32> buffer viewed as f32 bit patterns
// behind atomic<u32> (node i -> [2i]=x, [2i+1]=y) so per-edge threads can
// accumulate without lost updates on hub nodes. Same buffer, same bytes —
// other passes keep their array<vec2<f32>> view.
@group(0) @binding(2) var<storage, read_write> forces: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;
// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(5) var<storage, read> node_flags: array<u32>;

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
// The same buffer, map and contents springs_simple.comp.wgsl reads. The host
// sizes and fills it in simulation/pipeline.ts (uploadEdgeBundles,
// LOD_EDGE_SET_WORDS_PER_EDGE), where the same map is stated.
@group(0) @binding(6) var<storage, read> lod_edge_set: array<u32>;

// Per-node simulation mass (1.0 = one body); a proxy carries the mass of the
// subtree it replaces. See inverse_mass.
@group(0) @binding(7) var<storage, read> node_mass: array<f32>;

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

// t-FDP attraction between two slots:
//   F = [ alpha * d + beta * d / (1 + d^2) ] * direction
//
// Component 1 (linear spring): pulls proportional to distance — long-range structure
// Component 2 (t-force): bounded short-range boost — neighborhood preservation
//
// `count` is how many source edges this call stands for — 1.0 for a real edge,
// the bundle weight for an aggregated one. It multiplies the magnitude, and
// that is exactly superposition: `count` coincident edges between the same pair
// pull `count` times as hard however the law bends with distance. The t-force
// component is the one that makes the approximation visible: it is the term
// that decays with separation, so a bundle replacing short intra-cluster edges
// with one long proxy-to-proxy edge loses most of it. That is the intended
// reading — the short-range term exists to preserve neighbourhoods, and a
// collapsed subtree's neighbourhood is no longer on screen — while the linear
// term, which carries the long-range structure the visible layout is made of,
// aggregates exactly.
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
    count: f32,
    source_inv_mass: f32,
    target_inv_mass: f32,
) {
    // Edges touching an inert slot — dead or LOD-hidden — exert no force. The
    // mutation paths cascade-remove a node's edges before flagging its slot
    // dead, so this should never fire — the guard matches the other per-edge
    // attraction kernels (springs_simple, fa2_attraction, linlog_attraction)
    // so a dangling endpoint can never silently yank a live node to the origin.
    if (((node_flags[src] | node_flags[tgt]) & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let src_pos = positions[src];
    let tgt_pos = positions[tgt];

    let delta = tgt_pos - src_pos;
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    let dir = delta / dist;

    // Normalized distance (same scale as the repulsion kernel)
    let d = dist / uniforms.dist_scale;

    // Component 1: Linear spring (rest length = 0)
    //   F_spring = alpha * d * direction
    let spring_force = uniforms.alpha * d;

    // Component 2: Attractive t-force (phi = 1 per paper)
    //   F_tforce = beta * d / (1 + d^2) * direction
    let t_force = uniforms.beta * d / (1.0 + d * d);

    // Combined attractive force magnitude
    let force_magnitude = (spring_force + t_force) * count;
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

    apply_attraction(edge_sources[edge_idx], edge_targets[edge_idx], 1.0, 1.0, 1.0);
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
        apply_attraction(src, tgt, 1.0, inverse_mass(src), inverse_mass(tgt));
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
        f32(lod_edge_set[base + 2u]),
        inverse_mass(src),
        inverse_mass(tgt),
    );
}
