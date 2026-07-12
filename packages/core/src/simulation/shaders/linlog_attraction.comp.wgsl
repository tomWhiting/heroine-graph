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

struct LinLogUniforms {
    node_count: u32,
    edge_count: u32,
    kr: f32,                    // Repulsion scaling (unused here)
    kg: f32,                    // Gravity strength (unused here)
    edge_weight_influence: f32, // δ: exponent on edge weights
    flags: u32,                 // bit 0 = strong_gravity (unused here)
    _padding: vec2<u32>,
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
// Node state flags (bit 0 = dead slot from removal)
@group(0) @binding(6) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;

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

// LinLog attraction: F = w^δ * log(1 + d) * direction
// Where w = edge weight, δ = edge_weight_influence, d = distance between endpoints.
//
// Applied symmetrically: source pulled toward target, target pulled toward source.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    let src = edge_sources[edge_idx];
    let tgt = edge_targets[edge_idx];

    // Edges touching dead slots (holes from removals) exert no force
    if (((node_flags[src] | node_flags[tgt]) & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let src_pos = positions[src];
    let tgt_pos = positions[tgt];

    let delta = tgt_pos - src_pos;
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // Weight influence: w^δ (1.0 when δ=0, w when δ=1)
    let w = edge_weights[edge_idx];
    var weight_factor: f32;
    if (uniforms.edge_weight_influence == 0.0) {
        weight_factor = 1.0;
    } else if (uniforms.edge_weight_influence == 1.0) {
        weight_factor = w;
    } else {
        weight_factor = pow(max(w, MIN_DISTANCE), uniforms.edge_weight_influence);
    }

    // Logarithmic attraction: F = w^δ * log(1 + d)
    let force_magnitude = weight_factor * log(1.0 + dist);
    let dir = delta / dist;
    let force = dir * force_magnitude;

    // Apply equal-and-opposite forces (race-free, see atomic_add_f32)
    accumulate_force(src, force);
    accumulate_force(tgt, -force);
}
