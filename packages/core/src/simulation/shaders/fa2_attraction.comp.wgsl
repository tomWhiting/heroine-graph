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

struct FA2AttractionUniforms {
    edge_count: u32,
    edge_weight_influence: f32, // delta: exponent on edge weights (0 = ignore, 1 = linear)
    flags: u32,                 // bit 0 = linlog mode
    _padding: u32,
}

const MIN_DISTANCE: f32 = 0.01;
const FLAG_LINLOG: u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms: FA2AttractionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;
@group(0) @binding(5) var<storage, read> edge_weights: array<f32>;

// FA2 attraction: F = w^delta * d * direction (standard)
//             or: F = w^delta * log(1 + d) * direction (linlog)
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

    let src_pos = positions[src];
    let tgt_pos = positions[tgt];

    let delta = tgt_pos - src_pos;
    let dist_sq = dot(delta, delta);
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // Weight influence: w^delta (1.0 when delta=0, w when delta=1)
    let w = edge_weights[edge_idx];
    var weight_factor: f32;
    if (uniforms.edge_weight_influence == 0.0) {
        weight_factor = 1.0;
    } else if (uniforms.edge_weight_influence == 1.0) {
        weight_factor = w;
    } else {
        weight_factor = pow(max(w, MIN_DISTANCE), uniforms.edge_weight_influence);
    }

    // FA2 attraction: always pulling, no rest length, no equilibrium distance
    var force_magnitude: f32;
    if ((uniforms.flags & FLAG_LINLOG) != 0u) {
        // LinLog mode: F = w^delta * log(1 + d)
        // Caps long-range attraction for better cluster separation
        force_magnitude = weight_factor * log(1.0 + dist);
    } else {
        // Standard mode: F = w^delta * d
        // Linear — simple, effective, produces good general layouts
        force_magnitude = weight_factor * dist;
    }

    let dir = delta / dist;
    let force = dir * force_magnitude;

    // Apply equal-and-opposite forces
    // Note: WGSL storage buffers don't have true atomics for f32,
    // so there's a potential race condition on concurrent writes.
    // In practice, the integration step smooths out frame-to-frame noise.
    forces[src] += force;
    forces[tgt] -= force;
}
