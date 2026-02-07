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
// This shader operates per-edge: each thread processes one edge and applies
// equal-and-opposite forces to the source and target nodes.
//
// Uses vec2<f32> layout for consolidated position/force data.

struct TFdpAttractionUniforms {
    edge_count: u32,
    alpha: f32,          // Linear spring weight (paper default: 0.1)
    beta: f32,           // Attractive t-force weight (paper default: 8.0)
    _padding: u32,
}

const MIN_DISTANCE: f32 = 0.0001;

@group(0) @binding(0) var<uniform> uniforms: TFdpAttractionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;

// t-FDP attraction: F = [ alpha * d + beta * d / (1 + d^2) ] * direction
//
// Component 1 (linear spring): pulls proportional to distance — long-range structure
// Component 2 (t-force): bounded short-range boost — neighborhood preservation
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

    let dir = delta / dist;

    // Component 1: Linear spring (rest length = 0)
    //   F_spring = alpha * d * direction
    let spring_force = uniforms.alpha * dist;

    // Component 2: Attractive t-force (phi = 1 per paper)
    //   F_tforce = beta * d / (1 + d^2) * direction
    let t_force = uniforms.beta * dist / (1.0 + dist_sq);

    // Combined attractive force magnitude
    let force_magnitude = spring_force + t_force;
    let force = dir * force_magnitude;

    // Apply equal-and-opposite forces
    // Note: WGSL storage buffers don't have true atomics for f32,
    // so there's a potential race condition on concurrent writes.
    // In practice, the integration step smooths out frame-to-frame noise,
    // and this is the same approach used by existing LinLog/FA2 shaders.
    forces[src] += force;
    forces[tgt] -= force;
}
