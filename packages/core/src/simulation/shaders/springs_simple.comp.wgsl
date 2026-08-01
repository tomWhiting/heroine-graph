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
    _padding: u32,
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    let source_idx = edge_sources[edge_idx];
    let target_idx = edge_targets[edge_idx];

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
    let force_magnitude = uniforms.spring_strength * displacement;

    let force = dir * force_magnitude;

    // Add force to source (attractive toward target)
    // Subtract from target (attractive toward source)
    accumulate_force(source_idx, force);
    accumulate_force(target_idx, -force);
}
