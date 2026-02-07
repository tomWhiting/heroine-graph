// Tidy Tree Layout Force Compute Shader
// Applies spring forces pulling nodes toward pre-computed target positions.
// Target positions are computed by the Buchheim O(n) tidy tree algorithm
// running in WASM, then uploaded to a GPU storage buffer.

struct TreeUniforms {
    node_count: u32,
    stiffness: f32,     // Spring strength toward target (0-1, typically 0.1-0.5)
    damping: f32,       // Reduce force as node approaches target (0-1)
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: TreeUniforms;

// Current node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators (read-write) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Target positions from tidy tree layout (read only) - vec2<f32> per node
@group(0) @binding(3) var<storage, read> target_positions: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let current_pos = positions[node_idx];
    let target_pos = target_positions[node_idx];

    // Skip non-tree nodes marked with sentinel value (~f32::MAX = 3.4e+38).
    // The WASM layout sets non-tree node positions to this sentinel.
    // Using x >= threshold since no real layout position would be this large.
    if (target_pos.x >= 3.0e+38) {
        return;
    }

    // Spring force toward target position
    let delta = target_pos - current_pos;
    let dist = length(delta);

    // Apply damping: reduce force as node gets close to target
    // This prevents oscillation around the target
    let damped_strength = uniforms.stiffness * (1.0 - exp(-dist * uniforms.damping));

    let force = delta * damped_strength;

    forces[node_idx] += force;
}
