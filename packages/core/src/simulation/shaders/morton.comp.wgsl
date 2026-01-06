// Morton Code Generation Compute Shader
// Converts 2D positions to Morton codes (Z-order curve) for spatial sorting
//
// Morton codes interleave bits of x and y coordinates, creating a space-filling
// curve that preserves locality. Nodes that are spatially close will have
// similar Morton codes, making them adjacent after sorting.

struct SimulationUniforms {
    bounds_min: vec2<f32>,      // Bounding box minimum
    bounds_max: vec2<f32>,      // Bounding box maximum
    node_count: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: SimulationUniforms;
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> morton_codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> node_indices: array<u32>;

// Number of bits for each coordinate (16 bits each = 32-bit Morton code)
const MORTON_BITS: u32 = 16u;
const MORTON_SCALE: f32 = 65535.0;  // 2^16 - 1

// Expand bits by inserting zeros between each bit
// 0000 0000 0000 0000 abcd efgh ijkl mnop
// becomes:
// 0a0b 0c0d 0e0f 0g0h 0i0j 0k0l 0m0n 0o0p
fn expand_bits(v: u32) -> u32 {
    var x = v & 0x0000FFFFu;  // Only use lower 16 bits
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

// Generate Morton code by interleaving bits of x and y
fn morton2d(x: u32, y: u32) -> u32 {
    return expand_bits(x) | (expand_bits(y) << 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= uniforms.node_count) {
        return;
    }

    // Get node position
    let pos = vec2<f32>(positions_x[idx], positions_y[idx]);

    // Normalize to [0, 1] range within bounding box
    let bounds_size = uniforms.bounds_max - uniforms.bounds_min;
    let safe_size = max(bounds_size, vec2<f32>(1.0));  // Avoid division by zero
    let normalized = (pos - uniforms.bounds_min) / safe_size;

    // Clamp to valid range
    let clamped = clamp(normalized, vec2<f32>(0.0), vec2<f32>(1.0));

    // Scale to integer range
    let scaled_x = u32(clamped.x * MORTON_SCALE);
    let scaled_y = u32(clamped.y * MORTON_SCALE);

    // Generate Morton code
    morton_codes[idx] = morton2d(scaled_x, scaled_y);

    // Initialize node indices (will be sorted alongside Morton codes)
    node_indices[idx] = idx;
}
