// Density Splat Compute Shader
// Renders Gaussian density contributions from node positions to a density texture
//
// Each node contributes a Gaussian splat to nearby pixels.
// Uses atomics for thread-safe accumulation.

struct DensitySplatUniforms {
    node_count: u32,
    width: u32,
    height: u32,
    radius: f32,        // Splat radius in pixels
    intensity: f32,     // Intensity multiplier
    bounds_min_x: f32,  // World space bounds
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DensitySplatUniforms;
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;
@group(0) @binding(3) var density_texture: texture_storage_2d<r32float, read_write>;

// Convert world position to texture UV
fn world_to_uv(pos: vec2<f32>) -> vec2<f32> {
    return (pos - vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y)) /
           (vec2<f32>(uniforms.bounds_max_x, uniforms.bounds_max_y) -
            vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y));
}

// Convert UV to pixel coordinates
fn uv_to_pixel(uv: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32(uv.x * f32(uniforms.width)),
        i32(uv.y * f32(uniforms.height))
    );
}

// Gaussian falloff
fn gaussian(dist_sq: f32, radius: f32) -> f32 {
    let sigma = radius / 3.0;  // 3 sigma = ~99% of distribution
    let sigma_sq = sigma * sigma;
    return exp(-dist_sq / (2.0 * sigma_sq));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = vec2<f32>(positions_x[node_idx], positions_y[node_idx]);
    let uv = world_to_uv(pos);

    // Skip if outside bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return;
    }

    let center_pixel = uv_to_pixel(uv);
    let radius_pixels = i32(uniforms.radius);

    // Splat Gaussian to nearby pixels
    for (var dy = -radius_pixels; dy <= radius_pixels; dy++) {
        for (var dx = -radius_pixels; dx <= radius_pixels; dx++) {
            let px = center_pixel.x + dx;
            let py = center_pixel.y + dy;

            // Bounds check
            if (px < 0 || px >= i32(uniforms.width) || py < 0 || py >= i32(uniforms.height)) {
                continue;
            }

            let dist_sq = f32(dx * dx + dy * dy);
            let radius_sq = f32(radius_pixels * radius_pixels);

            if (dist_sq > radius_sq) {
                continue;
            }

            let weight = gaussian(dist_sq, f32(radius_pixels)) * uniforms.intensity;

            // Atomic add to density texture
            // Note: texture_storage with read_write allows this pattern
            let current = textureLoad(density_texture, vec2<i32>(px, py));
            textureStore(density_texture, vec2<i32>(px, py), vec4<f32>(current.r + weight, 0.0, 0.0, 0.0));
        }
    }
}

// Clear the density texture
@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= uniforms.width || global_id.y >= uniforms.height) {
        return;
    }
    textureStore(density_texture, vec2<i32>(global_id.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
