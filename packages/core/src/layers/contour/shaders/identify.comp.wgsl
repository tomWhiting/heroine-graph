// Contour Active Cell Identification Compute Shader
//
// First stage of marching squares: identifies which cells cross
// the isosurface threshold and computes their case index (0-15).

struct ContourUniforms {
    // Texture dimensions
    width: u32,
    height: u32,
    // Density threshold for this contour level (normalized 0-1)
    threshold: f32,
    // Maximum density value for normalization
    max_density: f32,
}

@group(0) @binding(0) var density_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: ContourUniforms;
// Output: case index per cell (0-15), or 255 if cell doesn't cross threshold
@group(0) @binding(2) var<storage, read_write> cell_cases: array<u32>;
// Output: count of active cells (for prefix sum)
@group(0) @binding(3) var<storage, read_write> active_count: atomic<u32>;

// Sample density at integer coordinates (normalized 0-1)
fn sample_density(x: i32, y: i32) -> f32 {
    let clamped_x = clamp(x, 0, i32(uniforms.width) - 1);
    let clamped_y = clamp(y, 0, i32(uniforms.height) - 1);
    let raw_density = textureLoad(density_texture, vec2<i32>(clamped_x, clamped_y), 0).r;
    // Normalize to 0-1 range using max_density
    return clamp(raw_density / max(uniforms.max_density, 0.001), 0.0, 1.0);
}

// Compute marching squares case index from 4 corner values
fn compute_case(v0: f32, v1: f32, v2: f32, v3: f32, threshold: f32) -> u32 {
    var case_index: u32 = 0u;

    // Bottom-left
    if (v0 >= threshold) {
        case_index |= 1u;
    }
    // Bottom-right
    if (v1 >= threshold) {
        case_index |= 2u;
    }
    // Top-right
    if (v2 >= threshold) {
        case_index |= 4u;
    }
    // Top-left
    if (v3 >= threshold) {
        case_index |= 8u;
    }

    return case_index;
}

// Number of line segments for each case (0-15)
// Cases 0 and 15 have 0 segments (all inside or all outside)
fn segments_for_case(case_index: u32) -> u32 {
    // Lookup table for segment counts
    // Case: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    //       0  1  1  1  1  2  1  1  1  1  2  1  1  1  1  0
    let counts = array<u32, 16>(
        0u, 1u, 1u, 1u, 1u, 2u, 1u, 1u,
        1u, 1u, 2u, 1u, 1u, 1u, 1u, 0u
    );
    return counts[case_index];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_x = global_id.x;
    let cell_y = global_id.y;

    // Grid is (width-1) x (height-1) cells
    let grid_width = uniforms.width - 1u;
    let grid_height = uniforms.height - 1u;

    if (cell_x >= grid_width || cell_y >= grid_height) {
        return;
    }

    let cell_index = cell_y * grid_width + cell_x;

    // Sample density at 4 corners of this cell
    let x = i32(cell_x);
    let y = i32(cell_y);

    let v0 = sample_density(x, y);           // Bottom-left
    let v1 = sample_density(x + 1, y);       // Bottom-right
    let v2 = sample_density(x + 1, y + 1);   // Top-right
    let v3 = sample_density(x, y + 1);       // Top-left

    // Compute case index
    let case_index = compute_case(v0, v1, v2, v3, uniforms.threshold);

    // Store case index
    cell_cases[cell_index] = case_index;

    // Count active cells (those that will generate segments)
    let segment_count = segments_for_case(case_index);
    if (segment_count > 0u) {
        atomicAdd(&active_count, segment_count);
    }
}
