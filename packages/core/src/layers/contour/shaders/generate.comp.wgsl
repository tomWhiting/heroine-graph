// Contour Vertex Generation Compute Shader
//
// Third stage of marching squares: generates line segment vertices
// for each active cell using the case index and prefix sum offsets.

struct GenerateUniforms {
    // Texture dimensions
    width: u32,
    height: u32,
    // Density threshold
    threshold: f32,
    // Padding
    _padding: u32,
}

@group(0) @binding(0) var density_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: GenerateUniforms;
@group(0) @binding(2) var<storage, read> cell_cases: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_sums: array<u32>;
// Output vertices: each line segment is 4 floats (x1, y1, x2, y2)
@group(0) @binding(4) var<storage, read_write> vertices: array<f32>;

// Sample density at integer coordinates
fn sample_density(x: i32, y: i32) -> f32 {
    let clamped_x = clamp(x, 0, i32(uniforms.width) - 1);
    let clamped_y = clamp(y, 0, i32(uniforms.height) - 1);
    return textureLoad(density_texture, vec2<i32>(clamped_x, clamped_y), 0).r;
}

// Linear interpolation factor between two values crossing threshold
fn interp_factor(v0: f32, v1: f32, threshold: f32) -> f32 {
    let denom = v1 - v0;
    if (abs(denom) < 0.0001) {
        return 0.5;
    }
    return clamp((threshold - v0) / denom, 0.0, 1.0);
}

// Edge midpoint positions (normalized 0-1 within cell)
// Edge 0: bottom (between v0 and v1)
// Edge 1: right (between v1 and v2)
// Edge 2: top (between v2 and v3)
// Edge 3: left (between v3 and v0)
fn edge_point(edge: u32, v0: f32, v1: f32, v2: f32, v3: f32, threshold: f32) -> vec2<f32> {
    switch (edge) {
        case 0u: { // Bottom edge
            let t = interp_factor(v0, v1, threshold);
            return vec2<f32>(t, 0.0);
        }
        case 1u: { // Right edge
            let t = interp_factor(v1, v2, threshold);
            return vec2<f32>(1.0, t);
        }
        case 2u: { // Top edge
            let t = interp_factor(v3, v2, threshold);
            return vec2<f32>(t, 1.0);
        }
        case 3u: { // Left edge
            let t = interp_factor(v0, v3, threshold);
            return vec2<f32>(0.0, t);
        }
        default: {
            return vec2<f32>(0.5, 0.5);
        }
    }
}

// Marching squares edge table
// For each case, defines which edges to connect
// Format: [edge1_start, edge1_end, edge2_start, edge2_end]
// Use 255 for unused segments
fn get_edges(case_index: u32) -> vec4<u32> {
    // Edge indices: 0=bottom, 1=right, 2=top, 3=left
    switch (case_index) {
        case 0u:  { return vec4<u32>(255u, 255u, 255u, 255u); } // No contour
        case 1u:  { return vec4<u32>(0u, 3u, 255u, 255u); }     // Bottom-left corner
        case 2u:  { return vec4<u32>(0u, 1u, 255u, 255u); }     // Bottom-right corner
        case 3u:  { return vec4<u32>(3u, 1u, 255u, 255u); }     // Bottom edge
        case 4u:  { return vec4<u32>(1u, 2u, 255u, 255u); }     // Top-right corner
        case 5u:  { return vec4<u32>(0u, 3u, 1u, 2u); }         // Diagonal (ambiguous)
        case 6u:  { return vec4<u32>(0u, 2u, 255u, 255u); }     // Right edge
        case 7u:  { return vec4<u32>(3u, 2u, 255u, 255u); }     // Top-left corner
        case 8u:  { return vec4<u32>(2u, 3u, 255u, 255u); }     // Top-left corner
        case 9u:  { return vec4<u32>(0u, 2u, 255u, 255u); }     // Left edge
        case 10u: { return vec4<u32>(0u, 1u, 2u, 3u); }         // Diagonal (ambiguous)
        case 11u: { return vec4<u32>(1u, 2u, 255u, 255u); }     // Top-right corner
        case 12u: { return vec4<u32>(1u, 3u, 255u, 255u); }     // Top edge
        case 13u: { return vec4<u32>(0u, 1u, 255u, 255u); }     // Bottom-right corner
        case 14u: { return vec4<u32>(0u, 3u, 255u, 255u); }     // Bottom-left corner
        case 15u: { return vec4<u32>(255u, 255u, 255u, 255u); } // All inside
        default:  { return vec4<u32>(255u, 255u, 255u, 255u); }
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_width = uniforms.width - 1u;
    let grid_height = uniforms.height - 1u;
    let total_cells = grid_width * grid_height;

    let cell_index = global_id.x;
    if (cell_index >= total_cells) {
        return;
    }

    let case_index = cell_cases[cell_index];

    // Skip cells with no segments
    if (case_index == 0u || case_index == 15u) {
        return;
    }

    // Get cell position
    let cell_x = cell_index % grid_width;
    let cell_y = cell_index / grid_width;

    // Sample corner densities
    let x = i32(cell_x);
    let y = i32(cell_y);
    let v0 = sample_density(x, y);
    let v1 = sample_density(x + 1, y);
    let v2 = sample_density(x + 1, y + 1);
    let v3 = sample_density(x, y + 1);

    // Get edge connections for this case
    let edges = get_edges(case_index);

    // Get output offset from prefix sum
    let output_offset = prefix_sums[cell_index];

    // Generate first segment (always present for active cells)
    if (edges.x != 255u) {
        let p0 = edge_point(edges.x, v0, v1, v2, v3, uniforms.threshold);
        let p1 = edge_point(edges.y, v0, v1, v2, v3, uniforms.threshold);

        // Convert to texture coordinates (0-1)
        let tex_x0 = (f32(cell_x) + p0.x) / f32(grid_width);
        let tex_y0 = (f32(cell_y) + p0.y) / f32(grid_height);
        let tex_x1 = (f32(cell_x) + p1.x) / f32(grid_width);
        let tex_y1 = (f32(cell_y) + p1.y) / f32(grid_height);

        // Write to output buffer (4 floats per segment)
        let base = output_offset * 4u;
        vertices[base + 0u] = tex_x0;
        vertices[base + 1u] = tex_y0;
        vertices[base + 2u] = tex_x1;
        vertices[base + 3u] = tex_y1;
    }

    // Generate second segment (for saddle point cases 5 and 10)
    if (edges.z != 255u) {
        let p0 = edge_point(edges.z, v0, v1, v2, v3, uniforms.threshold);
        let p1 = edge_point(edges.w, v0, v1, v2, v3, uniforms.threshold);

        let tex_x0 = (f32(cell_x) + p0.x) / f32(grid_width);
        let tex_y0 = (f32(cell_y) + p0.y) / f32(grid_height);
        let tex_x1 = (f32(cell_x) + p1.x) / f32(grid_width);
        let tex_y1 = (f32(cell_y) + p1.y) / f32(grid_height);

        // Write to output buffer (next segment)
        let base = (output_offset + 1u) * 4u;
        vertices[base + 0u] = tex_x0;
        vertices[base + 1u] = tex_y0;
        vertices[base + 2u] = tex_x1;
        vertices[base + 3u] = tex_y1;
    }
}
