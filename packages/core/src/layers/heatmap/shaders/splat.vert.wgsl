// Gaussian Splat Vertex Shader
//
// Renders point-sprite quads for each node, which will be blurred
// by the fragment shader to create smooth density falloff.

struct ViewportUniforms {
    // Transform matrix (3x3 stored as 3 vec4s for alignment)
    transform_col0: vec4<f32>,
    transform_col1: vec4<f32>,
    transform_col2: vec4<f32>,
    // Screen dimensions
    screen_size: vec2<f32>,
    // Scale factor
    scale: f32,
    inv_scale: f32,
    dpr: f32,
    _padding: f32,
}

struct HeatmapUniforms {
    // Splat radius in graph units
    radius: f32,
    // Intensity multiplier
    intensity: f32,
    // Padding
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;
@group(0) @binding(1) var<uniform> heatmap: HeatmapUniforms;

@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;
// Per-node intensity values from value stream (optional - defaults to 1.0 for all nodes)
@group(1) @binding(1) var<storage, read> node_intensities: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) intensity: f32,
}

// Apply viewport transform to get clip coordinates
fn transform_point(pos: vec2<f32>) -> vec2<f32> {
    let col0 = viewport.transform_col0.xyz;
    let col1 = viewport.transform_col1.xyz;
    let col2 = viewport.transform_col2.xyz;

    let x = col0.x * pos.x + col1.x * pos.y + col2.x;
    let y = col0.y * pos.x + col1.y * pos.y + col2.y;

    return vec2<f32>(x, y);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // Get node position
    let node_pos = positions[instance_index];

    // Calculate splat size in screen space
    let screen_radius = heatmap.radius * viewport.scale;

    // Generate quad vertices (2 triangles, 6 vertices per instance)
    // Vertex order: 0,1,2, 2,1,3 for two triangles
    var quad_offset: vec2<f32>;
    var uv: vec2<f32>;

    switch vertex_index % 6u {
        case 0u: { // Top-left
            quad_offset = vec2<f32>(-1.0, -1.0);
            uv = vec2<f32>(0.0, 0.0);
        }
        case 1u: { // Top-right
            quad_offset = vec2<f32>(1.0, -1.0);
            uv = vec2<f32>(1.0, 0.0);
        }
        case 2u: { // Bottom-left
            quad_offset = vec2<f32>(-1.0, 1.0);
            uv = vec2<f32>(0.0, 1.0);
        }
        case 3u: { // Bottom-left (second triangle)
            quad_offset = vec2<f32>(-1.0, 1.0);
            uv = vec2<f32>(0.0, 1.0);
        }
        case 4u: { // Top-right (second triangle)
            quad_offset = vec2<f32>(1.0, -1.0);
            uv = vec2<f32>(1.0, 0.0);
        }
        case 5u, default: { // Bottom-right
            quad_offset = vec2<f32>(1.0, 1.0);
            uv = vec2<f32>(1.0, 1.0);
        }
    }

    // Transform node center to clip space
    let center_clip = transform_point(node_pos);

    // Add quad offset in screen space (convert radius to clip space)
    let offset_clip = quad_offset * screen_radius * 2.0 / viewport.screen_size;

    let final_pos = center_clip + offset_clip;

    // Get per-node intensity from stream data
    // When using default buffer (single 1.0), all nodes safely read index 0
    // When using stream data, each node gets its own intensity value
    let buffer_len = arrayLength(&node_intensities);
    let intensity_index = select(0u, instance_index, buffer_len > 1u);
    let node_intensity = node_intensities[intensity_index];

    var output: VertexOutput;
    output.position = vec4<f32>(final_pos, 0.0, 1.0);
    output.uv = uv;
    // Final intensity = per-node value × global multiplier
    output.intensity = node_intensity * heatmap.intensity;

    return output;
}
