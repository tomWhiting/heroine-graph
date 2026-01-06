// Edge Vertex Shader
// Renders edges as thick lines with anti-aliased edges

struct ViewportUniforms {
    transform_col0: vec4<f32>,
    transform_col1: vec4<f32>,
    transform_col2: vec4<f32>,
    screen_size: vec2<f32>,
    scale: f32,
    inv_scale: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;

// Node positions (needed to get edge endpoints)
@group(1) @binding(0) var<storage, read> positions_x: array<f32>;
@group(1) @binding(1) var<storage, read> positions_y: array<f32>;

// Edge data: pairs of (source_idx, target_idx) packed as u32
@group(1) @binding(2) var<storage, read> edge_indices: array<u32>;

// Edge attributes (width, color, selected, hovered)
@group(1) @binding(3) var<storage, read> edge_attrs: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,        // For line distance
    @location(1) color: vec3<f32>,      // Edge color
    @location(2) half_width: f32,       // Half line width in pixels
    @location(3) state: vec2<f32>,      // (selected, hovered)
}

// Each edge is rendered as a quad (2 triangles, 6 vertices)
const QUAD_OFFSETS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0),  // Bottom-left
    vec2<f32>(1.0, -1.0),  // Bottom-right
    vec2<f32>(1.0,  1.0),  // Top-right
    vec2<f32>(0.0, -1.0),  // Bottom-left
    vec2<f32>(1.0,  1.0),  // Top-right
    vec2<f32>(0.0,  1.0),  // Top-left
);

fn transform_point(pos: vec2<f32>) -> vec2<f32> {
    let col0 = viewport.transform_col0.xyz;
    let col1 = viewport.transform_col1.xyz;
    let col2 = viewport.transform_col2.xyz;
    return vec2<f32>(
        col0.x * pos.x + col1.x * pos.y + col2.x,
        col0.y * pos.x + col1.y * pos.y + col2.y
    );
}

// Default edge width in pixels
const DEFAULT_WIDTH: f32 = 1.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    var output: VertexOutput;

    // Get edge endpoints
    let edge_base = instance_idx * 2u;
    let source_idx = edge_indices[edge_base];
    let target_idx = edge_indices[edge_base + 1u];

    let source_pos = vec2<f32>(
        positions_x[source_idx],
        positions_y[source_idx]
    );
    let target_pos = vec2<f32>(
        positions_x[target_idx],
        positions_y[target_idx]
    );

    // Read edge attributes (6 floats per edge)
    let attr_base = instance_idx * 6u;
    let width = edge_attrs[attr_base];
    let color_r = edge_attrs[attr_base + 1u];
    let color_g = edge_attrs[attr_base + 2u];
    let color_b = edge_attrs[attr_base + 3u];
    let selected = edge_attrs[attr_base + 4u];
    let hovered = edge_attrs[attr_base + 5u];

    // Use default width if not specified
    let actual_width = select(DEFAULT_WIDTH, width, width > 0.0);
    let half_width = actual_width * 0.5;

    // Transform endpoints to clip space
    let source_clip = transform_point(source_pos);
    let target_clip = transform_point(target_pos);

    // Calculate edge direction in clip space
    let edge_dir = target_clip - source_clip;
    let edge_length = length(edge_dir);

    // Handle degenerate edges
    if (edge_length < 0.0001) {
        output.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);  // Behind camera
        return output;
    }

    let edge_unit = edge_dir / edge_length;

    // Perpendicular direction for width
    let perp = vec2<f32>(-edge_unit.y, edge_unit.x);

    // Get quad vertex offset
    let quad_offset = QUAD_OFFSETS[vertex_idx % 6u];

    // Calculate position along edge (0 = source, 1 = target)
    let t = quad_offset.x;
    let base_pos = mix(source_clip, target_clip, t);

    // Add width offset (perpendicular)
    // Convert width from pixels to clip space
    let width_clip = half_width * 2.0 / viewport.screen_size;
    let aa_padding = 1.5;  // Extra pixels for anti-aliasing
    let total_width = width_clip + vec2<f32>(aa_padding * 2.0 / viewport.screen_size);
    let offset = perp * total_width * quad_offset.y;

    output.position = vec4<f32>(base_pos + offset, 0.0, 1.0);

    // UV for line distance calculation
    // x: position along line (0-1)
    // y: perpendicular distance in pixels
    output.uv = vec2<f32>(t, quad_offset.y * (half_width + aa_padding));

    output.color = vec3<f32>(color_r, color_g, color_b);
    output.half_width = half_width;
    output.state = vec2<f32>(selected, hovered);

    return output;
}
