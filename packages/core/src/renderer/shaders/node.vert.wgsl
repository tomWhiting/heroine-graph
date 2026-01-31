// Node Vertex Shader
// Renders nodes as instanced quads with circles via SDF in fragment shader

// Viewport uniforms for coordinate transformation
struct ViewportUniforms {
    transform_col0: vec4<f32>,
    transform_col1: vec4<f32>,
    transform_col2: vec4<f32>,
    screen_size: vec2<f32>,
    scale: f32,
    inv_scale: f32,
    _padding: vec2<f32>,
}

// Per-node attributes for rendering
struct NodeAttributes {
    // Packed: radius (f32), color RGB (3 x u8), flags (u8)
    radius: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    selected: f32,  // 0.0 or 1.0
    hovered: f32,   // 0.0 or 1.0
}

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;

// Position buffer (vec2 layout)
@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;

// Node attributes buffer
@group(1) @binding(1) var<storage, read> node_attrs: array<f32>;

// Vertex output to fragment shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,           // Local UV for SDF
    @location(1) color: vec3<f32>,         // Node color
    @location(2) radius_px: f32,           // Radius in pixels for AA
    @location(3) state: vec2<f32>,         // (selected, hovered)
}

// Quad vertices for instanced rendering
const QUAD_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

// Transform graph position to clip space
fn transform_point(pos: vec2<f32>) -> vec2<f32> {
    let col0 = viewport.transform_col0.xyz;
    let col1 = viewport.transform_col1.xyz;
    let col2 = viewport.transform_col2.xyz;
    return vec2<f32>(
        col0.x * pos.x + col1.x * pos.y + col2.x,
        col0.y * pos.x + col1.y * pos.y + col2.y
    );
}

// Default node radius in graph units
const DEFAULT_RADIUS: f32 = 5.0;
// Minimum radius in pixels (for visibility at extreme zoom-out)
const MIN_RADIUS_PX: f32 = 2.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    var output: VertexOutput;

    // Get node position from vec2 buffer
    let node_pos = positions[instance_idx];

    // Read node attributes (6 floats per node)
    let attr_base = instance_idx * 6u;
    let radius = node_attrs[attr_base];
    let color_r = node_attrs[attr_base + 1u];
    let color_g = node_attrs[attr_base + 2u];
    let color_b = node_attrs[attr_base + 3u];
    let selected = node_attrs[attr_base + 4u];
    let hovered = node_attrs[attr_base + 5u];

    // Use default radius if not specified (0 or NaN)
    let actual_radius = select(DEFAULT_RADIUS, radius, radius > 0.0);

    // Calculate radius in screen pixels
    let radius_px = max(actual_radius * viewport.scale, MIN_RADIUS_PX);

    // Get quad vertex offset
    let quad_vertex = QUAD_VERTICES[vertex_idx % 6u];

    // Transform node center to clip space
    let center_clip = transform_point(node_pos);

    // Calculate vertex offset in clip space
    // Need to convert radius from pixels to clip space units
    let offset_clip = quad_vertex * (radius_px * 2.0 / viewport.screen_size);

    // Final vertex position (with slight expansion for AA)
    let aa_padding = 1.5 / radius_px; // Anti-aliasing padding
    output.position = vec4<f32>(
        center_clip + offset_clip * (1.0 + aa_padding),
        0.0,
        1.0
    );

    // Pass UV for SDF circle rendering
    output.uv = quad_vertex * (1.0 + aa_padding);

    // Pass color
    output.color = vec3<f32>(color_r, color_g, color_b);

    // Pass radius for AA calculations in fragment shader
    output.radius_px = radius_px;

    // Pass selection/hover state
    output.state = vec2<f32>(selected, hovered);

    return output;
}
