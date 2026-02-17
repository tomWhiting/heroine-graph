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
    dpr: f32,
    _padding: f32,
}

// Per-node attributes (8 floats per node, matches NODE_ATTR_FLOATS)
struct NodeAttributes {
    radius: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    selected: f32,   // 0.0 or 1.0
    hovered: f32,    // 0.0 or 1.0
    birth_time: f32, // Animation clock time at birth (0 = no animation)
    tex_index: f32,  // Reserved for Stage 2 SVG avatars
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
    @location(4) dpr: f32,                 // Device pixel ratio for AA
    @location(5) pulse_factor: f32,        // Birth pulse animation factor
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

    // Read node attributes (8 floats per node)
    let attr_base = instance_idx * 8u;
    let radius = node_attrs[attr_base];
    let color_r = node_attrs[attr_base + 1u];
    let color_g = node_attrs[attr_base + 2u];
    let color_b = node_attrs[attr_base + 3u];
    let selected = node_attrs[attr_base + 4u];
    let hovered = node_attrs[attr_base + 5u];
    // Clip removed nodes (radius <= 0) by placing behind near plane
    if (radius <= 0.0) {
        output.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        return output;
    }

    // Birth pulse animation: one-shot (positive birth_time) or looping (negative)
    var pulse_factor = 0.0;
    var pulse_looping = false;
    let raw_birth = node_attrs[attr_base + 6u];
    let birth_time_abs = abs(raw_birth);
    if (birth_time_abs > 0.0 && config.birth_pulse_intensity > 0.0) {
        let elapsed = config.time - birth_time_abs;
        if (elapsed >= 0.0) {
            let period = config.birth_pulse_duration * 3.0;
            if (raw_birth < 0.0) {
                // Looping: wrap elapsed into [0, period)
                pulse_factor = config.birth_pulse_intensity * exp(-(elapsed % period) * 3.0 / config.birth_pulse_duration);
                pulse_looping = true;
            } else if (elapsed < period) {
                // One-shot: decay and expire
                pulse_factor = config.birth_pulse_intensity * exp(-elapsed * 3.0 / config.birth_pulse_duration);
            }
        }
    }

    // Size pop: node grows then shrinks back
    let actual_radius = radius * (1.0 + pulse_factor * 1.0);

    // Calculate radius in screen pixels
    let radius_px = max(actual_radius * viewport.scale, MIN_RADIUS_PX);

    // Expand quad to accommodate the splash ring extending beyond the node.
    // ring_progress goes 0→1 as pulse_factor decays from intensity→0.
    // Ring expands to UV 1.0 + rp*10.0 with half-width up to 0.6 (outer ≈ 1.6 + rp*9.4).
    // Quad must always exceed ring's outer edge: 1.7 + rp*10.5 provides margin.
    var ring_scale = 1.0;
    if (pulse_factor > 0.001 && config.birth_pulse_intensity > 0.0) {
        let ring_progress = 1.0 - pulse_factor / config.birth_pulse_intensity;
        ring_scale = 1.7 + ring_progress * 10.5;
    }

    // Get quad vertex offset
    let quad_vertex = QUAD_VERTICES[vertex_idx % 6u];

    // Transform node center to clip space
    let center_clip = transform_point(node_pos);

    // Calculate vertex offset in clip space
    let offset_clip = quad_vertex * (radius_px * 2.0 / viewport.screen_size);

    // Final vertex position (with AA padding and ring expansion)
    let aa_padding = 1.5 / (viewport.dpr * radius_px);
    output.position = vec4<f32>(
        center_clip + offset_clip * (1.0 + aa_padding) * ring_scale,
        0.0,
        1.0
    );

    // UV scaled to match quad expansion (SDF circle stays at radius 1.0)
    output.uv = quad_vertex * (1.0 + aa_padding) * ring_scale;

    // Pass color
    output.color = vec3<f32>(color_r, color_g, color_b);

    // Pass radius for AA calculations in fragment shader
    output.radius_px = radius_px;

    // Pass selection/hover state
    output.state = vec2<f32>(selected, hovered);

    // Pass DPR for fragment shader AA
    output.dpr = viewport.dpr;

    // Pass birth pulse factor for brightness flash in fragment shader.
    // Sign encodes loop mode: negative = looping (use session color in fragment).
    if (pulse_looping) {
        output.pulse_factor = -pulse_factor;
    } else {
        output.pulse_factor = pulse_factor;
    }

    return output;
}
