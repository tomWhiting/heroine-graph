// Edge Vertex Shader
// Renders edges as thick lines with anti-aliased edges
// Supports both straight and curved (conic Bezier) edges

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

// Curved edge configuration
struct CurveConfig {
    enabled: u32,           // 0 = straight, 1 = curved
    segments: u32,          // tessellation segments (default 19)
    weight: f32,            // rational curve weight (default 0.8)
    _pad: f32,
}

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;

// Node positions (needed to get edge endpoints)
@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;

// Edge data: pairs of (source_idx, target_idx) packed as u32
@group(1) @binding(1) var<storage, read> edge_indices: array<u32>;

// Edge attributes (width, color, selected, hovered, curvature, opacity)
// Layout: 8 floats per edge
@group(1) @binding(2) var<storage, read> edge_attrs: array<f32>;

// Curve configuration uniform
@group(1) @binding(3) var<storage, read> curve_config: CurveConfig;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,        // For line distance (t along curve, perpendicular dist)
    @location(1) color: vec3<f32>,      // Edge color
    @location(2) half_width: f32,       // Half line width in pixels
    @location(3) state: vec2<f32>,      // (selected, hovered)
    @location(4) dpr: f32,              // Device pixel ratio for AA
    @location(5) opacity: f32,          // Edge opacity (0.0 = hidden, 1.0 = fully visible)
}

// Each edge segment is rendered as a quad (2 triangles, 6 vertices)
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

// Rational quadratic (conic) Bezier curve evaluation
// A = start, B = end, C = control point, t = parameter [0,1], w = weight
fn conic_bezier(A: vec2<f32>, B: vec2<f32>, C: vec2<f32>, t: f32, w: f32) -> vec2<f32> {
    let one_minus_t = 1.0 - t;
    let dividend = one_minus_t * one_minus_t * A +
                   2.0 * one_minus_t * t * w * C +
                   t * t * B;
    let divisor = one_minus_t * one_minus_t +
                  2.0 * one_minus_t * t * w +
                  t * t;
    return dividend / divisor;
}

// Compute control point for curved edge
// curvature: positive = bend right, negative = bend left
fn compute_control_point(src: vec2<f32>, dst: vec2<f32>, curvature: f32) -> vec2<f32> {
    let midpoint = (src + dst) * 0.5;
    let edge_dir = dst - src;
    let edge_length = length(edge_dir);
    // Perpendicular direction (rotated 90 degrees)
    let perp = vec2<f32>(-edge_dir.y, edge_dir.x) / max(edge_length, 0.0001);
    // Control point offset
    return midpoint + perp * edge_length * curvature;
}

// Default edge width in pixels
const DEFAULT_WIDTH: f32 = 1.0;
// Number of floats per edge in attributes buffer
const EDGE_ATTR_STRIDE: u32 = 8u;

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

    let source_pos = positions[source_idx];
    let target_pos = positions[target_idx];

    // Read edge attributes (8 floats per edge)
    let attr_base = instance_idx * EDGE_ATTR_STRIDE;
    let width = edge_attrs[attr_base];
    let color_r = edge_attrs[attr_base + 1u];
    let color_g = edge_attrs[attr_base + 2u];
    let color_b = edge_attrs[attr_base + 3u];
    let selected = edge_attrs[attr_base + 4u];
    let hovered = edge_attrs[attr_base + 5u];
    let curvature = edge_attrs[attr_base + 6u];
    let opacity = edge_attrs[attr_base + 7u];

    // Discard fully transparent edges early (push behind camera)
    if (opacity <= 0.0) {
        output.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        return output;
    }

    // Use default width if not specified
    let actual_width = select(DEFAULT_WIDTH, width, width > 0.0);
    let half_width = actual_width * 0.5;

    // Determine if we're rendering curved or straight
    let is_curved = curve_config.enabled != 0u && abs(curvature) > 0.001;
    let segments = max(curve_config.segments, 1u);

    // For curved edges, we render multiple segments
    // vertex_idx is divided into: which segment (0..segments-1) and which vertex of quad (0..5)
    var segment_idx: u32 = 0u;
    var quad_vertex_idx: u32 = vertex_idx % 6u;

    if (is_curved) {
        segment_idx = (vertex_idx / 6u) % segments;
        quad_vertex_idx = vertex_idx % 6u;
    }

    // Calculate t values for this segment
    var t0: f32;
    var t1: f32;
    if (is_curved) {
        let seg_f = f32(segments);
        t0 = f32(segment_idx) / seg_f;
        t1 = f32(segment_idx + 1u) / seg_f;
    } else {
        t0 = 0.0;
        t1 = 1.0;
    }

    // Compute positions based on curve mode
    var p0: vec2<f32>;
    var p1: vec2<f32>;

    if (is_curved) {
        // Compute control point for curve
        let control = compute_control_point(source_pos, target_pos, curvature);
        let w = curve_config.weight;

        // Evaluate curve at segment endpoints
        p0 = conic_bezier(source_pos, target_pos, control, t0, w);
        p1 = conic_bezier(source_pos, target_pos, control, t1, w);
    } else {
        p0 = source_pos;
        p1 = target_pos;
    }

    // Transform to clip space
    let p0_clip = transform_point(p0);
    let p1_clip = transform_point(p1);

    // Calculate segment direction in clip space
    let seg_dir = p1_clip - p0_clip;
    let seg_length = length(seg_dir);

    // Handle degenerate segments
    if (seg_length < 0.0001) {
        output.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);  // Behind camera
        return output;
    }

    let seg_unit = seg_dir / seg_length;

    // Perpendicular direction for width
    let perp = vec2<f32>(-seg_unit.y, seg_unit.x);

    // Get quad vertex offset
    let quad_offset = QUAD_OFFSETS[quad_vertex_idx];

    // Calculate position along segment (0 = p0, 1 = p1)
    let local_t = quad_offset.x;
    let base_pos = mix(p0_clip, p1_clip, local_t);

    // Add width offset (perpendicular)
    // Convert width from pixels to clip space
    let width_clip = half_width * 2.0 / viewport.screen_size;
    let aa_padding = 1.5 / viewport.dpr;  // DPR-aware: physical-pixel AA
    let total_width = width_clip + vec2<f32>(aa_padding * 2.0 / viewport.screen_size);
    let offset = perp * total_width * quad_offset.y;

    output.position = vec4<f32>(base_pos + offset, 0.0, 1.0);

    // UV for line distance calculation
    // x: position along full edge (0-1), accounting for segment position
    // y: perpendicular distance in pixels
    let global_t = mix(t0, t1, local_t);
    output.uv = vec2<f32>(global_t, quad_offset.y * (half_width + aa_padding));

    output.color = vec3<f32>(color_r, color_g, color_b);
    output.half_width = half_width;
    output.state = vec2<f32>(selected, hovered);
    output.dpr = viewport.dpr;
    output.opacity = opacity;

    return output;
}
