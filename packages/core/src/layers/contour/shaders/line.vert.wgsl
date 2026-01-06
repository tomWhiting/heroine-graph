// Contour Line Vertex Shader
//
// Renders contour line segments as thick lines using instanced quads.
// Each instance is a line segment, expanded to a quad with the specified width.

struct LineUniforms {
    // Line width in pixels
    line_width: f32,
    // Screen dimensions
    screen_width: f32,
    screen_height: f32,
    // Padding
    _padding: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) alpha: f32,
}

@group(0) @binding(0) var<uniform> uniforms: LineUniforms;
// Line segments: [x1, y1, x2, y2] per segment, in UV coordinates (0-1)
@group(0) @binding(1) var<storage, read> segments: array<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Read segment endpoints (UV coordinates 0-1)
    let base = instance_index * 4u;
    let p1 = vec2<f32>(segments[base + 0u], segments[base + 1u]);
    let p2 = vec2<f32>(segments[base + 2u], segments[base + 3u]);

    // Convert UV to clip space (-1 to 1)
    // Note: Flip Y because texture UV has Y=0 at top, but clip space has Y=-1 at bottom
    let clip_p1 = vec2<f32>(p1.x * 2.0 - 1.0, 1.0 - p1.y * 2.0);
    let clip_p2 = vec2<f32>(p2.x * 2.0 - 1.0, 1.0 - p2.y * 2.0);

    // Check for zero-length segment (degenerate - discard by putting off-screen)
    let diff = clip_p2 - clip_p1;
    let len_sq = dot(diff, diff);
    if (len_sq < 0.0000001) {
        // Zero-length segment - output off-screen position
        var output: VertexOutput;
        output.position = vec4<f32>(-10.0, -10.0, 0.0, 1.0);
        output.alpha = 0.0;
        return output;
    }

    // Line direction and perpendicular
    let dir = diff / sqrt(len_sq);

    // Perpendicular for line thickness (in screen space)
    let perp = vec2<f32>(-dir.y, dir.x);

    // Half width in clip space
    let half_width_x = uniforms.line_width / uniforms.screen_width;
    let half_width_y = uniforms.line_width / uniforms.screen_height;
    let offset = perp * vec2<f32>(half_width_x, half_width_y);

    // Generate quad vertices (2 triangles = 6 vertices)
    // 0: p1 - offset
    // 1: p1 + offset
    // 2: p2 - offset
    // 3: p2 + offset
    var pos: vec2<f32>;
    var alpha: f32 = 1.0;

    switch (vertex_index) {
        case 0u: { pos = clip_p1 - offset; }
        case 1u: { pos = clip_p1 + offset; }
        case 2u: { pos = clip_p2 - offset; }
        case 3u: { pos = clip_p1 + offset; }
        case 4u: { pos = clip_p2 - offset; }
        case 5u: { pos = clip_p2 + offset; }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.alpha = alpha;
    return output;
}
