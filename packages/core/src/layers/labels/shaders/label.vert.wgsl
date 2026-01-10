// Label Vertex Shader
//
// Renders instanced quads for text glyphs. Each instance represents
// one character with its position, size, and UV coordinates.

struct ViewportUniforms {
    // Viewport center offset (graph space)
    offset: vec2<f32>,
    // Zoom scale
    scale: f32,
    // Canvas dimensions
    canvas_width: f32,
    canvas_height: f32,
    _padding: vec3<f32>,
}

struct LabelUniforms {
    // Text color (RGBA)
    color: vec4<f32>,
    // Font size in pixels
    font_size: f32,
    // MSDF distance range (must match atlas generation)
    distance_range: f32,
    // Atlas texture dimensions
    atlas_width: f32,
    atlas_height: f32,
}

// Per-glyph instance data (48 bytes, aligned to 16)
struct GlyphInstance {
    // Position in graph space (x, y)
    position: vec2<f32>,
    // Glyph size in pixels (width, height)
    size: vec2<f32>,
    // UV coordinates in atlas (u0, v0, u1, v1)
    uv: vec4<f32>,
    // Offset from baseline (xoffset, yoffset)
    offset: vec2<f32>,
    // Padding for 16-byte alignment
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;
@group(0) @binding(1) var<uniform> label_uniforms: LabelUniforms;

// Storage buffer containing all visible glyph instances
@group(1) @binding(0) var<storage, read> glyphs: array<GlyphInstance>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Quad vertices (2 triangles)
// 0---1
// |  /|
// | / |
// 2---3
const QUAD_POSITIONS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),  // 0 - top left
    vec2<f32>(1.0, 0.0),  // 1 - top right
    vec2<f32>(0.0, 1.0),  // 2 - bottom left
    vec2<f32>(1.0, 0.0),  // 1 - top right
    vec2<f32>(1.0, 1.0),  // 3 - bottom right
    vec2<f32>(0.0, 1.0),  // 2 - bottom left
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let glyph = glyphs[instance_index];
    let quad_pos = QUAD_POSITIONS[vertex_index];

    // Calculate glyph position in screen space
    // First, transform graph position to screen position
    let screen_center = vec2<f32>(viewport.canvas_width, viewport.canvas_height) * 0.5;
    let graph_offset = glyph.position - viewport.offset;
    let screen_pos = graph_offset * viewport.scale + screen_center;

    // Apply glyph offset and size (in screen pixels)
    let scale_factor = label_uniforms.font_size / 42.0; // 42 is typical atlas font size
    let glyph_offset = glyph.offset * scale_factor;
    let glyph_size = glyph.size * scale_factor;

    // Final vertex position
    let vertex_pos = screen_pos + glyph_offset + quad_pos * glyph_size;

    // Convert to clip space
    let clip_pos = vec2<f32>(
        (vertex_pos.x / viewport.canvas_width) * 2.0 - 1.0,
        1.0 - (vertex_pos.y / viewport.canvas_height) * 2.0
    );

    // Interpolate UV coordinates
    let uv = mix(glyph.uv.xy, glyph.uv.zw, quad_pos);

    var output: VertexOutput;
    output.position = vec4<f32>(clip_pos, 0.0, 1.0);
    output.uv = uv;

    return output;
}
