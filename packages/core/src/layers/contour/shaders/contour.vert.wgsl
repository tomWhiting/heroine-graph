// Contour Fullscreen Vertex Shader
//
// Renders a fullscreen triangle for the contour layer.
// Uses the standard trick of generating vertices from vertex_index.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV coordinates: (0,0) at bottom-left, (1,1) at top-right in texture space
    // Flip Y to match texture coordinates (Y=0 at top)
    output.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return output;
}
