// Contour Line Fragment Shader
//
// Renders contour lines with configurable color and opacity.

struct LineColorUniforms {
    // Line color (RGBA)
    color: vec4<f32>,
}

struct FragmentInput {
    @location(0) alpha: f32,
}

@group(0) @binding(2) var<uniform> color_uniforms: LineColorUniforms;

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(
        color_uniforms.color.rgb,
        color_uniforms.color.a * input.alpha
    );
}
