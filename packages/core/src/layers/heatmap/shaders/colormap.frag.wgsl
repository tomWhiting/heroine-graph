// Colormap Fragment Shader
//
// Samples the density texture and maps values to colors
// using a 1D color scale texture (gradient lookup).

struct ColormapUniforms {
    // Density value range for normalization
    min_density: f32,
    max_density: f32,
    // Opacity of the heatmap overlay
    opacity: f32,
    // Padding
    _padding: f32,
}

@group(0) @binding(0) var density_texture: texture_2d<f32>;
@group(0) @binding(1) var density_sampler: sampler;
@group(0) @binding(2) var colorscale_texture: texture_1d<f32>;
@group(0) @binding(3) var colorscale_sampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: ColormapUniforms;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample density from the accumulated texture
    let density_sample = textureSample(density_texture, density_sampler, input.uv);
    let density = density_sample.r;

    // Skip areas with essentially zero density
    if (density < 0.0001) {
        discard;
    }

    // Normalize density to [0, 1] range
    let range = uniforms.max_density - uniforms.min_density;
    var normalized: f32;
    if (range > 0.0) {
        normalized = clamp((density - uniforms.min_density) / range, 0.0, 1.0);
    } else {
        normalized = 0.0;
    }

    // Sample color from the 1D color scale texture
    let color = textureSample(colorscale_texture, colorscale_sampler, normalized);

    // Soft alpha fade: use pow(normalized, 0.5) for gentle falloff at edges
    // This preserves visibility of low-density areas while still fading smoothly
    let density_alpha = pow(normalized, 0.5);

    // Apply density-based alpha and global opacity
    return vec4<f32>(color.rgb, color.a * density_alpha * uniforms.opacity);
}
