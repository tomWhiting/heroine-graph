// Contour Fragment Shader
//
// Renders contour lines using a simple band test around density thresholds.
// Inspired by metaball outline mode - checks if density is within a band
// around the threshold value.

struct ContourUniforms {
    // Contour line color (RGBA)
    line_color: vec4<f32>,
    // Line thickness (controls the density band width)
    line_thickness: f32,
    // Anti-aliasing feather amount
    feather: f32,
    // Threshold value for contour (normalized 0-1)
    threshold: f32,
    // Maximum density for normalization
    max_density: f32,
}

@group(0) @binding(0) var density_texture: texture_2d<f32>;
@group(0) @binding(1) var density_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: ContourUniforms;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample density from heatmap texture
    let raw_density = textureSample(density_texture, density_sampler, input.uv).r;

    // Normalize density to 0-1 range
    let density = clamp(raw_density / max(uniforms.max_density, 0.001), 0.0, 1.0);

    // Skip areas with no density
    if (density < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Calculate how far we are from the threshold (in density space)
    let density_diff = abs(density - uniforms.threshold);

    // Band width in density space - line_thickness controls how wide the band is
    // Smaller values = thinner lines, larger values = thicker lines
    let band_width = uniforms.line_thickness * 0.01;
    let feather_width = uniforms.feather * 0.005;

    // Check if we're within the contour band
    if (density_diff < band_width + feather_width) {
        // Inside or near the band - calculate alpha for anti-aliasing
        var alpha: f32;
        if (density_diff < band_width) {
            // Fully inside the band
            alpha = 1.0;
        } else {
            // In the feather zone - fade out
            alpha = 1.0 - (density_diff - band_width) / feather_width;
        }

        return vec4<f32>(
            uniforms.line_color.rgb,
            uniforms.line_color.a * alpha
        );
    }

    // Outside the contour band
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
