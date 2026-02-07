// MSDF Label Fragment Shader
//
// Renders text using Multi-channel Signed Distance Field (MSDF) technique.
// This produces sharp, anti-aliased text at any zoom level.
//
// The MSDF texture stores distance-to-edge in RGB channels, where
// the median of the three values gives the actual distance.

struct LabelUniforms {
    // Text color (RGBA)
    color: vec4<f32>,
    // Font size in pixels
    font_size: f32,
    // MSDF distance range (must match atlas generation, typically 4)
    distance_range: f32,
    // Atlas texture dimensions
    atlas_width: f32,
    atlas_height: f32,
    // Font size used when generating the atlas
    atlas_font_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(1) var<uniform> label_uniforms: LabelUniforms;

@group(2) @binding(0) var font_atlas: texture_2d<f32>;
@group(2) @binding(1) var font_sampler: sampler;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

// Median of three values - core MSDF operation
fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample the MSDF texture
    let msdf = textureSample(font_atlas, font_sampler, input.uv);

    // Get the signed distance from the median of RGB channels
    let sd = median(msdf.r, msdf.g, msdf.b);

    // Calculate screen-space distance for anti-aliasing
    let screen_px_range = label_uniforms.distance_range * label_uniforms.font_size / label_uniforms.atlas_font_size;

    // Convert SDF to signed distance (inverted: 0.5 - sd because atlas convention)
    let screen_px_distance = screen_px_range * (0.5 - sd);

    // Anti-aliased edge using screen-space derivatives
    let fw = max(fwidth(screen_px_distance), 0.001);
    let opacity = clamp(screen_px_distance / fw + 0.5, 0.0, 1.0);

    // Discard fully transparent pixels
    if (opacity < 0.01) {
        discard;
    }

    return vec4<f32>(label_uniforms.color.rgb, label_uniforms.color.a * opacity);
}
