// Gaussian Splat Fragment Shader
//
// Computes Gaussian falloff for each fragment to create smooth
// density accumulation. Output is additively blended to the
// density texture.

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) intensity: f32,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Convert UV [0,1] to centered coordinates [-1,1]
    let centered = input.uv * 2.0 - 1.0;

    // Distance from center (squared for efficiency)
    let distSq = dot(centered, centered);

    // Discard fragments well outside the visible area
    if (distSq > 1.2) {
        discard;
    }

    // Gaussian falloff with wider spread for softer edges
    // sigma = 0.5 means the Gaussian naturally fades to ~13% at r=1
    let sigma = 0.5;
    let gaussian = exp(-distSq / (2.0 * sigma * sigma));

    // Additional smooth edge falloff starting earlier (from 30% to 100% of radius)
    // This creates a very gradual fade to zero at the boundary
    let edge_falloff = 1.0 - smoothstep(0.3, 1.0, distSq);

    // Output density value with intensity scaling and edge falloff
    let density = gaussian * edge_falloff * input.intensity;

    return vec4<f32>(density, density, density, density);
}
