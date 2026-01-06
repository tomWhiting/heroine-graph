// Edge Fragment Shader
// Renders anti-aliased lines with selection/hover highlighting

struct FragmentInput {
    @location(0) uv: vec2<f32>,         // (t along line, perpendicular distance)
    @location(1) color: vec3<f32>,
    @location(2) half_width: f32,        // Half width in pixels
    @location(3) state: vec2<f32>,       // (selected, hovered)
}

// Visual configuration
const SELECTION_COLOR: vec3<f32> = vec3<f32>(0.259, 0.522, 0.957);  // #4285f4
const HOVER_BRIGHTNESS: f32 = 1.3;

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let half_width = input.half_width;
    let perp_dist = abs(input.uv.y);  // Distance from line center in pixels
    let selected = input.state.x;
    let hovered = input.state.y;

    // Anti-aliasing: smooth falloff at edges
    let aa_width = 1.0;  // pixels
    let edge_dist = perp_dist - half_width;
    let alpha = 1.0 - smoothstep(-aa_width, aa_width, edge_dist);

    // Discard if outside line
    if (alpha < 0.01) {
        discard;
    }

    // Start with base color
    var final_color = input.color;

    // Apply hover effect
    if (hovered > 0.5) {
        final_color = min(final_color * HOVER_BRIGHTNESS, vec3<f32>(1.0));
    }

    // Apply selection effect (tint toward selection color)
    if (selected > 0.5) {
        final_color = mix(final_color, SELECTION_COLOR, 0.5);
    }

    return vec4<f32>(final_color, alpha);
}
