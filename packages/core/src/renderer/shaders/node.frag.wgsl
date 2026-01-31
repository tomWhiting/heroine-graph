// Node Fragment Shader
// Renders circular nodes using signed distance field with anti-aliasing

// Visual configuration (can be updated at runtime via uniform buffer)
struct RenderConfig {
    // Selection highlight color
    selection_color: vec3<f32>,
    selection_ring_width: f32,
    // Hover highlight
    hover_brightness: f32,
    // Border settings
    border_enabled: u32,  // 0 = disabled, 1 = enabled
    border_width: f32,
    _pad1: f32,
    border_color: vec3<f32>,
    _pad2: f32,
}

// Render config uniform buffer (group 2, binding 0)
// Default values are set when buffer is created in graph.ts
@group(2) @binding(0) var<uniform> config: RenderConfig;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) radius_px: f32,
    @location(3) state: vec2<f32>,  // (selected, hovered)
}

// SDF circle: distance from edge (negative inside)
fn sdf_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Smooth step for anti-aliasing
fn aa_step(d: f32, aa_width: f32) -> f32 {
    return 1.0 - smoothstep(-aa_width, aa_width, d);
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let radius_px = input.radius_px;
    let selected = input.state.x;
    let hovered = input.state.y;

    // Calculate anti-aliasing width based on radius
    // Larger nodes need proportionally smaller AA
    let aa_width = 1.5 / radius_px;

    // Distance from circle edge (in normalized space where radius = 1)
    let d = sdf_circle(uv, 1.0);

    // Base circle alpha with AA
    let circle_alpha = aa_step(d, aa_width);

    // Discard if outside circle
    if (circle_alpha < 0.01) {
        discard;
    }

    // Start with base color
    var final_color = input.color;

    // Apply hover effect (brighten)
    if (hovered > 0.5) {
        final_color = min(final_color * config.hover_brightness, vec3<f32>(1.0));
    }

    // Draw border (only if enabled and width > 0)
    if (config.border_enabled != 0u && config.border_width > 0.0) {
        let border_inner = 1.0 - config.border_width / radius_px;
        let d_border = sdf_circle(uv, border_inner);
        let border_alpha = aa_step(d_border, aa_width);
        final_color = mix(config.border_color, final_color, border_alpha);
    }

    // Draw selection ring
    if (selected > 0.5) {
        let ring_outer = 1.0 + config.selection_ring_width / radius_px;
        let ring_inner = 1.0;
        let d_outer = sdf_circle(uv, ring_outer);
        let d_inner = sdf_circle(uv, ring_inner);

        // Ring is where d_outer < 0 AND d_inner > 0
        let ring_alpha = aa_step(d_outer, aa_width) * (1.0 - aa_step(d_inner, aa_width));
        final_color = mix(final_color, config.selection_color, ring_alpha);

        // Extend overall alpha for the ring
        let extended_alpha = aa_step(d_outer, aa_width);
        return vec4<f32>(final_color, max(circle_alpha, extended_alpha * ring_alpha));
    }

    return vec4<f32>(final_color, circle_alpha);
}
