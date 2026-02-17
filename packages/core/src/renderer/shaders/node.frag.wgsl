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
    // Birth pulse animation
    time: f32,
    birth_pulse_duration: f32,
    birth_pulse_intensity: f32,
    _pad3: f32,
}

// Render config uniform buffer (group 2, binding 0)
// Default values are set when buffer is created in graph.ts
@group(2) @binding(0) var<uniform> config: RenderConfig;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) radius_px: f32,
    @location(3) state: vec2<f32>,  // (selected, hovered)
    @location(4) dpr: f32,          // Device pixel ratio for AA
    @location(5) pulse_factor: f32, // Birth pulse animation factor (0 = none)
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

    // Calculate anti-aliasing width based on radius and DPR
    // DPR-aware: constant physical-pixel AA width regardless of display density
    let aa_width = 1.5 / (input.dpr * input.radius_px);

    // Distance from circle edge (in normalized space where radius = 1)
    let d = sdf_circle(uv, 1.0);

    // Base circle alpha with AA
    let circle_alpha = aa_step(d, aa_width);

    // === Splash ring (expanding ripple on birth) ===
    var ring_alpha = 0.0;
    if (input.pulse_factor > 0.001 && config.birth_pulse_intensity > 0.0) {
        let dist = length(uv);
        // ring_progress: 0 at birth, 1 when pulse expires
        let ring_progress = 1.0 - input.pulse_factor / config.birth_pulse_intensity;
        // Ring center expands from node edge (1.0) outward to 11.0
        let ring_center = 1.0 + ring_progress * 10.0;
        // Ring starts thick and thins as it expands
        let ring_half_width = 0.15 + (1.0 - ring_progress) * 0.45;
        let d_ring = abs(dist - ring_center) - ring_half_width;
        // Ring fades as it expands (full brightness at start)
        ring_alpha = aa_step(d_ring, aa_width * 3.0) * (1.0 - ring_progress);
    }

    // Combined visibility: node circle OR splash ring
    let visible_alpha = max(circle_alpha, ring_alpha);
    if (visible_alpha < 0.01) {
        discard;
    }

    // Start with base color
    var final_color = input.color;

    // Apply hover effect (brighten)
    if (hovered > 0.5) {
        final_color = min(final_color * config.hover_brightness, vec3<f32>(1.0));
    }

    // Birth pulse brightness flash (inside circle only)
    if (input.pulse_factor > 0.001) {
        final_color = mix(final_color, vec3<f32>(1.0, 1.0, 1.0), min(input.pulse_factor * 0.6, 1.0));
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
        let sel_ring_alpha = aa_step(d_outer, aa_width) * (1.0 - aa_step(d_inner, aa_width));
        final_color = mix(final_color, config.selection_color, sel_ring_alpha);

        // Extend overall alpha for the selection ring and splash ring
        let extended_alpha = aa_step(d_outer, aa_width);
        return vec4<f32>(final_color, max(max(circle_alpha, extended_alpha * sel_ring_alpha), ring_alpha));
    }

    // === Composite node body + splash ring ===
    if (ring_alpha > 0.001) {
        // Blend: inside circle = node color, outside circle = white ring
        let mix_t = clamp(circle_alpha, 0.0, 1.0);
        let pixel_color = mix(vec3<f32>(1.0, 1.0, 1.0), final_color, mix_t);
        return vec4<f32>(pixel_color, max(circle_alpha, ring_alpha));
    }

    return vec4<f32>(final_color, circle_alpha);
}
