// Edge Fragment Shader
// Renders anti-aliased lines with selection/hover highlighting and flow animation

struct FragmentInput {
    @location(0) uv: vec2<f32>,         // (t along line, perpendicular distance)
    @location(1) color: vec3<f32>,
    @location(2) half_width: f32,        // Half width in pixels
    @location(3) state: vec2<f32>,       // (selected, hovered)
    @location(4) dpr: f32,              // Device pixel ratio for AA
}

// Flow layer uniforms
struct FlowLayer {
    enabled: f32,
    pulse_width: f32,
    pulse_count: f32,
    speed: f32,
    wave_shape: f32,      // 0 = square, 0.5 = triangle, 1 = sine
    brightness: f32,
    fade: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
    has_color: f32,
}

struct FlowUniforms {
    layer1: FlowLayer,
    layer2: FlowLayer,
    time: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(2) @binding(0) var<uniform> flow: FlowUniforms;

// Visual configuration
const SELECTION_COLOR: vec3<f32> = vec3<f32>(0.259, 0.522, 0.957);  // #4285f4
const HOVER_BRIGHTNESS: f32 = 1.3;
const PI: f32 = 3.14159265359;

// PWM wave function
// t: position (0-1), phase: animation phase, width: pulse width, shape: wave shape
fn pwm_wave(t: f32, phase: f32, width: f32, count: f32, shape: f32) -> f32 {
    // Scale position by pulse count and add phase
    let scaled_t = fract(t * count - phase);

    // Generate wave based on shape
    // shape: 0 = square, 0.5 = triangle, 1 = sine

    if (shape < 0.25) {
        // Square wave
        return select(0.0, 1.0, scaled_t < width);
    } else if (shape < 0.75) {
        // Triangle wave - interpolate between square and sine behavior
        let tri_factor = (shape - 0.25) * 2.0; // 0 to 1 as shape goes from 0.25 to 0.75

        // Base triangle
        let center = width * 0.5;
        let dist = abs(scaled_t - center);
        let tri = 1.0 - smoothstep(0.0, width * 0.5, dist);

        // Square base for blending
        let sq = select(0.0, 1.0, scaled_t < width);

        return mix(sq, tri, tri_factor);
    } else {
        // Sine wave
        if (scaled_t < width) {
            let normalized = scaled_t / width;
            return sin(normalized * PI);
        }
        return 0.0;
    }
}

// Calculate flow contribution for a single layer
fn calculate_flow_layer(layer: FlowLayer, t: f32, time: f32) -> vec4<f32> {
    if (layer.enabled < 0.5) {
        return vec4<f32>(0.0);
    }

    // Calculate animation phase
    let phase = time * layer.speed;

    // Get PWM intensity
    let intensity = pwm_wave(t, phase, layer.pulse_width, layer.pulse_count, layer.wave_shape);

    // Apply fade (creates trailing effect)
    let fade_phase = time * layer.speed * (1.0 - layer.fade * 0.5);
    let fade_intensity = pwm_wave(t, fade_phase, layer.pulse_width * (1.0 + layer.fade), layer.pulse_count, layer.wave_shape);
    let combined_intensity = max(intensity, fade_intensity * layer.fade);

    // Apply brightness
    let final_intensity = combined_intensity * layer.brightness;

    // Return color contribution
    if (layer.has_color > 0.5) {
        return vec4<f32>(layer.color_r, layer.color_g, layer.color_b, layer.color_a * final_intensity);
    } else {
        return vec4<f32>(1.0, 1.0, 1.0, final_intensity);
    }
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let half_width = input.half_width;
    let perp_dist = abs(input.uv.y);  // Distance from line center in pixels
    let selected = input.state.x;
    let hovered = input.state.y;

    // Anti-aliasing: smooth falloff at edges
    // DPR-aware: constant physical-pixel AA width regardless of display density
    let aa_width = 1.0 / input.dpr;
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

    // Apply flow animation
    let t = input.uv.x;  // Position along edge (0-1)

    // Calculate flow contributions
    let flow1 = calculate_flow_layer(flow.layer1, t, flow.time);
    let flow2 = calculate_flow_layer(flow.layer2, t, flow.time);

    // Combine flow layers with base color
    // Flow adds brightness (additive) or replaces color (if layer has custom color)
    var flow_brightness = 1.0;

    // Layer 1: Apply as brightness boost or color overlay
    if (flow.layer1.enabled > 0.5) {
        if (flow.layer1.has_color > 0.5) {
            // Blend with custom color
            final_color = mix(final_color, flow1.rgb, flow1.a);
        } else {
            // Just boost brightness
            flow_brightness += flow1.a;
        }
    }

    // Layer 2: Apply as brightness boost or color overlay
    if (flow.layer2.enabled > 0.5) {
        if (flow.layer2.has_color > 0.5) {
            // Blend with custom color additively
            final_color = final_color + flow2.rgb * flow2.a;
        } else {
            // Just boost brightness
            flow_brightness += flow2.a;
        }
    }

    // Apply combined brightness
    final_color = min(final_color * flow_brightness, vec3<f32>(1.0));

    return vec4<f32>(final_color, alpha);
}
