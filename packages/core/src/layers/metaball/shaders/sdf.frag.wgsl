// Metaball SDF Fragment Shader
//
// Evaluates a signed distance field for metaballs using smooth minimum.
// Renders organic blob-like shapes around node clusters.

struct MetaballUniforms {
    // Viewport transform
    viewport_offset: vec2<f32>,
    viewport_scale: f32,
    // Metaball parameters
    threshold: f32,
    blend_radius: f32,
    node_radius: f32,
    // Screen dimensions
    screen_width: f32,
    screen_height: f32,
    // Fill color
    fill_color: vec4<f32>,
    // Outline parameters
    outline_only: u32,
    outline_width: f32,
    // Node count
    node_count: u32,
    _padding: u32,
}

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: MetaballUniforms;
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Quadratic smooth minimum for SDF blending
// This creates organic blobs when SDFs are combined
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Signed distance to a circle
fn sd_circle(p: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

// Evaluate metaball field at a point
fn evaluate_field(p: vec2<f32>) -> f32 {
    var d = 1e10;

    // Blend all node circles together
    for (var i = 0u; i < uniforms.node_count; i++) {
        let center = vec2<f32>(positions_x[i], positions_y[i]);
        let circle_d = sd_circle(p, center, uniforms.node_radius);
        d = smin(d, circle_d, uniforms.blend_radius);
    }

    return d;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Early exit if no nodes
    if (uniforms.node_count == 0u) {
        discard;
    }

    // Convert UV to graph coordinates
    let screen_pos = input.uv * vec2<f32>(uniforms.screen_width, uniforms.screen_height);
    let graph_pos = (screen_pos - vec2<f32>(uniforms.screen_width, uniforms.screen_height) * 0.5)
                    / uniforms.viewport_scale
                    + uniforms.viewport_offset;

    // Evaluate SDF
    let d = evaluate_field(graph_pos);

    // For small node counts, use actual evaluation
    // For large counts, we'd need spatial acceleration (octree/BVH)
    // This simple version works well for up to ~1000 nodes

    if (uniforms.outline_only != 0u) {
        // Outline mode: only draw near the boundary
        let outline_half = uniforms.outline_width * 0.5 / uniforms.viewport_scale;

        if (abs(d) < outline_half) {
            // Inside outline band
            let edge_alpha = 1.0 - abs(d) / outline_half;
            return vec4<f32>(
                uniforms.fill_color.rgb,
                uniforms.fill_color.a * edge_alpha
            );
        } else {
            discard;
        }
    } else {
        // Filled mode
        if (d < 0.0) {
            // Inside the metaball
            // Smooth edge falloff
            let edge_dist = 5.0 / uniforms.viewport_scale; // 5 pixel falloff
            let alpha = clamp(-d / edge_dist, 0.0, 1.0);

            return vec4<f32>(
                uniforms.fill_color.rgb,
                uniforms.fill_color.a * alpha
            );
        } else {
            // Outside - apply soft edge
            let soft_edge = 2.0 / uniforms.viewport_scale;
            if (d < soft_edge) {
                let alpha = 1.0 - d / soft_edge;
                return vec4<f32>(
                    uniforms.fill_color.rgb,
                    uniforms.fill_color.a * alpha * 0.3
                );
            }
            discard;
        }
    }
}
