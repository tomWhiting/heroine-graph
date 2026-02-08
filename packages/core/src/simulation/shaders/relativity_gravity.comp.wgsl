// Relativity Atlas: Mass-Weighted Gravity Shader
// Applies gravitational attraction toward the center of mass.
//
// Unlike simple centering, this uses hierarchical mass:
// - High-mass nodes (with large subtrees) move slowly
// - Low-mass nodes (leaves) are pulled more strongly
// - Creates natural clustering around hubs
//
// Uses vec2<f32> layout for consolidated position/force data.

struct GravityUniforms {
    node_count: u32,
    gravity_strength: f32,
    center_x: f32,
    center_y: f32,
    mass_exponent: f32,    // How much mass affects gravity (0 = uniform, 1 = linear)
    gravity_curve: u32,    // 0=linear, 1=inverse, 2=soft, 3=custom
    gravity_exponent: f32, // Exponent for custom curve
    depth_decay_rate: f32, // Bubble mode: gravity *= decay^depth (1.0 = no effect)
}

@group(0) @binding(0) var<uniform> uniforms: GravityUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node masses
@group(0) @binding(3) var<storage, read> node_mass: array<f32>;

// Node depth in hierarchy (0 = root, bubble mode)
@group(0) @binding(4) var<storage, read> node_depth: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;

// Apply mass-weighted gravity toward center
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    let mass = max(node_mass[node_idx], 1.0);

    // Direction toward center
    let center = vec2<f32>(uniforms.center_x, uniforms.center_y);
    let to_center = center - pos;

    let dist_sq = dot(to_center, to_center);

    // Skip if already at center
    if (dist_sq < EPSILON) {
        return;
    }

    // Mass-weighted gravity: lighter nodes are pulled more strongly
    // gravity_factor = gravity_strength / mass^exponent
    let mass_factor = pow(mass, uniforms.mass_exponent);
    var gravity = uniforms.gravity_strength / max(mass_factor, 0.1);

    // Depth-decaying gravity (bubble mode): gravity *= decay^depth
    // Root (depth 0): full gravity. Deep leaves: near-zero gravity.
    let depth = node_depth[node_idx];
    if (depth > 0.0 && uniforms.depth_decay_rate < 1.0) {
        gravity *= pow(uniforms.depth_decay_rate, depth);
    }

    // Calculate distance for curve calculations
    let dist = sqrt(dist_sq);

    // Apply gravity curve to modulate force based on distance
    var distance_factor: f32;
    switch (uniforms.gravity_curve) {
        case 0u: {
            // Linear (current behavior): force scales with distance
            distance_factor = dist;
        }
        case 1u: {
            // Inverse-square: stable "orbits", weakens with distance
            distance_factor = 1.0 / max(dist, 1.0);
        }
        case 2u: {
            // Soft: gentle at close range, sqrt falloff
            distance_factor = sqrt(dist);
        }
        default: {
            // Custom: distance^exponent
            distance_factor = pow(dist, uniforms.gravity_exponent);
        }
    }

    // Apply gravity with distance curve
    let force = normalize(to_center) * gravity * distance_factor;

    forces[node_idx] += force;
}

