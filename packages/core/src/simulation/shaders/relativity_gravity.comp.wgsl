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
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: GravityUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node masses
@group(0) @binding(3) var<storage, read> node_mass: array<f32>;

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
    let gravity = uniforms.gravity_strength / max(mass_factor, 0.1);

    // Linear pull (not inverse-square, for stability)
    let force = to_center * gravity;

    forces[node_idx] += force;
}

// Compute center of mass for the entire graph
// Uses workgroup reduction
var<workgroup> shared_sum_x: array<f32, 256>;
var<workgroup> shared_sum_y: array<f32, 256>;
var<workgroup> shared_mass: array<f32, 256>;

@compute @workgroup_size(256)
fn compute_center_of_mass(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) group_id: vec3<u32>) {
    let node_idx = global_id.x;
    let tid = local_id.x;

    // Load data (or zero if out of bounds)
    if (node_idx < uniforms.node_count) {
        let mass = max(node_mass[node_idx], 1.0);
        let pos = positions[node_idx];
        shared_sum_x[tid] = pos.x * mass;
        shared_sum_y[tid] = pos.y * mass;
        shared_mass[tid] = mass;
    } else {
        shared_sum_x[tid] = 0.0;
        shared_sum_y[tid] = 0.0;
        shared_mass[tid] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            shared_sum_x[tid] += shared_sum_x[tid + stride];
            shared_sum_y[tid] += shared_sum_y[tid + stride];
            shared_mass[tid] += shared_mass[tid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 holds the partial sum for this workgroup.
    // The host code performs final reduction across workgroups.
}

// Apply gravity toward computed center of mass (requires pre-computed center)
@compute @workgroup_size(256)
fn apply_gravity_to_com(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Use pre-computed center of mass from uniforms
    let com = vec2<f32>(uniforms.center_x, uniforms.center_y);

    let pos = positions[node_idx];
    let mass = max(node_mass[node_idx], 1.0);

    let delta = com - pos;
    let dist = length(delta);

    if (dist < EPSILON) {
        return;
    }

    // Mass-weighted attraction
    let mass_factor = pow(mass, uniforms.mass_exponent);
    let force_mag = uniforms.gravity_strength / max(mass_factor, 0.1);

    forces[node_idx] += delta * force_mag;
}
