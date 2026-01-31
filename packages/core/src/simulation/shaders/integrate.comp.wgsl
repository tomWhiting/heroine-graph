// Integration Step Compute Shader
// Updates node positions and velocities using computed forces
//
// Implements velocity Verlet integration with adaptive damping
// for stable force-directed layout convergence.
//
// All position, velocity, and force data uses vec2<f32> layout.

struct IntegrationUniforms {
    node_count: u32,
    dt: f32,                    // Time step
    damping: f32,               // Velocity damping (0.9 typical)
    max_velocity: f32,          // Velocity cap to prevent instability
    alpha: f32,                 // Simulation "temperature" (decreases over time)
    alpha_decay: f32,           // Rate of alpha decrease per step
    alpha_min: f32,             // Minimum alpha (simulation stops below this)
    gravity_strength: f32,      // Pull toward center
    center_x: f32,              // Gravity center X
    center_y: f32,              // Gravity center Y
    _pad0: u32,                 // Padding for 16-byte alignment
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: IntegrationUniforms;

// Positions (read from one buffer, write to another for ping-pong) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> positions_out: array<vec2<f32>>;

// Velocities - vec2<f32> per node
@group(0) @binding(3) var<storage, read> velocities_in: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> velocities_out: array<vec2<f32>>;

// Forces (input) - vec2<f32> per node
@group(0) @binding(5) var<storage, read> forces: array<vec2<f32>>;

// Clamp vector magnitude
fn clamp_magnitude(v: vec2<f32>, max_mag: f32) -> vec2<f32> {
    let mag = length(v);
    if (mag > max_mag && mag > 0.0) {
        return v * (max_mag / mag);
    }
    return v;
}

// Main integration kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Read current state
    let pos = positions_in[node_idx];
    var vel = velocities_in[node_idx];
    let force = forces[node_idx];

    // Add gravity (pull toward center)
    let center = vec2<f32>(uniforms.center_x, uniforms.center_y);
    let to_center = center - pos;
    let gravity_force = to_center * uniforms.gravity_strength;
    let total_force = force + gravity_force;

    // Update velocity: v' = v * damping + (F/m) * dt * alpha
    // We assume unit mass, so F/m = F
    // Alpha acts as temperature, reducing force effect as simulation cools
    let acceleration = total_force * uniforms.alpha;
    vel = vel * uniforms.damping + acceleration * uniforms.dt;

    // Cap velocity to prevent instability
    vel = clamp_magnitude(vel, uniforms.max_velocity);

    // Update position: x' = x + v' * dt
    let new_pos = pos + vel * uniforms.dt;

    // Write new state
    positions_out[node_idx] = new_pos;
    velocities_out[node_idx] = vel;
}

// Workgroup shared memory for energy reduction (must be at module scope in WGSL)
var<workgroup> shared_energy: array<f32, 256>;

// Compute kinetic energy for convergence detection
@compute @workgroup_size(256)
fn compute_energy(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>) {
    let node_idx = global_id.x;
    let lid = local_id.x;

    // Calculate kinetic energy for this node
    var energy = 0.0;
    if (node_idx < uniforms.node_count) {
        let vel = velocities_in[node_idx];
        energy = 0.5 * dot(vel, vel);
    }

    shared_energy[lid] = energy;
    workgroupBarrier();

    // Reduce within workgroup
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            shared_energy[lid] += shared_energy[lid + stride];
        }
        workgroupBarrier();
    }

    // This would write to an output buffer for total energy
    // Used to detect when simulation has converged
}

// Reset velocities (for restarting simulation)
@compute @workgroup_size(256)
fn reset_velocities(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    velocities_out[node_idx] = vec2<f32>(0.0, 0.0);
}

// Apply random jitter (for escaping local minima)
@compute @workgroup_size(256)
fn apply_jitter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Pseudo-random jitter using hash-based RNG (node index + alpha as seed).
    let seed = f32(node_idx) * 12.9898 + uniforms.alpha * 78.233;
    let random_x = fract(sin(seed) * 43758.5453) * 2.0 - 1.0;
    let random_y = fract(sin(seed + 1.0) * 43758.5453) * 2.0 - 1.0;

    let jitter_strength = 1.0;
    let pos = positions_in[node_idx];
    positions_out[node_idx] = pos + vec2<f32>(random_x, random_y) * jitter_strength;
}
