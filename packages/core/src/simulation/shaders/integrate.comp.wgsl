// Integration Step Compute Shader
// Updates node positions and velocities using computed forces
//
// Implements velocity Verlet integration with adaptive damping
// for stable force-directed layout convergence.

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
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: IntegrationUniforms;

// Positions (read from one buffer, write to another for ping-pong)
@group(0) @binding(1) var<storage, read> positions_x_in: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> positions_x_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> positions_y_out: array<f32>;

// Velocities
@group(0) @binding(5) var<storage, read> velocities_x_in: array<f32>;
@group(0) @binding(6) var<storage, read> velocities_y_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> velocities_x_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> velocities_y_out: array<f32>;

// Forces (input)
@group(0) @binding(9) var<storage, read> forces_x: array<f32>;
@group(0) @binding(10) var<storage, read> forces_y: array<f32>;

// Node state flags (for pinned nodes)
@group(0) @binding(11) var<storage, read> node_flags: array<u32>;

// Flag bits
const FLAG_PINNED: u32 = 1u;
const FLAG_HIDDEN: u32 = 2u;

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

    // Check if node is pinned
    let flags = node_flags[node_idx];
    let is_pinned = (flags & FLAG_PINNED) != 0u;

    // Read current state
    let pos = vec2<f32>(positions_x_in[node_idx], positions_y_in[node_idx]);
    var vel = vec2<f32>(velocities_x_in[node_idx], velocities_y_in[node_idx]);
    let force = vec2<f32>(forces_x[node_idx], forces_y[node_idx]);

    // Pinned nodes don't move
    if (is_pinned) {
        positions_x_out[node_idx] = pos.x;
        positions_y_out[node_idx] = pos.y;
        velocities_x_out[node_idx] = 0.0;
        velocities_y_out[node_idx] = 0.0;
        return;
    }

    // Add gravity (pull toward center)
    let to_center = vec2<f32>(uniforms.center_x, uniforms.center_y) - pos;
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
    positions_x_out[node_idx] = new_pos.x;
    positions_y_out[node_idx] = new_pos.y;
    velocities_x_out[node_idx] = vel.x;
    velocities_y_out[node_idx] = vel.y;
}

// Compute kinetic energy for convergence detection
@compute @workgroup_size(256)
fn compute_energy(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Shared memory for reduction
    var<workgroup> shared_energy: array<f32, 256>;

    let node_idx = global_id.x;
    let lid = local_id.x;

    // Calculate kinetic energy for this node
    var energy = 0.0;
    if (node_idx < uniforms.node_count) {
        let vx = velocities_x_in[node_idx];
        let vy = velocities_y_in[node_idx];
        energy = 0.5 * (vx * vx + vy * vy);
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

    velocities_x_out[node_idx] = 0.0;
    velocities_y_out[node_idx] = 0.0;
}

// Apply random jitter (for escaping local minima)
@compute @workgroup_size(256)
fn apply_jitter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let flags = node_flags[node_idx];
    if ((flags & FLAG_PINNED) != 0u) {
        return;
    }

    // Simple pseudo-random jitter based on node index
    // A better implementation would use a proper RNG
    let seed = f32(node_idx) * 12.9898 + uniforms.alpha * 78.233;
    let random_x = fract(sin(seed) * 43758.5453) * 2.0 - 1.0;
    let random_y = fract(sin(seed + 1.0) * 43758.5453) * 2.0 - 1.0;

    let jitter_strength = 1.0;  // Could be a uniform
    positions_x_out[node_idx] = positions_x_in[node_idx] + random_x * jitter_strength;
    positions_y_out[node_idx] = positions_y_in[node_idx] + random_y * jitter_strength;
}
