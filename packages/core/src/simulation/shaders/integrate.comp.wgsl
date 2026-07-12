// Integration Step Compute Shader
// Updates node positions and velocities using computed forces
//
// Implements velocity Verlet integration with adaptive damping
// for stable force-directed layout convergence.
//
// Hierarchical settling: parents freeze before children via depth-scaled alpha.
// Deeper nodes retain higher effective alpha, so they keep adjusting position
// after their parents have settled. This gives children time to orbit into
// natural positions around already-stable parents.
//
// All position, velocity, and force data uses vec2<f32> layout.

struct IntegrationUniforms {
    node_count: u32,
    dt: f32,                    // Time step
    damping: f32,               // Velocity damping (0.9 typical)
    max_velocity: f32,          // Velocity cap to prevent instability
    alpha: f32,                 // Simulation "temperature" (decreases over time)
    depth_settling_spread: f32, // Depth-based settling: alpha *= (1 + depth * spread)
    alpha_min: f32,             // Minimum alpha (simulation stops below this)
    gravity_strength: f32,      // Pull toward center
    center_x: f32,              // Gravity center X
    center_y: f32,              // Gravity center Y
    pinned_node: u32,           // Index of pinned node (0xFFFFFFFF = none)
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

// Node depth from root (f32 per node, 0.0 = root or non-hierarchical)
@group(0) @binding(6) var<storage, read> node_depth: array<f32>;

// Node state flags (bit 0 = dead slot from removal, bit 1 = pinned)
@group(0) @binding(7) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_PINNED: u32 = 2u;

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

    // Dead slots don't integrate — carry the (zeroed) position through the
    // ping-pong so the output buffer never holds stale data for the slot.
    // Pinned nodes hold their externally-written position (pinNode / drag
    // writes both ping-pong buffers) with zero velocity.
    if ((node_flags[node_idx] & (NODE_FLAG_DEAD | NODE_FLAG_PINNED)) != 0u) {
        positions_out[node_idx] = positions_in[node_idx];
        velocities_out[node_idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Pinned node: keep at center with zero velocity
    if (node_idx == uniforms.pinned_node) {
        let center = vec2<f32>(uniforms.center_x, uniforms.center_y);
        positions_out[node_idx] = center;
        velocities_out[node_idx] = vec2<f32>(0.0, 0.0);
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

    // Hierarchical settling: deeper nodes retain more alpha (settle later).
    // When depth_settling_spread = 0, all nodes use raw alpha (no hierarchy).
    // When spread > 0, effective_alpha = alpha * (1 + depth * spread), clamped to 1.
    let depth = node_depth[node_idx];
    let effective_alpha = min(uniforms.alpha * (1.0 + depth * uniforms.depth_settling_spread), 1.0);

    // Update velocity: v' = v * damping + (F/m) * dt * alpha
    // We assume unit mass, so F/m = F
    // Alpha acts as temperature, reducing force effect as simulation cools
    let acceleration = total_force * effective_alpha;
    vel = vel * uniforms.damping + acceleration * uniforms.dt;

    // Cap velocity to prevent instability
    vel = clamp_magnitude(vel, uniforms.max_velocity);

    // Update position: x' = x + v' * dt
    let new_pos = pos + vel * uniforms.dt;

    // Write new state
    positions_out[node_idx] = new_pos;
    velocities_out[node_idx] = vel;
}
