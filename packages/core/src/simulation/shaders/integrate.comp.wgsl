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
    adaptive_speed: f32,        // Adaptive speed strength k (0 = disabled)
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

// Node state flags (bit 0 = dead slot, bit 1 = pinned, bit 2 = hidden by LOD)
@group(0) @binding(7) var<storage, read> node_flags: array<u32>;

// Previous tick's total force per node (for adaptive speed swing/traction)
@group(0) @binding(8) var<storage, read_write> prev_forces: array<vec2<f32>>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_PINNED: u32 = 2u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// Every slot that holds its position instead of integrating.
const NODE_FLAG_FROZEN: u32 = NODE_FLAG_DEAD | NODE_FLAG_PINNED | NODE_FLAG_HIDDEN_LOD;

// Clamp vector magnitude
fn clamp_magnitude(v: vec2<f32>, max_mag: f32) -> vec2<f32> {
    let mag = length(v);
    if (mag > max_mag && mag > 0.0) {
        return v * (max_mag / mag);
    }
    return v;
}

// Main integration kernel.
//
// Dispatched over ALL node slots, not over the active-index list the force
// passes use: positions ping-pong, so a slot this pass does not write keeps
// whatever the destination buffer held two ticks ago. A node that stops being
// active would then alternate between its last two positions forever. Carrying
// every inactive slot through is what makes freezing exact, and it is O(N) of
// pure streaming against the O(active²) the list saves in repulsion.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Frozen slots don't integrate — carry the position through the ping-pong
    // so the output buffer never holds stale data for the slot. Dead slots are
    // holes from removals; pinned nodes hold their externally-written position
    // (pinNode / drag writes both ping-pong buffers); LOD-hidden nodes hold the
    // arrangement they were collapsed with, which is what makes expanding them
    // again cheap and non-jarring.
    if ((node_flags[node_idx] & NODE_FLAG_FROZEN) != 0u) {
        positions_out[node_idx] = positions_in[node_idx];
        velocities_out[node_idx] = vec2<f32>(0.0, 0.0);
        prev_forces[node_idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Pinned node: keep at center with zero velocity
    if (node_idx == uniforms.pinned_node) {
        let center = vec2<f32>(uniforms.center_x, uniforms.center_y);
        positions_out[node_idx] = center;
        velocities_out[node_idx] = vec2<f32>(0.0, 0.0);
        prev_forces[node_idx] = vec2<f32>(0.0, 0.0);
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

    // Adaptive speed (ForceAtlas2-style swing/traction, per node):
    //   swing    = |F_t - F_{t-1}|  — oscillation (force flipping direction)
    //   traction = |F_t + F_{t-1}|/2 — steady progress in one direction
    //   factor   = k * traction / (traction + swing), clamped to [0.05, 1]
    // A node converging steadily gets factor ~ k (1.0 by default, no change);
    // a node whose force reverses every tick gets factor ~ 0, killing the
    // oscillation. Disabled when adaptive_speed = 0.
    if (uniforms.adaptive_speed > 0.0) {
        let prev_force = prev_forces[node_idx];
        let swing = length(total_force - prev_force);
        let traction = length(total_force + prev_force) * 0.5;
        let factor = clamp(
            uniforms.adaptive_speed * traction / (traction + swing + 1e-6),
            0.05,
            1.0,
        );
        vel *= factor;
    }
    prev_forces[node_idx] = total_force;

    // Cap velocity to prevent instability
    vel = clamp_magnitude(vel, uniforms.max_velocity);

    // Update position: x' = x + v' * dt
    let new_pos = pos + vel * uniforms.dt;

    // Write new state
    positions_out[node_idx] = new_pos;
    velocities_out[node_idx] = vel;
}
