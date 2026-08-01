// Relativity Atlas: Mass-Weighted Gravity Shader
// Applies gravitational attraction toward the center of mass.
//
// Hierarchy-safe gravity model:
// - Depth decay is the PRIMARY hierarchy knob: gravity *= decay^depth, so
//   deep leaves feel less external pull than their ancestors (the nested
//   bubble principle: children experience LESS external force than parents).
// - Inverse-mass modulation (light nodes pulled harder, creating hub
//   clustering) is BOUNDED while depth decay is active so it can never
//   invert the hierarchy: children always carry less mass than their parent
//   (mass = base + factor * sum(child_mass)), so with the mass factor
//   clamped to [1, 1/decay] the pull ratio at equal radius satisfies
//     pull_child / pull_parent = (massfac_parent / massfac_child) * decay
//                             <= (1/decay) * decay = 1.
//   The unbounded inverse-mass modulation (the hub-clustering behavior for
//   non-hierarchical graphs) remains available as an explicit opt-in by
//   disabling depth decay (depth_decay_rate >= 1).
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
    depth_decay_rate: f32, // gravity *= decay^depth (>= 1.0 disables decay)
}

@group(0) @binding(0) var<uniform> uniforms: GravityUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node masses
@group(0) @binding(3) var<storage, read> node_mass: array<f32>;

// Node depth in hierarchy (0 = root)
@group(0) @binding(4) var<storage, read> node_depth: array<f32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(5) var<storage, read> node_flags: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;
const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Apply mass-weighted gravity toward center
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Inert slots — dead or LOD-hidden — receive no forces
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
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

    // Inverse-mass modulation: gravity_factor = gravity_strength / mass^exponent.
    // mass >= 1 and mass_exponent >= 0, so mass_factor_raw >= 1.
    let mass_factor_raw = pow(mass, uniforms.mass_exponent);
    var gravity: f32;
    let depth = node_depth[node_idx];

    if (uniforms.depth_decay_rate < 1.0) {
        // Hierarchy mode: bound the inverse-mass factor to one level of depth
        // decay so a light child's pull never exceeds its heavy parent's at
        // equal radius (see module header for the proof sketch).
        let mass_factor = clamp(mass_factor_raw, 1.0, 1.0 / uniforms.depth_decay_rate);
        gravity = uniforms.gravity_strength / mass_factor;
        if (depth > 0.0) {
            gravity *= pow(uniforms.depth_decay_rate, depth);
        }
    } else {
        // Hub-clustering opt-in (depth decay disabled): full inverse-mass
        // modulation, lighter nodes pulled arbitrarily harder than hubs.
        gravity = uniforms.gravity_strength / max(mass_factor_raw, 0.1);
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
