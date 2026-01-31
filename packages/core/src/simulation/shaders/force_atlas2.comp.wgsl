// ForceAtlas2 Compute Shader
// A force-directed layout algorithm designed for network visualization
//
// Key differences from standard force-directed:
// - Linear attraction (not quadratic spring)
// - Degree-weighted repulsion
// - Optional LinLog mode for better cluster separation
// - Strong gravity option for disconnected components

struct ForceAtlas2Uniforms {
    node_count: u32,
    scaling: f32,              // Overall force scaling (kr)
    gravity: f32,              // Gravity strength (kg)
    edge_weight_influence: f32, // How much edge weights affect attraction
    flags: u32,                // Bit flags: 0=linlog, 1=strong_gravity, 2=prevent_overlap
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: ForceAtlas2Uniforms;

// Node positions
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Output forces (accumulated)
@group(0) @binding(3) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces_y: array<f32>;

// Node degrees (for degree-weighted repulsion)
@group(0) @binding(5) var<storage, read> degrees: array<u32>;

const MIN_DISTANCE: f32 = 0.01;
const FLAG_LINLOG: u32 = 1u;
const FLAG_STRONG_GRAVITY: u32 = 2u;
const FLAG_PREVENT_OVERLAP: u32 = 4u;

// ForceAtlas2 repulsion: F = kr * (degree(i) + 1) * (degree(j) + 1) / distance
// This is different from Coulomb repulsion which uses distance²
fn fa2_repulsion(dx: f32, dy: f32, degree_i: u32, degree_j: u32) -> vec2<f32> {
    let dist_sq = dx * dx + dy * dy;
    let dist = sqrt(max(dist_sq, MIN_DISTANCE * MIN_DISTANCE));

    // Degree-weighted mass
    let mass_i = f32(degree_i + 1u);
    let mass_j = f32(degree_j + 1u);

    // ForceAtlas2 uses linear distance in denominator (not squared)
    let force_magnitude = uniforms.scaling * mass_i * mass_j / dist;

    // Direction: pointing away from the other node
    let dir = vec2<f32>(dx, dy) / dist;

    return dir * force_magnitude;
}

// Main repulsion kernel - O(n²) but with FA2 force model
@compute @workgroup_size(256)
fn repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = vec2<f32>(positions_x[node_idx], positions_y[node_idx]);
    let node_degree = degrees[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    // Repulsion from all other nodes
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = vec2<f32>(positions_x[i], positions_y[i]);
        let other_degree = degrees[i];

        let dx = node_pos.x - other_pos.x;
        let dy = node_pos.y - other_pos.y;

        total_force += fa2_repulsion(dx, dy, node_degree, other_degree);
    }

    // Gravity toward center
    let gravity_force = -uniforms.gravity * node_pos;

    // Strong gravity option: scales with distance from center
    var final_gravity = gravity_force;
    if ((uniforms.flags & FLAG_STRONG_GRAVITY) != 0u) {
        let dist_from_center = length(node_pos);
        if (dist_from_center > MIN_DISTANCE) {
            final_gravity = gravity_force * dist_from_center;
        }
    }

    total_force += final_gravity;

    // Add to output forces
    forces_x[node_idx] += total_force.x;
    forces_y[node_idx] += total_force.y;
}

// ForceAtlas2 attraction: F = distance (linear, not spring-like)
// In LinLog mode: F = log(1 + distance)
struct AttractionUniforms {
    edge_count: u32,
    linlog: u32,
    weight_influence: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> attr_uniforms: AttractionUniforms;
@group(0) @binding(1) var<storage, read> attr_positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> attr_positions_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> attr_forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> attr_forces_y: array<f32>;
@group(0) @binding(5) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(6) var<storage, read> edge_targets: array<u32>;

@compute @workgroup_size(256)
fn attraction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= attr_uniforms.edge_count) {
        return;
    }

    let src_node = edge_sources[edge_idx];
    let dst_node = edge_targets[edge_idx];

    let source_pos = vec2<f32>(attr_positions_x[src_node], attr_positions_y[src_node]);
    let target_pos = vec2<f32>(attr_positions_x[dst_node], attr_positions_y[dst_node]);

    let delta = target_pos - source_pos;
    let dist = length(delta);

    if (dist < MIN_DISTANCE) {
        return;
    }

    // ForceAtlas2 attraction
    var attraction_force: f32;
    if (attr_uniforms.linlog != 0u) {
        // LinLog mode: log(1 + distance) for better cluster separation
        attraction_force = log(1.0 + dist);
    } else {
        // Standard FA2: linear attraction
        attraction_force = dist;
    }

    let dir = delta / dist;
    let force = dir * attraction_force;

    // Apply forces to both nodes (equal and opposite).
    // Direct accumulation is safe because each edge is processed exactly once.
    attr_forces_x[src_node] += force.x;
    attr_forces_y[src_node] += force.y;
    attr_forces_x[dst_node] -= force.x;
    attr_forces_y[dst_node] -= force.y;
}
