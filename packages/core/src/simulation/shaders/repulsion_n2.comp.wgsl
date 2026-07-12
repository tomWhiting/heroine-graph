// N^2 Repulsion Force Compute Shader
// Simple all-pairs repulsion calculation for small graphs
//
// Uses vec2<f32> layout for consolidated position/force data.
//
// Two entry points share the pair-force math:
// - main:        bindings 0-2, no slot masking (used by the N² algorithm
//                plugin, whose bind group has no node_flags buffer)
// - main_masked: bindings 0-3, skips dead slots (holes from removals) —
//                used by the core simulation pipeline
// Auto pipeline layouts only require bindings statically reachable from the
// chosen entry point, so main's layout stays 0-2.

struct RepulsionUniforms {
    node_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    max_distance: f32,
}

@group(0) @binding(0) var<uniform> uniforms: RepulsionUniforms;

// Node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators (read-write) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Node state flags (bit 0 = dead slot from removal) - main_masked only
@group(0) @binding(3) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;

// Coulomb-like repulsion between one pair: F = k / r^2
fn pair_force(node_pos: vec2<f32>, other_pos: vec2<f32>) -> vec2<f32> {
    let delta = node_pos - other_pos;
    let dist_sq = dot(delta, delta);

    // Skip nodes beyond max_distance (0 = no limit)
    if (uniforms.max_distance > 0.0 && dist_sq > uniforms.max_distance * uniforms.max_distance) {
        return vec2<f32>(0.0, 0.0);
    }

    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);
    let dist = sqrt(safe_dist_sq);

    let force_magnitude = uniforms.repulsion_strength / safe_dist_sq;

    return delta * (force_magnitude / dist);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = positions[node_idx];
    var force = vec2<f32>(0.0, 0.0);

    // Direct N^2 summation
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        force += pair_force(node_pos, positions[i]);
    }

    forces[node_idx] += force;
}

@compute @workgroup_size(256)
fn main_masked(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Dead slots (holes from removals, zeroed to the origin) neither
    // receive nor exert repulsion
    if ((node_flags[node_idx] & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let node_pos = positions[node_idx];
    var force = vec2<f32>(0.0, 0.0);

    // Direct N^2 summation over live slots
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }
        if ((node_flags[i] & NODE_FLAG_DEAD) != 0u) {
            continue;
        }

        force += pair_force(node_pos, positions[i]);
    }

    forces[node_idx] += force;
}
