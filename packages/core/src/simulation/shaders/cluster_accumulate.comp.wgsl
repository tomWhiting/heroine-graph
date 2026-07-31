// Cluster Force: Accumulate community centroids
//
// Each thread adds its node's position to its community's centroid
// accumulator using atomic operations.
//
// Fixed-point encoding (scale by 10) provides 0.1 precision with i32
// atomics, supporting positions up to +/-214,748.

struct AccumUniforms {
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: AccumUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> community_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> centroid_sum_x: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> centroid_sum_y: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> centroid_count: array<atomic<u32>>;
// Node state flags (bit 0 = dead slot from removal) - see pipeline.ts
@group(0) @binding(6) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.node_count) {
        return;
    }

    // Dead slots (holes from removals) keep the community id of their former
    // occupant while their position is zeroed to the origin. Folding one into
    // a centroid drags it toward the origin AND inflates the member count that
    // divides the attraction strength in cluster_attract, so the whole
    // community loosens. They must not contribute at all.
    if ((node_flags[idx] & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let comm = community_ids[idx];
    let pos = positions[idx];

    // Fixed-point accumulation: scale by 10 for 0.1 precision
    // Supports positions up to +/-214,748 without i32 overflow
    let fx = i32(pos.x * 10.0);
    let fy = i32(pos.y * 10.0);

    // Accumulate per-community centroid
    atomicAdd(&centroid_sum_x[comm], fx);
    atomicAdd(&centroid_sum_y[comm], fy);
    atomicAdd(&centroid_count[comm], 1u);
}
