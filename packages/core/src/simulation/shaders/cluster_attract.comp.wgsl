// Cluster Force: Attract toward own community centroid
//
// Each node is pulled toward the centroid of its community (Hooke's law).
// Inter-community separation is handled by the modulated repulsion shader,
// not here — this shader only does intra-cluster attraction.
//
// No slot mask here, deliberately: this pass only reads its own slot and
// writes to its own force accumulator, so an inert slot — dead or LOD-hidden —
// cannot reach an active node. Its stray force is discarded: clear_forces
// zeroes every slot each tick and integrate.comp.wgsl freezes inert slots
// without reading the accumulator. The masks that matter for this algorithm
// are in cluster_accumulate (an inert slot would poison the centroid it still
// points at) and repulsion_community (an inert slot would push every active
// node).
//
// CRITICAL: Force is mass-weighted (sqrt(degree+1)) to match repulsion and gravity.
// Without mass-weighting, centroid attraction is the only non-mass-scaled force,
// which causes high-degree parents to drift to the periphery while low-degree
// children cluster at the centroid. Mass-weighting makes the equilibrium
// condition mass-independent: all nodes settle at the same position regardless
// of degree, with degree only affecting transient dynamics.

struct AttractUniforms {
    node_count: u32,
    attraction_strength: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: AttractUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> community_ids: array<u32>;
// Centroid buffers read as plain i32/u32 (barrier ensures visibility after atomic writes)
@group(0) @binding(4) var<storage, read> centroid_sum_x: array<i32>;
@group(0) @binding(5) var<storage, read> centroid_sum_y: array<i32>;
@group(0) @binding(6) var<storage, read> centroid_count: array<u32>;
@group(0) @binding(7) var<storage, read> degrees: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.node_count) {
        return;
    }

    let comm = community_ids[idx];
    let count = centroid_count[comm];
    if (count == 0u) {
        return;
    }

    // Decode own community centroid from fixed-point (scale factor 10)
    let cx = f32(centroid_sum_x[comm]) / (10.0 * f32(count));
    let cy = f32(centroid_sum_y[comm]) / (10.0 * f32(count));

    let pos = positions[idx];

    // Mass = sqrt(degree+1), matching repulsion and gravity.
    // This makes the equilibrium position mass-independent:
    //   repulsion(pos) = gravity + centroid_attraction(pos)
    // All three terms scale by mass, so mass cancels at equilibrium.
    let mass = sqrt(f32(degrees[idx] + 1u));

    // Intra-cluster attraction: pull toward own community centroid (Hooke's law)
    // Scale by mass / sqrt(community_size)
    let dx = cx - pos.x;
    let dy = cy - pos.y;
    let force = vec2<f32>(dx, dy) * uniforms.attraction_strength * mass / sqrt(f32(count));

    forces[idx] += force;
}
