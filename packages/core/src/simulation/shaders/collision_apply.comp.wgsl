// Collision Displacement Apply Shader — pass 2 of collision resolution
//
// The resolve passes (collision.comp.wgsl, collision_grid.comp.wgsl) read
// positions and write per-node displacements. This pass folds those
// displacements back into the positions. Splitting the two is what makes
// collision resolution deterministic: within one iteration no thread can
// observe another thread's updated position, so the result no longer depends
// on how the driver schedules workgroups.
//
// It lives in its own module because positions must be declared read-only in
// the resolve shaders (a WGSL module has one access mode per binding) and
// read-write here.
//
// One thread per node; each thread touches only its own slot.

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> displacements: array<vec2<f32>>;

@compute @workgroup_size(256)
fn apply_displacements(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    // Bound by the shorter of the two buffers: the dispatch rounds up to a
    // whole workgroup, and the displacement buffer has a 4-slot floor
    // (createCollisionBuffers) so it can outrun a smaller position buffer.
    // An out-of-range write would not fault — it would clamp onto a live
    // node's position.
    if (node_idx >= min(arrayLength(&positions), arrayLength(&displacements))) {
        return;
    }

    let disp = displacements[node_idx];

    // Skip untouched nodes entirely rather than adding zero. Pinned nodes and
    // dead slots always land here, and their positions must come through
    // bit-identical — `x + 0.0` would turn a stored -0.0 into +0.0.
    if (disp.x == 0.0 && disp.y == 0.0) {
        return;
    }

    positions[node_idx] = positions[node_idx] + disp;
}
