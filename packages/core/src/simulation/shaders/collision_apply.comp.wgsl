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

// Node state flags (bit 0 = dead slot, bit 1 = pinned, bit 2 = hidden by LOD).
//
// This pass stays node-indexed — its body is a load, a compare and an add, so
// sweeping every slot costs nothing worth gathering for — but it can no longer
// trust that the resolve pass wrote every slot. Resolve now dispatches over the
// active-index list, and the slots it omits keep whatever displacement they
// were left with when they were last active. Those slots are exactly the inert
// ones, so masking IMMOVABLE here is what makes the shortened resolve dispatch
// safe: the mask is the guarantee, the list is the optimisation.
@group(0) @binding(2) var<storage, read> node_flags: array<u32>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_PINNED: u32 = 2u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// Slots collision never displaces: inert (dead or LOD-hidden) or pinned.
const NODE_FLAG_IMMOVABLE: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD | NODE_FLAG_PINNED;

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

    if ((node_flags[node_idx] & NODE_FLAG_IMMOVABLE) != 0u) {
        return;
    }

    let disp = displacements[node_idx];

    // Skip untouched nodes entirely rather than adding zero: a position must
    // come through bit-identical — `x + 0.0` would turn a stored -0.0 into
    // +0.0.
    if (disp.x == 0.0 && disp.y == 0.0) {
        return;
    }

    positions[node_idx] = positions[node_idx] + disp;
}
