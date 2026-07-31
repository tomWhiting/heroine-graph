// Collision Detection and Resolution Compute Shader
// Prevents node overlap by applying separation forces
//
// This shader runs after integration to push overlapping nodes apart.
// Each node checks against all other nodes (O(n^2)) or can use spatial
// partitioning for larger graphs.
//
// Uses vec2<f32> layout for consolidated position data.
//
// This is pass 1 of 2: positions are READ-ONLY here and every thread writes
// its own slot of `displacements`. Writing positions in place while other
// threads read them is a cross-workgroup data race (no ordering exists
// between workgroups within a dispatch), which made the resolved layout
// depend on GPU scheduling. collision_apply.comp.wgsl runs as pass 2 and
// adds the displacements to the positions, so every thread in an iteration
// sees the same position snapshot.

struct CollisionUniforms {
    node_count: u32,
    collision_strength: f32,   // How strongly nodes push apart (0-1)
    radius_multiplier: f32,    // Multiplier for node radii
    iterations: u32,           // Number of resolution iterations
    default_radius: f32,       // Default node radius if not specified
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: CollisionUniforms;

// Node positions (read-only snapshot for this iteration) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Node sizes/radii (read-only, optional - uses default if all zeros).
// A negative radius marks a dead slot (hole from removal) — those slots
// do not exist for collision purposes and are skipped entirely.
@group(0) @binding(2) var<storage, read> node_sizes: array<f32>;

// Node state flags (bit 0 = dead slot from removal, bit 1 = pinned).
// Pinned nodes are never displaced by collision resolution (the integrate
// pin pass-through would otherwise carry the drift forward every frame),
// but they still push other nodes away.
@group(0) @binding(3) var<storage, read> node_flags: array<u32>;

// Per-node displacement accumulated by this iteration, consumed by the apply
// pass. Every thread of the dispatch writes its own slot and no other, so
// there is no cross-thread hazard.
@group(0) @binding(4) var<storage, read_write> displacements: array<vec2<f32>>;

const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_PINNED: u32 = 2u;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;

// Accumulated separation for one node against every other node, or zero when
// the node takes no displacement (out of range, dead slot, or pinned).
fn resolve_displacement(node_idx: u32) -> vec2<f32> {
    if (node_idx >= uniforms.node_count) {
        return vec2<f32>(0.0, 0.0);
    }

    // Dead slots don't exist; pinned nodes are never displaced (they still
    // push others away — other threads read this node's position and size).
    if ((node_flags[node_idx] & (NODE_FLAG_DEAD | NODE_FLAG_PINNED)) != 0u) {
        return vec2<f32>(0.0, 0.0);
    }

    // Get this node's position and radius
    let pos = positions[node_idx];

    // Get node radius (negative = dead slot, use default if size buffer has zero)
    var radius_i = node_sizes[node_idx];
    if (radius_i < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    if (radius_i <= EPSILON) {
        radius_i = uniforms.default_radius;
    }
    radius_i *= uniforms.radius_multiplier;

    // Accumulated displacement
    var disp = vec2<f32>(0.0, 0.0);

    // Check against all other nodes
    for (var j = 0u; j < uniforms.node_count; j++) {
        if (j == node_idx) {
            continue;
        }

        // Get other node's radius (skip dead slots — flags and radius sentinel)
        if ((node_flags[j] & NODE_FLAG_DEAD) != 0u) {
            continue;
        }
        var radius_j = node_sizes[j];
        if (radius_j < 0.0) {
            continue;
        }
        if (radius_j <= EPSILON) {
            radius_j = uniforms.default_radius;
        }
        radius_j *= uniforms.radius_multiplier;

        let other_pos = positions[j];

        // Distance between centers
        let delta = pos - other_pos;
        let dist_sq = dot(delta, delta);
        let dist = sqrt(dist_sq);

        // Minimum distance to avoid overlap
        let min_dist = radius_i + radius_j;

        // Check for overlap
        if (dist < min_dist && dist > EPSILON) {
            // Calculate overlap amount
            let overlap = min_dist - dist;

            // Direction to push (from other node to this node)
            let n = delta / dist;

            // Add displacement to separate nodes
            // Only move this node half the distance (other node handles its half)
            let push = overlap * 0.5 * uniforms.collision_strength;
            disp += n * push;
        } else if (dist <= EPSILON && j > node_idx) {
            // Nodes are at exactly the same position
            // Use a deterministic offset based on indices to break symmetry
            let angle = f32(node_idx) * 0.618033988749895 * 6.28318530718;  // Golden ratio
            disp += vec2<f32>(cos(angle), sin(angle)) * uniforms.default_radius * uniforms.collision_strength;
        }
    }

    return disp;
}

// Main collision detection and resolution
// Uses iterative relaxation to separate overlapping nodes
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    // Every thread within the buffer writes its slot — including the
    // padding threads past node_count that the workgroup round-up adds —
    // so the apply pass can never pick up a displacement left by an
    // earlier iteration or an earlier (larger) node count.
    if (node_idx >= arrayLength(&displacements)) {
        return;
    }

    displacements[node_idx] = resolve_displacement(node_idx);
}

// Workgroup-optimized version using shared memory for positions
// More efficient for medium-sized graphs
var<workgroup> shared_pos: array<vec2<f32>, 256>;
var<workgroup> shared_radius: array<f32, 256>;

@compute @workgroup_size(256)
fn resolve_tiled(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>,
                 @builtin(workgroup_id) group_id: vec3<u32>) {
    let node_idx = global_id.x;
    let tid = local_id.x;

    // CRITICAL: All threads must reach workgroupBarrier() calls.
    // Out-of-bounds threads participate in barriers but skip actual work.
    // Dead slots (negative radius) participate in barriers but never collide.
    // Pinned nodes participate in barriers and tile loads (so they push
    // others away) but never receive displacement themselves.
    var is_valid = node_idx < uniforms.node_count;

    // Load this node's data (only valid threads read from global memory)
    var pos = vec2<f32>(0.0, 0.0);
    var radius = 0.0;
    if (is_valid) {
        pos = positions[node_idx];
        radius = node_sizes[node_idx];
        if (radius < 0.0 ||
            (node_flags[node_idx] & (NODE_FLAG_DEAD | NODE_FLAG_PINNED)) != 0u) {
            is_valid = false;
        } else if (radius <= EPSILON) {
            radius = uniforms.default_radius;
        }
        radius *= uniforms.radius_multiplier;
    }

    var disp = vec2<f32>(0.0, 0.0);

    // Process nodes in tiles using shared memory
    let num_tiles = (uniforms.node_count + 255u) / 256u;

    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile data into shared memory
        // All threads participate in loading to ensure shared memory is populated
        let tile_idx = tile * 256u + tid;
        if (tile_idx < uniforms.node_count) {
            shared_pos[tid] = positions[tile_idx];
            var r = node_sizes[tile_idx];
            if ((node_flags[tile_idx] & NODE_FLAG_DEAD) != 0u) {
                // Flag-dead slots get the same negative skip marker
                r = -1.0;
            } else if (r >= 0.0 && r <= EPSILON) {
                r = uniforms.default_radius;
            }
            // Dead slots keep their negative radius as a skip marker
            shared_radius[tid] = r * uniforms.radius_multiplier;
        } else {
            shared_pos[tid] = vec2<f32>(0.0, 0.0);
            shared_radius[tid] = -1.0;
        }

        // ALL threads must reach this barrier
        workgroupBarrier();

        // Process all nodes in this tile (only valid threads compute)
        if (is_valid) {
            let tile_start = tile * 256u;
            for (var j = 0u; j < 256u; j++) {
                let other_idx = tile_start + j;
                if (other_idx >= uniforms.node_count || other_idx == node_idx) {
                    continue;
                }

                let other_pos = shared_pos[j];
                let other_radius = shared_radius[j];

                // Skip dead slots (negative radius marker)
                if (other_radius < 0.0) {
                    continue;
                }

                let delta = pos - other_pos;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);
                let min_dist = radius + other_radius;

                if (dist < min_dist && dist > EPSILON) {
                    let overlap = min_dist - dist;
                    let n = delta / dist;
                    let push = overlap * 0.5 * uniforms.collision_strength;
                    disp += n * push;
                } else if (dist <= EPSILON && other_idx > node_idx) {
                    // Nodes at exact same position - use deterministic offset to break symmetry
                    // Only one node (the one with smaller index) applies the offset to avoid double-push
                    let angle = f32(node_idx) * 0.618033988749895 * 6.28318530718;  // Golden ratio
                    disp += vec2<f32>(cos(angle), sin(angle)) * uniforms.default_radius * uniforms.collision_strength;
                }
            }
        }

        // ALL threads must reach this barrier
        workgroupBarrier();
    }

    // Record the displacement. Every thread within the buffer writes its
    // slot (disp is still zero for out-of-range, dead and pinned nodes) so
    // the apply pass never reads a value left over from a previous
    // iteration — see the header note on the two-pass split.
    if (node_idx < arrayLength(&displacements)) {
        displacements[node_idx] = disp;
    }
}
