// Grid-Based Collision Detection Shader — Atomic Linked Lists
//
// Uses spatial hashing with per-cell atomic linked lists for O(n·k) collision
// detection where k is the average number of nodes per cell neighborhood.
//
// Pipeline (3 dispatches per iteration):
// 1. clear_cells  - Reset all cell head pointers to EMPTY
// 2. build_lists  - Each node atomically prepends itself to its cell's list
// 3. resolve_grid - Walk linked lists in 3x3 cell neighborhood for overlaps
//
// Cell size = 2 * max_radius * radius_multiplier, guaranteeing that any two
// overlapping nodes (distance < r_i + r_j <= 2 * max_radius) are in the same
// cell or adjacent cells.

struct GridCollisionUniforms {
    node_count: u32,
    grid_width: u32,
    grid_height: u32,
    cell_size: f32,
    bounds_min_x: f32,
    bounds_min_y: f32,
    collision_strength: f32,
    radius_multiplier: f32,
    default_radius: f32,
    total_cells: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: GridCollisionUniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> node_sizes: array<f32>;
@group(0) @binding(3) var<storage, read_write> cell_head: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> node_next: array<u32>;
@group(0) @binding(5) var<storage, read_write> node_cell: array<u32>;

const EPSILON: f32 = 0.0001;
const EMPTY: u32 = 0xFFFFFFFFu;
// Safety cap for linked list traversal. With cell_size = 2 * max_radius,
// average chain length is 1-5 nodes. 64 handles extreme clustering gracefully.
const MAX_CHAIN_LEN: u32 = 64u;

// Phase 1: Clear all cell head pointers to EMPTY.
// One thread per cell. Must complete before build_lists.
@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.total_cells) {
        return;
    }

    atomicStore(&cell_head[idx], EMPTY);
}

// Phase 2: Build per-cell linked lists via atomic prepend.
// Each node computes its grid cell, then atomically inserts itself at the
// head of that cell's list. The old head becomes the node's "next" pointer.
@compute @workgroup_size(256)
fn build_lists(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[idx];

    // Map position to grid cell, clamped to valid range.
    // Using i32 intermediate avoids undefined behavior when positions are
    // slightly outside bounds (negative float -> u32 is UB in WGSL).
    let rel_x = (pos.x - uniforms.bounds_min_x) / uniforms.cell_size;
    let rel_y = (pos.y - uniforms.bounds_min_y) / uniforms.cell_size;
    let cell_x = clamp(i32(floor(rel_x)), 0, i32(uniforms.grid_width) - 1);
    let cell_y = clamp(i32(floor(rel_y)), 0, i32(uniforms.grid_height) - 1);
    let cell_hash = u32(cell_y) * uniforms.grid_width + u32(cell_x);

    // Store cell assignment for resolve_grid to read without recomputing
    node_cell[idx] = cell_hash;

    // Atomic prepend: swap this node's index into the cell head,
    // link the previous head as this node's next pointer
    let old_head = atomicExchange(&cell_head[cell_hash], idx);
    node_next[idx] = old_head;
}

// Phase 3: Resolve collisions using spatial grid.
// For each node, walks the linked lists in its cell and 8 adjacent cells,
// checking for overlapping neighbors and accumulating displacement.
@compute @workgroup_size(256)
fn resolve_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = gid.x;
    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];

    // Get this node's radius
    var radius_i = node_sizes[node_idx];
    if (radius_i <= EPSILON) {
        radius_i = uniforms.default_radius;
    }
    radius_i *= uniforms.radius_multiplier;

    // Look up this node's cell from the build phase
    let my_cell = node_cell[node_idx];
    let cell_x = i32(my_cell % uniforms.grid_width);
    let cell_y = i32(my_cell / uniforms.grid_width);

    var disp = vec2<f32>(0.0, 0.0);

    // Check 3x3 neighborhood of cells
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = cell_x + dx;
            let ny = cell_y + dy;

            // Skip out-of-bounds cells
            if (nx < 0 || nx >= i32(uniforms.grid_width) || ny < 0 || ny >= i32(uniforms.grid_height)) {
                continue;
            }

            let neighbor_cell = u32(ny) * uniforms.grid_width + u32(nx);

            // Walk the linked list for this cell
            var current = atomicLoad(&cell_head[neighbor_cell]);
            var safety = 0u;

            while (current != EMPTY && safety < MAX_CHAIN_LEN) {
                let other_idx = current;

                // Advance to next node before any continue statements
                current = node_next[current];
                safety += 1u;

                // Don't collide with self
                if (other_idx == node_idx) {
                    continue;
                }

                let other_pos = positions[other_idx];

                // Get other node's radius
                var radius_j = node_sizes[other_idx];
                if (radius_j <= EPSILON) {
                    radius_j = uniforms.default_radius;
                }
                radius_j *= uniforms.radius_multiplier;

                // Check for overlap
                let delta = pos - other_pos;
                let dist_sq = dot(delta, delta);
                let dist = sqrt(dist_sq);
                let min_dist = radius_i + radius_j;

                if (dist < min_dist && dist > EPSILON) {
                    let overlap = min_dist - dist;
                    let n = delta / dist;
                    let push = overlap * 0.5 * uniforms.collision_strength;
                    disp += n * push;
                } else if (dist <= EPSILON && other_idx > node_idx) {
                    // Nodes at same position — deterministic offset to break symmetry.
                    // Only the lower-index node applies the offset to avoid double-push.
                    let angle = f32(node_idx) * 0.618033988749895 * 6.28318530718;
                    disp += vec2<f32>(cos(angle), sin(angle)) * uniforms.default_radius * uniforms.collision_strength;
                }
            }
        }
    }

    // Apply accumulated displacement
    if (any(abs(disp) > vec2<f32>(EPSILON))) {
        positions[node_idx] = pos + disp;
    }
}
