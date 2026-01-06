// Bottom-Up Quadtree Construction Compute Shader
// Builds a quadtree from Morton-sorted nodes for Barnes-Hut algorithm
//
// After nodes are sorted by Morton code, the quadtree structure can be
// implicitly defined: consecutive nodes with the same Morton prefix belong
// to the same quadtree cell at that level.

struct QuadtreeUniforms {
    node_count: u32,
    max_depth: u32,          // Maximum tree depth
    theta_squared: f32,      // Barnes-Hut opening angle squared
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: QuadtreeUniforms;

// Sorted node data (by Morton code)
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read> morton_codes: array<u32>;
@group(0) @binding(3) var<storage, read> positions_x: array<f32>;
@group(0) @binding(4) var<storage, read> positions_y: array<f32>;

// Quadtree node data (internal nodes)
// Each internal node stores: center of mass (x, y), total mass, child info
@group(0) @binding(5) var<storage, read_write> tree_nodes_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> tree_nodes_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> tree_nodes_mass: array<f32>;
@group(0) @binding(8) var<storage, read_write> tree_children: array<u32>;  // Packed child indices

// Range markers: for each tree node, store [start, end) of leaf range
@group(0) @binding(9) var<storage, read_write> tree_ranges: array<vec2<u32>>;

const MAX_TREE_NODES: u32 = 262144u;  // 256K internal nodes max

// Get the Morton prefix at a given level (0 = root, higher = deeper)
fn get_morton_prefix(code: u32, level: u32) -> u32 {
    let shift = (uniforms.max_depth - level) * 2u;
    return code >> shift;
}

// Find the first node with a different prefix at given level
fn find_range_end(start: u32, level: u32) -> u32 {
    if (start >= uniforms.node_count) {
        return start;
    }

    let start_prefix = get_morton_prefix(morton_codes[start], level);

    var end = start + 1u;
    while (end < uniforms.node_count) {
        let end_prefix = get_morton_prefix(morton_codes[end], level);
        if (end_prefix != start_prefix) {
            break;
        }
        end++;
    }

    return end;
}

// Build leaf-level tree nodes (each leaf corresponds to a small group of particles)
@compute @workgroup_size(256)
fn build_leaves(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf_idx = global_id.x;

    // Determine which nodes belong to this leaf
    // At max depth, each Morton code prefix defines a leaf cell
    let start = leaf_idx;
    if (start >= uniforms.node_count) {
        return;
    }

    let end = find_range_end(start, uniforms.max_depth);

    // Compute center of mass for this leaf cell
    var com_x = 0.0;
    var com_y = 0.0;
    var total_mass = 0.0;

    for (var i = start; i < end; i++) {
        let node_idx = sorted_indices[i];
        let x = positions_x[node_idx];
        let y = positions_y[node_idx];
        let mass = 1.0;  // Unit mass per node

        com_x += x * mass;
        com_y += y * mass;
        total_mass += mass;
    }

    if (total_mass > 0.0) {
        com_x /= total_mass;
        com_y /= total_mass;
    }

    // Store in tree node arrays
    // Leaf nodes are stored at the end of the tree node array
    let tree_idx = MAX_TREE_NODES - 1u - leaf_idx;
    tree_nodes_x[tree_idx] = com_x;
    tree_nodes_y[tree_idx] = com_y;
    tree_nodes_mass[tree_idx] = total_mass;
    tree_ranges[tree_idx] = vec2<u32>(start, end);
}

// Build internal nodes bottom-up (must be called after leaves are built)
@compute @workgroup_size(256)
fn build_internal(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(num_workgroups) num_groups: vec3<u32>) {
    // This shader builds one level of internal nodes
    // It must be dispatched multiple times, once per level from bottom to top

    let node_idx = global_id.x;

    // Determine level from dispatch parameters (passed via push constant or uniform)
    // For now, assume level is encoded in workgroup count

    // Each internal node aggregates 4 children (quadtree)
    let child_base = node_idx * 4u;

    var com_x = 0.0;
    var com_y = 0.0;
    var total_mass = 0.0;
    var child_mask = 0u;

    // Aggregate children
    for (var q = 0u; q < 4u; q++) {
        let child_idx = child_base + q;
        let child_mass = tree_nodes_mass[child_idx];

        if (child_mass > 0.0) {
            com_x += tree_nodes_x[child_idx] * child_mass;
            com_y += tree_nodes_y[child_idx] * child_mass;
            total_mass += child_mass;
            child_mask |= (1u << q);
        }
    }

    if (total_mass > 0.0) {
        com_x /= total_mass;
        com_y /= total_mass;
    }

    // Store internal node
    tree_nodes_x[node_idx] = com_x;
    tree_nodes_y[node_idx] = com_y;
    tree_nodes_mass[node_idx] = total_mass;
    tree_children[node_idx] = child_mask | (child_base << 4u);  // Pack child info
}
