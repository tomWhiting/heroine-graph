// Karras Binary Radix Tree Construction
// Implements Karras 2012 "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//
// Given N sorted Morton codes, constructs a binary radix tree with:
// - N-1 internal nodes (indices 0..N-2)
// - N leaf nodes (indices N-1..2N-2, or referenced as negative indices)
//
// Each internal node has exactly 2 children (binary tree).
// Children can be internal nodes (positive index) or leaves (negative index).
//
// Uses vec2<f32> layout for consolidated position data.

struct TreeUniforms {
    node_count: u32,        // Number of particles (N leaves)
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    root_size: f32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: TreeUniforms;

// Sorted Morton codes (from radix sort)
@group(0) @binding(1) var<storage, read> morton_codes: array<u32>;

// Sorted particle indices (from radix sort)
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;

// Original particle positions - vec2<f32> per particle
@group(0) @binding(3) var<storage, read> positions: array<vec2<f32>>;

// Tree structure (N-1 internal nodes)
// Negative values indicate leaf index: child = -(leaf_idx + 1)
@group(0) @binding(4) var<storage, read_write> left_child: array<i32>;
@group(0) @binding(5) var<storage, read_write> right_child: array<i32>;
@group(0) @binding(6) var<storage, read_write> parent: array<i32>;

// Node properties (2N-1 total: N-1 internal + N leaves)
// Internal nodes: indices 0..N-2
// Leaf nodes: indices N-1..2N-2
@group(0) @binding(7) var<storage, read_write> node_com_x: array<f32>;
@group(0) @binding(8) var<storage, read_write> node_com_y: array<f32>;
@group(0) @binding(9) var<storage, read_write> node_mass: array<f32>;
@group(0) @binding(10) var<storage, read_write> node_size: array<f32>;

// Atomic counter for bottom-up aggregation
@group(0) @binding(11) var<storage, read_write> visit_count: array<atomic<u32>>;

const WORKGROUP_SIZE: u32 = 256u;

// Count leading zeros in common prefix
fn clz(x: u32) -> u32 {
    if (x == 0u) {
        return 32u;
    }
    var n = 0u;
    if ((x & 0xFFFF0000u) == 0u) { n += 16u; x <<= 16u; }
    if ((x & 0xFF000000u) == 0u) { n += 8u; x <<= 8u; }
    if ((x & 0xF0000000u) == 0u) { n += 4u; x <<= 4u; }
    if ((x & 0xC0000000u) == 0u) { n += 2u; x <<= 2u; }
    if ((x & 0x80000000u) == 0u) { n += 1u; }
    return n;
}

// Compute length of common prefix between keys at indices i and j
fn delta(i: i32, j: i32, n: i32) -> i32 {
    // Handle out of range
    if (j < 0 || j >= n) {
        return -1;
    }

    let ki = morton_codes[i];
    let kj = morton_codes[u32(j)];

    // If keys are identical, use index as tiebreaker
    if (ki == kj) {
        return i32(32u + clz(u32(i) ^ u32(j)));
    }

    return i32(clz(ki ^ kj));
}

// Phase 1: Build tree topology using Karras algorithm
// Each thread processes one internal node (indices 0..N-2)
@compute @workgroup_size(256)
fn build_topology(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = i32(uniforms.node_count);

    // Internal nodes are 0..N-2 (total N-1 internal nodes)
    if (idx >= uniforms.node_count - 1u) {
        return;
    }

    let i = i32(idx);

    // Determine direction of the range
    let d_left = delta(i, i - 1, n);
    let d_right = delta(i, i + 1, n);
    let d = select(-1, 1, d_right > d_left);

    // Compute upper bound for the range length
    let delta_min = min(d_left, d_right);
    var l_max = 2;
    while (delta(i, i + l_max * d, n) > delta_min) {
        l_max *= 2;
    }

    // Binary search for the actual range length
    var l = 0;
    var t = l_max / 2;
    while (t >= 1) {
        if (delta(i, i + (l + t) * d, n) > delta_min) {
            l += t;
        }
        t /= 2;
    }
    let j = i + l * d;

    // Find split position
    let delta_node = delta(i, j, n);
    var s = 0;
    var div = 2;
    t = (l + div - 1) / div;  // Ceiling division
    while (t >= 1) {
        let new_split = i + (s + t) * d;
        if (new_split >= 0 && new_split < n) {
            if (delta(i, new_split, n) > delta_node) {
                s += t;
            }
        }
        div *= 2;
        t = (l + div - 1) / div;
    }
    let split = i + s * d + min(d, 0);

    // Set children
    // Left child
    if (min(i, j) == split) {
        // Left child is a leaf (leaf index = split)
        left_child[idx] = -(split + 1);  // Negative indicates leaf
        // Set parent of leaf
        parent[u32(n - 1 + split)] = i32(idx);
    } else {
        // Left child is internal node
        left_child[idx] = split;
        parent[u32(split)] = i32(idx);
    }

    // Right child
    if (max(i, j) == split + 1) {
        // Right child is a leaf (leaf index = split + 1)
        right_child[idx] = -(split + 2);  // Negative indicates leaf
        // Set parent of leaf
        parent[u32(n - 1 + split + 1)] = i32(idx);
    } else {
        // Right child is internal node
        right_child[idx] = split + 1;
        parent[u32(split + 1)] = i32(idx);
    }

    // Root has no parent
    if (idx == 0u) {
        parent[0] = -1;
    }

    // Initialize visit count for aggregation
    atomicStore(&visit_count[idx], 0u);
}

// Phase 2: Initialize leaf nodes with particle data
// Each thread processes one leaf (particle)
@compute @workgroup_size(256)
fn init_leaves(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf_idx = global_id.x;
    let n = uniforms.node_count;

    if (leaf_idx >= n) {
        return;
    }

    // Leaf nodes are stored at indices N-1..2N-2
    let node_idx = n - 1u + leaf_idx;

    // Get original particle index from sorted order
    let particle_idx = sorted_indices[leaf_idx];

    // Get position from consolidated vec2 buffer
    let pos = positions[particle_idx];

    // Set leaf properties
    node_com_x[node_idx] = pos.x;
    node_com_y[node_idx] = pos.y;
    node_mass[node_idx] = 1.0;  // Each particle has mass 1

    // Leaf bounding box size derived from tree depth (root_size / 256.0).
    // This provides the cell size used in the Barnes-Hut theta criterion.
    node_size[node_idx] = uniforms.root_size / 256.0;
}

// Phase 3: Bottom-up aggregation of centers of mass
// Each thread processes one leaf, then walks up the tree
// Using atomic counters to ensure both children are processed before parent
@compute @workgroup_size(256)
fn aggregate_bottom_up(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf_idx = global_id.x;
    let n = uniforms.node_count;

    if (leaf_idx >= n) {
        return;
    }

    // Start from leaf node
    let leaf_node_idx = n - 1u + leaf_idx;

    // Get parent of this leaf
    var current_parent = parent[leaf_node_idx];

    // Walk up the tree
    while (current_parent >= 0) {
        let parent_idx = u32(current_parent);

        // Increment visit count - returns old value
        let visit = atomicAdd(&visit_count[parent_idx], 1u);

        if (visit == 0u) {
            // First child to arrive - wait for sibling
            return;
        }

        // Second child to arrive - compute parent properties
        let left = left_child[parent_idx];
        let right = right_child[parent_idx];

        // Get child indices (convert negative leaf references)
        let left_idx = select(u32(left), n - 1u + u32(-(left + 1)), left < 0);
        let right_idx = select(u32(right), n - 1u + u32(-(right + 1)), right < 0);

        // Aggregate mass
        let left_mass = node_mass[left_idx];
        let right_mass = node_mass[right_idx];
        let total_mass = left_mass + right_mass;

        if (total_mass > 0.0) {
            // Weighted center of mass
            let com_x = (node_com_x[left_idx] * left_mass + node_com_x[right_idx] * right_mass) / total_mass;
            let com_y = (node_com_y[left_idx] * left_mass + node_com_y[right_idx] * right_mass) / total_mass;

            node_com_x[parent_idx] = com_x;
            node_com_y[parent_idx] = com_y;
            node_mass[parent_idx] = total_mass;

            // Node size is max of child sizes (approximate bounding box)
            let left_size = node_size[left_idx];
            let right_size = node_size[right_idx];
            node_size[parent_idx] = max(left_size, right_size) * 2.0;
        }

        // Move to next parent
        current_parent = parent[parent_idx];
    }
}

// Clear tree data for fresh build
@compute @workgroup_size(256)
fn clear_tree(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = uniforms.node_count;
    let total_nodes = 2u * n - 1u;  // N-1 internal + N leaves

    if (idx >= total_nodes) {
        return;
    }

    node_com_x[idx] = 0.0;
    node_com_y[idx] = 0.0;
    node_mass[idx] = 0.0;
    node_size[idx] = 0.0;

    // Clear internal node structure
    if (idx < n - 1u) {
        left_child[idx] = 0;
        right_child[idx] = 0;
        parent[idx] = -1;
        atomicStore(&visit_count[idx], 0u);
    }
}
