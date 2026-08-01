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
//
// Bottom-up aggregation is MULTI-PASS: WGSL atomics are relaxed-only and there
// is no device-scope fence, so the classic CUDA visit-counter pattern (second
// child to arrive reads the sibling's plain-stored mass/COM within the same
// dispatch) is a spec-level data race — a reader can observe the zeros written
// by clear_tree even though the sibling subtree's thread already stored its
// values. Instead, aggregate_pass is dispatched repeatedly; a node only
// consumes child data stamped in an EARLIER dispatch (inter-dispatch ordering
// guarantees visibility) or data this thread wrote itself (program order).
// With the in-pass upward walk, ceil(log2(N)) dispatches provably suffice:
// a node needs a later pass than its children only when both subtrees finish
// in the same pass, which requires each to hold >= 2^(pass-1) leaves.

struct TreeUniforms {
    node_count: u32,        // Number of particles (N leaves)
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    root_size: f32,
    aggregation_pass: u32,  // Current aggregate_pass dispatch (2, 3, ...); leaves are stamped 1
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: TreeUniforms;

// Sorted Morton codes (from radix sort)
@group(0) @binding(1) var<storage, read> morton_codes: array<u32>;

// Sorted particle indices (from radix sort). High bit set = dead slot
// (see morton.comp.wgsl DEAD_INDEX_BIT).
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
@group(0) @binding(7) var<storage, read_write> node_com: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> node_mass: array<f32>;
@group(0) @binding(9) var<storage, read_write> node_size: array<f32>;

// Aggregation readiness stamps, one per tree node (2N-1). 0 = not aggregated;
// leaves are stamped 1 by init_leaves; internal nodes are stamped with the
// aggregate_pass number that computed them. Atomic because stamps are read
// while other invocations of the same dispatch may store them — the VALUE
// read is race-free, and node data is only consumed for stamps from earlier
// dispatches (or this thread's own writes), never same-pass strangers.
@group(0) @binding(10) var<storage, read_write> agg_ready: array<atomic<u32>>;

// Per-particle simulation mass (f32 per slot, 1.0 = one body), seeding the
// leaves the bottom-up aggregation sums. Bindings 1-10 already exhaust the 10
// storage buffers per stage the device is asked for, so only init_leaves' own
// pipeline layout declares this one — the shared tree layout stays at 10.
@group(0) @binding(11) var<storage, read> particle_mass: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const DEAD_INDEX_BIT: u32 = 0x80000000u;

// Count leading zeros in common prefix
fn clz(x: u32) -> u32 {
    if (x == 0u) {
        return 32u;
    }
    var v = x;  // Copy to mutable variable (WGSL parameters are immutable)
    var n = 0u;
    if ((v & 0xFFFF0000u) == 0u) { n += 16u; v <<= 16u; }
    if ((v & 0xFF000000u) == 0u) { n += 8u; v <<= 8u; }
    if ((v & 0xF0000000u) == 0u) { n += 4u; v <<= 4u; }
    if ((v & 0xC0000000u) == 0u) { n += 2u; v <<= 2u; }
    if ((v & 0x80000000u) == 0u) { n += 1u; }
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

// Convert child reference to tree node index
// Negative values are leaves: -(leaf_idx + 1) -> node_idx = N - 1 + leaf_idx
fn child_to_node_idx(child: i32, n: u32) -> u32 {
    if (child < 0) {
        return n - 1u + u32(-(child + 1));
    }
    return u32(child);
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
    // SAFETY: Limit iterations to prevent infinite loop with duplicate Morton codes
    let delta_min = min(d_left, d_right);
    var l_max = 2;
    var search_iter = 0u;
    while (delta(i, i + l_max * d, n) > delta_min && search_iter < 32u) {
        l_max *= 2;
        search_iter += 1u;
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

    // Find split position: binary search with t halving from l via ceiling
    // division and an explicit final t=1 probe (Karras' published loop).
    // Terminates arithmetically in O(log l) probes — a `while (t >= 1)`
    // ceiling-division formulation never leaves t < 1 and would only exit
    // through i32 overflow wraparound after ~31 wasted iterations.
    let delta_node = delta(i, j, n);
    var s = 0;
    if (l > 0) {
        t = l;
        loop {
            t = (t + 1) / 2;  // Ceiling halving: l, ceil(l/2), ..., 2, 1
            if (delta(i, i + (s + t) * d, n) > delta_node) {
                s += t;
            }
            if (t <= 1) {
                break;
            }
        }
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

    // Get original particle index from sorted order; the high bit marks a
    // dead slot (morton.comp.wgsl), which must not exert repulsion.
    let particle_ref = sorted_indices[leaf_idx];
    let is_dead = (particle_ref & DEAD_INDEX_BIT) != 0u;
    let particle_idx = particle_ref & ~DEAD_INDEX_BIT;

    // Get position from consolidated vec2 buffer
    let pos = positions[particle_idx];

    // Set leaf properties
    node_com[node_idx] = pos;
    // Dead slots stay massless; live ones carry whatever mass they were given,
    // which is 1.0 unless a collapsed LOD subtree rolled up into this slot.
    node_mass[node_idx] = select(particle_mass[particle_idx], 0.0, is_dead);

    // Leaf size: minimum floor to prevent zero-size leaves breaking the
    // theta criterion (size/distance < theta would always pass with size=0).
    node_size[node_idx] = max(1.0, uniforms.root_size / 256.0);

    // Leaves are ready for aggregation from stamp 1 (aggregate passes are 2+)
    atomicStore(&agg_ready[node_idx], 1u);
}

// Phase 3: Bottom-up aggregation of centers of mass — one of several passes.
//
// Each thread owns one internal node. A node is aggregated once both children
// carry a readiness stamp from an EARLIER dispatch (their plain-stored
// mass/COM/size are then visible per WebGPU inter-dispatch ordering). After
// aggregating, the thread walks upward: it may also aggregate ancestors whose
// other child was stamped in an earlier pass, since its own just-written data
// is visible to itself in program order. Exactly one thread can aggregate a
// given node in a given pass (the sibling side stopped in the pass that
// stamped it, and the node's own thread rejects same-pass stamps), so there
// are no write conflicts on node data.
@compute @workgroup_size(256)
fn aggregate_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = uniforms.node_count;

    // Internal nodes are 0..N-2
    if (n < 2u || idx >= n - 1u) {
        return;
    }

    let p = uniforms.aggregation_pass;

    if (atomicLoad(&agg_ready[idx]) != 0u) {
        // Already aggregated in an earlier pass
        return;
    }

    var node = idx;
    // Tree node index of the child this thread aggregated itself this pass
    // (its data is program-order visible regardless of stamp); sentinel = none.
    var own_child = 0xFFFFFFFFu;

    // SAFETY: tree depth is bounded by the 64 distinct delta values, so 64
    // upward steps cover any well-formed tree; the cap breaks pointer cycles
    // from corrupted topology.
    for (var step = 0u; step < 64u; step += 1u) {
        let left_idx = child_to_node_idx(left_child[node], n);
        let right_idx = child_to_node_idx(right_child[node], n);

        // A child is consumable if this thread wrote it (own_child) or it was
        // stamped in an earlier dispatch (0 < stamp < p). Same-pass stamps
        // (== p) are rejected: their data writes are not yet visible here.
        var left_ok = left_idx == own_child;
        if (!left_ok) {
            let stamp = atomicLoad(&agg_ready[left_idx]);
            left_ok = stamp != 0u && stamp < p;
        }
        var right_ok = right_idx == own_child;
        if (!right_ok) {
            let stamp = atomicLoad(&agg_ready[right_idx]);
            right_ok = stamp != 0u && stamp < p;
        }
        if (!left_ok || !right_ok) {
            // Blocked; a later pass (or the sibling's walker) finishes this node
            return;
        }

        let left_mass = node_mass[left_idx];
        let right_mass = node_mass[right_idx];

        if (left_mass <= 0.0) {
            // Left subtree is all dead slots — inherit the right child so a
            // dead cluster contributes neither mass nor spurious extent
            node_com[node] = node_com[right_idx];
            node_mass[node] = right_mass;
            node_size[node] = node_size[right_idx];
        } else if (right_mass <= 0.0) {
            node_com[node] = node_com[left_idx];
            node_mass[node] = left_mass;
            node_size[node] = node_size[left_idx];
        } else {
            let left_com = node_com[left_idx];
            let right_com = node_com[right_idx];
            let total_mass = left_mass + right_mass;
            node_com[node] = (left_com * left_mass + right_com * right_mass) / total_mass;
            node_mass[node] = total_mass;

            // Node size = distance between children's centers + max child extent.
            // This gives a proper geometric measure of the subtree's spatial span
            // without requiring extra AABB buffers (WebGPU limits: 10 storage buffers).
            let child_dist = length(left_com - right_com);
            node_size[node] = child_dist + max(node_size[left_idx], node_size[right_idx]);
        }

        atomicStore(&agg_ready[node], p);

        // Walk up: this node's data is now program-order visible to this
        // thread; the parent can be aggregated if its other child is ready
        // from an earlier pass.
        let par = parent[node];
        if (par < 0) {
            return;
        }
        own_child = node;
        node = u32(par);
        if (atomicLoad(&agg_ready[node]) != 0u) {
            // Aggregated in an earlier pass (same-pass duplication is
            // impossible — see uniqueness argument above)
            return;
        }
    }
}

// Clear tree data for fresh build
@compute @workgroup_size(256)
fn clear_tree(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = uniforms.node_count;

    if (n == 0u) { return; }

    let total_nodes = 2u * n - 1u;  // N-1 internal + N leaves

    if (idx >= total_nodes) {
        return;
    }

    node_com[idx] = vec2<f32>(0.0, 0.0);
    node_mass[idx] = 0.0;
    node_size[idx] = 0.0;

    // Readiness stamps cover ALL nodes (internal + leaves)
    atomicStore(&agg_ready[idx], 0u);

    // CRITICAL: Initialize parent for ALL nodes (internal + leaves)
    // Without this, aggregation can follow garbage parent pointers
    // and enter an infinite loop. Leaves start at index n-1.
    parent[idx] = -1;

    // Clear internal node structure (children)
    if (idx < n - 1u) {
        left_child[idx] = 0;
        right_child[idx] = 0;
    }
}
