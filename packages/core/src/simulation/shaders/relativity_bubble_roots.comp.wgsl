// Relativity Atlas: Forest-Root Bubble Separation (bubble mode)
//
// Keeps the wells of the containment forest's ROOTS from overlapping. One
// thread per root; each walks its own slice of the root list and applies the
// same continuous quadratic overlap force the sibling pass uses for its
// phantom zones.
//
// # The induction this completes
//
// relativity_sibling.comp.wgsl holds a node inside its parent's well and holds
// siblings' wells apart. Those two clauses separate any pair whose lowest
// common ancestor is a real node, by descending to the sibling pair on the path
// between them. They say nothing about a pair in different trees, because such
// a pair has no common ancestor to descend from — which is precisely the case
// the algorithm's O(N + E) sibling model was missing, and which the density
// grid was standing in for. Separating the roots supplies the base case, and
// the three clauses together imply that every unrelated pair is separated.
//
// Nothing here consults a grid, a cell or a bucket: the force is a function of
// pairwise distance and the two well radii only, so its gradient is continuous
// everywhere and there is no cell boundary for a cross-hatch to form on.
//
// # Cost
//
// O(R * MAX_ROOT_INTERACTIONS) in wall time, not merely in force evaluations:
// the loop STRIDES by the group count, so a thread visits its own group and
// nothing else. Filtering inside a full scan computes exactly the same forces
// under the same cap while still touching every entry, which is O(R^2) work —
// 45 ms per dispatch at R = 35000 on an M-series adapter against 3.7 ms strided,
// three frame budgets for one of the algorithm's eight passes.
//
// That is worth the care because R = N is reachable from a legal input, not
// only a pathological one: a supplied hierarchy may declare every slot a root
// (it validates — the root subtree sizes still sum to nodeCount), and a heavily
// fragmented forest gets partway there on its own.
//
// The truncation itself remains an approximation, and a static one: two roots
// in different groups never repel, however deeply their wells overlap. Widening
// the cap only moves the cliff; the honest fix is a spatial structure over the
// root list, and simulation/algorithms/barnes-hut.ts already answers this exact
// question in O(R log R) — the work is to build its tree over the roots and
// read the far-field approximation here instead of a pairwise sweep. What that
// fix must not have to undo first is a performance cliff underneath it, which
// is what the scan was.

struct BubbleRootUniforms {
    // Slot bound; entries at or above it are ignored, mirroring every other pass.
    node_count: u32,
    // Live length of root_list.
    root_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
}

@group(0) @binding(0) var<uniform> uniforms: BubbleRootUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Well radii (bubble mode: subtree-based collision boundaries)
@group(0) @binding(3) var<storage, read> well_radius: array<f32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD)
@group(0) @binding(4) var<storage, read> node_flags: array<u32>;

// Compacted list of forest-root slots, ascending. Written at upload time from
// the retained hierarchy's parent column.
@group(0) @binding(5) var<storage, read> root_list: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;
const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Budget on force computations per root. Group selection keeps the expected
// number of selected partners at or below this.
const MAX_ROOT_INTERACTIONS: u32 = 64u;

// Number of modulo-groups for a truncated interaction list.
//
// Identical rule to relativity_sibling.comp.wgsl's sibling_group_count, and for
// the identical reason: `i % G == j % G` is symmetric in (i, j), and both
// endpoints derive the same G from the same shared list length, so pairwise
// visibility is MUTUAL. Newton's third law holds for every computed pair and
// truncation injects no net momentum. A head-prefix cap instead would push the
// tail of the root list against its head with no recoil, drifting the entire
// layout in one direction every frame.
//
// The group is taken over the ROOT LIST POSITION rather than the node index the
// sibling pass uses, and that is what makes the cost claim in the header true:
// positions are contiguous, so a group is an arithmetic progression the loop can
// stride over. Grouping by node index would be equally symmetric but only
// testable by a scan, since the list holds an arbitrary ascending subset of the
// slots. Mutuality is unaffected — both endpoints of a pair read the same list
// and so derive the same G.
fn root_group_count(list_len: u32, cap: u32) -> u32 {
    return max((list_len + cap - 1u) / max(cap, 1u), 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry = global_id.x;

    if (entry >= uniforms.root_count) {
        return;
    }

    let node_idx = root_list[entry];
    if (node_idx >= uniforms.node_count) {
        return;
    }
    if ((node_flags[node_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let pos = positions[node_idx];
    let radius_i = well_radius[node_idx];
    let groups = root_group_count(uniforms.root_count, MAX_ROOT_INTERACTIONS);

    var force = vec2<f32>(0.0, 0.0);

    // Stride, not scan: the first member of this thread's group is `entry %
    // groups` and every subsequent one is `groups` further along, so the loop
    // runs ceil(R / groups) <= MAX_ROOT_INTERACTIONS + 1 times whatever R is.
    for (var other = entry % groups; other < uniforms.root_count; other += groups) {
        let other_idx = root_list[other];

        if (other == entry || other_idx >= uniforms.node_count) {
            continue;
        }
        if ((node_flags[other_idx] & NODE_FLAG_INERT) != 0u) {
            continue;
        }

        let delta = pos - positions[other_idx];
        let dist_sq = dot(delta, delta);
        if (dist_sq < EPSILON) {
            continue;
        }

        let combined_radius = radius_i + well_radius[other_idx];
        let dist = sqrt(dist_sq);
        if (dist >= combined_radius || combined_radius <= 0.0) {
            continue;
        }

        // Quadratic ramp in the normalised overlap: zero value AND zero
        // gradient at the moment the bubbles touch, so a root crossing another's
        // boundary feels the force arrive smoothly rather than as a step.
        let overlap_ratio = (combined_radius - dist) / combined_radius;
        let dir = delta / max(dist, uniforms.min_distance);
        force += dir * uniforms.repulsion_strength * overlap_ratio * overlap_ratio;
    }

    forces[node_idx] += force;
}
