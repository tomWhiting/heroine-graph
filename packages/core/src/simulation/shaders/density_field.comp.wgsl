// Density Field Compute Shader
// O(n) force computation using a density grid
//
// Phase 1: Clear grid
// Phase 2: Accumulate density from nodes (continuous splat)
// Phase 3: Sample gradient (bilinear) and apply forces
//
// CONTINUITY: the force field must be a continuous function of world
// position. The original implementation truncated each node to its cell for
// both splatting and gradient sampling, making the field piecewise-constant
// per cell — force gradients stepped at every cell edge, producing the
// grid/cross-hatch artifacts and boundary jitter (a node crossing a cell
// edge saw its force flip discontinuously every frame). This version:
// - splats with kernel weights computed from the node's EXACT position
//   (distance from each cell center to the continuous grid coordinate),
//   using a tail-subtracted Gaussian that reaches exactly 0 at the cutoff
//   radius, so density is C0-continuous in node position;
// - samples the gradient bilinearly at the node's exact position
//   (central differences at the 4 surrounding cell centers, blended);
// - normalizes the gradient by the world-space cell size, so the force
//   magnitude is grid-resolution independent and weakens as the layout
//   (and therefore each cell) grows instead of feeding expansion.
//
// Uses a 2D grid stored as a 1D buffer for simplicity.
// Uses vec2<f32> layout for consolidated position/force data.

struct DensityUniforms {
    node_count: u32,
    grid_width: u32,
    grid_height: u32,
    repulsion_strength: f32,
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    splat_radius: f32,  // In grid cells
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DensityUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> density_grid: array<atomic<u32>>;

// Well radii (bubble mode: per-node splat radius from subtree)
@group(0) @binding(4) var<storage, read> well_radius: array<f32>;

// Node state flags (bit 0 = dead slot from removal). Dead slots sit at a
// zeroed position; without masking they splat phantom density near the
// world origin and push live nodes away from it.
@group(0) @binding(5) var<storage, read> node_flags: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const DENSITY_SCALE: f32 = 1000.0;  // Scale for atomic integer accumulation
const NODE_FLAG_DEAD: u32 = 1u;
const EPSILON: f32 = 0.0001;

// Kernel value at the cutoff radius: gaussian(r*r, r) with sigma = r/2 is
// exp(-2) for every r. Subtracted (and renormalized) so the splat reaches
// exactly zero at the cutoff instead of clipping a 13.5% step there.
const GAUSSIAN_TAIL: f32 = 0.1353352832;

// Reference cell size in world units (the 128-cell default grid over a
// ~2048-unit layout). The gradient is normalized by the actual world cell
// size for scale independence; this constant keeps the default force
// magnitude at the historical tuning scale.
const REFERENCE_CELL_SIZE: f32 = 16.0;

// Convert world position to CONTINUOUS grid coordinates. Cell (i, j) covers
// grid coords [i, i+1) x [j, j+1) and has its center at (i+0.5, j+0.5).
fn world_to_grid_f(pos: vec2<f32>) -> vec2<f32> {
    let bounds_min = vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y);
    let bounds_max = vec2<f32>(uniforms.bounds_max_x, uniforms.bounds_max_y);
    let extent = max(bounds_max - bounds_min, vec2<f32>(EPSILON, EPSILON));
    let normalized = clamp((pos - bounds_min) / extent, vec2<f32>(0.0), vec2<f32>(1.0));
    return normalized * vec2<f32>(f32(uniforms.grid_width), f32(uniforms.grid_height));
}

// Convert grid cell to 1D index
fn grid_index(cell: vec2<i32>) -> u32 {
    return u32(cell.y) * uniforms.grid_width + u32(cell.x);
}

// Read density from grid (convert from atomic u32)
fn read_density(cell: vec2<i32>) -> f32 {
    if (cell.x < 0 || cell.x >= i32(uniforms.grid_width) ||
        cell.y < 0 || cell.y >= i32(uniforms.grid_height)) {
        return 0.0;
    }
    let idx = grid_index(cell);
    return f32(atomicLoad(&density_grid[idx])) / DENSITY_SCALE;
}

// Tail-subtracted Gaussian splat kernel: 1 at the center, exactly 0 at
// dist = radius (C0-continuous support, no cutoff step).
fn splat_kernel(dist_sq: f32, radius: f32) -> f32 {
    let sigma = radius / 2.0;
    let g = exp(-dist_sq / (2.0 * sigma * sigma));
    return max((g - GAUSSIAN_TAIL) / (1.0 - GAUSSIAN_TAIL), 0.0);
}

// Central-difference density gradient at a cell center, in density per cell.
fn density_gradient_at(cell: vec2<i32>) -> vec2<f32> {
    return vec2<f32>(
        read_density(cell + vec2<i32>(1, 0)) - read_density(cell + vec2<i32>(-1, 0)),
        read_density(cell + vec2<i32>(0, 1)) - read_density(cell + vec2<i32>(0, -1)),
    ) * 0.5;
}

// Phase 1: Clear density grid
@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = uniforms.grid_width * uniforms.grid_height;

    if (idx >= total_cells) {
        return;
    }

    atomicStore(&density_grid[idx], 0u);
}

// Phase 2: Accumulate density from nodes
@compute @workgroup_size(256)
fn accumulate_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Dead slots contribute no density
    if ((node_flags[node_idx] & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let pos = positions[node_idx];
    let gpos = world_to_grid_f(pos);

    // Per-node splat radius: use wellRadius scaled to grid cells when available
    let bounds_extent = max(
        uniforms.bounds_max_x - uniforms.bounds_min_x,
        uniforms.bounds_max_y - uniforms.bounds_min_y
    );
    let cell_size = bounds_extent / f32(uniforms.grid_width);
    let node_well = well_radius[node_idx];
    // Use the larger of default splat radius or wellRadius-based radius, capped at 16
    let effective_splat = min(max(uniforms.splat_radius, node_well / max(cell_size, 0.001)), 16.0);

    // Cells whose CENTER (cx + 0.5, cy + 0.5) lies within effective_splat of
    // the node's continuous grid position.
    let lo = vec2<i32>(floor(gpos - effective_splat - 0.5));
    let hi = vec2<i32>(floor(gpos + effective_splat - 0.5)) + vec2<i32>(1, 1);

    for (var cy = lo.y; cy <= hi.y; cy++) {
        for (var cx = lo.x; cx <= hi.x; cx++) {
            // Bounds check
            if (cx < 0 || cx >= i32(uniforms.grid_width) ||
                cy < 0 || cy >= i32(uniforms.grid_height)) {
                continue;
            }

            // Distance from the node's EXACT position to the cell center —
            // the splat is continuous in node position, not snapped to cells.
            let cell_center = vec2<f32>(f32(cx) + 0.5, f32(cy) + 0.5);
            let d = cell_center - gpos;
            let dist_sq = dot(d, d);
            if (dist_sq > effective_splat * effective_splat) {
                continue;
            }

            let weight = splat_kernel(dist_sq, effective_splat);
            let contribution = u32(weight * DENSITY_SCALE);

            if (contribution > 0u) {
                let idx = grid_index(vec2<i32>(cx, cy));
                atomicAdd(&density_grid[idx], contribution);
            }
        }
    }
}

// Phase 3: Compute gradient and apply forces
@compute @workgroup_size(256)
fn apply_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Dead slots receive no forces
    if ((node_flags[node_idx] & NODE_FLAG_DEAD) != 0u) {
        return;
    }

    let pos = positions[node_idx];

    // Sample point in cell-center space: cell centers sit at integer coords
    // of g. Bilinearly blend the central-difference gradients of the 4
    // surrounding cell centers — the resulting force field is continuous in
    // position instead of constant per cell with jumps at every boundary.
    let g = world_to_grid_f(pos) - vec2<f32>(0.5, 0.5);
    let base = vec2<i32>(floor(g));
    let f = g - vec2<f32>(base);

    let g00 = density_gradient_at(base);
    let g10 = density_gradient_at(base + vec2<i32>(1, 0));
    let g01 = density_gradient_at(base + vec2<i32>(0, 1));
    let g11 = density_gradient_at(base + vec2<i32>(1, 1));

    // Gradient points toward higher density (density per cell)
    let gradient_cells = mix(mix(g00, g10, f.x), mix(g01, g11, f.x), f.y);

    // Normalize to world units per axis: makes the force magnitude
    // independent of grid resolution, and — because cells grow with the
    // layout bounds — repulsion weakens as the layout expands instead of
    // staying constant and feeding runaway expansion.
    let bounds_min = vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y);
    let bounds_max = vec2<f32>(uniforms.bounds_max_x, uniforms.bounds_max_y);
    let cell_size_world = max(
        (bounds_max - bounds_min) /
            vec2<f32>(f32(uniforms.grid_width), f32(uniforms.grid_height)),
        vec2<f32>(EPSILON, EPSILON),
    );
    let gradient_world = gradient_cells / cell_size_world * REFERENCE_CELL_SIZE;

    // Force is opposite to gradient (away from high density)
    let force = -gradient_world * uniforms.repulsion_strength;

    // Accumulate force
    forces[node_idx] += force;
}
