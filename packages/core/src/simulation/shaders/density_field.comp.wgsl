// Density Field Compute Shader
// O(n) force computation using a density grid
//
// Phase 1: Clear grid
// Phase 2: Accumulate density from nodes
// Phase 3: Compute gradient and apply forces
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

const WORKGROUP_SIZE: u32 = 256u;
const DENSITY_SCALE: f32 = 1000.0;  // Scale for atomic integer accumulation

// Convert world position to grid cell
fn world_to_grid(pos: vec2<f32>) -> vec2<i32> {
    let bounds_min = vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y);
    let bounds_max = vec2<f32>(uniforms.bounds_max_x, uniforms.bounds_max_y);
    let normalized = (pos - bounds_min) / (bounds_max - bounds_min);
    return vec2<i32>(
        clamp(i32(normalized.x * f32(uniforms.grid_width)), 0, i32(uniforms.grid_width) - 1),
        clamp(i32(normalized.y * f32(uniforms.grid_height)), 0, i32(uniforms.grid_height) - 1)
    );
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

// Gaussian falloff for splatting
fn gaussian(dist_sq: f32, radius: f32) -> f32 {
    let sigma = radius / 2.0;
    return exp(-dist_sq / (2.0 * sigma * sigma));
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

    let pos = positions[node_idx];
    let center_cell = world_to_grid(pos);

    // Per-node splat radius: use wellRadius scaled to grid cells when available
    let bounds_extent = max(
        uniforms.bounds_max_x - uniforms.bounds_min_x,
        uniforms.bounds_max_y - uniforms.bounds_min_y
    );
    let cell_size = bounds_extent / f32(uniforms.grid_width);
    let node_well = well_radius[node_idx];
    // Use the larger of default splat radius or wellRadius-based radius, capped at 16
    let effective_splat = min(max(uniforms.splat_radius, node_well / max(cell_size, 0.001)), 16.0);
    let radius = i32(ceil(effective_splat));

    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let cell = center_cell + vec2<i32>(dx, dy);

            // Bounds check
            if (cell.x < 0 || cell.x >= i32(uniforms.grid_width) ||
                cell.y < 0 || cell.y >= i32(uniforms.grid_height)) {
                continue;
            }

            let dist_sq = f32(dx * dx + dy * dy);
            if (dist_sq > effective_splat * effective_splat) {
                continue;
            }

            let weight = gaussian(dist_sq, effective_splat);
            let contribution = u32(weight * DENSITY_SCALE);

            if (contribution > 0u) {
                let idx = grid_index(cell);
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

    let pos = positions[node_idx];
    let cell = world_to_grid(pos);

    // Compute gradient using central differences
    // Sample 4 neighbors
    let left = read_density(cell + vec2<i32>(-1, 0));
    let right = read_density(cell + vec2<i32>(1, 0));
    let down = read_density(cell + vec2<i32>(0, -1));
    let up = read_density(cell + vec2<i32>(0, 1));

    // Gradient points toward higher density
    let gradient = vec2<f32>(right - left, up - down) * 0.5;

    // Force is opposite to gradient (away from high density)
    let force = -gradient * uniforms.repulsion_strength;

    // Accumulate force
    forces[node_idx] += force;
}
