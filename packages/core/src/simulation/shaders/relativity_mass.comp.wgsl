// Relativity Atlas: Mass Aggregation Shader
// Computes hierarchical mass for nodes based on their subtree.
//
// Mass formula: mass[i] = 1 + 0.5 * sum(child_mass)
// This creates "gravitational wells" around high-connectivity nodes.
//
// Iteration strategy: Runs for a fixed number of iterations rather than using
// GPU-to-CPU convergence readback. The convergence flag is still written for
// potential debugging/monitoring use, but is not read back due to the high cost
// of GPU synchronization. The iteration count may need adjustment for very deep
// hierarchies.

struct MassUniforms {
    node_count: u32,
    edge_count: u32,
    iteration: u32,              // Current iteration number
    convergence_threshold: f32,  // Mass change below this = converged
    base_mass: f32,              // Starting mass for all nodes (default: 1.0)
    child_mass_factor: f32,      // How much children contribute to parent mass (default: 0.5)
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: MassUniforms;

// CSR format for outgoing edges (to children)
@group(0) @binding(1) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> csr_targets: array<u32>;

// Degrees (for leaf detection)
@group(0) @binding(3) var<storage, read> degrees: array<u32>;

// Mass values (ping-pong for iteration)
@group(0) @binding(4) var<storage, read> mass_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> mass_out: array<f32>;

// Convergence flag (atomic - any thread can set to 0 if not converged).
// NOTE: This flag is written by the shader but NOT read back by the CPU.
// We use fixed iterations instead of readback due to GPU sync latency costs.
// The flag is retained for debugging and potential future use.
@group(0) @binding(6) var<storage, read_write> converged: atomic<u32>;

const WORKGROUP_SIZE: u32 = 256u;

// Initialize mass values (first iteration)
@compute @workgroup_size(256)
fn init_mass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Initial mass based on degree
    let out_degree = degrees[node_idx * 2u];
    let in_degree = degrees[node_idx * 2u + 1u];
    let total_degree = out_degree + in_degree;

    // Leaves (no children) get base mass
    // Non-leaves get slightly higher initial mass
    if (out_degree == 0u) {
        mass_out[node_idx] = uniforms.base_mass;
    } else {
        mass_out[node_idx] = uniforms.base_mass + f32(total_degree) * 0.1;
    }
}

// Aggregate child masses (one iteration)
@compute @workgroup_size(256)
fn aggregate_mass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let edge_start = csr_offsets[node_idx];
    let edge_end = csr_offsets[node_idx + 1u];
    let out_degree = edge_end - edge_start;

    // Sum child masses
    // SAFETY: Limit iterations to prevent infinite loops from corrupted CSR data
    // Max 10000 children per node should handle any reasonable graph
    let max_edges = min(out_degree, 10000u);
    var child_mass_sum = 0.0;
    for (var e = edge_start; e < edge_start + max_edges; e++) {
        let child_idx = csr_targets[e];
        if (child_idx < uniforms.node_count) {
            child_mass_sum += mass_in[child_idx];
        }
    }

    // New mass = base + factor * child_sum
    let new_mass = uniforms.base_mass + uniforms.child_mass_factor * child_mass_sum;
    let old_mass = mass_in[node_idx];

    mass_out[node_idx] = new_mass;

    // Check convergence
    let diff = abs(new_mass - old_mass);
    if (diff > uniforms.convergence_threshold) {
        atomicStore(&converged, 0u);  // Not converged
    }
}

