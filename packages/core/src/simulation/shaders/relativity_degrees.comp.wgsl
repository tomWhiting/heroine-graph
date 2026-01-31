// Relativity Atlas: Degree Computation Shader
// Computes out-degree and in-degree for each node from CSR edge data.
//
// For Relativity Atlas, degrees inform:
// - Mass calculation (more connections = higher mass)
// - Sibling identification (nodes with same parent)

struct DegreesUniforms {
    node_count: u32,
    edge_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: DegreesUniforms;

// CSR format: offsets[node_count+1], targets[edge_count]
// outgoing edges: offsets[i]..offsets[i+1] are targets of edges from node i
@group(0) @binding(1) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> csr_targets: array<u32>;

// Inverse CSR format: incoming edges (parents)
@group(0) @binding(3) var<storage, read> csr_inverse_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> csr_inverse_sources: array<u32>;

// Output: degrees[i*2] = out_degree, degrees[i*2+1] = in_degree
@group(0) @binding(5) var<storage, read_write> degrees: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

// Compute degrees from CSR offsets
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    // Out-degree: difference between consecutive offsets
    let out_start = csr_offsets[node_idx];
    let out_end = csr_offsets[node_idx + 1u];
    let out_degree = out_end - out_start;

    // In-degree: from inverse CSR offsets
    let in_start = csr_inverse_offsets[node_idx];
    let in_end = csr_inverse_offsets[node_idx + 1u];
    let in_degree = in_end - in_start;

    // Store degrees
    degrees[node_idx * 2u] = out_degree;
    degrees[node_idx * 2u + 1u] = in_degree;
}

// Alternative: compute total degree only
@compute @workgroup_size(256)
fn compute_total_degree(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let out_degree = csr_offsets[node_idx + 1u] - csr_offsets[node_idx];
    let in_degree = csr_inverse_offsets[node_idx + 1u] - csr_inverse_offsets[node_idx];

    // Store total degree in first slot, 0 in second
    degrees[node_idx * 2u] = out_degree + in_degree;
    degrees[node_idx * 2u + 1u] = 0u;
}
