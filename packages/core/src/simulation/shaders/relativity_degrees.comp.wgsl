// Relativity Atlas: Degree Computation Shader
// Computes out-degree and in-degree for each node from CSR edge data.
//
// For Relativity Atlas, degrees inform:
// - Mass calculation (more connections = higher mass)
// - Sibling identification (nodes with same parent)

struct DegreesUniforms {
    node_count: u32,
    edge_count: u32,
    // First element of csr_inverse's source region; see the region map below.
    csr_inverse_sources_base: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: DegreesUniforms;

// CSR format: offsets[node_count+1], targets[edge_count]
// outgoing edges: offsets[i]..offsets[i+1] are targets of edges from node i
@group(0) @binding(1) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> csr_targets: array<u32>;

// Inverse CSR: incoming edges (parents).
//
// INVERSE CSR REGION MAP (S = uniforms.csr_inverse_sources_base, the ALLOCATED
// offset-row length, not node_count + 1 — the regions must not move when the
// live node count does), in u32 elements:
//   [0, S)      offset row: node i's incoming edges are the source entries
//               [csr_inverse[i], csr_inverse[i+1])
//   [S, ...)    source row: the slot each incoming edge comes from
//
// One binding rather than two so the sibling pass, which reads this and the
// forward pair, binds eight storage buffers rather than nine — see
// ForceAlgorithmInfo.minStorageBuffersPerShaderStage. The host sizes, fills
// and states the same map at csrInverseSourcesBase in
// simulation/algorithms/relativity-atlas.ts.
@group(0) @binding(3) var<storage, read> csr_inverse: array<u32>;

// Output: degrees[i*2] = out_degree, degrees[i*2+1] = in_degree
@group(0) @binding(4) var<storage, read_write> degrees: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

// In-degree is a difference of offsets, so this pass reads the offset row
// only; the source row is addressed in relativity_sibling.comp.wgsl.
fn csr_inverse_offset(node_idx: u32) -> u32 {
    return csr_inverse[node_idx];
}

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
    let in_start = csr_inverse_offset(node_idx);
    let in_end = csr_inverse_offset(node_idx + 1u);
    let in_degree = in_end - in_start;

    // Store degrees
    degrees[node_idx * 2u] = out_degree;
    degrees[node_idx * 2u + 1u] = in_degree;
}

