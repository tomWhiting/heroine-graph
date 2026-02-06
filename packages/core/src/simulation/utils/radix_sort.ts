/**
 * Shared GPU Radix Sort Utility
 *
 * Provides reusable 32-bit key-value radix sort pipelines, buffers, bind groups,
 * and command encoding for GPU compute passes. Used by Barnes-Hut tree construction
 * and grid-based collision detection.
 *
 * Sort algorithm: LSD radix sort with 4-bit digits (8 passes for 32-bit keys).
 * Prefix scan: Blelloch exclusive scan, three-phase for large arrays.
 *
 * @module
 */

import { calculateWorkgroups } from "../../renderer/commands.ts";

// Import shader sources
import RADIX_SORT_WGSL from "../shaders/radix_sort.comp.wgsl";
import PREFIX_SCAN_WGSL from "../shaders/prefix_scan.comp.wgsl";

// Constants
const WORKGROUP_SIZE = 256;
const RADIX_PASSES = 8; // 32-bit keys / 4 bits per pass
const RADIX_SIZE = 16; // 2^4 = 16 buckets per radix digit

/**
 * Maximum number of workgroups the prefix scan can handle.
 * This limit comes from the Blelloch scan's 512-element shared memory array.
 * 512 workgroups * 256 threads = 131,072 elements maximum.
 */
const MAX_SCAN_WORKGROUPS = 512;

/**
 * GPU pipelines for radix sort and prefix scan.
 */
export interface RadixSortPipeline {
  /** Clear the histogram buffer (atomicStore 0) */
  clearHistogram: GPUComputePipeline;
  /** Build per-workgroup histogram counts for each radix digit */
  histogram: GPUComputePipeline;
  /** Scatter elements to sorted positions using prefix sums */
  scatter: GPUComputePipeline;
  /** Simple O(n^2) counting sort fallback for small arrays */
  simpleSort: GPUComputePipeline;
  /** Single-pass Blelloch scan for small histograms (<=512 elements) */
  scanSinglePass: GPUComputePipeline;
  /** Phase 1: Per-workgroup reduction for large histograms */
  scanReduce: GPUComputePipeline;
  /** Phase 2: Scan of workgroup sums */
  scanWorkgroupSums: GPUComputePipeline;
  /** Phase 3: Propagate base offsets into per-element prefix sums */
  scanPropagate: GPUComputePipeline;
  /** Bind group layout for sort passes (7 bindings) */
  sortLayout: GPUBindGroupLayout;
  /** Bind group layout for scan passes (3 bindings) */
  scanLayout: GPUBindGroupLayout;
}

/**
 * GPU buffers for radix sort working data.
 *
 * Keys and values are stored as ping-pong pairs. After an even number of passes
 * (the standard 8), results are in keysA/valuesA. After an odd number (e.g.,
 * simple sort = 1 pass), results are in keysB/valuesB.
 */
export interface RadixSortBuffers {
  /** Sort uniform buffer (active, 16 bytes, copied from staging per pass) */
  sortUniforms: GPUBuffer;
  /** Pre-computed uniforms for all 8 passes (128 bytes), used as copy source */
  sortUniformsStaging: GPUBuffer;
  /** Scan uniform buffer (16 bytes) */
  scanUniforms: GPUBuffer;
  /** Keys buffer A (input for even passes, output for odd passes) */
  keysA: GPUBuffer;
  /** Keys buffer B (output for even passes, input for odd passes) */
  keysB: GPUBuffer;
  /** Values buffer A (input for even passes, output for odd passes) */
  valuesA: GPUBuffer;
  /** Values buffer B (output for even passes, input for odd passes) */
  valuesB: GPUBuffer;
  /** Per-workgroup histogram (workgroupCount * 16 atomic u32) */
  histogram: GPUBuffer;
  /** Prefix sums (workgroupCount * 17 u32, extra space for workgroup sums) */
  prefixSums: GPUBuffer;
  /** Maximum element count this buffer set supports */
  maxElements: number;
}

/**
 * Bind groups for radix sort command recording.
 */
export interface RadixSortBindGroups {
  /** One bind group per radix pass (8 total, alternating ping-pong) */
  sortPass: GPUBindGroup[];
  /** Bind group for prefix scan */
  scan: GPUBindGroup;
}

/**
 * Creates radix sort and prefix scan compute pipelines.
 *
 * @param device - GPU device
 * @param label - Label prefix for pipeline debug names
 * @returns RadixSortPipeline with all pipelines and layouts
 */
export function createRadixSortPipeline(
  device: GPUDevice,
  label: string = "RadixSort",
): RadixSortPipeline {
  const radixSortModule = device.createShaderModule({
    label: `${label} Radix Sort Shader`,
    code: RADIX_SORT_WGSL,
  });

  const prefixScanModule = device.createShaderModule({
    label: `${label} Prefix Scan Shader`,
    code: PREFIX_SCAN_WGSL,
  });

  // Sort layout: uniforms, keys_in, values_in, keys_out, values_out, histogram, prefix_sums
  const sortLayout = device.createBindGroupLayout({
    label: `${label} Sort Layout`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });

  // Scan layout: uniforms, histogram, prefix_sums
  const scanLayout = device.createBindGroupLayout({
    label: `${label} Scan Layout`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const sortPipelineLayout = device.createPipelineLayout({
    label: `${label} Sort Pipeline Layout`,
    bindGroupLayouts: [sortLayout],
  });

  const scanPipelineLayout = device.createPipelineLayout({
    label: `${label} Scan Pipeline Layout`,
    bindGroupLayouts: [scanLayout],
  });

  return {
    clearHistogram: device.createComputePipeline({
      label: `${label} Clear Histogram`,
      layout: sortPipelineLayout,
      compute: { module: radixSortModule, entryPoint: "clear_histogram" },
    }),
    histogram: device.createComputePipeline({
      label: `${label} Histogram`,
      layout: sortPipelineLayout,
      compute: { module: radixSortModule, entryPoint: "histogram_count" },
    }),
    scatter: device.createComputePipeline({
      label: `${label} Scatter`,
      layout: sortPipelineLayout,
      compute: { module: radixSortModule, entryPoint: "scatter" },
    }),
    simpleSort: device.createComputePipeline({
      label: `${label} Simple Sort`,
      layout: sortPipelineLayout,
      compute: { module: radixSortModule, entryPoint: "simple_sort" },
    }),
    scanSinglePass: device.createComputePipeline({
      label: `${label} Scan Single Pass`,
      layout: scanPipelineLayout,
      compute: { module: prefixScanModule, entryPoint: "scan" },
    }),
    scanReduce: device.createComputePipeline({
      label: `${label} Scan Reduce`,
      layout: scanPipelineLayout,
      compute: { module: prefixScanModule, entryPoint: "reduce" },
    }),
    scanWorkgroupSums: device.createComputePipeline({
      label: `${label} Scan Workgroup Sums`,
      layout: scanPipelineLayout,
      compute: { module: prefixScanModule, entryPoint: "scan_workgroup_sums" },
    }),
    scanPropagate: device.createComputePipeline({
      label: `${label} Scan Propagate`,
      layout: scanPipelineLayout,
      compute: { module: prefixScanModule, entryPoint: "propagate" },
    }),
    sortLayout,
    scanLayout,
  };
}

/**
 * Creates GPU buffers for radix sort working data.
 *
 * @param device - GPU device
 * @param maxElements - Maximum number of key-value pairs to sort
 * @param label - Label prefix for buffer debug names
 * @returns RadixSortBuffers
 */
export function createRadixSortBuffers(
  device: GPUDevice,
  maxElements: number,
  label: string = "RadixSort",
): RadixSortBuffers {
  const safeMax = Math.max(maxElements, 4);
  const elementBytes = safeMax * 4;
  const workgroupCount = calculateWorkgroups(safeMax, WORKGROUP_SIZE);
  const histogramBytes = Math.max(workgroupCount * RADIX_SIZE * 4, 64);
  // Extra space for workgroup sums during three-phase scan
  const prefixSumsBytes = Math.max(workgroupCount * (RADIX_SIZE + 1) * 4, 64);

  const sortUniforms = device.createBuffer({
    label: `${label} Sort Uniforms`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const sortUniformsStaging = device.createBuffer({
    label: `${label} Sort Uniforms Staging`,
    size: RADIX_PASSES * 16,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const scanUniforms = device.createBuffer({
    label: `${label} Scan Uniforms`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const keysA = device.createBuffer({
    label: `${label} Keys A`,
    size: elementBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const keysB = device.createBuffer({
    label: `${label} Keys B`,
    size: elementBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const valuesA = device.createBuffer({
    label: `${label} Values A`,
    size: elementBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const valuesB = device.createBuffer({
    label: `${label} Values B`,
    size: elementBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const histogram = device.createBuffer({
    label: `${label} Histogram`,
    size: histogramBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const prefixSums = device.createBuffer({
    label: `${label} Prefix Sums`,
    size: prefixSumsBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    sortUniforms,
    sortUniformsStaging,
    scanUniforms,
    keysA,
    keysB,
    valuesA,
    valuesB,
    histogram,
    prefixSums,
    maxElements: safeMax,
  };
}

/**
 * Destroys all radix sort buffers.
 */
export function destroyRadixSortBuffers(buffers: RadixSortBuffers): void {
  buffers.sortUniforms.destroy();
  buffers.sortUniformsStaging.destroy();
  buffers.scanUniforms.destroy();
  buffers.keysA.destroy();
  buffers.keysB.destroy();
  buffers.valuesA.destroy();
  buffers.valuesB.destroy();
  buffers.histogram.destroy();
  buffers.prefixSums.destroy();
}

/**
 * Creates bind groups for radix sort passes.
 *
 * @param device - GPU device
 * @param pipeline - Radix sort pipeline
 * @param buffers - Radix sort buffers
 * @returns RadixSortBindGroups
 */
export function createRadixSortBindGroups(
  device: GPUDevice,
  pipeline: RadixSortPipeline,
  buffers: RadixSortBuffers,
): RadixSortBindGroups {
  // Create one bind group per radix pass with alternating ping-pong buffers.
  // Even passes: A → B, Odd passes: B → A
  const sortPass: GPUBindGroup[] = [];
  for (let pass = 0; pass < RADIX_PASSES; pass++) {
    const isEven = pass % 2 === 0;
    sortPass.push(
      device.createBindGroup({
        label: `RadixSort Pass ${pass}`,
        layout: pipeline.sortLayout,
        entries: [
          { binding: 0, resource: { buffer: buffers.sortUniforms } },
          { binding: 1, resource: { buffer: isEven ? buffers.keysA : buffers.keysB } },
          { binding: 2, resource: { buffer: isEven ? buffers.valuesA : buffers.valuesB } },
          { binding: 3, resource: { buffer: isEven ? buffers.keysB : buffers.keysA } },
          { binding: 4, resource: { buffer: isEven ? buffers.valuesB : buffers.valuesA } },
          { binding: 5, resource: { buffer: buffers.histogram } },
          { binding: 6, resource: { buffer: buffers.prefixSums } },
        ],
      }),
    );
  }

  const scan = device.createBindGroup({
    label: "RadixSort Scan",
    layout: pipeline.scanLayout,
    entries: [
      { binding: 0, resource: { buffer: buffers.scanUniforms } },
      { binding: 1, resource: { buffer: buffers.histogram } },
      { binding: 2, resource: { buffer: buffers.prefixSums } },
    ],
  });

  return { sortPass, scan };
}

/**
 * Writes sort and scan uniform buffers for a given element count.
 *
 * Must be called before recordRadixSort when the element count changes.
 *
 * @param device - GPU device
 * @param buffers - Radix sort buffers
 * @param elementCount - Number of elements to sort
 */
export function updateRadixSortUniforms(
  device: GPUDevice,
  buffers: RadixSortBuffers,
  elementCount: number,
): void {
  if (elementCount > buffers.maxElements) {
    throw new Error(
      `RadixSort buffer overflow: elementCount (${elementCount}) exceeds buffer capacity (${buffers.maxElements}). ` +
      `Buffers must be recreated with createRadixSortBuffers() when element count increases.`
    );
  }

  const workgroupCount = calculateWorkgroups(elementCount, WORKGROUP_SIZE);

  // Scan uniforms (16 bytes)
  // struct ScanUniforms { element_count: u32, pass_number: u32, workgroup_count: u32, _padding: u32 }
  const scanData = new ArrayBuffer(16);
  const scanView = new DataView(scanData);
  scanView.setUint32(0, workgroupCount * RADIX_SIZE, true);  // element_count
  scanView.setUint32(4, 0, true);                            // pass_number (unused)
  scanView.setUint32(8, workgroupCount, true);                // workgroup_count
  scanView.setUint32(12, 0, true);                           // _padding
  device.queue.writeBuffer(buffers.scanUniforms, 0, scanData);

  // Sort uniforms staging (128 bytes = 8 passes * 16 bytes)
  // Pre-compute uniforms for all passes so copyBufferToBuffer can be used during encoding
  // struct SortUniforms { node_count: u32, pass_number: u32, _padding: vec2<u32> }
  const stagingData = new ArrayBuffer(RADIX_PASSES * 16);
  const stagingView = new DataView(stagingData);
  for (let pass = 0; pass < RADIX_PASSES; pass++) {
    const offset = pass * 16;
    stagingView.setUint32(offset + 0, elementCount, true);  // node_count
    stagingView.setUint32(offset + 4, pass, true);          // pass_number (0-7)
    stagingView.setUint32(offset + 8, 0, true);             // _padding[0]
    stagingView.setUint32(offset + 12, 0, true);            // _padding[1]
  }
  device.queue.writeBuffer(buffers.sortUniformsStaging, 0, stagingData);
}

/**
 * Records GPU commands for a full 32-bit radix sort.
 *
 * For small arrays (< 1024 elements), uses a simple counting sort (1 pass).
 * For larger arrays, uses LSD radix sort with 8 passes of 4-bit digits.
 *
 * After sorting:
 * - Simple sort (1 pass): results are in keysB/valuesB
 * - Full radix sort (8 passes): results are in keysA/valuesA
 *
 * @param encoder - GPU command encoder
 * @param pipeline - Radix sort pipeline
 * @param bindGroups - Radix sort bind groups
 * @param buffers - Radix sort buffers (needed for copyBufferToBuffer)
 * @param elementCount - Number of elements to sort
 * @param label - Label prefix for compute pass debug names
 * @returns true if sort was recorded, false if skipped (e.g., workgroup overflow)
 */
export function recordRadixSort(
  encoder: GPUCommandEncoder,
  pipeline: RadixSortPipeline,
  bindGroups: RadixSortBindGroups,
  buffers: RadixSortBuffers,
  elementCount: number,
  label: string = "Sort",
): boolean {
  if (elementCount < 2) {
    return true;
  }

  const nodeWorkgroups = calculateWorkgroups(elementCount, WORKGROUP_SIZE);
  const useSimpleSort = elementCount < 1024;

  if (useSimpleSort) {
    // Simple counting sort writes to keysB/valuesB (pass 0: A → B)
    encoder.copyBufferToBuffer(
      buffers.sortUniformsStaging, 0,
      buffers.sortUniforms, 0,
      16,
    );
    const pass = encoder.beginComputePass({ label: `${label} Simple Sort` });
    pass.setPipeline(pipeline.simpleSort);
    pass.setBindGroup(0, bindGroups.sortPass[0]);
    pass.dispatchWorkgroups(nodeWorkgroups);
    pass.end();
  } else {
    // Validate workgroup count against prefix scan capacity
    if (nodeWorkgroups > MAX_SCAN_WORKGROUPS) {
      console.error(
        `RadixSort: Element count ${elementCount} requires ${nodeWorkgroups} workgroups, ` +
        `but prefix scan supports max ${MAX_SCAN_WORKGROUPS} (~131K elements).`
      );
      return false;
    }
    const histogramWorkgroups = calculateWorkgroups(nodeWorkgroups * RADIX_SIZE, WORKGROUP_SIZE);

    for (let pass = 0; pass < RADIX_PASSES; pass++) {
      // Copy per-pass uniforms from staging buffer
      encoder.copyBufferToBuffer(
        buffers.sortUniformsStaging, pass * 16,
        buffers.sortUniforms, 0,
        16,
      );

      // Clear histogram
      {
        const computePass = encoder.beginComputePass({ label: `${label} Clear Histogram ${pass}` });
        computePass.setPipeline(pipeline.clearHistogram);
        computePass.setBindGroup(0, bindGroups.sortPass[pass]);
        computePass.dispatchWorkgroups(histogramWorkgroups);
        computePass.end();
      }

      // Build histogram
      {
        const computePass = encoder.beginComputePass({ label: `${label} Histogram ${pass}` });
        computePass.setPipeline(pipeline.histogram);
        computePass.setBindGroup(0, bindGroups.sortPass[pass]);
        computePass.dispatchWorkgroups(nodeWorkgroups);
        computePass.end();
      }

      // Prefix scan
      const totalHistogramElements = nodeWorkgroups * RADIX_SIZE;
      const useSinglePassScan = totalHistogramElements <= 512;

      if (useSinglePassScan) {
        const computePass = encoder.beginComputePass({ label: `${label} Scan Single ${pass}` });
        computePass.setPipeline(pipeline.scanSinglePass);
        computePass.setBindGroup(0, bindGroups.scan);
        computePass.dispatchWorkgroups(1);
        computePass.end();
      } else {
        // Three-phase parallel scan for large histograms
        {
          const computePass = encoder.beginComputePass({ label: `${label} Scan Reduce ${pass}` });
          computePass.setPipeline(pipeline.scanReduce);
          computePass.setBindGroup(0, bindGroups.scan);
          computePass.dispatchWorkgroups(nodeWorkgroups);
          computePass.end();
        }
        {
          const computePass = encoder.beginComputePass({ label: `${label} Scan WG Sums ${pass}` });
          computePass.setPipeline(pipeline.scanWorkgroupSums);
          computePass.setBindGroup(0, bindGroups.scan);
          computePass.dispatchWorkgroups(1);
          computePass.end();
        }
        {
          const computePass = encoder.beginComputePass({ label: `${label} Scan Propagate ${pass}` });
          computePass.setPipeline(pipeline.scanPropagate);
          computePass.setBindGroup(0, bindGroups.scan);
          computePass.dispatchWorkgroups(nodeWorkgroups);
          computePass.end();
        }
      }

      // Scatter
      {
        const computePass = encoder.beginComputePass({ label: `${label} Scatter ${pass}` });
        computePass.setPipeline(pipeline.scatter);
        computePass.setBindGroup(0, bindGroups.sortPass[pass]);
        computePass.dispatchWorkgroups(nodeWorkgroups);
        computePass.end();
      }
    }
  }

  return true;
}

/**
 * Returns true if the last sort used simple sort (results in keysB/valuesB),
 * false if it used full radix sort (results in keysA/valuesA).
 *
 * @param elementCount - Number of elements that were sorted
 * @returns true if simple sort was used
 */
export function wasSimpleSort(elementCount: number): boolean {
  return elementCount < 1024;
}
