/**
 * Collision Detection and Resolution Module
 *
 * Provides GPU-accelerated collision detection and resolution for node overlap
 * prevention. Runs as an optional post-integration pass.
 *
 * @module
 */

import type { GPUContext } from "../webgpu/context.ts";
import { calculateWorkgroups } from "../renderer/commands.ts";
import type { FullForceConfig } from "./config.ts";

// Import shader source
import COLLISION_WGSL from "./shaders/collision.comp.wgsl";

const WORKGROUP_SIZE = 256;
const DEFAULT_RADIUS = 5.0;

/**
 * Collision pipeline resources
 */
export interface CollisionPipeline {
  /** Main collision resolution pipeline */
  resolve: GPUComputePipeline;
  /** Tiled version for larger graphs */
  resolveTiled: GPUComputePipeline;
  /** Bind group layout */
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Collision-specific buffers
 */
export interface CollisionBuffers {
  /** Uniform buffer for collision parameters */
  uniforms: GPUBuffer;
  /** Node sizes/radii buffer */
  nodeSizes: GPUBuffer;
  /** Maximum nodes this buffer set supports */
  maxNodes: number;
}

/**
 * Collision bind group
 */
export interface CollisionBindGroup {
  /** Bind group for collision pass */
  bindGroup: GPUBindGroup;
}

/**
 * Creates the collision compute pipeline
 *
 * @param context - GPU context
 * @returns Collision pipeline resources
 */
export function createCollisionPipeline(context: GPUContext): CollisionPipeline {
  const { device } = context;

  const shaderModule = device.createShaderModule({
    label: "Collision Shader",
    code: COLLISION_WGSL,
  });

  // Bind group layout for collision pass
  // Bindings: uniforms, positions (vec2, read-write), node_sizes
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Collision Bind Group Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Collision Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const resolve = device.createComputePipeline({
    label: "Collision Resolve Pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const resolveTiled = device.createComputePipeline({
    label: "Collision Resolve Tiled Pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "resolve_tiled" },
  });

  return {
    resolve,
    resolveTiled,
    bindGroupLayout,
  };
}

/**
 * Creates collision-specific buffers
 *
 * @param device - GPU device
 * @param maxNodes - Maximum number of nodes
 * @returns Collision buffers
 */
export function createCollisionBuffers(
  device: GPUDevice,
  maxNodes: number,
): CollisionBuffers {
  const safeMaxNodes = Math.max(maxNodes, 4);
  const nodeBytes = safeMaxNodes * 4;

  // Uniform buffer (32 bytes for CollisionUniforms struct)
  const uniforms = device.createBuffer({
    label: "Collision Uniforms",
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Node sizes buffer
  const nodeSizes = device.createBuffer({
    label: "Node Sizes",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    uniforms,
    nodeSizes,
    maxNodes: safeMaxNodes,
  };
}

/**
 * Creates collision bind group
 *
 * @param device - GPU device
 * @param pipeline - Collision pipeline
 * @param collisionBuffers - Collision-specific buffers
 * @param positions - Position buffer (vec2, read-write)
 * @returns Collision bind group
 */
export function createCollisionBindGroup(
  device: GPUDevice,
  pipeline: CollisionPipeline,
  collisionBuffers: CollisionBuffers,
  positions: GPUBuffer,
): CollisionBindGroup {
  const bindGroup = device.createBindGroup({
    label: "Collision Bind Group",
    layout: pipeline.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: collisionBuffers.uniforms } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: collisionBuffers.nodeSizes } },
    ],
  });

  return { bindGroup };
}

/**
 * Updates collision uniform buffer
 *
 * @param device - GPU device
 * @param collisionBuffers - Collision buffers
 * @param nodeCount - Number of nodes
 * @param forceConfig - Force configuration
 */
export function updateCollisionUniforms(
  device: GPUDevice,
  collisionBuffers: CollisionBuffers,
  nodeCount: number,
  forceConfig: FullForceConfig,
): void {
  // CollisionUniforms struct (32 bytes):
  // node_count: u32, collision_strength: f32, radius_multiplier: f32, iterations: u32,
  // default_radius: f32, _pad0: f32, _pad1: f32, _pad2: f32
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, nodeCount, true);
  view.setFloat32(4, forceConfig.collisionStrength, true);
  view.setFloat32(8, forceConfig.collisionRadiusMultiplier, true);
  view.setUint32(12, forceConfig.collisionIterations, true);
  view.setFloat32(16, DEFAULT_RADIUS, true);
  view.setFloat32(20, 0.0, true);  // _pad0
  view.setFloat32(24, 0.0, true);  // _pad1
  view.setFloat32(28, 0.0, true);  // _pad2
  device.queue.writeBuffer(collisionBuffers.uniforms, 0, data);
}

/**
 * Uploads node sizes to GPU
 *
 * @param device - GPU device
 * @param collisionBuffers - Collision buffers
 * @param nodeSizes - Array of node sizes/radii
 */
export function uploadNodeSizes(
  device: GPUDevice,
  collisionBuffers: CollisionBuffers,
  nodeSizes: Float32Array,
): void {
  // Create a proper ArrayBuffer copy to satisfy BufferSource type
  const buffer = new ArrayBuffer(nodeSizes.byteLength);
  new Float32Array(buffer).set(nodeSizes);
  device.queue.writeBuffer(collisionBuffers.nodeSizes, 0, buffer);
}

/**
 * Records collision detection pass(es) to command encoder
 *
 * @param encoder - Command encoder
 * @param pipeline - Collision pipeline
 * @param bindGroup - Collision bind group
 * @param nodeCount - Number of nodes
 * @param iterations - Number of collision resolution iterations
 * @param useTiled - Use tiled version for large graphs (>5000 nodes)
 */
export function recordCollisionPass(
  encoder: GPUCommandEncoder,
  pipeline: CollisionPipeline,
  bindGroup: CollisionBindGroup,
  nodeCount: number,
  iterations: number = 1,
  useTiled: boolean = false,
): void {
  const workgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
  const selectedPipeline = useTiled ? pipeline.resolveTiled : pipeline.resolve;

  for (let i = 0; i < iterations; i++) {
    const pass = encoder.beginComputePass({
      label: `Collision Resolution ${i + 1}/${iterations}`,
    });
    pass.setPipeline(selectedPipeline);
    pass.setBindGroup(0, bindGroup.bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }
}

/**
 * Destroys collision buffers
 *
 * @param buffers - Collision buffers to destroy
 */
export function destroyCollisionBuffers(buffers: CollisionBuffers): void {
  buffers.uniforms.destroy();
  buffers.nodeSizes.destroy();
}
