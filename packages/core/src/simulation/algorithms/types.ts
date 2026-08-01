/**
 * Force Algorithm Type Definitions
 *
 * Defines the interface for pluggable force algorithms.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import { ErrorCode, GraphMotherError } from "../../errors.ts";
import type { FullForceConfig } from "../config.ts";

/**
 * Available force algorithm types
 */
export type ForceAlgorithmType =
  | "n2"
  | "barnes-hut"
  | "force-atlas2"
  | "density"
  | "relativity-atlas"
  | "tidy-tree"
  | "linlog"
  | "t-fdp"
  | "community"
  | "codebase";

/**
 * Algorithm metadata for display
 */
export interface ForceAlgorithmInfo {
  /** Unique identifier */
  readonly id: ForceAlgorithmType;
  /** Display name */
  readonly name: string;
  /** Description of the algorithm */
  readonly description: string;
  /** Minimum recommended node count */
  readonly minNodes: number;
  /** Maximum recommended node count (-1 for unlimited) */
  readonly maxNodes: number;
  /** Time complexity description */
  readonly complexity: string;
  /**
   * Storage buffers per compute stage this algorithm's bind group layouts
   * need, when that is more than the WebGPU default of 8.
   *
   * Exceeding the device limit does not fail loudly: the bind group layout is
   * invalid, binding it poisons the compute pass, the poisoned pass poisons
   * the command encoder, and `submit()` discards the ENTIRE frame — springs
   * and integration included. The simulation freezes with console errors
   * rather than degrading, which is why this is declared as data: both
   * {@link ForceAlgorithm.createPipelines} guards and the registry's
   * device-aware selection read it (see supportsAlgorithmOnDevice).
   *
   * Omitted when the algorithm fits inside the 8-buffer default.
   */
  readonly minStorageBuffersPerShaderStage?: number;
}

/**
 * Whether `device` can run `info`'s pipelines.
 *
 * Only the storage-buffer-per-stage limit varies meaningfully across adapters
 * in practice; everything else GraphMother requests is at or below every
 * conformant implementation's default.
 */
export function supportsAlgorithmOnDevice(
  info: ForceAlgorithmInfo,
  device: Pick<GPUDevice, "limits">,
): boolean {
  const required = info.minStorageBuffersPerShaderStage;
  return required === undefined ||
    device.limits.maxStorageBuffersPerShaderStage >= required;
}

/**
 * Throws when `device` cannot run `info`'s pipelines. Called at the top of the
 * affected algorithms' createPipelines so the failure names the limit instead
 * of surfacing as a frozen simulation with WebGPU validation spam.
 */
export function assertAlgorithmSupportedOnDevice(
  info: ForceAlgorithmInfo,
  device: Pick<GPUDevice, "limits">,
): void {
  if (supportsAlgorithmOnDevice(info, device)) return;
  const required = info.minStorageBuffersPerShaderStage;
  throw new GraphMotherError(
    ErrorCode.PIPELINE_CREATION_FAILED,
    `The ${info.name} layout needs maxStorageBuffersPerShaderStage >= ${required}, ` +
      `but this device supports ${device.limits.maxStorageBuffersPerShaderStage}. ` +
      "Its bind group layouts would be invalid, which discards every command " +
      "buffer they are recorded into — the simulation would freeze rather than " +
      "degrade.",
    { algorithm: info.id, required, available: device.limits.maxStorageBuffersPerShaderStage },
    'Choose an algorithm that fits the device (setForceAlgorithm("n2") always ' +
      "does), or request the higher limit when creating the GPU device.",
  );
}

/**
 * Buffers specific to an algorithm (beyond the standard simulation buffers)
 */
export interface AlgorithmBuffers {
  /** Dispose of all algorithm-specific buffers */
  destroy(): void;
}

/**
 * Empty algorithm buffers for algorithms that don't need extra buffers
 */
export class EmptyAlgorithmBuffers implements AlgorithmBuffers {
  destroy(): void {
    // No buffers to destroy
  }
}

/**
 * Bind groups for algorithm-specific compute passes
 */
export interface AlgorithmBindGroups {
  /** Repulsion pass bind group */
  repulsion: GPUBindGroup;
}

/**
 * Pipelines for algorithm-specific compute passes
 */
export interface AlgorithmPipelines {
  /** Repulsion compute pipeline */
  repulsion: GPUComputePipeline;
}

/**
 * Context passed to algorithm for rendering
 *
 * All position and force buffers use vec2<f32> layout (8 bytes per node).
 */
export interface AlgorithmRenderContext {
  /** GPU device */
  device: GPUDevice;
  /** Position buffer - vec2<f32> per node (interleaved X,Y) */
  positions: GPUBuffer;
  /** Force accumulator buffer - vec2<f32> per node (interleaved X,Y) */
  forces: GPUBuffer;
  /** Number of nodes */
  nodeCount: number;
  /** Number of edges (required for Relativity Atlas CSR traversal) */
  edgeCount: number;
  /** Current force configuration */
  forceConfig: FullForceConfig;
  /** Viewport bounds (for spatial algorithms like Barnes-Hut and Density Field) */
  bounds?: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  } | undefined;
  /** Edge source indices buffer (for algorithms that handle their own springs) */
  edgeSources?: GPUBuffer;
  /** Edge target indices buffer (for algorithms that handle their own springs) */
  edgeTargets?: GPUBuffer;
  /** CPU-side edge source indices (for degree computation without GPU readback) */
  edgeSourcesData?: Uint32Array | undefined;
  /** CPU-side edge target indices (for degree computation without GPU readback) */
  edgeTargetsData?: Uint32Array | undefined;
  /**
   * Node state flags buffer (u32 per slot; bit 0 = NODE_FLAG_DEAD, bit 1 =
   * NODE_FLAG_PINNED — see pipeline.ts). nodeCount is the slot high-water
   * mark, so slots of removed nodes remain in range with their positions
   * zeroed to the origin. Any algorithm shader that iterates node slots
   * (repulsion, gravity, mass aggregation, ...) MUST bind this buffer and
   * skip slots with NODE_FLAG_DEAD set — otherwise dead slots act as phantom
   * force sources at (0,0). See repulsion_n2.comp.wgsl `main_masked` for the
   * reference pattern.
   */
  nodeFlags?: GPUBuffer | undefined;
  /**
   * Per-node simulation mass buffer (f32 per slot; 1.0 = one body — see
   * pipeline.ts `nodeMass`). A collapsed LOD subtree rolls its members' mass
   * into the visible proxy, so any repulsion shader that treats every body as
   * unit mass silently shrinks the layout when a subtree collapses. Algorithms
   * that cannot obtain this buffer must fall back to a 1.0-filled one, not a
   * zeroed one: zero mass is no repulsion.
   */
  nodeMass?: GPUBuffer | undefined;
  /**
   * Active-index list (u32 per entry; see pipeline.ts `liveIndices`). The
   * first {@link AlgorithmRenderContext.activeCount} entries are the slots
   * taking part in this tick, ascending.
   *
   * A node-indexed pass should dispatch one thread per ENTRY and gather
   * `live_idx[entry]`, which is what turns hiding a subtree into work that is
   * not done rather than threads that return early. The gather never replaces
   * the {@link AlgorithmRenderContext.nodeFlags} mask: a stale list can only
   * over-include, and masking makes over-inclusion inert, so a host that never
   * refreshes the list is slower and never wrong.
   *
   * Absent means "the whole graph, in slot order" — bind the identity list
   * (see createAlgorithmLodFallbacks), never a zeroed buffer.
   */
  liveIndices?: GPUBuffer | undefined;
  /**
   * Entries of {@link AlgorithmRenderContext.liveIndices} the passes cover.
   * Defaults to `nodeCount`, which is what the identity list describes.
   *
   * Per-frame state: it belongs in a uniform written by
   * {@link ForceAlgorithm.updateUniforms}, never captured in a bind group.
   */
  activeCount?: number | undefined;
  /**
   * Whether an LOD edge aggregation is uploaded (see pipeline.ts
   * `lodEdgesActive`). Distinct from the two counts below being zero, which is
   * a legal aggregation covering nothing — a cut can leave every edge inside a
   * collapsed subtree, and that must dispatch no attraction at all rather than
   * fall back to the whole edge array.
   */
  lodEdgesActive?: boolean | undefined;
  /**
   * Source edges with both endpoints in the visible cut (u32 per entry;
   * pipeline.ts `liveEdgeIndices`), and the aggregated cross-boundary edges
   * standing in for everything else (three u32 per bundle — source, target,
   * weight; pipeline.ts `edgeBundles`).
   *
   * An algorithm that supplies its own attraction (`handlesSprings`) MUST
   * consume these under a cut. Skipping an edge because an endpoint is hidden
   * does not defer that edge's pull, it deletes it: a collapsed module would
   * exert nothing on the modules it imports, and the collapsed layout would
   * stop resembling the expanded one (SC-002).
   */
  liveEdgeIndices?: GPUBuffer | undefined;
  edgeBundles?: GPUBuffer | undefined;
  /** Entries of each of the two lists above; meaningful while lodEdgesActive. */
  activeEdgeCount?: number | undefined;
  bundleCount?: number | undefined;
}

/**
 * Force algorithm interface
 *
 * Algorithms implement this interface to provide custom repulsion force calculations.
 * The standard spring forces and integration are handled by the main pipeline.
 */
export interface ForceAlgorithm {
  /** Algorithm info */
  readonly info: ForceAlgorithmInfo;

  /**
   * Whether this algorithm applies its own center gravity.
   * When true, the integration shader skips its built-in gravity
   * to avoid double-applying the center pull.
   */
  readonly handlesGravity: boolean;

  /**
   * Whether this algorithm provides its own edge/spring forces.
   * When true, the simulation skips the default edge spring pass
   * to avoid competing forces. Used by layout algorithms (e.g. tidy-tree)
   * that compute exact target positions rather than using force-directed springs.
   */
  readonly handlesSprings?: boolean;

  /**
   * Whether this algorithm wants FA2-style per-node adaptive speed
   * (swing/traction velocity scaling in the integrate pass). Forwarded to
   * updateSimulationUniforms as adaptiveSpeedOverride; when undefined the
   * forceConfig.adaptiveSpeed setting applies. Algorithms with unbounded
   * attraction (F = d) such as ForceAtlas2 and Relativity Atlas should set
   * this to true to suppress overshoot oscillation.
   */
  readonly prefersAdaptiveSpeed?: boolean;

  /**
   * Create GPU pipelines for this algorithm
   *
   * @param context - GPU context
   * @returns Algorithm-specific pipelines
   */
  createPipelines(context: GPUContext): AlgorithmPipelines;

  /**
   * Create algorithm-specific buffers
   *
   * @param device - GPU device
   * @param maxNodes - Maximum number of nodes to support
   * @returns Algorithm-specific buffers
   */
  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers;

  /**
   * Create bind groups for the repulsion pass.
   *
   * CONTRACT — called once per ping-pong parity, never per frame. The host
   * calls this exactly twice per buffer allocation: once with
   * `context.positions` set to each of the two ping-pong position buffers,
   * and caches both results (see `buildForBothParities` in ../pipeline.ts).
   * Per frame it only flips which cached set it binds.
   *
   * The implementation MUST therefore depend only on GPU buffer IDENTITIES —
   * those reachable from `context` and `algorithmBuffers`, plus buffers the
   * implementation owns on `this` (tidy-tree's uniform/target buffers,
   * community's and codebase's community-id/degree/centroid buffers). Two
   * calls with the same `context.positions` must produce equivalent bind
   * groups.
   *
   * Instance-owned buffers are allowed, with one obligation: they may only be
   * created or replaced inside {@link ForceAlgorithm.createBuffers}, and every
   * call site of `createBuffers` must be followed by a rebuild of BOTH parity
   * variants before the next frame is recorded (graph.ts does this at each of
   * its four call sites). Otherwise the cached bind groups keep naming a
   * destroyed buffer. Note that the registry hands out process-global
   * singletons (`getAlgorithmRegistry`), so two GraphMother instances — with
   * two different GPUDevices — share one algorithm object and therefore one
   * set of instance-owned buffers; the last `createBuffers` wins, and only the
   * immediately-following rebuild keeps that instance consistent. Moving these
   * buffers onto the returned {@link AlgorithmBuffers} would remove that
   * hazard entirely.
   *
   * It must not capture `context.nodeCount`, `context.bounds`, or
   * `context.forceConfig` in the returned bind groups — those change every
   * frame and belong in uniforms written by
   * {@link ForceAlgorithm.updateUniforms}. An algorithm whose bind groups
   * genuinely cannot satisfy this (because they must reference GPU state that
   * changes mid-run for a reason other than the ping-pong swap) cannot use the
   * cached path and must be given an explicit opt-out here before it is
   * registered; all currently registered algorithms satisfy the contract.
   * Algorithm-internal ping-pong (e.g. Relativity Atlas's mass iteration) is
   * unaffected: it lives entirely in `algorithmBuffers` and is reproduced
   * identically in both parity variants.
   *
   * The host rebuilds both variants whenever any referenced buffer is
   * reallocated (load, capacity growth, algorithm switch); a variant left
   * pointing at a replaced positions buffer is caught by the ParitySlot
   * staleness check in ../pipeline.ts rather than corrupting silently.
   *
   * @param device - GPU device
   * @param pipelines - Algorithm pipelines
   * @param context - Render context with buffers for one ping-pong parity
   * @param algorithmBuffers - Algorithm-specific buffers
   * @returns Bind groups valid for that parity
   */
  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups;

  /**
   * Update algorithm-specific uniform buffers
   *
   * @param device - GPU device
   * @param algorithmBuffers - Algorithm-specific buffers
   * @param context - Render context
   */
  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void;

  /**
   * Record the repulsion compute pass
   *
   * @param encoder - Command encoder
   * @param pipelines - Algorithm pipelines
   * @param bindGroups - Algorithm bind groups
   * @param nodeCount - Number of nodes
   */
  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void;
}
