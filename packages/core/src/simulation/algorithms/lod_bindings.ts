/**
 * LOD bindings for algorithm plugins.
 *
 * The core simulation pipeline reads the LOD state directly off
 * {@link SimulationBuffers}. A plugin cannot: it only sees an
 * {@link AlgorithmRenderContext}, and that context is built by whatever host is
 * driving it — GraphMother, a GPU test harness, or an embedder. This module is
 * the one place that turns "what the host offered" into "what a bind group
 * needs", so every plugin resolves the same way and the fallbacks are stated
 * once.
 *
 * The fallbacks are the discipline WP-D established for the node-flag mask and
 * WP-F for mass: a host that declares nothing gets a set of buffers describing
 * *the whole graph, unmasked* — the identity active list, unit mass, no flags —
 * so the pass is correct-but-slow rather than wrong. None of them can be
 * left zero-filled: a zeroed mass buffer is a graph with no repulsion, and a
 * zeroed index list is every thread hammering slot 0.
 *
 * @module
 */

import type { AlgorithmBuffers, AlgorithmRenderContext } from "./types.ts";
import { NODE_MASS_UNIT } from "../../lod/mass.ts";

/**
 * Buffers a plugin binds for LOD state its host did not supply.
 *
 * Sized at the `maxNodes` its owning algorithm's `createBuffers` was given, so
 * a node count that outgrows them is the same reallocation event that already
 * forces `createBuffers` to run again.
 */
export class AlgorithmLodFallbacks implements AlgorithmBuffers {
  constructor(
    /** `[0, maxNodes)` ascending: the whole graph, in slot order. */
    readonly liveIndices: GPUBuffer,
    /** {@link NODE_MASS_UNIT} everywhere: every slot stands for one body. */
    readonly nodeMass: GPUBuffer,
    /** All zero: no slot dead, pinned or hidden. */
    readonly nodeFlags: GPUBuffer,
    /**
     * A single zeroed `u32` standing in for both LOD edge lists. Never read:
     * the bundled entry points are dispatched only when the host declares an
     * aggregation, and declaring one means supplying the real buffers. It
     * exists so the bundled bind group can be built unconditionally, which
     * keeps bind-group construction independent of per-frame state (the
     * ForceAlgorithm.createBindGroups contract).
     */
    readonly emptyEdgeList: GPUBuffer,
  ) {}

  destroy(): void {
    this.liveIndices.destroy();
    this.nodeMass.destroy();
    this.nodeFlags.destroy();
    this.emptyEdgeList.destroy();
  }
}

/**
 * Allocate and fill the fallback set for a `maxNodes`-slot graph.
 */
export function createAlgorithmLodFallbacks(
  device: GPUDevice,
  maxNodes: number,
  label: string,
): AlgorithmLodFallbacks {
  const slots = Math.max(maxNodes, 1);

  const liveIndices = device.createBuffer({
    label: `${label} Fallback Live Indices`,
    size: slots * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const identity = new Uint32Array(slots);
  for (let i = 0; i < slots; i++) identity[i] = i;
  device.queue.writeBuffer(liveIndices, 0, identity);

  const nodeMass = device.createBuffer({
    label: `${label} Fallback Node Mass`,
    size: slots * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nodeMass, 0, new Float32Array(slots).fill(NODE_MASS_UNIT));

  // Zero is the right content here and the buffer is created zeroed, so unlike
  // the two above this one needs no write.
  const nodeFlags = device.createBuffer({
    label: `${label} Fallback Node Flags`,
    size: slots * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const emptyEdgeList = device.createBuffer({
    label: `${label} Fallback LOD Edge List`,
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return new AlgorithmLodFallbacks(liveIndices, nodeMass, nodeFlags, emptyEdgeList);
}

/** The active-index list to gather from, or the identity list. */
export function lodLiveIndices(
  context: AlgorithmRenderContext,
  fallbacks: AlgorithmLodFallbacks,
): GPUBuffer {
  return context.liveIndices ?? fallbacks.liveIndices;
}

/** Per-node mass, or unit mass everywhere. */
export function lodNodeMass(
  context: AlgorithmRenderContext,
  fallbacks: AlgorithmLodFallbacks,
): GPUBuffer {
  return context.nodeMass ?? fallbacks.nodeMass;
}

/** Per-node state flags, or all-live. */
export function lodNodeFlags(
  context: AlgorithmRenderContext,
  fallbacks: AlgorithmLodFallbacks,
): GPUBuffer {
  return context.nodeFlags ?? fallbacks.nodeFlags;
}

/**
 * Entries of the active list a node-indexed pass covers.
 *
 * Defaults to the slot high-water mark, which is exactly what the identity
 * fallback describes. A host that supplies a list but no count gets the same
 * number: over-including is masked and therefore slow rather than wrong, while
 * under-including would silently freeze the tail of the graph.
 */
export function lodActiveCount(context: AlgorithmRenderContext): number {
  return context.activeCount ?? context.nodeCount;
}

/** How a per-edge pass is dispatched this frame. */
export interface LodEdgeDispatch {
  /** Entries of `liveEdgeIndices` covered. */
  readonly activeEdgeCount: number;
  /** Bundles after them. */
  readonly bundleCount: number;
  /** Threads the bundled entry point needs: the two counts, in that order. */
  readonly total: number;
}

/**
 * The aggregated edge set the host declared, or `null` when there is none and
 * the pass should run over every source edge.
 *
 * `null` and a zero total are different answers, and the difference is
 * load-bearing: a cut can leave every edge inside a collapsed subtree, which is
 * a live aggregation covering nothing, and must dispatch no attraction pass at
 * all rather than fall back to attracting along the whole edge array.
 */
export function lodEdgeDispatch(context: AlgorithmRenderContext): LodEdgeDispatch | null {
  if (!context.lodEdgesActive) return null;
  const activeEdgeCount = context.activeEdgeCount ?? 0;
  const bundleCount = context.bundleCount ?? 0;
  return { activeEdgeCount, bundleCount, total: activeEdgeCount + bundleCount };
}

/** The LOD active-edge list, or the never-read placeholder. */
export function lodLiveEdgeIndices(
  context: AlgorithmRenderContext,
  fallbacks: AlgorithmLodFallbacks,
): GPUBuffer {
  return context.liveEdgeIndices ?? fallbacks.emptyEdgeList;
}

/** The LOD bundle table, or the never-read placeholder. */
export function lodEdgeBundles(
  context: AlgorithmRenderContext,
  fallbacks: AlgorithmLodFallbacks,
): GPUBuffer {
  return context.edgeBundles ?? fallbacks.emptyEdgeList;
}
