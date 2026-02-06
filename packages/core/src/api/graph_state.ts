/**
 * MutableGraphState
 *
 * Tracks the live graph state on the CPU side, managing the mapping between
 * user IDs, internal slot indices, and GPU buffer offsets. This is the
 * single source of truth for incremental mutations.
 *
 * Node slots use a free-list with a high-water mark. Removed node slots
 * are zeroed (radius=0 → invisible, position=0 → no forces) and reused
 * on subsequent adds.
 *
 * Edge slots use dense packing with swap-remove. The last edge is swapped
 * into the removed slot, keeping the array contiguous.
 *
 * @module
 */

import type { IdLike, IdMap } from "../graph/id_map.ts";
import { createIdMap } from "../graph/id_map.ts";
import type { ParsedGraph } from "../graph/parser.ts";
import { initialCapacity } from "./buffer_capacity.ts";

/**
 * Mutable graph state for incremental mutations
 */
export class MutableGraphState {
  // Node tracking
  nodeCount: number;
  nodeCapacity: number;
  /** Highest slot index in use + 1 (used as instance count for draw calls) */
  nodeHighWater: number;
  /** Reusable slot indices from removals */
  nodeFreeList: number[];
  nodeIdMap: IdMap<IdLike>;

  // Edge tracking (dense, no gaps via swap-remove)
  edgeCount: number;
  edgeCapacity: number;
  edgeIdMap: IdMap<IdLike>;

  // CPU shadow arrays (kept in sync with GPU buffers)
  positionsX: Float32Array;
  positionsY: Float32Array;
  /** 6 floats per slot: radius, r, g, b, selected, hovered */
  nodeAttributes: Float32Array;
  /** Dense edge source indices */
  edgeSources: Uint32Array;
  /** Dense edge target indices */
  edgeTargets: Uint32Array;
  /** 8 floats per edge: width, r, g, b, selected, hovered, curvature, reserved */
  edgeAttributes: Float32Array;

  // Metadata
  nodeMetadata: Map<number, Record<string, unknown>>;
  edgeMetadata: Map<number, Record<string, unknown>>;
  nodeTypes: (string | undefined)[];
  edgeTypes: (string | undefined)[];

  // Node adjacency for efficient edge cascade on node removal
  /** nodeSlotIndex → set of edge slot indices */
  nodeEdges: Map<number, Set<number>>;

  private constructor() {
    this.nodeCount = 0;
    this.nodeCapacity = 0;
    this.nodeHighWater = 0;
    this.nodeFreeList = [];
    this.nodeIdMap = createIdMap<IdLike>();
    this.edgeCount = 0;
    this.edgeCapacity = 0;
    this.edgeIdMap = createIdMap<IdLike>();
    this.positionsX = new Float32Array(0);
    this.positionsY = new Float32Array(0);
    this.nodeAttributes = new Float32Array(0);
    this.edgeSources = new Uint32Array(0);
    this.edgeTargets = new Uint32Array(0);
    this.edgeAttributes = new Float32Array(0);
    this.nodeMetadata = new Map();
    this.edgeMetadata = new Map();
    this.nodeTypes = [];
    this.edgeTypes = [];
    this.nodeEdges = new Map();
  }

  /**
   * Create from a ParsedGraph (on initial load)
   */
  static fromParsedGraph(parsed: ParsedGraph): MutableGraphState {
    const state = new MutableGraphState();

    state.nodeCount = parsed.nodeCount;
    state.nodeCapacity = initialCapacity(parsed.nodeCount);
    state.nodeHighWater = parsed.nodeCount;
    state.nodeFreeList = [];

    state.edgeCount = parsed.edgeCount;
    state.edgeCapacity = initialCapacity(parsed.edgeCount);

    // Copy ID maps
    state.nodeIdMap = parsed.nodeIdMap;
    state.edgeIdMap = parsed.edgeIdMap;

    // Create CPU shadow arrays at capacity size
    state.positionsX = new Float32Array(state.nodeCapacity);
    state.positionsY = new Float32Array(state.nodeCapacity);
    state.nodeAttributes = new Float32Array(state.nodeCapacity * 6);
    state.edgeSources = new Uint32Array(state.edgeCapacity);
    state.edgeTargets = new Uint32Array(state.edgeCapacity);
    state.edgeAttributes = new Float32Array(state.edgeCapacity * 8);

    // Copy data from parsed graph
    state.positionsX.set(parsed.positionsX);
    state.positionsY.set(parsed.positionsY);
    state.nodeAttributes.set(parsed.nodeAttributes);
    state.edgeSources.set(parsed.edgeSources);
    state.edgeTargets.set(parsed.edgeTargets);
    state.edgeAttributes.set(parsed.edgeAttributes);

    // Copy metadata
    state.nodeMetadata = new Map(parsed.nodeMetadata);
    state.edgeMetadata = new Map(parsed.edgeMetadata);
    state.nodeTypes = parsed.nodeTypes ? [...parsed.nodeTypes] : [];
    state.edgeTypes = parsed.edgeTypes ? [...parsed.edgeTypes] : [];

    // Build node-edge adjacency
    state.nodeEdges = new Map();
    for (let i = 0; i < parsed.edgeCount; i++) {
      const src = parsed.edgeSources[i];
      const tgt = parsed.edgeTargets[i];
      if (!state.nodeEdges.has(src)) state.nodeEdges.set(src, new Set());
      if (!state.nodeEdges.has(tgt)) state.nodeEdges.set(tgt, new Set());
      state.nodeEdges.get(src)!.add(i);
      state.nodeEdges.get(tgt)!.add(i);
    }

    return state;
  }

  // ===========================================================================
  // Node Slot Management
  // ===========================================================================

  /**
   * Allocate a node slot. Returns the slot index.
   * Reuses freed slots from the free list, or extends the high-water mark.
   */
  allocateNodeSlot(): number {
    this.nodeCount++;
    if (this.nodeFreeList.length > 0) {
      return this.nodeFreeList.pop()!;
    }
    const slot = this.nodeHighWater;
    this.nodeHighWater++;
    return slot;
  }

  /**
   * Free a node slot. Zeros the slot data and adds to free list.
   */
  freeNodeSlot(index: number): void {
    this.nodeCount--;

    // Zero the slot
    this.positionsX[index] = 0;
    this.positionsY[index] = 0;
    const attrBase = index * 6;
    for (let i = 0; i < 6; i++) {
      this.nodeAttributes[attrBase + i] = 0;
    }

    this.nodeFreeList.push(index);

    // Recalculate high water mark if this was the last slot
    if (index === this.nodeHighWater - 1) {
      while (this.nodeHighWater > 0 && this.nodeFreeList.includes(this.nodeHighWater - 1)) {
        const idx = this.nodeFreeList.indexOf(this.nodeHighWater - 1);
        this.nodeFreeList.splice(idx, 1);
        this.nodeHighWater--;
      }
    }

    // Clean up adjacency
    this.nodeEdges.delete(index);
  }

  /**
   * Check if adding nodes would exceed capacity
   */
  needsNodeReallocation(additionalNodes: number): boolean {
    // Check if the highest slot we'd use exceeds capacity
    const slotsFromFreeList = Math.min(this.nodeFreeList.length, additionalNodes);
    const newSlotsNeeded = additionalNodes - slotsFromFreeList;
    const requiredHighWater = this.nodeHighWater + newSlotsNeeded;
    return requiredHighWater > this.nodeCapacity;
  }

  /**
   * Grow node capacity. Resizes all CPU shadow arrays.
   */
  growNodeCapacity(newCapacity: number): void {
    const oldPosX = this.positionsX;
    const oldPosY = this.positionsY;
    const oldAttrs = this.nodeAttributes;

    this.positionsX = new Float32Array(newCapacity);
    this.positionsY = new Float32Array(newCapacity);
    this.nodeAttributes = new Float32Array(newCapacity * 6);

    this.positionsX.set(oldPosX);
    this.positionsY.set(oldPosY);
    this.nodeAttributes.set(oldAttrs);

    this.nodeCapacity = newCapacity;
  }

  // ===========================================================================
  // Edge Slot Management
  // ===========================================================================

  /**
   * Allocate an edge slot. Returns the index (always appended at end).
   */
  allocateEdgeSlot(): number {
    const slot = this.edgeCount;
    this.edgeCount++;
    return slot;
  }

  /**
   * Free an edge slot using swap-remove.
   * Swaps the last edge into the vacated slot and decrements edgeCount.
   * Returns the index of the edge that was swapped (or -1 if it was the last).
   */
  freeEdgeSlot(index: number): number {
    const lastIndex = this.edgeCount - 1;

    if (index < lastIndex) {
      // Swap last edge into vacated slot
      this.edgeSources[index] = this.edgeSources[lastIndex];
      this.edgeTargets[index] = this.edgeTargets[lastIndex];

      const srcAttr = lastIndex * 8;
      const dstAttr = index * 8;
      for (let i = 0; i < 8; i++) {
        this.edgeAttributes[dstAttr + i] = this.edgeAttributes[srcAttr + i];
      }

      // Update adjacency for the swapped edge
      const swappedSrc = this.edgeSources[index];
      const swappedTgt = this.edgeTargets[index];
      this.nodeEdges.get(swappedSrc)?.delete(lastIndex);
      this.nodeEdges.get(swappedSrc)?.add(index);
      this.nodeEdges.get(swappedTgt)?.delete(lastIndex);
      this.nodeEdges.get(swappedTgt)?.add(index);
    }

    // Clean up the last slot's adjacency
    const removedSrc = index < lastIndex ? this.edgeSources[lastIndex] : this.edgeSources[index];
    const removedTgt = index < lastIndex ? this.edgeTargets[lastIndex] : this.edgeTargets[index];
    if (index === lastIndex) {
      this.nodeEdges.get(removedSrc)?.delete(index);
      this.nodeEdges.get(removedTgt)?.delete(index);
    }

    this.edgeCount--;
    return index < lastIndex ? lastIndex : -1;
  }

  /**
   * Check if adding edges would exceed capacity
   */
  needsEdgeReallocation(additionalEdges: number): boolean {
    return this.edgeCount + additionalEdges > this.edgeCapacity;
  }

  /**
   * Grow edge capacity. Resizes all CPU shadow arrays.
   */
  growEdgeCapacity(newCapacity: number): void {
    const oldSources = this.edgeSources;
    const oldTargets = this.edgeTargets;
    const oldAttrs = this.edgeAttributes;

    this.edgeSources = new Uint32Array(newCapacity);
    this.edgeTargets = new Uint32Array(newCapacity);
    this.edgeAttributes = new Float32Array(newCapacity * 8);

    this.edgeSources.set(oldSources);
    this.edgeTargets.set(oldTargets);
    this.edgeAttributes.set(oldAttrs);

    this.edgeCapacity = newCapacity;
  }

  // ===========================================================================
  // Edge Adjacency Helpers
  // ===========================================================================

  /**
   * Register an edge in the node adjacency map
   */
  addEdgeAdjacency(edgeIndex: number, sourceSlot: number, targetSlot: number): void {
    if (!this.nodeEdges.has(sourceSlot)) this.nodeEdges.set(sourceSlot, new Set());
    if (!this.nodeEdges.has(targetSlot)) this.nodeEdges.set(targetSlot, new Set());
    this.nodeEdges.get(sourceSlot)!.add(edgeIndex);
    this.nodeEdges.get(targetSlot)!.add(edgeIndex);
  }

  /**
   * Get all edge indices connected to a node slot
   */
  getConnectedEdges(nodeSlot: number): Set<number> {
    return this.nodeEdges.get(nodeSlot) ?? new Set();
  }

  // ===========================================================================
  // CSR Generation
  // ===========================================================================

  /**
   * Generate forward CSR (outgoing edges) from the edge arrays.
   *
   * For each node slot, lists the target nodes of its outgoing edges.
   * Offsets array has `nodeHighWater + 1` elements. Dead node slots
   * (from removals) naturally get zero-length edge lists since no live
   * edges reference them as sources.
   */
  generateForwardCSR(): { offsets: Uint32Array; targets: Uint32Array } {
    const hw = this.nodeHighWater;
    const ec = this.edgeCount;

    // Count outgoing edges per node slot
    const counts = new Uint32Array(hw);
    for (let i = 0; i < ec; i++) {
      const src = this.edgeSources[i];
      if (src < hw) counts[src]++;
    }

    // Prefix sum → offsets
    const offsets = new Uint32Array(hw + 1);
    for (let i = 0; i < hw; i++) {
      offsets[i + 1] = offsets[i] + counts[i];
    }

    // Build targets array
    const targets = new Uint32Array(ec);
    const currentOffset = new Uint32Array(hw);
    for (let i = 0; i < ec; i++) {
      const src = this.edgeSources[i];
      if (src < hw) {
        const idx = offsets[src] + currentOffset[src];
        targets[idx] = this.edgeTargets[i];
        currentOffset[src]++;
      }
    }

    return { offsets, targets };
  }

  /**
   * Generate inverse CSR (incoming edges) from the edge arrays.
   *
   * For each node slot, lists the source nodes of its incoming edges.
   * Same structure as forward CSR but with source/target roles swapped.
   */
  generateInverseCSR(): { offsets: Uint32Array; sources: Uint32Array } {
    const hw = this.nodeHighWater;
    const ec = this.edgeCount;

    // Count incoming edges per node slot
    const counts = new Uint32Array(hw);
    for (let i = 0; i < ec; i++) {
      const tgt = this.edgeTargets[i];
      if (tgt < hw) counts[tgt]++;
    }

    // Prefix sum → offsets
    const offsets = new Uint32Array(hw + 1);
    for (let i = 0; i < hw; i++) {
      offsets[i + 1] = offsets[i] + counts[i];
    }

    // Build sources array
    const sources = new Uint32Array(ec);
    const currentOffset = new Uint32Array(hw);
    for (let i = 0; i < ec; i++) {
      const tgt = this.edgeTargets[i];
      if (tgt < hw) {
        const idx = offsets[tgt] + currentOffset[tgt];
        sources[idx] = this.edgeSources[i];
        currentOffset[tgt]++;
      }
    }

    return { offsets, sources };
  }
}
