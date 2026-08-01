/**
 * MutableGraphState
 *
 * Tracks the live graph state on the CPU side, managing the mapping between
 * user IDs, internal slot indices, and GPU buffer offsets. This is the
 * single source of truth for incremental mutations.
 *
 * Node slots use a free-list with a high-water mark. Removed node slots
 * are zeroed (radius=0 → invisible), flagged dead on the GPU (excluded
 * from forces/integration/collision) and reused on subsequent adds.
 *
 * Edge slots use dense packing with swap-remove. The last edge is swapped
 * into the removed slot, keeping the array contiguous.
 *
 * @module
 */

import type { IdLike, IdMap } from "../graph/id_map.ts";
import { createIdMap } from "../graph/id_map.ts";
import type { ParsedGraph } from "../graph/parser.ts";
import { growCapacity, initialCapacity } from "./buffer_capacity.ts";

/** Number of f32 values per node in the attribute buffer */
export const NODE_ATTR_FLOATS = 8;
/** Byte stride per node in the attribute buffer (NODE_ATTR_FLOATS * 4) */
export const NODE_ATTR_BYTES = 32;

/** Collision radius given to a live slot whose attribute radius is unset. */
export const DEFAULT_COLLISION_RADIUS = 5;

/**
 * Mutable graph state for incremental mutations
 */
export class MutableGraphState {
  // Node tracking
  nodeCount: number;
  nodeCapacity: number;
  /** Highest slot index in use + 1 (used as instance count for draw calls) */
  nodeHighWater: number;
  /** Reusable slot indices from removals (stack for LIFO pop) */
  nodeFreeList: number[];
  /** O(1) membership check for nodeFreeList */
  nodeFreeSet: Set<number>;
  nodeIdMap: IdMap<IdLike>;

  // Edge tracking (dense, no gaps via swap-remove)
  edgeCount: number;
  edgeCapacity: number;
  edgeIdMap: IdMap<IdLike>;

  // CPU shadow arrays (kept in sync with GPU buffers)
  positionsX: Float32Array;
  positionsY: Float32Array;
  /** 8 floats per slot: radius, r, g, b, selected, hovered, birth_time, tex_index */
  nodeAttributes: Float32Array;
  /**
   * Per-slot node state flags (NODE_FLAG_* bits from simulation/pipeline.ts).
   *
   * The single CPU-side authority for the GPU nodeFlags buffer. Liveness,
   * pinning and visibility are independent bits owned by different callers, so
   * every writer edits only its own bits here ({@link setNodeFlagBits}) and the
   * GPU write is a copy of this array — no writer ever recomposes the word from
   * the others' state, which is how a bit used to get dropped.
   */
  nodeFlagsShadow: Uint32Array;
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
    this.nodeFreeSet = new Set();
    this.nodeIdMap = createIdMap<IdLike>();
    this.edgeCount = 0;
    this.edgeCapacity = 0;
    this.edgeIdMap = createIdMap<IdLike>();
    this.positionsX = new Float32Array(0);
    this.positionsY = new Float32Array(0);
    this.nodeAttributes = new Float32Array(0);
    this.nodeFlagsShadow = new Uint32Array(0);
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
    state.nodeAttributes = new Float32Array(state.nodeCapacity * NODE_ATTR_FLOATS);
    // Zero = every slot live, unpinned and visible, matching the freshly
    // created (zero-filled) GPU nodeFlags buffer.
    state.nodeFlagsShadow = new Uint32Array(state.nodeCapacity);
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
      const slot = this.nodeFreeList.pop()!;
      this.nodeFreeSet.delete(slot);
      return slot;
    }
    const slot = this.nodeHighWater;
    this.nodeHighWater++;
    return slot;
  }

  /**
   * Free a node slot. Zeros the slot data and adds to free list.
   *
   * Freed slots are kept in the free list even when they form a trailing
   * run below nodeHighWater. The pure LIFO push/pop order must mirror
   * petgraph StableGraph's vacancy list in the WASM engine, which never
   * reclaims trailing slots — pruning here would make the two sides reuse
   * different slots and break the NodeId == slot contract. Holes are
   * reclaimed by the compacting batch paths (removeNodesBatch) instead.
   */
  freeNodeSlot(index: number): void {
    this.nodeCount--;

    // Zero the slot
    this.positionsX[index] = 0;
    this.positionsY[index] = 0;
    const attrBase = index * NODE_ATTR_FLOATS;
    for (let i = 0; i < NODE_ATTR_FLOATS; i++) {
      this.nodeAttributes[attrBase + i] = 0;
    }

    this.nodeFreeList.push(index);
    this.nodeFreeSet.add(index);

    // Clean up adjacency
    this.nodeEdges.delete(index);
  }

  /**
   * Set or clear `mask` in one slot's flag word, leaving every other bit
   * untouched. Returns whether the word actually changed, so callers can skip
   * the GPU write for a no-op.
   */
  setNodeFlagBits(slot: number, mask: number, enabled: boolean): boolean {
    const before = this.nodeFlagsShadow[slot];
    const after = enabled ? before | mask : before & ~mask;
    if (after === before) return false;
    this.nodeFlagsShadow[slot] = after;
    return true;
  }

  /**
   * Replace one slot's flag word outright. Only for slot (re)assignment, where
   * the previous occupant's pin and visibility are gone by definition; every
   * other writer must go through {@link setNodeFlagBits}.
   */
  resetNodeFlags(slot: number, flags: number): void {
    this.nodeFlagsShadow[slot] = flags;
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
    const oldFlags = this.nodeFlagsShadow;

    this.positionsX = new Float32Array(newCapacity);
    this.positionsY = new Float32Array(newCapacity);
    this.nodeAttributes = new Float32Array(newCapacity * NODE_ATTR_FLOATS);
    this.nodeFlagsShadow = new Uint32Array(newCapacity);

    this.positionsX.set(oldPosX);
    this.positionsY.set(oldPosY);
    this.nodeAttributes.set(oldAttrs);
    this.nodeFlagsShadow.set(oldFlags);

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

    // Save the removed edge's adjacency BEFORE any swap overwrites it
    const removedSrc = this.edgeSources[index];
    const removedTgt = this.edgeTargets[index];

    // Clean up the removed edge's adjacency
    this.nodeEdges.get(removedSrc)?.delete(index);
    this.nodeEdges.get(removedTgt)?.delete(index);

    if (index < lastIndex) {
      // Swap last edge into vacated slot
      this.edgeSources[index] = this.edgeSources[lastIndex];
      this.edgeTargets[index] = this.edgeTargets[lastIndex];

      const srcAttr = lastIndex * 8;
      const dstAttr = index * 8;
      for (let i = 0; i < 8; i++) {
        this.edgeAttributes[dstAttr + i] = this.edgeAttributes[srcAttr + i];
      }

      // Update adjacency for the swapped edge (was at lastIndex, now at index)
      const swappedSrc = this.edgeSources[index];
      const swappedTgt = this.edgeTargets[index];
      this.nodeEdges.get(swappedSrc)?.delete(lastIndex);
      this.nodeEdges.get(swappedSrc)?.add(index);
      this.nodeEdges.get(swappedTgt)?.delete(lastIndex);
      this.nodeEdges.get(swappedTgt)?.add(index);
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

/**
 * Re-point every column of `parsed` at the live state, so the two describe one
 * graph rather than two.
 *
 * `MutableGraphState` is the authority for everything a mutation touches, and
 * `ParsedGraph` is the read surface the renderer, the hit tester and the card
 * source were written against. Aliasing rather than copying is what makes that
 * safe: a compaction rebuilds the id maps and the metadata maps against new
 * slots, and a reader still holding the load-time objects would answer with a
 * neighbour's identity — its label, its content reference — with no error
 * anywhere.
 *
 * The typed arrays are aliased whole (their contents are slot-indexed and the
 * state owns them); the edge views are re-sliced because a mutation moves the
 * live edge count.
 */
export function aliasParsedGraphToState(parsed: ParsedGraph, gs: MutableGraphState): void {
  parsed.positionsX = gs.positionsX;
  parsed.positionsY = gs.positionsY;
  parsed.nodeAttributes = gs.nodeAttributes;
  parsed.edgeSources = gs.edgeSources.subarray(0, gs.edgeCount);
  parsed.edgeTargets = gs.edgeTargets.subarray(0, gs.edgeCount);
  parsed.edgeAttributes = gs.edgeAttributes.subarray(0, gs.edgeCount * 8);
  parsed.nodeCount = gs.nodeHighWater; // slot space, not live count: draw calls index it
  parsed.edgeCount = gs.edgeCount;
  parsed.nodeIdMap = gs.nodeIdMap;
  parsed.edgeIdMap = gs.edgeIdMap;
  parsed.nodeTypes = gs.nodeTypes as string[];
  parsed.edgeTypes = gs.edgeTypes as string[];
  parsed.nodeMetadata = gs.nodeMetadata;
  parsed.edgeMetadata = gs.edgeMetadata;
}

/**
 * Shift a slot-indexed column down to match a compaction, in place.
 *
 * `remap` is `oldSlot -> newSlot` for the survivors of a batch removal, which
 * only ever moves a slot downward; the tail above the last survivor is zeroed
 * so a dropped node's tag or weight cannot be read back through a slot nobody
 * occupies. Absent columns (the input carried none) stay absent.
 */
export function compactNodeColumn(
  column: Uint16Array | Float32Array | undefined,
  remap: ReadonlyMap<number, number>,
  highWater: number,
): void {
  if (!column) return;
  const end = Math.min(highWater, column.length);
  let written = 0;
  for (let oldSlot = 0; oldSlot < end; oldSlot++) {
    const newSlot = remap.get(oldSlot);
    if (newSlot === undefined) continue;
    column[newSlot] = column[oldSlot];
    written = newSlot + 1;
  }
  column.fill(0, written, end);
}

/**
 * The same column, sized to hold `slot`.
 *
 * These columns are allocated for the loaded graph and the slot space outgrows
 * them, so an added node needs the column extended before it can be written.
 * Growth follows the GPU buffers' doubling rather than fitting exactly: adds
 * arrive one at a time, and a reallocation per node would make a 10 000-node
 * insert quadratic.
 *
 * @returns the column to write through — the original when it already fits
 */
export function growSlotColumn<T extends Uint16Array | Float32Array>(
  column: T | undefined,
  slot: number,
  make: new (length: number) => T,
): T {
  if (column && slot < column.length) return column;
  const grown = new make(growCapacity(slot + 1, column?.length ?? 0));
  if (column) grown.set(column);
  return grown;
}

/** What {@link collisionRadiusColumn} has to ask about a slot it cannot see. */
export interface CollisionRadiusSource {
  /** Whether the slot holds no live node, and so takes {@link deadRadius}. */
  isDead(slot: number): boolean;
  /**
   * The radius a slot had before it started rendering as a collapsed proxy, or
   * `undefined` when it is not one (`ProxyRadiusTable.savedRadiusOf`).
   */
  proxyRadius(slot: number): number | undefined;
  /**
   * The collision sentinel for a dead slot (`DEAD_SLOT_RADIUS`), passed in
   * because pipeline.ts cannot be imported outside a bundle.
   */
  deadRadius: number;
}

/**
 * The collision radius of every slot in `[0, count)`, and the largest live one.
 *
 * Collision reads a radius the *physics* owns, which is not always the radius
 * in the attribute row: a collapsed LOD proxy borrows that row to render at its
 * well radius, which can be two orders of magnitude larger than the node's own.
 * Letting that reach collision would have the proxy physically shove every node
 * inside the bubble it is only *drawing*, and would coarsen the spatial-hash
 * cell size for the whole graph. `proxyRadius` gives the pre-inflation radius
 * back for a slot currently standing in for a subtree.
 *
 * The maximum is returned rather than accumulated by the caller: it sizes the
 * collision grid, so a radius that only ratchets upward leaves the grid coarse
 * for the rest of the session after a single inflated read.
 */
export function collisionRadiusColumn(
  attributes: Float32Array,
  count: number,
  source: CollisionRadiusSource,
): { sizes: Float32Array; maxRadius: number } {
  const { isDead, proxyRadius, deadRadius } = source;
  const sizes = new Float32Array(count);
  let maxRadius = 0;
  for (let i = 0; i < count; i++) {
    if (isDead(i)) {
      sizes[i] = deadRadius;
      continue;
    }
    const own = proxyRadius(i) ?? attributes[i * NODE_ATTR_FLOATS];
    const radius = own > 0 ? own : DEFAULT_COLLISION_RADIUS;
    sizes[i] = radius;
    if (radius > maxRadius) maxRadius = radius;
  }
  return { sizes, maxRadius };
}
