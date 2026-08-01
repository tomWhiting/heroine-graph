/**
 * Node Identity — Slot ↔ Producer Identifier
 *
 * A `NodeId` is a GPU slot index: an allocation detail that is recycled on
 * removal and means nothing outside the instance that issued it. Anything
 * that keeps its own record per node — a DOM card, a selection set, a search
 * result — has to key on the identifier the producer supplied, and until now
 * the only mapping back was `nodeIdMap` on private state, so consumers kept a
 * duplicate slot table of their own.
 *
 * These two functions are that mapping. They back `CardNode.externalId` and
 * the public `GraphMother.getExternalId` / `GraphMother.getNodeId`, and they
 * are pure so both directions are testable without a GPU device.
 *
 * @module
 */

import type { NodeId } from "../types.ts";
import type { IdLike, IdMap } from "../graph/id_map.ts";

/**
 * The slice of `MutableGraphState` identity resolution reads.
 *
 * Liveness is checked against `nodeFreeSet` as well as the id map's own free
 * list: the two are kept in lockstep by the mutation path, and agreeing with
 * only one of them would let a half-applied mutation resolve a dead slot.
 */
export interface NodeIdentitySource {
  /** Highest slot index in use + 1. */
  readonly nodeHighWater: number;
  /** Slots freed by removals and awaiting reuse. */
  readonly nodeFreeSet: ReadonlySet<number>;
  readonly nodeIdMap: Pick<IdMap<IdLike>, "get" | "getId">;
}

/**
 * Resolve a node slot to the identifier the producer supplied for it.
 *
 * Returns `undefined` — never throws — for a slot that is out of range, was
 * freed by a removal, or was never mapped, so a stale `NodeId` held across a
 * mutation is safe to look up.
 */
export function externalIdForSlot(
  source: NodeIdentitySource,
  slot: NodeId,
): IdLike | undefined {
  if (!Number.isInteger(slot) || slot < 0 || slot >= source.nodeHighWater) {
    return undefined;
  }
  if (source.nodeFreeSet.has(slot)) return undefined;
  return source.nodeIdMap.getId(slot);
}

/**
 * Resolve a producer identifier to the node slot currently holding it.
 *
 * Returns `undefined` if the identifier was never loaded or its node has been
 * removed. Slots are recycled, so the returned value is only valid until the
 * next mutation.
 */
export function slotForExternalId(
  source: NodeIdentitySource,
  externalId: IdLike,
): NodeId | undefined {
  const slot = source.nodeIdMap.get(externalId);
  if (slot === undefined) return undefined;
  if (slot < 0 || slot >= source.nodeHighWater) return undefined;
  if (source.nodeFreeSet.has(slot)) return undefined;
  return slot;
}
