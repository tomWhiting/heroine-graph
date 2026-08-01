/**
 * Aggregate mass for collapsed subtrees.
 *
 * Collapsing a subtree replaces thousands of bodies with one proxy. If that
 * proxy repels like a single node the visible layout contracts on collapse and
 * re-inflates on expand, so the arrangement of what stays on screen changes as
 * you zoom — the thing semantic LOD must not do. The fix is to give the proxy
 * the mass of everything it stands for, and to give the nodes it hides no mass
 * at all.
 *
 * Mass is per-slot and multiplies the *source* side of a repulsion pair. The
 * integrator carries no mass term, so a proxy of mass `M` receiving the force
 * a single body would receive accelerates like the centre of mass of the `M`
 * bodies it replaces (`a = M·F / M`), which is what keeps the visible layout
 * still.
 *
 * @module
 */

import { ErrorCode, GraphMotherError } from "../errors.ts";
import { type ChildrenCsr, HIERARCHY_ROOT } from "../graph/hierarchy.ts";

/**
 * Mass of a node that stands only for itself.
 *
 * Every slot carries this until a collapse says otherwise, which is what makes
 * LOD-off output bit-identical: multiplying a force by exactly 1.0 is the IEEE
 * identity, so the shaders compute the same bits they did before mass existed.
 */
export const NODE_MASS_UNIT = 1;

/**
 * Mass of a node hidden inside a collapsed subtree.
 *
 * Hidden nodes are excluded from force exchange by their flag, but their
 * buffer value must still be zero: their mass now lives in the proxy, and
 * leaving it at 1.0 would double-count it the moment any pass reads the buffer
 * without consulting the flag.
 */
export const NODE_MASS_HIDDEN = 0;

/** No visible ancestor is accumulating this branch — the node is visible. */
const NO_CARRIER = 0xFFFFFFFF;

/**
 * The half-open slot range a mass upload has to cover. An unchanged upload
 * reports `{ lo: 0, hi: 0 }`; callers test `lo === hi` and skip the write.
 */
export interface MassDirtyRange {
  readonly lo: number;
  readonly hi: number;
}

/**
 * Copy `mass` into a CPU shadow of the GPU buffer and report what changed.
 *
 * Collapse and expand touch a contiguous run of slots when the producer emits
 * slots depth-first, and a wide but sparse one otherwise; either way a single
 * range write beats one queue operation per slot, and an unchanged transition
 * costs no write at all.
 *
 * `mass` covers `[0, mass.length)` and slots beyond it are left as they are —
 * that tail is buffer capacity the simulation never dispatches over.
 *
 * @throws GraphMotherError if `mass` is longer than the shadow, or carries a
 *   value that is not a finite non-negative number. A NaN or negative mass
 *   does not fail loudly on the GPU: it turns the layout to NaN, or inverts a
 *   repulsion into an attraction.
 */
export function commitNodeMass(shadow: Float32Array, mass: Float32Array): MassDirtyRange {
  const count = mass.length;
  if (count > shadow.length) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `node mass has ${count} entries, exceeding the shadow length (${shadow.length})`,
    );
  }

  let lo = 0;
  while (lo < count && shadow[lo] === mass[lo]) lo++;
  if (lo === count) return { lo: 0, hi: 0 };
  let hi = count;
  while (hi > lo && shadow[hi - 1] === mass[hi - 1]) hi--;

  for (let i = lo; i < hi; i++) {
    if (!(mass[i] >= 0) || !Number.isFinite(mass[i])) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `node mass for slot ${i} is ${mass[i]}; expected a finite non-negative number`,
      );
    }
  }

  shadow.set(mass.subarray(lo, hi), lo);
  return { lo, hi };
}

/**
 * Working arrays {@link rollUpMass} would otherwise allocate on every call.
 *
 * A transition runs the roll-up over the whole slot space, so at 220K nodes
 * the three arrays are 1.9 MB of garbage per zoom step — produced on the
 * interaction path, where a collection pause is a dropped frame. Passing a
 * scratch set makes the walk allocation-free; omitting it is still correct.
 *
 * Build it with {@link createRollUpMassScratch} and keep it for as long as the
 * hierarchy it is sized for.
 */
export interface RollUpMassScratch {
  /** Membership of the collapsed-root set, one byte per slot. */
  readonly collapsed: Uint8Array;
  /** Explicit descent stack: the slot at each level. */
  readonly stackSlot: Uint32Array;
  /** Explicit descent stack: the visible ancestor accumulating that slot. */
  readonly stackCarrier: Uint32Array;
}

/**
 * Allocate a {@link RollUpMassScratch} for a `nodeCount`-slot hierarchy.
 *
 * Sized exactly: every slot is pushed at most once (it has one parent), so
 * `nodeCount` entries bound the stack for any forest over that slot space.
 */
export function createRollUpMassScratch(nodeCount: number): RollUpMassScratch {
  return {
    collapsed: new Uint8Array(nodeCount),
    stackSlot: new Uint32Array(nodeCount),
    stackCarrier: new Uint32Array(nodeCount),
  };
}

/**
 * Compute per-slot mass for a set of collapsed subtree roots.
 *
 * Every node hidden by the collapse contributes its unit of mass to its
 * *lowest visible ancestor* and keeps none, so nesting resolves naturally: a
 * collapsed root inside another collapsed subtree is itself hidden, and its
 * subtree's mass passes through it to the outer proxy. A collapsed root with
 * no children hides nothing and stays at {@link NODE_MASS_UNIT}.
 *
 * Total mass is conserved by construction — every slot's unit lands in exactly
 * one place — so the sum over the result is always `parent.length`, whatever
 * the collapse set. Hidden nodes are exactly the slots that come back
 * {@link NODE_MASS_HIDDEN}, which is the set a caller must also mark
 * `NODE_FLAG_HIDDEN_LOD`.
 *
 * O(nodeCount), one top-down forest walk, no recursion.
 *
 * @param parent Containment parent per slot; {@link HIERARCHY_ROOT} marks a
 *   forest root.
 * @param children Parent→children CSR over the same slot space, as
 *   `buildChildrenCsr` produces.
 * @param collapsedRoots Slots whose subtrees collapse into them. May be empty,
 *   may nest, and may repeat.
 * @param out Optional destination, reused across transitions to keep the
 *   per-transition path allocation-free. Must be at least `parent.length` long;
 *   entries beyond that are left untouched.
 * @param scratch Optional working arrays, reused for the same reason as `out`.
 *   Each must be at least `parent.length` long; entries beyond that are left
 *   untouched. Contents on entry are irrelevant — the walk seeds everything it
 *   reads — so one set can serve every transition over one hierarchy.
 * @returns `out`, or a fresh array, holding one mass per slot.
 * @throws GraphMotherError if the CSR does not describe `parent.length` slots,
 *   `out` or any scratch array is too short, or a collapsed root is out of
 *   range.
 */
export function rollUpMass(
  parent: Uint32Array,
  children: ChildrenCsr,
  collapsedRoots: Iterable<number>,
  out?: Float32Array,
  scratch?: RollUpMassScratch,
): Float32Array {
  const nodeCount = parent.length;
  if (children.offsets.length !== nodeCount + 1) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `children CSR has ${children.offsets.length} offsets, expected nodeCount + 1 ` +
        `(${nodeCount + 1})`,
    );
  }
  if (out && out.length < nodeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `mass output has ${out.length} entries, expected at least nodeCount (${nodeCount})`,
    );
  }
  if (scratch) {
    for (
      const [name, length] of [
        ["collapsed", scratch.collapsed.length],
        ["stackSlot", scratch.stackSlot.length],
        ["stackCarrier", scratch.stackCarrier.length],
      ] as const
    ) {
      if (length < nodeCount) {
        throw new GraphMotherError(
          ErrorCode.INVALID_GRAPH_DATA,
          `mass scratch ${name} has ${length} entries, expected at least nodeCount ` +
            `(${nodeCount})`,
        );
      }
    }
  }

  const mass = out ?? new Float32Array(nodeCount);
  // A slot unreachable from any root is never visited below. Seeding the whole
  // range keeps such a slot at unit mass instead of inheriting a stale value
  // from a previous transition through `out`.
  mass.fill(NODE_MASS_UNIT, 0, nodeCount);

  // A reused scratch carries the previous transition's collapsed set, and a
  // stale 1 here would hide a subtree that is now expanded.
  const collapsed = scratch?.collapsed ?? new Uint8Array(nodeCount);
  if (scratch) collapsed.fill(0, 0, nodeCount);
  let anyCollapsed = false;
  for (const root of collapsedRoots) {
    if (!Number.isInteger(root) || root < 0 || root >= nodeCount) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `collapsed root ${root} is not a slot in [0, ${nodeCount})`,
      );
    }
    collapsed[root] = 1;
    anyCollapsed = true;
  }
  if (!anyCollapsed) return mass;

  // Every slot is pushed exactly once (one parent each), so nodeCount entries
  // bound the stack. No seeding needed on a reused pair: an entry is always
  // written before it is read.
  const stackSlot = scratch?.stackSlot ?? new Uint32Array(nodeCount);
  const stackCarrier = scratch?.stackCarrier ?? new Uint32Array(nodeCount);
  let top = 0;
  for (let i = 0; i < nodeCount; i++) {
    if (parent[i] !== HIERARCHY_ROOT) continue;
    stackSlot[top] = i;
    stackCarrier[top] = NO_CARRIER;
    top++;
  }

  const { offsets, children: childList } = children;
  while (top > 0) {
    top--;
    const slot = stackSlot[top];
    const carrier = stackCarrier[top];

    let childCarrier: number;
    if (carrier === NO_CARRIER) {
      mass[slot] = NODE_MASS_UNIT;
      // A collapsed node carries its descendants; anything else passes the
      // visible frontier down unchanged.
      childCarrier = collapsed[slot] === 1 ? slot : NO_CARRIER;
    } else {
      mass[slot] = NODE_MASS_HIDDEN;
      // The carrier was assigned when it was popped, which is strictly before
      // any of its descendants, so this only ever adds to a settled value.
      mass[carrier] += NODE_MASS_UNIT;
      childCarrier = carrier;
    }

    for (let c = offsets[slot]; c < offsets[slot + 1]; c++) {
      stackSlot[top] = childList[c];
      stackCarrier[top] = childCarrier;
      top++;
    }
  }

  return mass;
}
