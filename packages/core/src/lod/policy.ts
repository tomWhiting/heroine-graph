/**
 * Semantic-LOD policy hook.
 *
 * A purely geometric threshold cannot know that `src/api/` should stay
 * expanded while `node_modules/` stays collapsed. It also cannot know that a
 * single-child directory chain carries no information at all, which is endemic
 * in code trees and the highest-value thing a producer can tell the layout.
 *
 * The hook is deliberately narrow. Core hands the policy two opaque per-node
 * columns it never interprets — {@link LodCandidate.tag}, an index into a table
 * only the producer has, and {@link LodCandidate.weight}, a 0..1 importance —
 * and takes back one verb. No strings are compared, no metadata map is walked,
 * and nothing the policy returns can put the cut into a state the geometric
 * rule could not have reached on its own.
 *
 * @module
 */

import type { NodeId } from "../types.ts";

/**
 * What a policy wants done with one candidate.
 *
 * - `default` — delegate to the built-in geometric rule.
 * - `expand` / `collapse` — override it for this evaluation.
 * - `pass-through` — this node carries no information: render its children as
 *   if it were not here. The node leaves the cut entirely and its children are
 *   considered at its parent's level, so a chain of them collapses to nothing.
 * - `force-card` — promote to DOM regardless of screen size, for a focus or
 *   search hit. Expansion still follows the geometric rule.
 */
export type LodDecision =
  | "default"
  | "expand"
  | "collapse"
  | "pass-through"
  | "force-card";

/**
 * One node offered to the policy.
 *
 * **Reused and mutated between calls.** Core allocates one of these per
 * controller and overwrites it for every candidate, so a policy that retains it
 * observes a different node on the next call. Read what you need and return.
 */
export interface LodCandidate {
  /** Node slot. */
  readonly node: NodeId;
  /** Containment depth; forest roots are 0. */
  readonly depth: number;
  /** Direct children in the containment tree. */
  readonly childCount: number;
  /** Nodes in this node's subtree, including itself. */
  readonly subtreeSize: number;
  /** Subtree extent in graph units. */
  readonly wellRadius: number;
  /** Subtree extent in screen pixels: `wellRadius * zoom`. */
  readonly screenRadius: number;
  /** Whether the subtree's extent intersects the viewport rect. */
  readonly onScreen: boolean;
  /** Producer-supplied tag index; 0 when the graph carries no tag column. */
  readonly tag: number;
  /** Producer-supplied importance in 0..1; 0 when the graph carries none. */
  readonly weight: number;
}

/** The evaluation the candidate belongs to. Stable for all candidates in one pass. */
export interface LodContext {
  /** Quantised zoom the evaluation ran at. */
  readonly zoom: number;
  /** Screen extent at or above which the geometric rule expands. */
  readonly expand: number;
  /** Screen extent below which the geometric rule collapses. */
  readonly collapse: number;
  /** Leaf screen radius at or above which a node is promoted to DOM. */
  readonly dom: number;
  /** Nodes the host has declared as focus. */
  readonly focus: ReadonlySet<NodeId>;
}

/**
 * Decide what happens to one candidate.
 *
 * **MUST be pure, synchronous and allocation-free.** It runs inside the
 * controller's evaluation pass, which is on the interaction path, and it is
 * called once per node that crossed a band — a policy that allocates turns a
 * cheap pass into a GC pause exactly when the camera is moving.
 */
export type LodPolicy = (candidate: LodCandidate, context: LodContext) => LodDecision;

/**
 * The built-in policy: geometry only.
 *
 * Registering nothing and registering this are the same thing. It exists so a
 * consumer policy can delegate the cases it has no opinion about without
 * knowing what the geometric rule is.
 */
export const defaultLodPolicy: LodPolicy = () => "default";
