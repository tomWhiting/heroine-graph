/**
 * The demo's semantic LOD policy.
 *
 * Two things a purely geometric threshold cannot know about a code tree, and
 * nothing else. The policy runs inside the controller's evaluation pass, once
 * per node that crossed a band, so it is pure, synchronous and allocates
 * nothing — including no closure over the candidate, which core reuses and
 * overwrites between calls.
 *
 * @module
 */

// Typed against the module that declares it, not the barrel — see the note at
// the top of `repo.ts`.
import type { LodPolicy } from "../../../packages/core/src/lod/policy.ts";
import { NODE_KIND } from "./repo.ts";

/**
 * - **A repository root never stands in for its repository.** Folded, it is a
 *   single sprite for the whole graph: technically the correct answer to "this
 *   is too small to read", and useless. Its children are the top-level layout,
 *   so the root always unfolds.
 * - **A directory with one child carries no information.** `src/main/java/...`
 *   is one edge repeated; rendering the chain spends slots and screen space on
 *   nodes that say nothing the child does not. `pass-through` drops the node
 *   from the cut and considers its children at its parent's level, so a whole
 *   chain of them collapses to nothing rather than one link per evaluation.
 *
 * Everything else — how big a subtree must be before it unfolds, when a leaf
 * earns a DOM card — is geometry, and geometry is core's job.
 */
export const codeGraphLodPolicy: LodPolicy = (candidate) => {
  if (candidate.depth === 0) return "expand";
  if (candidate.tag === NODE_KIND.dir && candidate.childCount === 1) return "pass-through";
  return "default";
};
