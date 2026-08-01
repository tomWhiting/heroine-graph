/**
 * The demo's producer: a synthetic code repository.
 *
 * A code graph is a containment tree — repo → directory → file → symbol — with
 * cross-cutting dependency edges laid over it. This module builds one at a
 * chosen size and emits it in the shape GraphMother's typed fast path wants:
 * flat slot-indexed columns, no per-node objects, and the containment
 * hierarchy precomputed rather than re-derived from the edges at load.
 *
 * It stands in for a real indexer. Everything a real one would ship — paths,
 * node kinds, importance, the four hierarchy columns — is produced here, so
 * swapping this file for a `libcorpus`-style producer changes nothing
 * downstream.
 *
 * The shape parameters and the three scales match `tests/fixtures/code_tree.ts`
 * (2 500 / 35 000 / 220 000 nodes), so the demo and the physics harness argue
 * about the same workload.
 *
 * # Producer contracts honoured here
 *
 * - **Depth-first slot order.** Slots are assigned in DFS pre-order, so every
 *   subtree is a contiguous slot range and every child has a higher slot than
 *   its parent. Core neither verifies nor requires this, but its LOD expand
 *   fix-up becomes one buffer write per subtree instead of one per node — and
 *   it lets the bottom-up sweeps below run as a single reverse scan.
 * - **Supplied hierarchy.** `wellRadius` is computed with the same formula and
 *   the same constants as the WASM derivation, so supplying the columns and
 *   letting core derive them produce the same layout.
 * - **Containment typing.** Dependency edges are tagged with a different kind
 *   than containment edges, so nothing can reparent a file under a module that
 *   merely imports it.
 *
 * @module
 */

// The app layer of this example imports from the public barrel, as a consumer
// would. This file and the other pure modules (`knobs.ts`, `lod_policy.ts`)
// import from the core modules that declare what they use instead: the barrel
// is a bundler entry point — it imports WGSL as text — and these modules are
// unit-tested under `deno test`, which resolves the whole graph. Everything
// named here is re-exported unchanged by `mod.ts`.
import type { GraphTypedInput } from "../../../packages/core/src/types.ts";
import {
  HIERARCHY_ROOT,
  type HierarchyColumns,
} from "../../../packages/core/src/graph/hierarchy.ts";

// =============================================================================
// Node kinds — the `tag` column
// =============================================================================

/**
 * Producer-defined node kinds, as they appear in `GraphTypedInput.tag`.
 *
 * Core never interprets these; it compares them for the LOD policy and hands
 * them back on `CardNode.tag`. The table lives on this side of the boundary
 * only.
 */
export const NODE_KIND = {
  repo: 0,
  dir: 1,
  file: 2,
  symbol: 3,
} as const;

/** Name of a {@link NODE_KIND} entry. */
export type NodeKindName = keyof typeof NODE_KIND;

/** Numeric kind as it appears in the `tag` column. */
export type NodeKind = typeof NODE_KIND[NodeKindName];

/** Display name per kind, indexed by the numeric kind. */
export const KIND_NAMES: readonly NodeKindName[] = ["repo", "dir", "file", "symbol"];

/** Sprite colour per kind as `[r, g, b]` in 0..1, indexed by the numeric kind. */
const KIND_COLORS: readonly (readonly [number, number, number])[] = [
  [0.93, 0.93, 0.97], // repo   — near-white
  [0.95, 0.69, 0.20], // dir    — amber
  [0.31, 0.55, 1.00], // file   — blue
  [0.27, 0.82, 0.64], // symbol — green
];

/** Sprite radius per kind, in graph units, indexed by the numeric kind. */
const KIND_RADII: readonly number[] = [9, 5, 3, 1.8];

/**
 * Edge kind values in `GraphTypedInput.edgeKinds`.
 *
 * Core reads exactly one of them — the one named by `containmentKind` — and
 * treats every other value as "not containment". Tagging imports separately is
 * what stops a file being reparented under a module that merely imports it.
 */
const CONTAINMENT_EDGE_KIND = 0;
const IMPORT_EDGE_KIND = 1;

// =============================================================================
// Bubble constants
// =============================================================================

/**
 * Leaf bubble radius, and the padding added to an internal node's.
 *
 * These are the producer's half of a shared contract: `main.ts` passes the same
 * two numbers to `setForceConfig`, so the radii in the supplied hierarchy and
 * the ones the layout would have derived agree. Changing one without the other
 * silently rescales the layout.
 */
export const BUBBLE_BASE_RADIUS = 10;
export const BUBBLE_PADDING = 5;

/**
 * Circle-packing efficiency used when sizing a parent bubble from its children.
 *
 * Matches `BubbleConfig::default()` in `packages/wasm/src/layout/bubble.rs`.
 */
const BUBBLE_PACKING_EFFICIENCY = 0.82;

// =============================================================================
// Scales
// =============================================================================

/** Identifier of one entry in {@link REPO_SCALES}. */
export type RepoScaleId = "small" | "medium" | "large";

/** One selectable dataset size. */
export interface RepoScale {
  readonly id: RepoScaleId;
  /** Menu label. */
  readonly label: string;
  readonly nodeCount: number;
  /** What this scale is for; shown next to the control. */
  readonly note: string;
}

/**
 * The three sizes the library's performance envelope is stated at.
 *
 * `large` is included because the envelope claims it, not because it is a
 * pleasant demo: generating it takes a few seconds and it is past the point
 * where GPU labels are affordable (see {@link rankLabelCandidates}). It is the
 * stress case, and it is labelled as one.
 */
export const REPO_SCALES: readonly RepoScale[] = [
  { id: "small", label: "2.5K", nodeCount: 2_500, note: "demo repo — the default" },
  { id: "medium", label: "35K", nodeCount: 35_000, note: "Meridian-sized repo with symbols" },
  {
    id: "large",
    label: "220K",
    nodeCount: 220_000,
    note: "stress case: slow to build, not pretty",
  },
];

// =============================================================================
// Generated repository
// =============================================================================

/**
 * A generated repository: the typed input core consumes, plus the producer-side
 * columns core has no use for but the demo's cards and labels do.
 *
 * Every array is indexed by GPU slot. The demo never mutates the graph after
 * load, so a slot stays the same node for the lifetime of the dataset and these
 * columns can be read with the slot core hands back on a card or an event.
 */
export interface RepoGraph {
  readonly input: GraphTypedInput;
  readonly nodeCount: number;
  readonly edgeCount: number;
  /** Leading edges `[0, containmentEdgeCount)` are containment; the rest are imports. */
  readonly containmentEdgeCount: number;
  readonly hierarchy: HierarchyColumns;
  /** Node kind per slot; values are {@link NODE_KIND}. */
  readonly kind: Uint8Array;
  /** Importance per slot in 0..1 — the same column supplied as `weight`. */
  readonly weight: Float32Array;
  /** Seeded positions as `[x0, y0, x1, y1, ...]` — the same column supplied as `positions`. */
  readonly positions: Float32Array;
  /** Full path per slot, e.g. `atlas/src/render/frame.ts#drawFrame`. */
  readonly path: readonly string[];
  /** Last path segment per slot — what a label or a card title shows. */
  readonly name: readonly string[];
  /** Direct containment children per slot. */
  readonly childCount: Uint32Array;
  /** How many slots each kind accounts for, indexed by the numeric kind. */
  readonly kindCounts: readonly number[];
}

/** Options for {@link generateRepo}. */
export interface RepoOptions {
  /** Total node count, including the repository root. Minimum 1. */
  readonly nodeCount: number;
  /** PRNG seed; the same seed always yields the same repository. */
  readonly seed?: number;
  /** Cross-cutting import edges as a fraction of node count. */
  readonly importRatio?: number;
}

// =============================================================================
// Vocabulary
// =============================================================================

const DIR_WORDS = [
  "src",
  "core",
  "render",
  "layout",
  "graph",
  "simulation",
  "overlay",
  "runtime",
  "codec",
  "transport",
  "schema",
  "adapters",
  "internal",
  "shared",
  "widgets",
  "hooks",
  "utils",
  "workers",
  "shaders",
  "bindings",
  "telemetry",
  "storage",
  "indexer",
  "cli",
];

const FILE_WORDS = [
  "index",
  "mod",
  "engine",
  "buffer",
  "pipeline",
  "context",
  "session",
  "registry",
  "resolver",
  "walker",
  "parser",
  "emitter",
  "scheduler",
  "cache",
  "queue",
  "config",
  "errors",
  "types",
  "state",
  "view",
];

const FILE_EXTENSIONS = ["ts", "ts", "ts", "tsx", "rs", "wgsl"];

const SYMBOL_VERBS = [
  "build",
  "resolve",
  "encode",
  "decode",
  "flush",
  "commit",
  "walk",
  "select",
  "measure",
  "apply",
  "sync",
  "drain",
];

const SYMBOL_NOUNS = [
  "Node",
  "Edge",
  "Frame",
  "Buffer",
  "Cursor",
  "Batch",
  "Range",
  "Bounds",
  "Slot",
  "Column",
  "Chunk",
  "Handle",
];

// =============================================================================
// Shape
// =============================================================================

/** Directories stop nesting past this depth; below it everything is files. */
const MAX_DIR_DEPTH = 5;

/**
 * Slots a directory must be able to spend before nesting is worth it.
 *
 * Without a floor the budget split produces directories holding one small file,
 * which is the single-child chain the LOD policy then has to fold away.
 */
const MIN_DIR_BUDGET = 24;

/** Symbols per file, before the remaining budget clamps it. */
const MIN_SYMBOLS_PER_FILE = 2;
const MAX_SYMBOLS_PER_FILE = 14;

/** Children a directory aims for; the budget decides how many it gets. */
const MIN_FANOUT = 3;
const MAX_FANOUT = 8;

/** Fraction of import edges that land near their source in DFS order. */
const IMPORT_LOCALITY = 0.65;

/** How far a "local" import reaches, in leaf-array positions. */
const IMPORT_LOCALITY_SPAN = 96;

// =============================================================================
// PRNG (mulberry32, as in tests/fixtures/prng.ts)
// =============================================================================

type Rng = () => number;

function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic integer in `[min, max]`, inclusive. */
function randInt(rng: Rng, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min + 1));
}

function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)];
}

/** Append a numeric suffix until the name is unique among its siblings. */
function unique(taken: Set<string>, base: string): string {
  if (!taken.has(base)) {
    taken.add(base);
    return base;
  }
  for (let n = 2;; n++) {
    const candidate = `${base}-${n}`;
    if (!taken.has(candidate)) {
      taken.add(candidate);
      return candidate;
    }
  }
}

// =============================================================================
// Generation
// =============================================================================

/**
 * Build a deterministic repository of exactly `nodeCount` nodes.
 *
 * The tree is grown depth-first under a slot budget: each directory splits its
 * remaining budget between its children, and every child consumes at least one
 * slot and never more than its share. The budget therefore reaches exactly zero
 * — the requested node count is hit on the nose, not approached.
 */
export function generateRepo(options: RepoOptions): RepoGraph {
  const nodeCount = Math.max(1, Math.trunc(options.nodeCount));
  const rng = mulberry32(options.seed ?? 0x5EED);
  const importRatio = options.importRatio ?? 0.15;

  const parent = new Uint32Array(nodeCount);
  const depth = new Uint16Array(nodeCount);
  const kind = new Uint8Array(nodeCount);
  const childCount = new Uint32Array(nodeCount);
  const childIndex = new Uint32Array(nodeCount);
  const name: string[] = new Array(nodeCount);
  const path: string[] = new Array(nodeCount);

  let emitted = 0;

  function emit(parentSlot: number, nodeKind: NodeKind, label: string): number {
    const slot = emitted++;
    kind[slot] = nodeKind;
    name[slot] = label;
    if (parentSlot === HIERARCHY_ROOT) {
      parent[slot] = HIERARCHY_ROOT;
      depth[slot] = 0;
      path[slot] = label;
    } else {
      parent[slot] = parentSlot;
      depth[slot] = depth[parentSlot] + 1;
      childIndex[slot] = childCount[parentSlot];
      childCount[parentSlot]++;
      // A symbol lives inside a file rather than beside it, and `#` is how
      // every code indexer writes that.
      path[slot] = nodeKind === NODE_KIND.symbol
        ? `${path[parentSlot]}#${label}`
        : `${path[parentSlot]}/${label}`;
    }
    return slot;
  }

  /** Fill `budget` slots under `dirSlot` with directories, files and symbols. */
  function expand(dirSlot: number, dirDepth: number, budget: number): void {
    let remaining = budget;
    const fanout = randInt(rng, MIN_FANOUT, MAX_FANOUT);
    const share = Math.max(1, Math.ceil(budget / fanout));
    const taken = new Set<string>();

    while (remaining > 0) {
      const spend = Math.min(remaining, share);
      const before = emitted;

      if (dirDepth < MAX_DIR_DEPTH && spend >= MIN_DIR_BUDGET) {
        const slot = emit(dirSlot, NODE_KIND.dir, unique(taken, pick(rng, DIR_WORDS)));
        expand(slot, dirDepth + 1, spend - 1);
      } else {
        const base = `${pick(rng, FILE_WORDS)}.${pick(rng, FILE_EXTENSIONS)}`;
        const slot = emit(dirSlot, NODE_KIND.file, unique(taken, base));
        const symbols = Math.min(
          spend - 1,
          randInt(rng, MIN_SYMBOLS_PER_FILE, MAX_SYMBOLS_PER_FILE),
        );
        const symbolNames = new Set<string>();
        for (let s = 0; s < symbols; s++) {
          const symbol = `${pick(rng, SYMBOL_VERBS)}${pick(rng, SYMBOL_NOUNS)}`;
          emit(slot, NODE_KIND.symbol, unique(symbolNames, symbol));
        }
      }

      remaining -= emitted - before;
    }
  }

  const root = emit(HIERARCHY_ROOT, NODE_KIND.repo, "atlas");
  expand(root, 0, nodeCount - 1);

  const hierarchy = buildHierarchy(parent, depth, childCount, nodeCount);
  const positions = seedPositions(hierarchy, parent, childIndex, childCount, nodeCount);
  const { edgePairs, edgeKinds, containmentEdgeCount } = buildEdges(
    rng,
    parent,
    kind,
    nodeCount,
    importRatio,
  );

  const kindCounts = new Array<number>(KIND_NAMES.length).fill(0);
  for (let i = 0; i < nodeCount; i++) kindCounts[kind[i]]++;

  const weight = weightsFromSubtree(hierarchy.subtreeSize, nodeCount);

  return {
    input: {
      nodeCount,
      edgeCount: edgeKinds.length,
      positions,
      edgePairs,
      edgeKinds,
      containmentKind: CONTAINMENT_EDGE_KIND,
      hierarchy,
      // Paths double as the external identifiers: in a code graph that is what
      // a producer's ids actually are, and it is what a card shows.
      nodeIds: path,
      tag: kindToTag(kind, nodeCount),
      weight,
      nodeRadii: radiiFromKind(kind, nodeCount),
      nodeColors: colorsFromKind(kind, nodeCount),
      edgeColors: edgeColorsFromKind(edgeKinds),
    },
    nodeCount,
    edgeCount: edgeKinds.length,
    containmentEdgeCount,
    hierarchy,
    kind,
    weight,
    positions,
    path,
    name,
    childCount,
    kindCounts,
  };
}

/**
 * Bottom-up hierarchy columns.
 *
 * Slots are in DFS pre-order, so every child has a higher slot than its parent
 * and one reverse scan visits each node after all of its children. `wellRadius`
 * uses the same formula as `compute_bubble_hierarchy` in WASM: a leaf gets the
 * base radius; a parent gets the radius of the circle enclosing its children's
 * combined area at the packing efficiency, floored at the base radius, plus
 * padding.
 */
function buildHierarchy(
  parent: Uint32Array,
  depth: Uint16Array,
  childCount: Uint32Array,
  nodeCount: number,
): HierarchyColumns {
  const wellRadius = new Float32Array(nodeCount);
  const subtreeSize = new Uint32Array(nodeCount).fill(1);
  const childArea = new Float64Array(nodeCount);

  for (let slot = nodeCount - 1; slot >= 0; slot--) {
    const radius = childCount[slot] === 0 ? BUBBLE_BASE_RADIUS : Math.max(
      Math.sqrt(childArea[slot] / (Math.PI * BUBBLE_PACKING_EFFICIENCY)),
      BUBBLE_BASE_RADIUS,
    ) + BUBBLE_PADDING;
    wellRadius[slot] = radius;

    const p = parent[slot];
    if (p !== HIERARCHY_ROOT) {
      childArea[p] += Math.PI * radius * radius;
      subtreeSize[p] += subtreeSize[slot];
    }
  }

  return { parent, wellRadius, depth, subtreeSize };
}

/** Golden angle, so successive siblings do not line up. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Seed positions by nesting rather than by scattering.
 *
 * Children are placed on a sunflower spiral inside their parent's bubble, so
 * the simulation starts from a configuration the nested-bubble forces already
 * roughly agree with. Slots are in DFS pre-order, so a forward scan sees every
 * parent placed before its children.
 */
function seedPositions(
  hierarchy: HierarchyColumns,
  parent: Uint32Array,
  childIndex: Uint32Array,
  childCount: Uint32Array,
  nodeCount: number,
): Float32Array {
  const positions = new Float32Array(nodeCount * 2);
  for (let slot = 1; slot < nodeCount; slot++) {
    const p = parent[slot];
    const siblings = Math.max(1, childCount[p]);
    const rank = childIndex[slot];
    const radius = hierarchy.wellRadius[p] * Math.sqrt((rank + 0.5) / siblings);
    const angle = GOLDEN_ANGLE * rank;
    positions[slot * 2] = positions[p * 2] + radius * Math.cos(angle);
    positions[slot * 2 + 1] = positions[p * 2 + 1] + radius * Math.sin(angle);
  }
  return positions;
}

/**
 * Containment edges followed by cross-cutting imports.
 *
 * Imports run between files and symbols only — directories do not import — and
 * most of them land near their source in slot order. Because slots are DFS
 * pre-order, "near in slot order" means "in a nearby subtree", which is what
 * import locality looks like in a real repository.
 */
function buildEdges(
  rng: Rng,
  parent: Uint32Array,
  kind: Uint8Array,
  nodeCount: number,
  importRatio: number,
): { edgePairs: Uint32Array; edgeKinds: Uint16Array; containmentEdgeCount: number } {
  const leaves: number[] = [];
  for (let slot = 0; slot < nodeCount; slot++) {
    if (kind[slot] === NODE_KIND.file || kind[slot] === NODE_KIND.symbol) leaves.push(slot);
  }

  const containmentEdgeCount = nodeCount - 1;
  const importCount = leaves.length < 2 ? 0 : Math.round(importRatio * nodeCount);
  const edgeCount = containmentEdgeCount + importCount;

  const edgePairs = new Uint32Array(edgeCount * 2);
  const edgeKinds = new Uint16Array(edgeCount);

  for (let slot = 1; slot < nodeCount; slot++) {
    edgePairs[(slot - 1) * 2] = parent[slot];
    edgePairs[(slot - 1) * 2 + 1] = slot;
  }

  for (let i = 0; i < importCount; i++) {
    const sourceRank = randInt(rng, 0, leaves.length - 1);
    let targetRank: number;
    if (rng() < IMPORT_LOCALITY) {
      const offset = randInt(rng, 1, IMPORT_LOCALITY_SPAN) * (rng() < 0.5 ? -1 : 1);
      // Wrapping rather than reflecting: reflection can still land outside the
      // array when the repository has fewer leaves than the locality span.
      targetRank = (((sourceRank + offset) % leaves.length) + leaves.length) % leaves.length;
    } else {
      targetRank = randInt(rng, 0, leaves.length - 1);
    }
    if (targetRank === sourceRank) {
      targetRank = (sourceRank + 1) % leaves.length;
    }

    const edge = containmentEdgeCount + i;
    edgePairs[edge * 2] = leaves[sourceRank];
    edgePairs[edge * 2 + 1] = leaves[targetRank];
    edgeKinds[edge] = IMPORT_EDGE_KIND;
  }

  return { edgePairs, edgeKinds, containmentEdgeCount };
}

function kindToTag(kind: Uint8Array, nodeCount: number): Uint16Array {
  const tag = new Uint16Array(nodeCount);
  for (let slot = 0; slot < nodeCount; slot++) tag[slot] = kind[slot];
  return tag;
}

/**
 * Importance in 0..1 from subtree size, on a log scale.
 *
 * Core uses `weight` to break ties between nodes of equal screen size even with
 * no policy registered, and the demo ranks label candidates by it. Subtree size
 * spans four orders of magnitude at the large scale, so a linear normalisation
 * would leave every file and symbol at 0.
 */
function weightsFromSubtree(subtreeSize: Uint32Array, nodeCount: number): Float32Array {
  const weight = new Float32Array(nodeCount);
  const scale = 1 / Math.log2(1 + nodeCount);
  for (let slot = 0; slot < nodeCount; slot++) {
    weight[slot] = Math.log2(1 + subtreeSize[slot]) * scale;
  }
  return weight;
}

function radiiFromKind(kind: Uint8Array, nodeCount: number): Float32Array {
  const radii = new Float32Array(nodeCount);
  for (let slot = 0; slot < nodeCount; slot++) radii[slot] = KIND_RADII[kind[slot]];
  return radii;
}

function colorsFromKind(kind: Uint8Array, nodeCount: number): Float32Array {
  const colors = new Float32Array(nodeCount * 3);
  for (let slot = 0; slot < nodeCount; slot++) {
    const rgb = KIND_COLORS[kind[slot]];
    colors[slot * 3] = rgb[0];
    colors[slot * 3 + 1] = rgb[1];
    colors[slot * 3 + 2] = rgb[2];
  }
  return colors;
}

/** Containment edges recede; imports are the thing worth seeing. */
const CONTAINMENT_EDGE_COLOR: readonly [number, number, number] = [0.20, 0.22, 0.30];
const IMPORT_EDGE_COLOR: readonly [number, number, number] = [0.72, 0.35, 0.62];

function edgeColorsFromKind(edgeKinds: Uint16Array): Float32Array {
  const colors = new Float32Array(edgeKinds.length * 3);
  for (let edge = 0; edge < edgeKinds.length; edge++) {
    const rgb = edgeKinds[edge] === CONTAINMENT_EDGE_KIND
      ? CONTAINMENT_EDGE_COLOR
      : IMPORT_EDGE_COLOR;
    colors[edge * 3] = rgb[0];
    colors[edge * 3 + 1] = rgb[1];
    colors[edge * 3 + 2] = rgb[2];
  }
  return colors;
}

// =============================================================================
// Label candidates
// =============================================================================

/**
 * Every slot worth handing to the GPU label layer, most important first.
 *
 * Symbols are excluded: a symbol legible as a label is a symbol close enough to
 * card, and cards are what they are for. The rest are ranked by producer weight,
 * which is subtree size, so directories outrank the files inside them and any
 * prefix of this array is a sensible label set.
 *
 * Ranked once per dataset rather than once per density change: the density knob
 * takes a prefix, and re-sorting 90 000 slots on every frame of a slider drag is
 * not something a demo should model.
 */
export function rankLabelCandidates(repo: RepoGraph): Uint32Array {
  const { weight } = repo;
  const eligible: number[] = [];
  for (let slot = 0; slot < repo.nodeCount; slot++) {
    if (repo.kind[slot] !== NODE_KIND.symbol) eligible.push(slot);
  }
  // Slot index breaks ties, so the ranking is a function of the repository alone
  // and does not depend on the sort's stability.
  eligible.sort((a, b) => weight[b] - weight[a] || a - b);
  return Uint32Array.from(eligible);
}
