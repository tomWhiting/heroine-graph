/**
 * GraphMother — Code Graph.
 *
 * One dataset shape (a code repository), one layout (nested bubbles), one
 * detail model (semantic LOD promoting to DOM cards), five knobs. Everything
 * here is wiring: the dataset is `src/repo.ts`, what the knobs mean is
 * `src/knobs.ts`, what a card looks like is `src/cards.ts`, and the readout's
 * arithmetic is `src/hud.ts`.
 *
 * The shape of the app is the argument. A repository is a containment tree with
 * dependency edges across it, so the layout is told about the tree (bubble
 * mode over the supplied hierarchy), the detail model is told about the tree
 * (LOD folds subtrees, the policy knows a single-child directory says nothing),
 * and at high zoom leaves stop being sprites and become real DOM.
 *
 * @module
 */

import {
  createGraphMother,
  type FullForceConfig,
  getSupportInfo,
  type GraphMother,
  type LabelData,
  type NodeId,
} from "../../packages/core/mod.ts";
import { createRepoCardProvider, type RepoCardProvider } from "./src/cards.ts";
import { HudModel } from "./src/hud.ts";
import {
  DEFAULT_KNOBS,
  type KnobState,
  labelCandidateLimit,
  labelConfigFromKnobs,
  lodConfigFromKnobs,
} from "./src/knobs.ts";
import { codeGraphLodPolicy } from "./src/lod_policy.ts";
import { mountHud, mountPanel } from "./src/panel.ts";
import {
  BUBBLE_BASE_RADIUS,
  BUBBLE_PADDING,
  generateRepo,
  rankLabelCandidates,
  REPO_SCALES,
  type RepoGraph,
  type RepoScaleId,
} from "./src/repo.ts";

// =============================================================================
// Fixed configuration
// =============================================================================

/**
 * Nested-bubble layout.
 *
 * The base radius and padding must be the two the producer sized its
 * `wellRadius` column with, or the supplied hierarchy and the forces reading it
 * describe differently-scaled trees. `relativityDepthDecay` is what keeps a
 * symbol from being dragged past its file by global gravity: external forces
 * fall off as `decay^depth`, so a leaf feels little but its parent's spring.
 */
const BUBBLE_FORCES: Partial<FullForceConfig> = {
  relativityBubbleMode: true,
  relativityBubbleBaseRadius: BUBBLE_BASE_RADIUS,
  relativityBubblePadding: BUBBLE_PADDING,
  relativityDepthDecay: 0.7,
  relativityBubbleOrbitScale: 0.6,
};

/** Label styling; the density knob owns `maxLabels` and nothing else does. */
const LABEL_STYLE = {
  fontSize: 13,
  fontColor: "#d8dbe6",
  minZoom: 0.05,
  verticalOffset: 12,
} as const;

/** How often the readout is rewritten. Every frame would be DOM churn. */
const HUD_INTERVAL_MS = 250;

// =============================================================================
// Bootstrap
// =============================================================================

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`missing element #${id}`);
  return element;
}

function setStatus(text: string | null): void {
  const status = requireElement("status");
  status.textContent = text ?? "";
  status.classList.toggle("is-visible", text !== null);
}

/** Yield to the browser so a status message paints before a long build. */
function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function main(): Promise<void> {
  const support = await getSupportInfo();
  if (!support.supported) {
    setStatus(support.reason ?? "WebGPU is not available in this browser.");
    return;
  }

  const stage = requireElement("stage");
  const canvas = requireElement("canvas") as HTMLCanvasElement;
  resizeCanvas(canvas, stage);

  const graph = await createGraphMother({ canvas });

  graph.setForceAlgorithm("relativity-atlas");
  graph.setForceConfig(BUBBLE_FORCES);
  graph.setLodPolicy(codeGraphLodPolicy);
  // The overlay's host must be a positioned element the canvas fills; #stage is
  // exactly that, and passing it explicitly keeps the cards inside the app's
  // own stacking context rather than wherever the canvas happens to sit.
  graph.setDomOverlay({ enabled: true, host: stage, className: "card-container" });

  let labelsAvailable = true;
  try {
    await graph.enableLabels(LABEL_STYLE);
  } catch (error) {
    // The MSDF atlas is served from the repository's `dist/`; without a build
    // it is simply absent, and that must not take the demo down with it.
    labelsAvailable = false;
    console.warn("[code-graph] labels unavailable:", error);
  }

  await run(graph, stage, canvas, labelsAvailable);
}

/** Size the drawing buffer to the stage in device pixels. */
function resizeCanvas(canvas: HTMLCanvasElement, stage: HTMLElement): void {
  const rect = stage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
}

// =============================================================================
// Application
// =============================================================================

async function run(
  graph: GraphMother,
  stage: HTMLElement,
  canvas: HTMLCanvasElement,
  labelsAvailable: boolean,
): Promise<void> {
  let knobs: KnobState = DEFAULT_KNOBS;
  let repo: RepoGraph | null = null;
  let cards: RepoCardProvider | null = null;
  /** Label candidates for `repo`, ranked once at load; the knob takes a prefix. */
  let ranked: Uint32Array = new Uint32Array(0);

  const hud = new HudModel();
  const hudView = mountHud(requireElement("hud"));

  const panel = mountPanel(requireElement("panel"), knobs, {
    onChange(next) {
      const previous = knobs;
      knobs = next;
      applyKnobs(previous, next);
    },
    onReheat() {
      // The one legitimate reheat: an explicit user action. Nothing else in
      // this app — not a camera move, not an LOD transition, not a card mount
      // — raises simulation alpha.
      graph.restartSimulation();
    },
  });

  function applyKnobs(previous: KnobState, next: KnobState): void {
    graph.setLodConfig(lodConfigFromKnobs(next));
    if (labelsAvailable) {
      graph.setLabelsConfig(labelConfigFromKnobs(next));
      if (next.labelDensity !== previous.labelDensity) publishLabels(next);
    }
    if (next.running !== previous.running) {
      if (next.running) {
        graph.startSimulation();
      } else {
        graph.pauseSimulation();
        hud.simulationStopped();
      }
    }
    if (next.scale !== previous.scale) void loadScale(next.scale);
  }

  /** Hand the label layer the top of the ranked candidate set. */
  function publishLabels(state: KnobState): void {
    if (repo === null || !labelsAvailable) return;
    const current = repo;
    const limit = Math.min(ranked.length, labelCandidateLimit(state));
    const labels: LabelData[] = new Array(limit);
    for (let i = 0; i < limit; i++) {
      const slot = ranked[i];
      labels[i] = {
        nodeId: slot,
        text: current.name[slot],
        x: current.positions[slot * 2],
        y: current.positions[slot * 2 + 1],
        priority: current.weight[slot],
      };
    }
    graph.setLabels(labels);
  }

  async function loadScale(id: RepoScaleId): Promise<void> {
    const scale = REPO_SCALES.find((entry) => entry.id === id);
    if (scale === undefined) return;

    panel.setBusy(true);
    setStatus(`Building a ${scale.label} repository…`);
    await nextFrame();

    const built = generateRepo({ nodeCount: scale.nodeCount });
    repo = built;
    ranked = rankLabelCandidates(built);

    // A card renderer is bound to one dataset's producer columns, so it is
    // replaced with the dataset. Swapping providers releases every mounted card
    // first, which is what keeps the HUD's count honest across a reload.
    cards = createRepoCardProvider(built);
    graph.setCardProvider(cards);

    setStatus(`Uploading ${built.nodeCount.toLocaleString()} nodes…`);
    await nextFrame();
    await graph.load(built.input);
    fitOnSettle = true;

    graph.setLodConfig(lodConfigFromKnobs(knobs));
    if (labelsAvailable) {
      graph.setLabelsConfig(labelConfigFromKnobs(knobs));
      publishLabels(knobs);
    }
    if (!knobs.running) graph.pauseSimulation();

    hud.loaded(built.nodeCount, built.edgeCount);
    setStatus(null);
    panel.setBusy(false);
  }

  // --- Events ------------------------------------------------------------

  graph.on("lod:change", (event) => hud.lodChanged(event));
  graph.on("node:collapse", () => hud.nodeFolded());
  graph.on("node:expand", () => hud.nodeUnfolded());
  graph.on("simulation:tick", () => hud.tick(performance.now()));

  // `load` fits the camera to the seeded positions, but the settled bubble
  // layout is much larger than the seed: one containment spring holds back a
  // whole subtree's aggregate repulsion, so top-level bubbles equilibrate far
  // apart and drift out of the load-time frame. Refit once, when the layout
  // first settles after a load; after that the camera belongs to the user.
  let fitOnSettle = false;
  graph.on("simulation:end", () => {
    hud.simulationStopped();
    if (fitOnSettle) {
      fitOnSettle = false;
      graph.fitToView();
    }
  });

  // A click declares focus: the node is carded whatever its screen size says.
  graph.on("node:click", (event) => graph.setLodFocus([event.nodeId]));
  graph.on("background:click", () => graph.setLodFocus([]));
  graph.on("node:dblclick", (event) => toggleFold(event.nodeId));

  /**
   * Fold or unfold the subtree under `node`.
   *
   * Whether it is currently folded is read off its first child: if the cut
   * hides that child, the node is standing in for the subtree. The first child
   * is `node + 1` because the producer emits slots in DFS pre-order, which is
   * also why a subtree is a contiguous slot range.
   */
  function toggleFold(node: NodeId): void {
    if (repo === null || repo.childCount[node] === 0) return;
    const firstChild = node + 1;
    if (graph.getVisibleAncestor(firstChild) === firstChild) {
      graph.collapseNode(node);
    } else {
      graph.expandNode(node);
    }
  }

  globalThis.addEventListener("resize", () => {
    resizeCanvas(canvas, stage);
    graph.resize(canvas.width, canvas.height);
  });

  // --- Readout -----------------------------------------------------------

  let lastHudAt = 0;
  function frame(now: number): void {
    hud.frame(now);
    if (now - lastHudAt >= HUD_INTERVAL_MS) {
      lastHudAt = now;
      hud.cardsMounted(cards?.mountedCount() ?? 0);
      hudView.render(hud.snapshot());
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  await loadScale(knobs.scale);
}

main().catch((error: unknown) => {
  console.error(error);
  setStatus(`Failed to start: ${error instanceof Error ? error.message : String(error)}`);
});
