/**
 * HeroineGraph Mission Control
 *
 * Compact bottom-drawer UI with tabbed controls for all visualization features.
 *
 * @module
 */

import {
  createHeroineGraph,
  getSupportInfo,
  type EdgeInput,
  type GraphInput,
  type HeroineGraph,
  type NodeInput,
} from "../../packages/core/mod.ts";

// ============================================================================
// Types & Constants
// ============================================================================

interface AppState {
  graph: HeroineGraph | null;
  graphData: GraphInput | null;
  nodeCount: number;
  edgeCount: number;
  drawerOpen: boolean;
  lastFrameTime: number;
  frameCount: number;
  fps: number;
  codebaseData: CodebaseData | null;
  codebaseMetrics: Map<number, NodeMetrics> | null;
  currentEdgeColors: Float32Array | null;
}

interface NodeMetrics {
  errors: number;
  warnings: number;
  complexity: number;
  lines: number;
}

interface CodebaseNode {
  id: number;
  label: string;
  type: string;
  metrics: NodeMetrics;
}

interface CodebaseEdge {
  source: number;
  target: number;
  type: string;
}

interface CodebaseData {
  name: string;
  description: string;
  nodes: CodebaseNode[];
  edges: CodebaseEdge[];
  typeStyles: Record<string, { color: number[]; label: string }>;
  edgeTypeStyles: Record<string, { color: number[]; label: string }>;
}

interface FaderHandle {
  setValue(v: number): void;
  getValue(): number;
  el: HTMLElement;
}

// Type colors for JSON-loaded graphs
const TYPE_PALETTE: Record<string, string> = {
  repository: "#ff6b6b",
  directory: "#feca57",
  file: "#a0a0a0",
  function: "#4facfe",
  class: "#a18cd1",
  method: "#54a0ff",
  variable: "#1dd1a1",
  interface: "#48dbfb",
  type: "#ff9ff3",
  import: "#ff6b6b33",
  export: "#54a0ff33",
  contains: "#feca5733",
  // Generic fallbacks
  root: "#ff6b6b",
  "level-1": "#feca57",
  "level-2": "#48dbfb",
  "level-3": "#1dd1a1",
  "level-4": "#5f27cd",
  "level-5": "#ff9ff3",
  "level-6": "#54a0ff",
};

const THEMES: Record<string, { background: string; labelColor: string }> = {
  dark: { background: "#0a0a0f", labelColor: "#ffffff" },
  light: { background: "#f5f5f5", labelColor: "#1a1a1a" },
  midnight: { background: "#0d1117", labelColor: "#c9d1d9" },
  contrast: { background: "#000000", labelColor: "#ffffff" },
};

// ============================================================================
// DOM Helpers
// ============================================================================

const $ = (id: string) => document.getElementById(id)!;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function formatNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function hexToRgba(hex: string): [number, number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b, 1.0];
}



/** Compact vertical fader component */
function fader(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  initial: number,
  step: number,
  onChange: (v: number) => void,
  format?: (v: number) => string,
): FaderHandle {
  const fmt = format ?? ((v) => step >= 1 ? v.toFixed(0) : v.toFixed(2));
  let current = initial;

  const container = el("div", "fader");
  const valueEl = el("div", "fader-value", fmt(initial));
  const track = el("div", "fader-track");
  const thumb = el("div", "fader-thumb");
  const labelEl = el("div", "fader-label", label);

  track.appendChild(thumb);
  container.append(valueEl, track, labelEl);
  parent.appendChild(container);

  function setPosition(v: number, notify = true) {
    current = Math.round(Math.max(min, Math.min(max, v)) / step) * step;
    // Fix floating point
    current = parseFloat(current.toFixed(10));
    const pct = max === min ? 0 : ((current - min) / (max - min)) * 100;
    thumb.style.bottom = `${pct}%`;
    valueEl.textContent = fmt(current);
    if (notify) onChange(current);
  }

  function pctToValue(pct: number): number {
    return min + Math.max(0, Math.min(1, pct)) * (max - min);
  }

  // Pointer drag handling
  let dragging = false;

  function onPointerDown(e: PointerEvent) {
    e.preventDefault();
    dragging = true;
    thumb.classList.add("dragging");
    thumb.setPointerCapture(e.pointerId);
    const rect = track.getBoundingClientRect();
    const pct = 1 - (e.clientY - rect.top) / rect.height;
    setPosition(pctToValue(pct));
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragging) return;
    const rect = track.getBoundingClientRect();
    const pct = 1 - (e.clientY - rect.top) / rect.height;
    setPosition(pctToValue(pct));
  }

  function onPointerUp() {
    dragging = false;
    thumb.classList.remove("dragging");
  }

  track.addEventListener("pointerdown", onPointerDown);
  track.addEventListener("pointermove", onPointerMove);
  track.addEventListener("pointerup", onPointerUp);
  track.addEventListener("pointercancel", onPointerUp);

  // Also handle click on track (not just thumb)
  track.addEventListener("click", (e) => {
    if (dragging) return;
    const rect = track.getBoundingClientRect();
    const pct = 1 - (e.clientY - rect.top) / rect.height;
    setPosition(pctToValue(pct));
  });

  setPosition(initial, false);

  return {
    setValue(v: number) { setPosition(v, false); },
    getValue() { return current; },
    el: container,
  };
}

/** Toggle switch */
function toggle(
  parent: HTMLElement,
  checked: boolean,
  onChange: (v: boolean) => void,
): { el: HTMLElement; set(v: boolean): void; get(): boolean } {
  const label = el("label", "toggle");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  const track = el("span", "toggle-track");
  label.append(input, track);
  parent.appendChild(label);
  input.addEventListener("change", () => onChange(input.checked));
  return {
    el: label,
    set(v: boolean) { input.checked = v; },
    get() { return input.checked; },
  };
}

/** Small select dropdown */
function dropdown(
  parent: HTMLElement,
  options: Array<{ value: string; label: string }>,
  initial: string,
  onChange: (v: string) => void,
): { el: HTMLSelectElement; set(v: string): void; get(): string } {
  const sel = el("select", "sm-select");
  for (const opt of options) {
    const o = el("option");
    o.value = opt.value;
    o.textContent = opt.label;
    sel.appendChild(o);
  }
  sel.value = initial;
  parent.appendChild(sel);
  sel.addEventListener("change", () => onChange(sel.value));
  return { el: sel, set(v) { sel.value = v; }, get() { return sel.value; } };
}

/** Color picker */
function colorPicker(
  parent: HTMLElement,
  initial: string,
  onChange: (v: string) => void,
): { el: HTMLInputElement; set(v: string): void } {
  const input = document.createElement("input");
  input.type = "color";
  input.className = "sm-color";
  input.value = initial;
  parent.appendChild(input);
  input.addEventListener("input", () => onChange(input.value));
  return { el: input, set(v) { input.value = v; } };
}

/** Inline range slider for layer cards */
function miniSlider(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  initial: number,
  step: number,
  onChange: (v: number) => void,
  format?: (v: number) => string,
): FaderHandle {
  const fmt = format ?? ((v) => step >= 1 ? v.toFixed(0) : v.toFixed(2));
  const container = el("div", "mini-slider");
  const row = el("div", "ctrl-row");
  const lbl = el("span", "ctrl-label", label);
  const val = el("span", undefined, fmt(initial));
  val.style.cssText = "font-size:10px;font-variant-numeric:tabular-nums;min-width:30px;text-align:right";
  row.append(lbl, val);
  container.appendChild(row);

  const input = document.createElement("input");
  input.type = "range";
  input.className = "sm-range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(initial);
  container.appendChild(input);
  parent.appendChild(container);

  let current = initial;
  input.addEventListener("input", () => {
    current = parseFloat(input.value);
    val.textContent = fmt(current);
    onChange(current);
  });

  return {
    setValue(v) { current = v; input.value = String(v); val.textContent = fmt(v); },
    getValue() { return current; },
    el: container,
  };
}

/** Tab system */
function tabSystem(
  tabBar: HTMLElement,
  body: HTMLElement,
  tabs: Array<{ id: string; label: string; build: (panel: HTMLElement) => void }>,
): { activate(id: string): void } {
  const tabEls: HTMLElement[] = [];
  const panels: HTMLElement[] = [];

  for (const t of tabs) {
    const tabEl = el("div", "tab", t.label);
    tabBar.appendChild(tabEl);
    tabEls.push(tabEl);

    const panel = el("div", "tab-panel");
    panel.id = `tab-${t.id}`;
    body.appendChild(panel);
    panels.push(panel);
    t.build(panel);

    tabEl.addEventListener("click", (e) => {
      e.stopPropagation();
      activate(t.id);
    });
  }

  function activate(id: string) {
    const idx = tabs.findIndex((t) => t.id === id);
    if (idx < 0) return;
    for (let i = 0; i < tabs.length; i++) {
      tabEls[i].classList.toggle("active", i === idx);
      panels[i].classList.toggle("active", i === idx);
    }
  }

  activate(tabs[0]?.id ?? "");
  return { activate };
}

// ============================================================================
// Data Generation
// ============================================================================

function generateHierarchicalGraph(
  nodeCount: number,
  branchFactor: number = 4,
  crossTalk: number = 0,
): GraphInput {
  const nodes: Array<{
    id: string; x: number; y: number; radius: number;
    color: string; metadata: Record<string, unknown>;
  }> = [];
  const edges: Array<{ source: string; target: string; width: number; color: string }> = [];
  const edgeSet = new Set<string>();

  const depthColors = [
    "#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1",
    "#5f27cd", "#ff9ff3", "#54a0ff", "#00d2d3",
  ];
  const TREE_EDGE_COLOR = "#4facfe44";
  const CROSS_EDGE_COLOR = "#ff6b6b33";

  const addEdge = (src: string, tgt: string, color: string, width: number): boolean => {
    const key = src < tgt ? `${src}-${tgt}` : `${tgt}-${src}`;
    if (edgeSet.has(key) || src === tgt) return false;
    edgeSet.add(key);
    edges.push({ source: src, target: tgt, width, color });
    return true;
  };

  const nodeParents: (number | undefined)[] = [];
  let currentIdx = 0;

  // Root
  nodes.push({
    id: "node-0", x: 0, y: 0, radius: 15,
    color: depthColors[0],
    metadata: { type: "root", depth: 0, name: "Root" },
  });
  nodeParents.push(undefined);
  currentIdx++;

  let currentLevel = [0];
  let depth = 1;

  while (currentIdx < nodeCount && currentLevel.length > 0) {
    const nextLevel: number[] = [];

    for (const parentIdx of currentLevel) {
      if (currentIdx >= nodeCount) break;

      const roll = Math.random();
      let numChildren: number;
      if (roll < 0.2) numChildren = 0;
      else if (roll < 0.3) numChildren = Math.floor(branchFactor * (1.5 + Math.random() * 1.5));
      else numChildren = Math.max(1, Math.floor(1 + Math.random() * branchFactor));

      const parentNode = nodes[parentIdx];

      for (let c = 0; c < numChildren && currentIdx < nodeCount; c++) {
        const angle = (c / numChildren) * Math.PI * 2 + Math.random() * 0.3;
        const dist = 30 + depth * 20 + Math.random() * 20;
        const x = (parentNode.x ?? 0) + Math.cos(angle) * dist;
        const y = (parentNode.y ?? 0) + Math.sin(angle) * dist;

        nodes.push({
          id: `node-${currentIdx}`, x, y,
          radius: Math.max(4, 12 - depth),
          color: depthColors[depth % depthColors.length],
          metadata: { type: `level-${depth}`, depth, parent: parentIdx, name: `Node ${currentIdx}` },
        });
        nodeParents.push(parentIdx);
        addEdge(`node-${parentIdx}`, `node-${currentIdx}`, TREE_EDGE_COLOR, 1.5);
        nextLevel.push(currentIdx);
        currentIdx++;
      }
    }
    currentLevel = nextLevel;
    depth++;
    if (depth > 20) break;
  }

  // Cross-talk
  if (crossTalk > 0 && nodes.length > 2) {
    const crossCount = Math.floor(edges.length * (crossTalk / 100));
    let added = 0, attempts = 0;
    while (added < crossCount && attempts < crossCount * 10) {
      attempts++;
      const a = Math.floor(Math.random() * nodes.length);
      const b = Math.floor(Math.random() * nodes.length);
      if (a === b || nodeParents[a] === b || nodeParents[b] === a) continue;
      if (addEdge(`node-${a}`, `node-${b}`, CROSS_EDGE_COLOR, 0.8)) added++;
    }
  }

  console.log(`Generated tree: ${nodes.length} nodes, ${edges.length} edges, depth ${depth - 1}`);
  return { nodes, edges };
}

function addCrossTalkEdges(
  graphData: GraphInput,
  count: number,
): { newEdges: Array<{ source: string; target: string; width: number; color: string }>; totalAdded: number } {
  const existing = new Set<string>();
  for (const edge of graphData.edges) {
    const key = String(edge.source) < String(edge.target)
      ? `${edge.source}-${edge.target}` : `${edge.target}-${edge.source}`;
    existing.add(key);
  }

  const newEdges: Array<{ source: string; target: string; width: number; color: string }> = [];
  const nodeCount = graphData.nodes.length;
  if (nodeCount < 2) return { newEdges, totalAdded: 0 };

  let added = 0, attempts = 0;
  while (added < count && attempts < count * 20) {
    attempts++;
    const a = Math.floor(Math.random() * nodeCount);
    const b = Math.floor(Math.random() * nodeCount);
    if (a === b) continue;
    const srcId = String(graphData.nodes[a].id);
    const tgtId = String(graphData.nodes[b].id);
    const key = srcId < tgtId ? `${srcId}-${tgtId}` : `${tgtId}-${srcId}`;
    if (existing.has(key)) continue;
    existing.add(key);
    newEdges.push({ source: srcId, target: tgtId, width: 0.8, color: "#ff6b6b55" });
    added++;
  }
  return { newEdges, totalAdded: added };
}

// ============================================================================
// Main Application
// ============================================================================

async function main(): Promise<void> {
  const state: AppState = {
    graph: null, graphData: null,
    nodeCount: 0, edgeCount: 0,
    drawerOpen: false,
    lastFrameTime: performance.now(), frameCount: 0, fps: 0,
    codebaseData: null, codebaseMetrics: null, currentEdgeColors: null,
  };

  // ---- WebGPU Init ----
  const support = await getSupportInfo();
  if (!support.supported) {
    $("loading-overlay").classList.add("hidden");
    $("error-text").textContent = support.reason || "WebGPU not supported.";
    $("error-message").classList.add("visible");
    return;
  }

  const canvas = $("graph-canvas") as HTMLCanvasElement;
  const container = $("graph-container");
  const rect = container.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;

  try {
    state.graph = await createHeroineGraph({ canvas, debug: false });
  } catch (err) {
    console.error("Init failed:", err);
    $("loading-overlay").classList.add("hidden");
    $("error-text").textContent = `Init failed: ${err}`;
    $("error-message").classList.add("visible");
    return;
  }
  $("loading-overlay").classList.add("hidden");

  // Resize
  window.addEventListener("resize", () => {
    const r = container.getBoundingClientRect();
    canvas.width = r.width * devicePixelRatio;
    canvas.height = r.height * devicePixelRatio;
    state.graph?.resize(canvas.width, canvas.height);
  });

  // ---- Stats elements (populated later by header builder) ----
  let statNodes: HTMLElement;
  let statEdges: HTMLElement;
  let statFps: HTMLElement;

  function updateStats() {
    if (statNodes) statNodes.textContent = formatNum(state.nodeCount);
    if (statEdges) statEdges.textContent = formatNum(state.edgeCount);
  }

  // FPS
  function fpsLoop() {
    const now = performance.now();
    state.frameCount++;
    if (now - state.lastFrameTime >= 1000) {
      state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
      state.frameCount = 0;
      state.lastFrameTime = now;
      if (statFps) statFps.textContent = String(state.fps);
    }
    requestAnimationFrame(fpsLoop);
  }
  requestAnimationFrame(fpsLoop);

  // ---- Core data functions ----

  // Fader/control references (filled by tab builders, read by config loader)
  const F: Record<string, FaderHandle> = {};
  const T: Record<string, ReturnType<typeof toggle>> = {};
  const D: Record<string, ReturnType<typeof dropdown>> = {};
  const C: Record<string, ReturnType<typeof colorPicker>> = {};

  function setEdgeColorsWithTracking(colors: Float32Array) {
    state.graph?.setEdgeColors(colors);
    state.currentEdgeColors = new Float32Array(colors);
  }

  async function loadNodes(count: number) {
    if (!state.graph) return;
    const branchFactor = F["branch"]?.getValue() ?? 4;
    const data = generateHierarchicalGraph(count, branchFactor, 0);
    await state.graph.load(data);
    state.graphData = data;
    state.nodeCount = data.nodes.length;
    state.edgeCount = data.edges.length;
    updateStats();
    state.codebaseData = null;
    state.codebaseMetrics = null;
    // Default edge color tracking
    const trackedColors = new Float32Array(state.edgeCount * 4);
    for (let i = 0; i < state.edgeCount; i++) {
      trackedColors[i * 4 + 0] = 0.5;
      trackedColors[i * 4 + 1] = 0.5;
      trackedColors[i * 4 + 2] = 0.5;
      trackedColors[i * 4 + 3] = 0.4;
    }
    state.currentEdgeColors = trackedColors;
    if (T["labels"]?.get()) updateLabels();
  }

  async function loadCodebase() {
    if (!state.graph) return;
    const response = await fetch("./data/codebase.json");
    const codebase: CodebaseData = await response.json();
    state.codebaseData = codebase;
    state.codebaseMetrics = new Map();

    const rgbaToHex = (rgba: number[]) => {
      const r = Math.round((rgba[0] || 0) * 255);
      const g = Math.round((rgba[1] || 0) * 255);
      const b = Math.round((rgba[2] || 0) * 255);
      return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
    };

    const graphData: GraphInput = {
      nodes: codebase.nodes.map((node) => {
        state.codebaseMetrics!.set(node.id, node.metrics);
        return { id: node.id, label: node.label, type: node.type };
      }),
      edges: codebase.edges.map((edge) => ({
        source: edge.source, target: edge.target, type: edge.type,
      })),
    };

    await state.graph.load(graphData);

    // Type-based styling
    const nodeStyles: Record<string, { color: string }> = {};
    for (const [typeName, style] of Object.entries(codebase.typeStyles)) {
      nodeStyles[typeName] = { color: rgbaToHex(style.color) };
    }
    state.graph.setNodeTypeStyles(nodeStyles);

    const edgeStyles: Record<string, { color: string; opacity?: number }> = {};
    for (const [typeName, style] of Object.entries(codebase.edgeTypeStyles)) {
      edgeStyles[typeName] = { color: rgbaToHex(style.color), opacity: style.color[3] || 0.5 };
    }
    state.graph.setEdgeTypeStyles(edgeStyles);

    // Track colors
    const edgeCount = graphData.edges.length;
    const trackedColors = new Float32Array(edgeCount * 4);
    for (let i = 0; i < edgeCount; i++) {
      const edge = graphData.edges[i];
      const typeStyle = (edge.type as string) ? codebase.edgeTypeStyles[edge.type as string] : null;
      if (typeStyle) {
        trackedColors[i * 4] = typeStyle.color[0] || 0.5;
        trackedColors[i * 4 + 1] = typeStyle.color[1] || 0.5;
        trackedColors[i * 4 + 2] = typeStyle.color[2] || 0.5;
        trackedColors[i * 4 + 3] = typeStyle.color[3] || 0.5;
      } else {
        trackedColors[i * 4] = trackedColors[i * 4 + 1] = trackedColors[i * 4 + 2] = 0.5;
        trackedColors[i * 4 + 3] = 0.5;
      }
    }
    state.currentEdgeColors = trackedColors;
    state.graphData = graphData;
    state.nodeCount = graphData.nodes.length;
    state.edgeCount = edgeCount;
    updateStats();
    if (T["labels"]?.get()) updateLabels();
    console.log(`Loaded codebase: ${codebase.name} (${state.nodeCount} nodes, ${state.edgeCount} edges)`);
  }

  async function loadJSONFile(file: File) {
    if (!state.graph) return;
    const text = await file.text();
    const data = JSON.parse(text);
    await loadGraphJSON(data);
  }

  async function loadGraphJSON(data: Record<string, unknown>) {
    if (!state.graph) return;

    const rawNodes = data.nodes as Record<string, unknown>[] | undefined;
    if (!rawNodes || !Array.isArray(rawNodes)) {
      console.error("Invalid JSON: missing nodes array");
      return;
    }

    // Drop duplicate node IDs — keep first occurrence, discard the rest.
    // Edges referencing dropped nodes are also dropped.
    const seenIds = new Set<string>();
    const keptNodes: { id: string; label: string; type: string }[] = [];
    let droppedCount = 0;

    for (const n of rawNodes) {
      const id = String(n.id ?? n.name ?? keptNodes.length);
      if (seenIds.has(id)) {
        droppedCount++;
        continue;
      }
      seenIds.add(id);
      keptNodes.push({
        id,
        label: String(n.label ?? n.name ?? n.id ?? keptNodes.length),
        type: String(n.type ?? n.node_type ?? "default"),
      });
    }

    const rawEdges = (data.edges ?? data.links ?? []) as Record<string, unknown>[];
    const keptEdges: EdgeInput[] = [];
    const seenEdges = new Set<string>();
    let droppedEdges = 0;
    for (const e of rawEdges) {
      const source = String(e.source ?? e.from);
      const target = String(e.target ?? e.to);
      if (!seenIds.has(source) || !seenIds.has(target)) {
        droppedEdges++;
        continue;
      }
      const edgeKey = `${source}\0${target}`;
      if (seenEdges.has(edgeKey)) {
        droppedEdges++;
        continue;
      }
      seenEdges.add(edgeKey);
      keptEdges.push({ source, target, type: String(e.type ?? e.kind ?? "default") } as EdgeInput);
    }

    const graphData: GraphInput = { nodes: keptNodes, edges: keptEdges };

    if (droppedCount > 0 || droppedEdges > 0) {
      console.log(`Loaded: ${keptNodes.length} nodes, ${keptEdges.length} edges (dropped ${droppedCount} duplicate nodes, ${droppedEdges} duplicate/orphaned edges)`);
    }

    await state.graph.load(graphData);

    // Auto-assign type colors for nodes
    const nodeTypes = new Set(graphData.nodes.map((n) => n.type as string));
    const nodeStyles: Record<string, { color: string }> = {};
    const palette = Object.values(TYPE_PALETTE);
    let pi = 0;
    for (const type of nodeTypes) {
      if (type && type !== "default") {
        nodeStyles[type] = { color: TYPE_PALETTE[type] ?? palette[pi % palette.length] };
        pi++;
      }
    }
    if (Object.keys(nodeStyles).length > 0) {
      state.graph.setNodeTypeStyles(nodeStyles);
    }

    // Auto-assign type colors for edges
    const edgeTypes = new Set(graphData.edges.map((e) => (e as Record<string, unknown>).type as string));
    const edgeStyles: Record<string, { color: string; opacity?: number }> = {};
    const edgePalette = ["#feca5733", "#54a0ff33", "#ff6b6b33", "#1dd1a133", "#ff9ff333"];
    let ei = 0;
    for (const type of edgeTypes) {
      if (type && type !== "default") {
        edgeStyles[type] = {
          color: TYPE_PALETTE[type] ?? edgePalette[ei % edgePalette.length],
          opacity: type === "contains" ? 0.15 : 0.5,
        };
        ei++;
      }
    }
    if (Object.keys(edgeStyles).length > 0) {
      state.graph.setEdgeTypeStyles(edgeStyles);
    }

    state.graphData = graphData;
    state.nodeCount = graphData.nodes.length;
    state.edgeCount = graphData.edges.length;
    state.codebaseData = null;
    state.codebaseMetrics = null;
    state.currentEdgeColors = null;
    updateStats();
    if (T["labels"]?.get()) updateLabels();
    console.log(`Loaded JSON: ${state.nodeCount} nodes, ${state.edgeCount} edges`);
  }

  async function loadIndexData(filename = "index-data.json") {
    try {
      const resp = await fetch(`./${filename}?t=${Date.now()}`, { cache: "no-store" });
      if (!resp.ok) { console.warn(`${filename} not found, falling back to generated tree`); return false; }
      const data = await resp.json();
      await loadGraphJSON(data);
      return true;
    } catch (err) {
      console.warn(`Failed to load ${filename}:`, err);
      return false;
    }
  }

  function updateLabels() {
    if (!state.graph || !state.graphData) return;
    const labels = state.graphData.nodes.map((node, index) => ({
      nodeId: index,
      text: String((node as Record<string, unknown>).label ?? node.id),
      x: (node as Record<string, unknown>).x as number ?? 0,
      y: (node as Record<string, unknown>).y as number ?? 0,
      priority: 1 - index / state.graphData!.nodes.length,
    }));
    state.graph.setLabels(labels);
  }

  // ---- Build Drawer Header ----
  const drawerEl = $("drawer");
  const headerEl = $("drawer-header");

  // Stats
  const sn = el("span", "stat");
  sn.innerHTML = '<b id="sn">0</b> nodes';
  const se = el("span", "stat");
  se.innerHTML = '<b id="se">0</b> edges';
  const sf = el("span", "stat");
  sf.innerHTML = '<b id="sf">0</b> fps';

  // Buttons
  const fitBtn = el("button", "header-btn", "Fit");
  fitBtn.addEventListener("click", (e) => { e.stopPropagation(); state.graph?.fitToView(); });

  const playBtn = el("button", "header-btn", "||");
  let simRunning = true;
  playBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (simRunning) { state.graph?.stopSimulation(); playBtn.textContent = ">"; }
    else { state.graph?.startSimulation(); playBtn.textContent = "||"; }
    simRunning = !simRunning;
  });

  // Algorithm dropdown in header
  const algoSelect = document.createElement("select");
  algoSelect.id = "algo-select";
  const algorithms = [
    { value: "n2", label: "N2" },
    { value: "force-atlas2", label: "FA2" },
    { value: "barnes-hut", label: "Barnes-Hut" },
    { value: "density", label: "Density" },
    { value: "relativity-atlas", label: "Relativity" },
    { value: "tidy-tree", label: "Tidy Tree" },
    { value: "linlog", label: "LinLog" },
    { value: "t-fdp", label: "t-FDP" },
    { value: "community", label: "Community" },
    { value: "codebase", label: "Codebase" },
  ];
  for (const a of algorithms) {
    const o = el("option");
    o.value = a.value;
    o.textContent = a.label;
    algoSelect.appendChild(o);
  }

  const chevron = el("span", undefined, "\u25B2");
  chevron.id = "drawer-chevron";

  const spacer = el("span", "header-spacer");
  headerEl.append(sn, se, sf, spacer, fitBtn, playBtn, algoSelect, chevron);

  statNodes = document.getElementById("sn")!;
  statEdges = document.getElementById("se")!;
  statFps = document.getElementById("sf")!;

  // Algorithm-specific control panels (filled by forces tab, shown/hidden on switch)
  const algoPanels: Record<string, HTMLElement> = {};

  algoSelect.addEventListener("change", () => {
    const type = algoSelect.value;
    try {
      state.graph?.setForceAlgorithm(type as "n2");
      // Show/hide algorithm-specific panels
      for (const [key, panel] of Object.entries(algoPanels)) {
        panel.style.display = key === type ? "" : "none";
      }
      // Auto-compute for structure-aware algorithms
      if (type === "tidy-tree") state.graph?.computeTreeLayout();
      else if (type === "community") state.graph?.computeCommunityLayout();
      else if (type === "codebase") {
        const categories = buildCodebaseCategories();
        state.graph?.computeCodebaseLayout(categories);
      }
    } catch (e) { console.error("Algorithm switch failed:", e); }
  });

  function buildCodebaseCategories(): Uint8Array | undefined {
    if (!state.codebaseData || !state.graph) return undefined;
    const nodeBound = state.graph.nodeCount;
    const categories = new Uint8Array(nodeBound);
    for (const node of state.codebaseData.nodes) {
      if (node.id < nodeBound) {
        switch (node.type) {
          case "repository": categories[node.id] = 0; break;
          case "directory": categories[node.id] = 1; break;
          case "file": categories[node.id] = 2; break;
          case "function": case "class": case "method": case "variable": case "interface": case "type":
            categories[node.id] = 3; break;
          default: categories[node.id] = 4; break;
        }
      }
    }
    return categories;
  }

  // Drawer open/close
  function toggleDrawer() {
    state.drawerOpen = !state.drawerOpen;
    drawerEl.classList.toggle("open", state.drawerOpen);
  }
  headerEl.addEventListener("click", toggleDrawer);

  // ========================================================================
  // Build Tabs
  // ========================================================================

  const tabBar = $("drawer-tabs");
  const drawerBody = $("drawer-body");

  tabSystem(tabBar, drawerBody, [
    { id: "data", label: "Data", build: buildDataTab },
    { id: "forces", label: "Forces", build: buildForcesTab },
    { id: "layers", label: "Layers", build: buildLayersTab },
    { id: "flow", label: "Flow", build: buildFlowTab },
    { id: "style", label: "Style", build: buildStyleTab },
  ]);

  // ========================================================================
  // DATA TAB
  // ========================================================================

  function buildDataTab(panel: HTMLElement) {
    // Generate tree section
    const genLabel = el("div", "section-label", "Generate Tree");
    panel.appendChild(genLabel);

    const genRow = el("div", "btn-row");
    for (const size of [100, 1000, 5000, 10000, 50000]) {
      const b = el("button", "btn", size >= 1000 ? `${size / 1000}K` : String(size));
      b.addEventListener("click", () => loadNodes(size));
      genRow.appendChild(b);
    }
    panel.appendChild(genRow);

    // Branching factor fader
    const bfRow = el("div", "fader-row");
    bfRow.style.height = "100px";
    F["branch"] = fader(bfRow, "Branch", 2, 8, 4, 1, () => {}, (v) => v.toFixed(0));
    panel.appendChild(bfRow);

    // Load JSON
    const jsonLabel = el("div", "section-label", "Load JSON");
    panel.appendChild(jsonLabel);

    const dropZone = el("div", "drop-zone", "Drop JSON file here or click to browse");
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json";
    fileInput.style.display = "none";

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer?.files[0];
      if (file) loadJSONFile(file).catch(console.error);
    });
    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (file) loadJSONFile(file).catch(console.error);
    });

    panel.append(dropZone, fileInput);

    // Load codebase
    const cbLabel = el("div", "section-label", "Bundled Data");
    panel.appendChild(cbLabel);

    const cbBtn = el("button", "btn", "Load codebase.json");
    cbBtn.addEventListener("click", () => loadCodebase().catch(console.error));
    panel.appendChild(cbBtn);

    const idxBtn = el("button", "btn", "Load index-data.json");
    idxBtn.addEventListener("click", () => loadIndexData().catch(console.error));
    panel.appendChild(idxBtn);

    // Mutations
    const mutLabel = el("div", "section-label", "Mutations");
    panel.appendChild(mutLabel);

    const mutRow = el("div", "btn-row");
    const addNodesBtn = el("button", "btn", "+10 Nodes");
    addNodesBtn.addEventListener("click", async () => {
      if (!state.graph || !state.graphData) return;
      const newNodes: NodeInput[] = [];
      const colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96e6a1", "#dda0dd"];
      for (let i = 0; i < 10; i++) {
        newNodes.push({
          id: `mut_${state.nodeCount + i}_${Date.now()}`,
          color: colors[i % colors.length],
          radius: 4 + Math.random() * 6,
        });
      }
      const ids = await state.graph.addNodes(newNodes);
      state.nodeCount += ids.length;
      // Add edges to random existing nodes
      if (state.graphData.nodes.length > 0) {
        const edgesToAdd: EdgeInput[] = newNodes.map((n) => ({
          source: n.id,
          target: state.graphData!.nodes[Math.floor(Math.random() * state.graphData!.nodes.length)].id,
        }));
        const results = await state.graph.addEdges(edgesToAdd);
        const addedEdges = edgesToAdd.filter((_, i) => results[i] !== undefined);
        state.edgeCount += addedEdges.length;
        state.graphData = {
          nodes: [...state.graphData.nodes, ...newNodes],
          edges: [...state.graphData.edges, ...addedEdges],
        };
        state.currentEdgeColors = null;
      } else {
        state.graphData = { ...state.graphData, nodes: [...state.graphData.nodes, ...newNodes] };
      }
      updateStats();
    });

    const addEdgesBtn = el("button", "btn", "+20 Edges");
    addEdgesBtn.addEventListener("click", async () => {
      if (!state.graph || !state.graphData || state.graphData.nodes.length < 2) return;
      const newEdges: EdgeInput[] = [];
      const nodes = state.graphData.nodes;
      for (let i = 0; i < 20; i++) {
        const s = Math.floor(Math.random() * nodes.length);
        let t = Math.floor(Math.random() * nodes.length);
        if (t === s) t = (s + 1) % nodes.length;
        newEdges.push({ source: nodes[s].id, target: nodes[t].id });
      }
      const results = await state.graph.addEdges(newEdges);
      const addedEdges = newEdges.filter((_, i) => results[i] !== undefined);
      state.edgeCount += addedEdges.length;
      state.graphData = { ...state.graphData, edges: [...state.graphData.edges, ...addedEdges] };
      state.currentEdgeColors = null;
      updateStats();
    });

    const rmNodesBtn = el("button", "btn danger", "-5 Nodes");
    rmNodesBtn.addEventListener("click", async () => {
      if (!state.graph || !state.graphData || state.graphData.nodes.length < 5) return;
      const toRemove: (string | number)[] = [];
      const copy = [...state.graphData.nodes];
      for (let i = 0; i < 5 && copy.length > 0; i++) {
        const idx = Math.floor(Math.random() * copy.length);
        toRemove.push(copy[idx].id);
        copy.splice(idx, 1);
      }
      const removed = await state.graph.removeNodes(toRemove);
      state.nodeCount -= removed;
      const removedSet = new Set(toRemove);
      state.graphData = {
        nodes: state.graphData.nodes.filter((n) => !removedSet.has(n.id)),
        edges: state.graphData.edges.filter((e) => !removedSet.has(e.source) && !removedSet.has(e.target)),
      };
      state.edgeCount = state.graphData.edges.length;
      state.currentEdgeColors = null;
      updateStats();
    });

    const crossBtn = el("button", "btn", "+Cross");
    crossBtn.addEventListener("click", async () => {
      if (!state.graph || !state.graphData || state.graphData.nodes.length === 0) return;
      const { newEdges, totalAdded } = addCrossTalkEdges(state.graphData, 50);
      if (totalAdded === 0) return;
      const results = await state.graph.addEdges(newEdges);
      const addedEdges = newEdges.filter((_, i) => results[i] !== undefined);
      state.graphData = { ...state.graphData, edges: [...state.graphData.edges, ...addedEdges] };
      state.edgeCount += addedEdges.length;
      state.currentEdgeColors = null;
      updateStats();
    });

    mutRow.append(addNodesBtn, addEdgesBtn, rmNodesBtn, crossBtn);
    panel.appendChild(mutRow);
  }

  // ========================================================================
  // FORCES TAB
  // ========================================================================

  function buildForcesTab(panel: HTMLElement) {
    // Common force faders
    const commonLabel = el("div", "section-label", "Common Forces");
    panel.appendChild(commonLabel);

    const commonRow = el("div", "fader-row");
    F["rep"] = fader(commonRow, "Rep", 1, 1000, 50, 1,
      (v) => state.graph?.setForceConfig({ repulsionStrength: -v }));
    F["spring"] = fader(commonRow, "Spring", 0.01, 1, 0.1, 0.01,
      (v) => state.graph?.setForceConfig({ springStrength: v }));
    F["length"] = fader(commonRow, "Length", 1, 500, 30, 1,
      (v) => state.graph?.setForceConfig({ springLength: v }));
    F["gravity"] = fader(commonRow, "Gravity", 0, 0.5, 0.01, 0.01,
      (v) => state.graph?.setForceConfig({ centerStrength: v }));
    F["damping"] = fader(commonRow, "Damp", 0.1, 0.9, 0.4, 0.05,
      (v) => state.graph?.setForceConfig({ velocityDecay: v }));
    F["theta"] = fader(commonRow, "Theta", 0.3, 1.5, 0.8, 0.1,
      (v) => state.graph?.setForceConfig({ theta: v }));
    F["maxvel"] = fader(commonRow, "MaxVel", 10, 200, 50, 10,
      (v) => state.graph?.setForceConfig({ maxVelocity: v }));
    panel.appendChild(commonRow);

    // Pin root toggle
    const pinRow = el("div", "ctrl-row");
    const pinLabel = el("span", "ctrl-label", "Pin Root");
    pinRow.appendChild(pinLabel);
    T["pinRoot"] = toggle(pinRow, true,
      (v) => state.graph?.setForceConfig({ pinnedNode: v ? 0 : 0xFFFFFFFF }));
    panel.appendChild(pinRow);

    // ---- Algorithm-specific panels ----

    // Relativity Atlas
    const raPanel = el("div");
    raPanel.style.display = algoSelect.value === "relativity-atlas" ? "" : "none";
    algoPanels["relativity-atlas"] = raPanel;

    const raLabel = el("div", "section-label", "Relativity Atlas");
    raPanel.appendChild(raLabel);
    const raRow = el("div", "fader-row");
    F["ra-baseMass"] = fader(raRow, "Mass", 0.1, 10, 1.0, 0.1,
      (v) => state.graph?.setForceConfig({ relativityBaseMass: v }), (v) => v.toFixed(1));
    F["ra-childFactor"] = fader(raRow, "Child", 0, 1, 0.5, 0.05,
      (v) => state.graph?.setForceConfig({ relativityChildMassFactor: v }));
    F["ra-massExp"] = fader(raRow, "MExp", 0, 2, 0.5, 0.1,
      (v) => state.graph?.setForceConfig({ relativityMassExponent: v }));
    F["ra-maxSib"] = fader(raRow, "MaxSib", 10, 500, 100, 10,
      (v) => state.graph?.setForceConfig({ relativityMaxSiblings: Math.floor(v) }), (v) => v.toFixed(0));
    F["ra-parentChild"] = fader(raRow, "P-C", 0, 1, 0.15, 0.05,
      (v) => state.graph?.setForceConfig({ relativityParentChildMultiplier: v }));
    F["ra-density"] = fader(raRow, "DenRep", 0, 2, 0.5, 0.05,
      (v) => state.graph?.setForceConfig({ relativityDensityRepulsion: v }));
    F["ra-orbit"] = fader(raRow, "Orbit", 0, 20, 1.0, 0.5,
      (v) => state.graph?.setForceConfig({ relativityOrbitStrength: v }), (v) => v.toFixed(1));
    F["ra-tang"] = fader(raRow, "Tang", 1, 20, 2.0, 0.5,
      (v) => state.graph?.setForceConfig({ relativityTangentialMultiplier: v }), (v) => v.toFixed(1));
    F["ra-orbitRad"] = fader(raRow, "OrbRad", 1, 200, 25, 1,
      (v) => state.graph?.setForceConfig({ relativityOrbitRadius: v }), (v) => v.toFixed(0));
    raPanel.appendChild(raRow);

    // Toggles: Cousin, Phantom
    const raToggles = el("div");
    raToggles.style.cssText = "display:flex;gap:16px;margin:4px 0";
    const cousRow = el("div", "ctrl-row");
    cousRow.style.fontSize = "10px";
    cousRow.appendChild(el("span", "ctrl-label", "Cousin"));
    T["ra-cousin"] = toggle(cousRow, false,
      (v) => state.graph?.setForceConfig({ relativityCousinRepulsion: v }));
    const phantRow = el("div", "ctrl-row");
    phantRow.style.fontSize = "10px";
    phantRow.appendChild(el("span", "ctrl-label", "Phantom"));
    T["ra-phantom"] = toggle(phantRow, false,
      (v) => state.graph?.setForceConfig({ relativityPhantomZone: v }));
    raToggles.append(cousRow, phantRow);
    raPanel.appendChild(raToggles);

    // Bubble mode sub-panel
    const bubbleLabel = el("div", "section-label", "Bubble Mode");
    raPanel.appendChild(bubbleLabel);
    const bubbleRow = el("div");
    bubbleRow.style.cssText = "display:flex;align-items:center;gap:8px";
    const bubbleTxt = el("span", "ctrl-label", "Enable");
    bubbleTxt.style.fontSize = "10px";
    bubbleRow.appendChild(bubbleTxt);
    T["ra-bubble"] = toggle(bubbleRow, false,
      (v) => state.graph?.setForceConfig({ relativityBubbleMode: v }));
    raPanel.appendChild(bubbleRow);
    const bubbleFaders = el("div", "fader-row");
    F["ra-bubbleRad"] = fader(bubbleFaders, "BaseR", 1, 100, 10, 1,
      (v) => state.graph?.setForceConfig({ relativityBubbleBaseRadius: v }), (v) => v.toFixed(0));
    F["ra-bubblePad"] = fader(bubbleFaders, "Pad", 0, 50, 5, 1,
      (v) => state.graph?.setForceConfig({ relativityBubblePadding: v }), (v) => v.toFixed(0));
    F["ra-depthDecay"] = fader(bubbleFaders, "Decay", 0, 1, 0.7, 0.05,
      (v) => state.graph?.setForceConfig({ relativityDepthDecay: v }));
    F["ra-bubbleOrbit"] = fader(bubbleFaders, "OrbScl", 0.1, 2, 0.6, 0.1,
      (v) => state.graph?.setForceConfig({ relativityBubbleOrbitScale: v }), (v) => v.toFixed(1));
    raPanel.appendChild(bubbleFaders);
    panel.appendChild(raPanel);

    // Tidy Tree
    const ttPanel = el("div");
    ttPanel.style.display = "none";
    algoPanels["tidy-tree"] = ttPanel;
    ttPanel.appendChild(el("div", "section-label", "Tidy Tree"));
    const ttRow = el("div", "fader-row");
    F["tt-levelSep"] = fader(ttRow, "LvlSep", 20, 300, 80, 5,
      (v) => state.graph?.setForceConfig({ tidyTreeLevelSeparation: v }), (v) => v.toFixed(0));
    F["tt-sibSep"] = fader(ttRow, "SibSep", 0.1, 5, 1.0, 0.1,
      (v) => state.graph?.setForceConfig({ tidyTreeSiblingSeparation: v }));
    F["tt-subSep"] = fader(ttRow, "SubSep", 0.5, 10, 2.0, 0.5,
      (v) => state.graph?.setForceConfig({ tidyTreeSubtreeSeparation: v }));
    F["tt-stiff"] = fader(ttRow, "Stiff", 0.01, 1, 0.3, 0.01,
      (v) => state.graph?.setForceConfig({ tidyTreeStiffness: v }));
    F["tt-damp"] = fader(ttRow, "Damp", 0.01, 2, 0.5, 0.01,
      (v) => state.graph?.setForceConfig({ tidyTreeDamping: v }));
    ttPanel.appendChild(ttRow);
    const ttBtns = el("div", "ctrl-row");
    const ttRadialRow = el("div", "ctrl-row");
    ttRadialRow.appendChild(el("span", "ctrl-label", "Radial"));
    T["tt-radial"] = toggle(ttRadialRow, true,
      (v) => state.graph?.setForceConfig({ tidyTreeRadial: v }));
    const ttCompute = el("button", "btn primary", "Recompute");
    ttCompute.addEventListener("click", () => state.graph?.computeTreeLayout());
    ttBtns.append(ttRadialRow, ttCompute);
    ttPanel.appendChild(ttBtns);
    panel.appendChild(ttPanel);

    // Community
    const commPanel = el("div");
    commPanel.style.display = "none";
    algoPanels["community"] = commPanel;
    commPanel.appendChild(el("div", "section-label", "Community Layout"));
    const commRow = el("div", "fader-row");
    F["comm-res"] = fader(commRow, "Res", 0.1, 5, 1.0, 0.1,
      (v) => state.graph?.setForceConfig({ communityResolution: v }));
    F["comm-space"] = fader(commRow, "Space", 5, 500, 50, 5,
      (v) => state.graph?.setForceConfig({ communitySpacing: v }), (v) => v.toFixed(0));
    F["comm-nodeSpace"] = fader(commRow, "NdSpc", 1, 100, 10, 1,
      (v) => state.graph?.setForceConfig({ communityNodeSpacing: v }), (v) => v.toFixed(0));
    F["comm-spread"] = fader(commRow, "Spread", 0.1, 5, 1.5, 0.1,
      (v) => state.graph?.setForceConfig({ communitySpreadFactor: v }));
    commPanel.appendChild(commRow);
    const commCompute = el("button", "btn primary", "Recompute Communities");
    commCompute.addEventListener("click", () => state.graph?.computeCommunityLayout());
    commPanel.appendChild(commCompute);
    panel.appendChild(commPanel);

    // Codebase
    const cbPanel = el("div");
    cbPanel.style.display = "none";
    algoPanels["codebase"] = cbPanel;
    cbPanel.appendChild(el("div", "section-label", "Codebase Layout"));
    const cbRow = el("div", "fader-row");
    F["cb-dirPad"] = fader(cbRow, "DirPad", 1, 100, 15, 1,
      (v) => state.graph?.setForceConfig({ codebaseDirectoryPadding: v }), (v) => v.toFixed(0));
    F["cb-filePad"] = fader(cbRow, "FilePd", 1, 50, 8, 1,
      (v) => state.graph?.setForceConfig({ codebaseFilePadding: v }), (v) => v.toFixed(0));
    F["cb-spread"] = fader(cbRow, "Spread", 0.1, 5, 1.5, 0.1,
      (v) => state.graph?.setForceConfig({ codebaseSpreadFactor: v }));
    cbPanel.appendChild(cbRow);
    const cbCompute = el("button", "btn primary", "Recompute Layout");
    cbCompute.addEventListener("click", () => {
      const cats = buildCodebaseCategories();
      state.graph?.computeCodebaseLayout(cats);
    });
    cbPanel.appendChild(cbCompute);
    panel.appendChild(cbPanel);

    // LinLog
    const llPanel = el("div");
    llPanel.style.display = "none";
    algoPanels["linlog"] = llPanel;
    llPanel.appendChild(el("div", "section-label", "LinLog"));
    const llRow = el("div", "fader-row");
    F["ll-scaling"] = fader(llRow, "Scale", 0.1, 100, 10, 0.1,
      (v) => state.graph?.setForceConfig({ linlogScaling: v }), (v) => v.toFixed(1));
    F["ll-gravity"] = fader(llRow, "Grav", 0, 10, 1, 0.1,
      (v) => state.graph?.setForceConfig({ linlogGravity: v }), (v) => v.toFixed(1));
    F["ll-weight"] = fader(llRow, "EdgeW", 0, 2, 1, 0.05,
      (v) => state.graph?.setForceConfig({ linlogEdgeWeightInfluence: v }));
    llPanel.appendChild(llRow);
    const llStrong = el("div", "ctrl-row");
    llStrong.appendChild(el("span", "ctrl-label", "Strong Gravity"));
    T["ll-strong"] = toggle(llStrong, false,
      (v) => state.graph?.setForceConfig({ linlogStrongGravity: v }));
    llPanel.appendChild(llStrong);
    panel.appendChild(llPanel);

    // t-FDP
    const tfdpPanel = el("div");
    tfdpPanel.style.display = "none";
    algoPanels["t-fdp"] = tfdpPanel;
    tfdpPanel.appendChild(el("div", "section-label", "t-FDP"));
    const tfdpRow = el("div", "fader-row");
    F["tfdp-gamma"] = fader(tfdpRow, "Gamma", 1, 5, 2, 0.1,
      (v) => state.graph?.setForceConfig({ tFdpGamma: v }));
    F["tfdp-rep"] = fader(tfdpRow, "Rep", 0.1, 50, 1, 0.1,
      (v) => state.graph?.setForceConfig({ tFdpRepulsionScale: v }));
    F["tfdp-alpha"] = fader(tfdpRow, "Alpha", 0.01, 1, 0.1, 0.01,
      (v) => state.graph?.setForceConfig({ tFdpAlpha: v }));
    F["tfdp-beta"] = fader(tfdpRow, "Beta", 0, 20, 8, 0.5,
      (v) => state.graph?.setForceConfig({ tFdpBeta: v }));
    tfdpPanel.appendChild(tfdpRow);
    panel.appendChild(tfdpPanel);
  }

  // ========================================================================
  // LAYERS TAB
  // ========================================================================

  function buildLayersTab(panel: HTMLElement) {
    const grid = el("div", "layer-grid");
    panel.appendChild(grid);

    // ---- Heatmap ----
    const hmCard = el("div", "card");
    const hmHeader = el("div", "card-header");
    hmHeader.appendChild(el("span", "card-title", "Heatmap"));
    T["heatmap"] = toggle(hmHeader, false, (v) => {
      if (v) {
        state.graph?.enableHeatmap({
          colorScale: D["hm-scale"]?.get() as "viridis" ?? "viridis",
          radius: F["hm-radius"]?.getValue() ?? 50,
          intensity: F["hm-intensity"]?.getValue() ?? 1.0,
          opacity: F["hm-opacity"]?.getValue() ?? 0.8,
        });
      } else state.graph?.disableHeatmap();
    });
    hmCard.appendChild(hmHeader);

    D["hm-scale"] = dropdown(hmCard,
      ["viridis", "plasma", "inferno", "magma", "turbo", "spectral", "coolwarm", "blues", "reds", "greens"]
        .map((v) => ({ value: v, label: v })),
      "viridis",
      (v) => T["heatmap"]?.get() && state.graph?.setHeatmapConfig({ colorScale: v as "viridis" }));
    D["hm-source"] = dropdown(hmCard,
      [{ value: "density", label: "Density" }, { value: "errors", label: "Errors" },
        { value: "warnings", label: "Warnings" }, { value: "complexity", label: "Complexity" },
        { value: "lines", label: "Lines" }],
      "density",
      (v) => {
        if (!T["heatmap"]?.get()) return;
        if (v === "density") { state.graph?.setHeatmapDataSource(v); return; }
        // Create stream on demand
        if (state.graph && !state.graph.hasValueStream(v)) {
          const configs: Record<string, { domain: [number, number]; stops: Array<{ position: number; color: [number, number, number, number] }> }> = {
            errors: { domain: [0, 5], stops: [{ position: 0, color: [0, 0, 0, 0] }, { position: 0.5, color: [1, 0.2, 0.1, 0.5] }, { position: 1, color: [1, 0.1, 0.05, 1] }] },
            warnings: { domain: [0, 10], stops: [{ position: 0, color: [0, 0, 0, 0] }, { position: 0.5, color: [1, 0.8, 0.2, 0.5] }, { position: 1, color: [1, 0.9, 0.3, 1] }] },
            complexity: { domain: [0, 30], stops: [{ position: 0, color: [0, 0, 0, 0] }, { position: 0.5, color: [0.5, 0.3, 0.8, 0.5] }, { position: 1, color: [0.8, 0.2, 0.9, 1] }] },
            lines: { domain: [0, 350], stops: [{ position: 0, color: [0, 0, 0, 0] }, { position: 0.5, color: [0.3, 0.7, 0.4, 0.5] }, { position: 1, color: [0.4, 0.9, 0.5, 1] }] },
          };
          const cfg = configs[v];
          if (cfg) {
            state.graph.defineValueStream({ id: v, name: v, colorScale: cfg, blendMode: "additive", opacity: 1.0 });
            const data: Array<{ nodeIndex: number; value: number }> = [];
            if (state.codebaseMetrics) {
              for (const [idx, metrics] of state.codebaseMetrics) {
                const val = metrics[v as keyof NodeMetrics];
                if (val > 0) data.push({ nodeIndex: idx, value: val });
              }
            } else {
              const [mn, mx] = cfg.domain;
              for (let i = 0; i < state.nodeCount; i++) {
                if (Math.random() < 0.3) data.push({ nodeIndex: i, value: mn + Math.pow(Math.random(), 2) * (mx - mn) });
              }
            }
            state.graph.setStreamValues(v, data);
          }
        }
        state.graph?.setHeatmapDataSource(v);
      });

    F["hm-radius"] = miniSlider(hmCard, "Radius", 10, 200, 50, 1,
      (v) => T["heatmap"]?.get() && state.graph?.setHeatmapConfig({ radius: v }), (v) => v.toFixed(0));
    F["hm-intensity"] = miniSlider(hmCard, "Intensity", 0.1, 5, 1.0, 0.1,
      (v) => T["heatmap"]?.get() && state.graph?.setHeatmapConfig({ intensity: v }), (v) => v.toFixed(1));
    F["hm-opacity"] = miniSlider(hmCard, "Opacity", 0, 1, 0.8, 0.05,
      (v) => T["heatmap"]?.get() && state.graph?.setHeatmapConfig({ opacity: v }));
    grid.appendChild(hmCard);

    // ---- Contours ----
    const ctCard = el("div", "card");
    const ctHeader = el("div", "card-header");
    ctHeader.appendChild(el("span", "card-title", "Contours"));
    T["contours"] = toggle(ctHeader, false, (v) => {
      if (v) {
        if (!T["heatmap"]?.get()) { T["contours"]!.set(false); return; }
        state.graph?.enableContour({
          strokeWidth: F["ct-width"]?.getValue() ?? 2,
          strokeColor: C["ct-color"]?.el.value ?? "#ffffff",
          thresholds: getContourThresholds(),
        });
      } else state.graph?.disableContour();
    });
    ctCard.appendChild(ctHeader);

    F["ct-width"] = miniSlider(ctCard, "Width", 0.5, 8, 2, 0.5,
      (v) => T["contours"]?.get() && state.graph?.setContourConfig({ strokeWidth: v }));
    const ctColorRow = el("div", "ctrl-row");
    ctColorRow.appendChild(el("span", "ctrl-label", "Color"));
    C["ct-color"] = colorPicker(ctColorRow, "#ffffff",
      (v) => T["contours"]?.get() && state.graph?.setContourConfig({ strokeColor: v }));
    ctCard.appendChild(ctColorRow);
    F["ct-thresh"] = miniSlider(ctCard, "Thresholds", 1, 10, 4, 1,
      () => T["contours"]?.get() && state.graph?.setContourConfig({ thresholds: getContourThresholds() }), (v) => v.toFixed(0));
    F["ct-min"] = miniSlider(ctCard, "Min", 0.02, 0.5, 0.1, 0.02,
      () => T["contours"]?.get() && state.graph?.setContourConfig({ thresholds: getContourThresholds() }));
    grid.appendChild(ctCard);

    function getContourThresholds(): number[] {
      const count = Math.round(F["ct-thresh"]?.getValue() ?? 4);
      const min = F["ct-min"]?.getValue() ?? 0.1;
      return Array.from({ length: count }, (_, i) => min + ((0.9 - min) * (i + 1)) / (count + 1));
    }

    // ---- Metaballs ----
    const mbCard = el("div", "card");
    const mbHeader = el("div", "card-header");
    mbHeader.appendChild(el("span", "card-title", "Metaballs"));
    T["metaballs"] = toggle(mbHeader, false, (v) => {
      if (v) {
        state.graph?.enableMetaball({
          fillColor: C["mb-color"]?.el.value ?? "#4f8cff",
          threshold: F["mb-thresh"]?.getValue() ?? 0.5,
          opacity: F["mb-opacity"]?.getValue() ?? 0.6,
        });
      } else state.graph?.disableMetaball();
    });
    mbCard.appendChild(mbHeader);

    const mbColorRow = el("div", "ctrl-row");
    mbColorRow.appendChild(el("span", "ctrl-label", "Color"));
    C["mb-color"] = colorPicker(mbColorRow, "#4f8cff",
      (v) => T["metaballs"]?.get() && state.graph?.setMetaballConfig({ fillColor: v }));
    mbCard.appendChild(mbColorRow);
    F["mb-thresh"] = miniSlider(mbCard, "Threshold", 0.1, 1, 0.5, 0.05,
      (v) => T["metaballs"]?.get() && state.graph?.setMetaballConfig({ threshold: v }));
    F["mb-opacity"] = miniSlider(mbCard, "Opacity", 0, 1, 0.6, 0.05,
      (v) => T["metaballs"]?.get() && state.graph?.setMetaballConfig({ opacity: v }));
    grid.appendChild(mbCard);

    // ---- Labels ----
    const lbCard = el("div", "card");
    const lbHeader = el("div", "card-header");
    lbHeader.appendChild(el("span", "card-title", "Labels"));
    T["labels"] = toggle(lbHeader, false, async (v) => {
      if (v) {
        await state.graph?.enableLabels({
          fontSize: Math.round(F["lb-size"]?.getValue() ?? 14),
          fontColor: C["lb-color"]?.el.value ?? "#ffffff",
          maxLabels: Math.round(F["lb-max"]?.getValue() ?? 100),
        });
        updateLabels();
      } else state.graph?.disableLabels();
    });
    lbCard.appendChild(lbHeader);

    F["lb-size"] = miniSlider(lbCard, "Size", 8, 32, 14, 1,
      (v) => T["labels"]?.get() && state.graph?.setLabelsConfig({ fontSize: v }), (v) => v.toFixed(0));
    const lbColorRow = el("div", "ctrl-row");
    lbColorRow.appendChild(el("span", "ctrl-label", "Color"));
    C["lb-color"] = colorPicker(lbColorRow, "#ffffff",
      (v) => T["labels"]?.get() && state.graph?.setLabelsConfig({ fontColor: v }));
    lbCard.appendChild(lbColorRow);
    F["lb-max"] = miniSlider(lbCard, "Max", 10, 500, 100, 10,
      (v) => T["labels"]?.get() && state.graph?.setLabelsConfig({ maxLabels: v }), (v) => v.toFixed(0));
    grid.appendChild(lbCard);

    // ---- Borders ----
    const brCard = el("div", "card");
    const brHeader = el("div", "card-header");
    brHeader.appendChild(el("span", "card-title", "Borders"));
    T["borders"] = toggle(brHeader, false, (v) => {
      if (v) state.graph?.enableNodeBorder(F["br-width"]?.getValue() ?? 2, C["br-color"]?.el.value ?? "#000000");
      else state.graph?.disableNodeBorder();
    });
    brCard.appendChild(brHeader);

    F["br-width"] = miniSlider(brCard, "Width", 0.5, 5, 2, 0.5,
      (v) => T["borders"]?.get() && state.graph?.setNodeBorder({ width: v }));
    const brColorRow = el("div", "ctrl-row");
    brColorRow.appendChild(el("span", "ctrl-label", "Color"));
    C["br-color"] = colorPicker(brColorRow, "#000000",
      (v) => T["borders"]?.get() && state.graph?.setNodeBorder({ color: v }));
    brCard.appendChild(brColorRow);
    grid.appendChild(brCard);

    // ---- Curves ----
    const cvCard = el("div", "card");
    const cvHeader = el("div", "card-header");
    cvHeader.appendChild(el("span", "card-title", "Curves"));
    T["curves"] = toggle(cvHeader, false, (v) => {
      if (v) {
        state.graph?.enableCurvedEdges(
          Math.round(F["cv-segs"]?.getValue() ?? 19),
          F["cv-weight"]?.getValue() ?? 0.8,
        );
        applyCurvature();
      } else state.graph?.disableCurvedEdges();
    });
    cvCard.appendChild(cvHeader);

    F["cv-curv"] = miniSlider(cvCard, "Curvature", -0.5, 0.5, 0.25, 0.01,
      () => T["curves"]?.get() && applyCurvature());
    F["cv-segs"] = miniSlider(cvCard, "Segments", 3, 50, 19, 1,
      (v) => T["curves"]?.get() && state.graph?.setCurvedEdges({ segments: v }), (v) => v.toFixed(0));
    F["cv-weight"] = miniSlider(cvCard, "Weight", 0.1, 2, 0.8, 0.05,
      (v) => T["curves"]?.get() && state.graph?.setCurvedEdges({ weight: v }));
    grid.appendChild(cvCard);

    function applyCurvature() {
      if (!state.graph || state.edgeCount === 0) return;
      const curv = F["cv-curv"]?.getValue() ?? 0.25;
      const curvatures = new Float32Array(state.edgeCount);
      curvatures.fill(curv);
      state.graph.setEdgeCurvatures(curvatures);
    }
  }

  // ========================================================================
  // FLOW TAB
  // ========================================================================

  function buildFlowTab(panel: HTMLElement) {
    // Preset
    const presetRow = el("div", "ctrl-row");
    presetRow.appendChild(el("span", "ctrl-label", "Preset"));
    D["flow-preset"] = dropdown(presetRow,
      ["none", "particles", "waves", "dataStream", "sparks", "energy", "warning", "dualLayer"]
        .map((v) => ({ value: v, label: v })),
      "none",
      (v) => {
        if (v === "none") {
          state.graph?.disableEdgeFlow();
          T["flow-on"]?.set(false);
        } else {
          state.graph?.setEdgeFlowPreset(v as "particles");
          T["flow-on"]?.set(true);
          syncFlowSlidersFromConfig();
        }
      });
    panel.appendChild(presetRow);

    function syncFlowSlidersFromConfig() {
      const config = state.graph?.getEdgeFlowConfig();
      if (!config) return;
      F["f1-speed"]?.setValue(config.layer1.speed);
      F["f1-width"]?.setValue(config.layer1.pulseWidth);
      F["f1-count"]?.setValue(config.layer1.pulseCount);
      F["f1-bright"]?.setValue(config.layer1.brightness);
      F["f1-fade"]?.setValue(config.layer1.fade);
      D["f1-wave"]?.set(config.layer1.waveShape);
      if (config.layer2) {
        T["f2-on"]?.set(config.layer2.enabled);
        F["f2-speed"]?.setValue(config.layer2.speed);
        F["f2-width"]?.setValue(config.layer2.pulseWidth);
        F["f2-count"]?.setValue(config.layer2.pulseCount);
        F["f2-bright"]?.setValue(config.layer2.brightness);
        F["f2-fade"]?.setValue(config.layer2.fade);
        D["f2-wave"]?.set(config.layer2.waveShape);
      }
    }

    // Layer 1
    const l1Label = el("div", "section-label", "Layer 1");
    panel.appendChild(l1Label);

    const l1Row = el("div");
    l1Row.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:4px";
    l1Row.appendChild(el("span", "ctrl-label", "Enable"));
    T["flow-on"] = toggle(l1Row, false, (v) => {
      if (v) updateFlowFromSliders();
      else state.graph?.disableEdgeFlow();
    });
    panel.appendChild(l1Row);

    const f1Faders = el("div", "fader-row");
    F["f1-speed"] = fader(f1Faders, "Speed", 0.05, 2, 0.3, 0.05, () => updateFlowFromSliders());
    F["f1-width"] = fader(f1Faders, "Width", 0.01, 0.5, 0.1, 0.01, () => updateFlowFromSliders());
    F["f1-count"] = fader(f1Faders, "Count", 1, 8, 3, 1, () => updateFlowFromSliders(), (v) => v.toFixed(0));
    F["f1-bright"] = fader(f1Faders, "Bright", 1, 5, 2, 0.1, () => updateFlowFromSliders(), (v) => v.toFixed(1));
    F["f1-fade"] = fader(f1Faders, "Fade", 0, 1, 0.3, 0.05, () => updateFlowFromSliders());
    panel.appendChild(f1Faders);

    const f1Extra = el("div");
    f1Extra.style.cssText = "display:flex;gap:12px;align-items:center;margin:4px 0";
    D["f1-wave"] = dropdown(f1Extra,
      [{ value: "sine", label: "Sine" }, { value: "square", label: "Square" }, { value: "triangle", label: "Triangle" }],
      "sine", () => updateFlowFromSliders());
    C["f1-color"] = colorPicker(f1Extra, "#00ffff", () => updateFlowFromSliders());
    const f1UseEdge = el("label");
    f1UseEdge.style.cssText = "font-size:9px;display:flex;align-items:center;gap:3px";
    const f1UseEdgeCb = document.createElement("input");
    f1UseEdgeCb.type = "checkbox";
    f1UseEdgeCb.addEventListener("change", () => updateFlowFromSliders());
    f1UseEdge.append(f1UseEdgeCb, document.createTextNode("edge color"));
    f1Extra.appendChild(f1UseEdge);
    panel.appendChild(f1Extra);

    // Layer 2
    panel.appendChild(el("div", "section-label", "Layer 2"));
    const l2Row = el("div");
    l2Row.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:4px";
    l2Row.appendChild(el("span", "ctrl-label", "Enable"));
    T["f2-on"] = toggle(l2Row, false, () => T["flow-on"]?.get() && updateFlowFromSliders());
    panel.appendChild(l2Row);

    const f2Faders = el("div", "fader-row");
    F["f2-speed"] = fader(f2Faders, "Speed", 0.05, 2, 0.5, 0.05, () => updateFlowFromSliders());
    F["f2-width"] = fader(f2Faders, "Width", 0.01, 0.5, 0.05, 0.01, () => updateFlowFromSliders());
    F["f2-count"] = fader(f2Faders, "Count", 1, 12, 6, 1, () => updateFlowFromSliders(), (v) => v.toFixed(0));
    F["f2-bright"] = fader(f2Faders, "Bright", 0.5, 5, 1.5, 0.1, () => updateFlowFromSliders(), (v) => v.toFixed(1));
    F["f2-fade"] = fader(f2Faders, "Fade", 0, 1, 0.2, 0.05, () => updateFlowFromSliders());
    panel.appendChild(f2Faders);

    const f2Extra = el("div");
    f2Extra.style.cssText = "display:flex;gap:12px;align-items:center;margin:4px 0";
    D["f2-wave"] = dropdown(f2Extra,
      [{ value: "sine", label: "Sine" }, { value: "square", label: "Square" }, { value: "triangle", label: "Triangle" }],
      "square", () => updateFlowFromSliders());
    C["f2-color"] = colorPicker(f2Extra, "#ff6b6b", () => updateFlowFromSliders());
    const f2UseEdge = el("label");
    f2UseEdge.style.cssText = "font-size:9px;display:flex;align-items:center;gap:3px";
    const f2UseEdgeCb = document.createElement("input");
    f2UseEdgeCb.type = "checkbox";
    f2UseEdgeCb.addEventListener("change", () => updateFlowFromSliders());
    f2UseEdge.append(f2UseEdgeCb, document.createTextNode("edge color"));
    f2Extra.appendChild(f2UseEdge);
    panel.appendChild(f2Extra);

    function getFlowColor(colorPick: ReturnType<typeof colorPicker>, useEdgeCb: HTMLInputElement): [number, number, number, number] | null {
      if (useEdgeCb.checked) return null;
      return hexToRgba(colorPick.el.value);
    }

    function updateFlowFromSliders() {
      if (!T["flow-on"]?.get()) return;
      state.graph?.setEdgeFlowConfig({
        layer1: {
          enabled: true,
          speed: F["f1-speed"]?.getValue() ?? 0.3,
          pulseWidth: F["f1-width"]?.getValue() ?? 0.1,
          pulseCount: Math.round(F["f1-count"]?.getValue() ?? 3),
          brightness: F["f1-bright"]?.getValue() ?? 2,
          fade: F["f1-fade"]?.getValue() ?? 0.3,
          waveShape: (D["f1-wave"]?.get() ?? "sine") as "sine" | "square" | "triangle",
          color: getFlowColor(C["f1-color"]!, f1UseEdgeCb),
        },
        layer2: {
          enabled: T["f2-on"]?.get() ?? false,
          speed: F["f2-speed"]?.getValue() ?? 0.5,
          pulseWidth: F["f2-width"]?.getValue() ?? 0.05,
          pulseCount: Math.round(F["f2-count"]?.getValue() ?? 6),
          brightness: F["f2-bright"]?.getValue() ?? 1.5,
          fade: F["f2-fade"]?.getValue() ?? 0.2,
          waveShape: (D["f2-wave"]?.get() ?? "square") as "sine" | "square" | "triangle",
          color: getFlowColor(C["f2-color"]!, f2UseEdgeCb),
        },
      });
    }
  }

  // ========================================================================
  // STYLE TAB
  // ========================================================================

  function buildStyleTab(panel: HTMLElement) {
    // Theme buttons
    panel.appendChild(el("div", "section-label", "Theme"));
    const themeRow = el("div", "btn-row");
    for (const name of Object.keys(THEMES)) {
      const b = el("button", "btn", name);
      b.addEventListener("click", () => {
        const theme = THEMES[name];
        C["bg-color"]?.set(theme.background);
        state.graph?.setBackgroundColor(theme.background);
        if (T["labels"]?.get()) {
          C["lb-color"]?.set(theme.labelColor);
          state.graph?.setLabelsConfig({ fontColor: theme.labelColor });
        }
      });
      themeRow.appendChild(b);
    }
    panel.appendChild(themeRow);

    // Background color
    const bgRow = el("div", "ctrl-row");
    bgRow.appendChild(el("span", "ctrl-label", "Background"));
    C["bg-color"] = colorPicker(bgRow, "#0a0a0f",
      (v) => state.graph?.setBackgroundColor(v));
    panel.appendChild(bgRow);

    // Edge opacity
    panel.appendChild(el("div", "section-label", "Edges"));
    const edgeFaders = el("div", "fader-row");
    edgeFaders.style.height = "100px";
    F["edge-opacity"] = fader(edgeFaders, "Opacity", 0.05, 1, 0.4, 0.05, (v) => {
      if (!state.graph || state.edgeCount === 0) return;
      const colors = new Float32Array(state.edgeCount * 4);
      for (let i = 0; i < state.edgeCount; i++) {
        let r = 0.5, g = 0.5, b = 0.5;
        if (state.currentEdgeColors && state.currentEdgeColors.length === state.edgeCount * 4) {
          r = state.currentEdgeColors[i * 4]; g = state.currentEdgeColors[i * 4 + 1]; b = state.currentEdgeColors[i * 4 + 2];
        }
        colors[i * 4] = r; colors[i * 4 + 1] = g; colors[i * 4 + 2] = b; colors[i * 4 + 3] = v;
      }
      setEdgeColorsWithTracking(colors);
    });
    F["edge-widthScale"] = fader(edgeFaders, "Width", 0.1, 5, 1, 0.1, (v) => {
      if (!state.graph || !state.graphData || state.edgeCount === 0) return;
      const widths = new Float32Array(state.edgeCount);
      for (let i = 0; i < state.edgeCount; i++) {
        const edge = state.graphData.edges[i];
        const baseWidth = edge ? ((edge as Record<string, unknown>).width as number ?? 1) : 1;
        widths[i] = baseWidth * v;
      }
      state.graph.setEdgeWidths(widths);
    }, (v) => v.toFixed(1));
    panel.appendChild(edgeFaders);

    // Node size
    panel.appendChild(el("div", "section-label", "Nodes"));
    const nodeFaders = el("div", "fader-row");
    nodeFaders.style.height = "100px";
    F["node-sizeScale"] = fader(nodeFaders, "Size", 0.25, 4, 1, 0.25, (scale) => {
      if (!state.graph || !state.graphData || state.nodeCount === 0) return;
      const sizes = new Float32Array(state.nodeCount);
      for (let i = 0; i < state.nodeCount; i++) {
        const node = state.graphData.nodes[i];
        const baseRadius = node ? ((node as Record<string, unknown>).radius as number ?? 5) : 5;
        sizes[i] = baseRadius * scale;
      }
      state.graph.setNodeSizes(sizes);
    }, (v) => v.toFixed(1));
    panel.appendChild(nodeFaders);
  }

  // ========================================================================
  // Config Loading
  // ========================================================================

  async function loadConfig(): Promise<void> {
    let config: Record<string, unknown>;
    try {
      const resp = await fetch("/config.json");
      if (!resp.ok) { await loadNodes(1000); return; }
      config = await resp.json();
      console.log("Loaded config.json");
    } catch {
      await loadNodes(1000);
      return;
    }

    const g = config.generator as Record<string, unknown> | undefined;
    const f = config.force as Record<string, unknown> | undefined;
    const ra = config.relativityAtlas as Record<string, unknown> | undefined;
    const ll = config.linlog as Record<string, unknown> | undefined;
    const tfdp = config.tFdp as Record<string, unknown> | undefined;
    const tt = config.tidyTree as Record<string, unknown> | undefined;
    const comm = config.community as Record<string, unknown> | undefined;
    const cb = config.codebase as Record<string, unknown> | undefined;
    const disp = config.display as Record<string, unknown> | undefined;
    const hm = config.heatmap as Record<string, unknown> | undefined;
    const ct = config.contours as Record<string, unknown> | undefined;
    const mb = config.metaballs as Record<string, unknown> | undefined;
    const lb = config.labels as Record<string, unknown> | undefined;
    const ns = config.nodeStyle as Record<string, unknown> | undefined;
    const es = config.edgeStyle as Record<string, unknown> | undefined;
    const ef = config.edgeFlow as Record<string, unknown> | undefined;

    // Generator
    if (g?.branching != null) F["branch"]?.setValue(Number(g.branching));

    // Algorithm
    if (f?.algorithm) {
      const alg = String(f.algorithm);
      algoSelect.value = alg;
      state.graph?.setForceAlgorithm(alg as "n2");
      for (const [key, panel] of Object.entries(algoPanels)) {
        panel.style.display = key === alg ? "" : "none";
      }
    }

    // Force config
    const fc: Record<string, unknown> = {};

    // Common forces
    if (f?.repulsion != null) { fc.repulsionStrength = -Math.abs(Number(f.repulsion)); F["rep"]?.setValue(Math.abs(Number(f.repulsion))); }
    if (f?.springStrength != null) { fc.springStrength = Number(f.springStrength); F["spring"]?.setValue(Number(f.springStrength)); }
    if (f?.linkDistance != null) { fc.springLength = Number(f.linkDistance); F["length"]?.setValue(Number(f.linkDistance)); }
    if (f?.centerGravity != null) { fc.centerStrength = Number(f.centerGravity); F["gravity"]?.setValue(Number(f.centerGravity)); }
    if (f?.damping != null) { fc.velocityDecay = Number(f.damping); F["damping"]?.setValue(Number(f.damping)); }
    if (f?.theta != null) { fc.theta = Number(f.theta); F["theta"]?.setValue(Number(f.theta)); }
    if (f?.maxVelocity != null) { fc.maxVelocity = Number(f.maxVelocity); F["maxvel"]?.setValue(Number(f.maxVelocity)); }
    if (f?.pinRootNode != null) { fc.pinnedNode = f.pinRootNode ? 0 : 0xFFFFFFFF; T["pinRoot"]?.set(Boolean(f.pinRootNode)); }

    // Relativity Atlas
    if (ra) {
      if (ra.baseMass != null) { fc.relativityBaseMass = Number(ra.baseMass); F["ra-baseMass"]?.setValue(Number(ra.baseMass)); }
      if (ra.childFactor != null) { fc.relativityChildMassFactor = Number(ra.childFactor); F["ra-childFactor"]?.setValue(Number(ra.childFactor)); }
      if (ra.massExponent != null) { fc.relativityMassExponent = Number(ra.massExponent); F["ra-massExp"]?.setValue(Number(ra.massExponent)); }
      if (ra.maxSiblings != null) { fc.relativityMaxSiblings = Number(ra.maxSiblings); F["ra-maxSib"]?.setValue(Number(ra.maxSiblings)); }
      if (ra.parentChildMult != null) { fc.relativityParentChildMultiplier = Number(ra.parentChildMult); F["ra-parentChild"]?.setValue(Number(ra.parentChildMult)); }
      if (ra.densityRepulsion != null) { fc.relativityDensityRepulsion = Number(ra.densityRepulsion); F["ra-density"]?.setValue(Number(ra.densityRepulsion)); }
      if (ra.orbitStrength != null) { fc.relativityOrbitStrength = Number(ra.orbitStrength); F["ra-orbit"]?.setValue(Number(ra.orbitStrength)); }
      if (ra.tangentialMultiplier != null) { fc.relativityTangentialMultiplier = Number(ra.tangentialMultiplier); F["ra-tang"]?.setValue(Number(ra.tangentialMultiplier)); }
      if (ra.orbitRadius != null) { fc.relativityOrbitRadius = Number(ra.orbitRadius); F["ra-orbitRad"]?.setValue(Number(ra.orbitRadius)); }
      if (ra.cousinRepulsion != null) { fc.relativityCousinRepulsion = Boolean(ra.cousinRepulsion); T["ra-cousin"]?.set(Boolean(ra.cousinRepulsion)); }
      if (ra.phantomZone != null) { fc.relativityPhantomZone = Boolean(ra.phantomZone); T["ra-phantom"]?.set(Boolean(ra.phantomZone)); }
      // Bubble mode
      if (ra.bubbleMode != null) { fc.relativityBubbleMode = Boolean(ra.bubbleMode); T["ra-bubble"]?.set(Boolean(ra.bubbleMode)); }
      if (ra.bubbleBaseRadius != null) { fc.relativityBubbleBaseRadius = Number(ra.bubbleBaseRadius); F["ra-bubbleRad"]?.setValue(Number(ra.bubbleBaseRadius)); }
      if (ra.bubblePadding != null) { fc.relativityBubblePadding = Number(ra.bubblePadding); F["ra-bubblePad"]?.setValue(Number(ra.bubblePadding)); }
      if (ra.depthDecay != null) { fc.relativityDepthDecay = Number(ra.depthDecay); F["ra-depthDecay"]?.setValue(Number(ra.depthDecay)); }
      if (ra.bubbleOrbitScale != null) { fc.relativityBubbleOrbitScale = Number(ra.bubbleOrbitScale); F["ra-bubbleOrbit"]?.setValue(Number(ra.bubbleOrbitScale)); }
    }

    // LinLog
    if (ll) {
      if (ll.scaling != null) { fc.linlogScaling = Number(ll.scaling); F["ll-scaling"]?.setValue(Number(ll.scaling)); }
      if (ll.gravity != null) { fc.linlogGravity = Number(ll.gravity); F["ll-gravity"]?.setValue(Number(ll.gravity)); }
      if (ll.edgeWeightInfluence != null) { fc.linlogEdgeWeightInfluence = Number(ll.edgeWeightInfluence); F["ll-weight"]?.setValue(Number(ll.edgeWeightInfluence)); }
      if (ll.strongGravity != null) { fc.linlogStrongGravity = Boolean(ll.strongGravity); T["ll-strong"]?.set(Boolean(ll.strongGravity)); }
    }

    // t-FDP
    if (tfdp) {
      if (tfdp.gamma != null) { fc.tFdpGamma = Number(tfdp.gamma); F["tfdp-gamma"]?.setValue(Number(tfdp.gamma)); }
      if (tfdp.repulsionScale != null) { fc.tFdpRepulsionScale = Number(tfdp.repulsionScale); F["tfdp-rep"]?.setValue(Number(tfdp.repulsionScale)); }
      if (tfdp.alpha != null) { fc.tFdpAlpha = Number(tfdp.alpha); F["tfdp-alpha"]?.setValue(Number(tfdp.alpha)); }
      if (tfdp.beta != null) { fc.tFdpBeta = Number(tfdp.beta); F["tfdp-beta"]?.setValue(Number(tfdp.beta)); }
    }

    // Tidy Tree
    if (tt) {
      if (tt.levelSeparation != null) { fc.tidyTreeLevelSeparation = Number(tt.levelSeparation); F["tt-levelSep"]?.setValue(Number(tt.levelSeparation)); }
      if (tt.siblingSeparation != null) { fc.tidyTreeSiblingSeparation = Number(tt.siblingSeparation); F["tt-sibSep"]?.setValue(Number(tt.siblingSeparation)); }
      if (tt.subtreeSeparation != null) { fc.tidyTreeSubtreeSeparation = Number(tt.subtreeSeparation); F["tt-subSep"]?.setValue(Number(tt.subtreeSeparation)); }
      if (tt.stiffness != null) { fc.tidyTreeStiffness = Number(tt.stiffness); F["tt-stiff"]?.setValue(Number(tt.stiffness)); }
      if (tt.damping != null) { fc.tidyTreeDamping = Number(tt.damping); F["tt-damp"]?.setValue(Number(tt.damping)); }
      if (tt.coordMode != null) { fc.tidyTreeRadial = String(tt.coordMode) === "radial"; T["tt-radial"]?.set(String(tt.coordMode) === "radial"); }
    }

    // Community
    if (comm) {
      if (comm.resolution != null) { fc.communityResolution = Number(comm.resolution); F["comm-res"]?.setValue(Number(comm.resolution)); }
      if (comm.spacing != null) { fc.communitySpacing = Number(comm.spacing); F["comm-space"]?.setValue(Number(comm.spacing)); }
      if (comm.nodeSpacing != null) { fc.communityNodeSpacing = Number(comm.nodeSpacing); F["comm-nodeSpace"]?.setValue(Number(comm.nodeSpacing)); }
      if (comm.spreadFactor != null) { fc.communitySpreadFactor = Number(comm.spreadFactor); F["comm-spread"]?.setValue(Number(comm.spreadFactor)); }
    }

    // Codebase
    if (cb) {
      if (cb.directoryPadding != null) { fc.codebaseDirectoryPadding = Number(cb.directoryPadding); F["cb-dirPad"]?.setValue(Number(cb.directoryPadding)); }
      if (cb.filePadding != null) { fc.codebaseFilePadding = Number(cb.filePadding); F["cb-filePad"]?.setValue(Number(cb.filePadding)); }
      if (cb.spreadFactor != null) { fc.codebaseSpreadFactor = Number(cb.spreadFactor); F["cb-spread"]?.setValue(Number(cb.spreadFactor)); }
    }

    // Apply all force config
    if (Object.keys(fc).length > 0) state.graph?.setForceConfig(fc);

    // Load data based on generator type
    const genType = g?.type ? String(g.type) : "hierarchical";
    if (genType === "json") {
      const file = g?.file ? String(g.file) : "index-data.json";
      const loaded = await loadIndexData(file);
      if (!loaded) {
        await loadNodes(1000);
      }
    } else {
      const nodeCount = g?.nodes ? Number(g.nodes) : 1000;
      await loadNodes(nodeCount);
    }

    // Display
    if (disp?.backgroundColor) {
      C["bg-color"]?.set(String(disp.backgroundColor));
      state.graph?.setBackgroundColor(String(disp.backgroundColor));
    }

    // Heatmap
    if (hm) {
      if (hm.colorScale) D["hm-scale"]?.set(String(hm.colorScale));
      if (hm.dataSource) D["hm-source"]?.set(String(hm.dataSource));
      if (hm.radius != null) F["hm-radius"]?.setValue(Number(hm.radius));
      if (hm.intensity != null) F["hm-intensity"]?.setValue(Number(hm.intensity));
      if (hm.opacity != null) F["hm-opacity"]?.setValue(Number(hm.opacity));
      T["heatmap"]?.set(Boolean(hm.enabled));
      if (hm.enabled) {
        state.graph?.enableHeatmap({
          colorScale: String(hm.colorScale ?? "viridis") as "viridis",
          radius: Number(hm.radius ?? 50),
          intensity: Number(hm.intensity ?? 1),
          opacity: Number(hm.opacity ?? 0.8),
        });
      }
    }

    // Contours
    if (ct) {
      if (ct.strokeWidth != null) F["ct-width"]?.setValue(Number(ct.strokeWidth));
      if (ct.strokeColor) C["ct-color"]?.set(String(ct.strokeColor));
      if (ct.thresholdCount != null) F["ct-thresh"]?.setValue(Number(ct.thresholdCount));
      if (ct.minThreshold != null) F["ct-min"]?.setValue(Number(ct.minThreshold));
      T["contours"]?.set(Boolean(ct.enabled));
      if (ct.enabled) {
        const count = Number(ct.thresholdCount ?? 4);
        const min = Number(ct.minThreshold ?? 0.1);
        state.graph?.enableContour({
          thresholds: Array.from({ length: count }, (_, i) => min + (i * (1 - min)) / count),
          strokeWidth: Number(ct.strokeWidth ?? 2),
          strokeColor: String(ct.strokeColor ?? "#ffffff"),
        });
      }
    }

    // Metaballs
    if (mb) {
      if (mb.threshold != null) F["mb-thresh"]?.setValue(Number(mb.threshold));
      if (mb.opacity != null) F["mb-opacity"]?.setValue(Number(mb.opacity));
      if (mb.fillColor) C["mb-color"]?.set(String(mb.fillColor));
      T["metaballs"]?.set(Boolean(mb.enabled));
      if (mb.enabled) {
        state.graph?.enableMetaball({
          fillColor: String(mb.fillColor ?? "#4f8cff"),
          threshold: Number(mb.threshold ?? 0.5),
          opacity: Number(mb.opacity ?? 0.6),
        });
      }
    }

    // Labels
    if (lb) {
      if (lb.fontSize != null) F["lb-size"]?.setValue(Number(lb.fontSize));
      if (lb.textColor) C["lb-color"]?.set(String(lb.textColor));
      if (lb.maxLabels != null) F["lb-max"]?.setValue(Number(lb.maxLabels));
      T["labels"]?.set(Boolean(lb.enabled));
      if (lb.enabled) {
        await state.graph?.enableLabels({
          fontSize: Number(lb.fontSize ?? 14),
          fontColor: String(lb.textColor ?? "#ffffff"),
          maxLabels: Number(lb.maxLabels ?? 100),
        });
      }
    }

    // Node style
    if (ns) {
      if (ns.sizeScale != null) F["node-sizeScale"]?.setValue(Number(ns.sizeScale));
      const borders = ns.borders as Record<string, unknown> | undefined;
      if (borders) {
        if (borders.width != null) F["br-width"]?.setValue(Number(borders.width));
        if (borders.color) C["br-color"]?.set(String(borders.color));
        T["borders"]?.set(Boolean(borders.enabled));
        if (borders.enabled) state.graph?.enableNodeBorder(Number(borders.width ?? 2), String(borders.color ?? "#000000"));
      }
    }

    // Edge style
    if (es) {
      if (es.opacity != null) F["edge-opacity"]?.setValue(Number(es.opacity));
      if (es.widthScale != null) F["edge-widthScale"]?.setValue(Number(es.widthScale));
      const curved = es.curved as Record<string, unknown> | undefined;
      if (curved) {
        if (curved.segments != null) F["cv-segs"]?.setValue(Number(curved.segments));
        if (curved.weight != null) F["cv-weight"]?.setValue(Number(curved.weight));
        if (curved.curvature != null) F["cv-curv"]?.setValue(Number(curved.curvature));
        T["curves"]?.set(Boolean(curved.enabled));
        if (curved.enabled) state.graph?.enableCurvedEdges(Number(curved.segments ?? 19), Number(curved.weight ?? 0.8));
      }
    }

    // Edge flow
    if (ef) {
      const l1 = ef.layer1 as Record<string, unknown> | undefined;
      const l2 = ef.layer2 as Record<string, unknown> | undefined;
      if (l1) {
        if (l1.speed != null) F["f1-speed"]?.setValue(Number(l1.speed));
        if (l1.pulseWidth != null) F["f1-width"]?.setValue(Number(l1.pulseWidth));
        if (l1.pulseCount != null) F["f1-count"]?.setValue(Number(l1.pulseCount));
        if (l1.brightness != null) F["f1-bright"]?.setValue(Number(l1.brightness));
        if (l1.fade != null) F["f1-fade"]?.setValue(Number(l1.fade));
        if (l1.waveShape) D["f1-wave"]?.set(String(l1.waveShape));
        if (l1.color) C["f1-color"]?.set(String(l1.color));
      }
      if (l2) {
        if (l2.speed != null) F["f2-speed"]?.setValue(Number(l2.speed));
        if (l2.pulseWidth != null) F["f2-width"]?.setValue(Number(l2.pulseWidth));
        if (l2.pulseCount != null) F["f2-count"]?.setValue(Number(l2.pulseCount));
        if (l2.brightness != null) F["f2-bright"]?.setValue(Number(l2.brightness));
        if (l2.fade != null) F["f2-fade"]?.setValue(Number(l2.fade));
        if (l2.waveShape) D["f2-wave"]?.set(String(l2.waveShape));
        if (l2.color) C["f2-color"]?.set(String(l2.color));
        T["f2-on"]?.set(Boolean(l2.enabled));
      }
      T["flow-on"]?.set(Boolean(ef.enabled));
      if (ef.enabled && l1) {
        state.graph?.setEdgeFlowConfig({
          layer1: {
            enabled: true,
            speed: Number(l1.speed ?? 0.3), pulseWidth: Number(l1.pulseWidth ?? 0.1),
            pulseCount: Number(l1.pulseCount ?? 3), brightness: Number(l1.brightness ?? 2),
            fade: Number(l1.fade ?? 0.3), waveShape: String(l1.waveShape ?? "sine") as "sine",
            color: l1.useEdgeColor ? null : hexToRgba(String(l1.color ?? "#00ffff")),
          },
          layer2: l2 ? {
            enabled: Boolean(l2.enabled),
            speed: Number(l2.speed ?? 0.5), pulseWidth: Number(l2.pulseWidth ?? 0.05),
            pulseCount: Number(l2.pulseCount ?? 6), brightness: Number(l2.brightness ?? 1.5),
            fade: Number(l2.fade ?? 0.2), waveShape: String(l2.waveShape ?? "square") as "sine",
            color: l2.useEdgeColor ? null : hexToRgba(String(l2.color ?? "#ff6b6b")),
          } : undefined,
        });
      }
    }

    // Auto-compute for structure-aware algorithms
    if (f?.algorithm) {
      const alg = String(f.algorithm);
      try {
        if (alg === "tidy-tree") state.graph?.computeTreeLayout();
        else if (alg === "community") state.graph?.computeCommunityLayout();
        else if (alg === "codebase") state.graph?.computeCodebaseLayout(buildCodebaseCategories());
      } catch (err) { console.warn("Auto-compute failed:", err); }
    }

    console.log("Config applied");
  }

  // ========================================================================
  // Keyboard shortcuts
  // ========================================================================

  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.code === "Space") { e.preventDefault(); toggleDrawer(); }
    else if (e.code === "KeyR") { e.preventDefault(); state.graph?.restartSimulation(); }
    else if (e.code === "KeyF") { e.preventDefault(); state.graph?.fitToView(); }
  });

  // ========================================================================
  // Start
  // ========================================================================

  await loadConfig();
}

main().catch(console.error);
