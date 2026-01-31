/**
 * HeroineGraph - Large Graph Stress Test
 *
 * Tests the performance of the library with large graphs (up to 1M nodes).
 * Use this to benchmark and verify the 30fps @ 500K nodes target.
 *
 * @module
 */

import { createHeroineGraph, type GraphInput, type HeroineGraph } from "../../packages/core/mod.ts";

// DOM elements
const canvas = document.getElementById("graph-canvas") as HTMLCanvasElement;
const nodeCountSelect = document.getElementById("node-count") as HTMLSelectElement;
const edgeRatioSelect = document.getElementById("edge-ratio") as HTMLSelectElement;
const btnLoad = document.getElementById("btn-load") as HTMLButtonElement;
const btnSimulation = document.getElementById("btn-simulation") as HTMLButtonElement;
const btnFit = document.getElementById("btn-fit") as HTMLButtonElement;
const loadingOverlay = document.getElementById("loading") as HTMLDivElement;
const loadingText = document.getElementById("loading-text") as HTMLDivElement;

// Stats elements
const statNodes = document.getElementById("stat-nodes") as HTMLSpanElement;
const statEdges = document.getElementById("stat-edges") as HTMLSpanElement;
const statFps = document.getElementById("stat-fps") as HTMLSpanElement;
const statFrametime = document.getElementById("stat-frametime") as HTMLSpanElement;
const statSimulation = document.getElementById("stat-simulation") as HTMLSpanElement;
const statAlpha = document.getElementById("stat-alpha") as HTMLSpanElement;

// State
let graph: HeroineGraph | null = null;
let isSimulationRunning = false;

// FPS tracking
const fpsSamples: number[] = [];
let lastFrameTime = 0;

/**
 * Generate a scale-free graph (Barabási–Albert model)
 * This creates more realistic network topology
 */
function generateScaleFreeGraph(nodeCount: number, edgesPerNode: number): GraphInput {
  const nodes: GraphInput["nodes"] = [];
  const edges: GraphInput["edges"] = [];

  // Create nodes
  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      id: i,
      radius: 2, // Small radius for large graphs
      color: "#4ecdc4",
    });
  }

  // Keep track of node degrees for preferential attachment
  const degrees = new Array(nodeCount).fill(0);

  // Create initial small clique
  const m0 = Math.min(edgesPerNode + 1, 10);
  for (let i = 0; i < m0; i++) {
    for (let j = i + 1; j < m0; j++) {
      edges.push({ source: i, target: j });
      degrees[i]++;
      degrees[j]++;
    }
  }

  // Add remaining nodes with preferential attachment
  let totalDegree = edges.length * 2;

  for (let i = m0; i < nodeCount; i++) {
    const connections = new Set<number>();

    // Connect to existing nodes with probability proportional to degree
    while (connections.size < edgesPerNode && connections.size < i) {
      // Sample a node with probability proportional to degree
      let targetSum = Math.random() * totalDegree;
      let target = 0;

      for (let j = 0; j < i; j++) {
        targetSum -= degrees[j] + 1; // +1 to avoid zero probability
        if (targetSum <= 0) {
          target = j;
          break;
        }
      }

      if (!connections.has(target)) {
        connections.add(target);
        edges.push({ source: i, target });
        degrees[i]++;
        degrees[target]++;
        totalDegree += 2;
      }
    }
  }

  return { nodes, edges };
}

/**
 * Generate a random graph (Erdős–Rényi model)
 */
function generateRandomGraph(nodeCount: number, edgesPerNode: number): GraphInput {
  const nodes: GraphInput["nodes"] = [];
  const edges: GraphInput["edges"] = [];

  // Create nodes
  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      id: i,
      radius: 2,
      color: "#4ecdc4",
    });
  }

  // Create edges
  const targetEdges = nodeCount * edgesPerNode;
  const edgeSet = new Set<string>();

  while (edges.length < targetEdges) {
    const source = Math.floor(Math.random() * nodeCount);
    const target = Math.floor(Math.random() * nodeCount);

    if (source !== target) {
      const key = source < target ? `${source}-${target}` : `${target}-${source}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push({ source, target });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Show loading overlay
 */
function showLoading(message: string) {
  loadingOverlay.classList.remove("hidden");
  loadingText.textContent = message;
}

/**
 * Hide loading overlay
 */
function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

/**
 * Update FPS display
 */
function updateFps() {
  const now = performance.now();
  if (lastFrameTime > 0) {
    const frameTime = now - lastFrameTime;
    fpsSamples.push(frameTime);
    if (fpsSamples.length > 60) {
      fpsSamples.shift();
    }

    const avgFrameTime = fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length;
    const fps = 1000 / avgFrameTime;

    statFps.textContent = fps.toFixed(1);
    statFps.className = `stats-value ${fps >= 30 ? "good" : fps >= 15 ? "" : "warning"}`;

    statFrametime.textContent = `${avgFrameTime.toFixed(1)}ms`;
    statFrametime.className = `stats-value ${avgFrameTime <= 33 ? "good" : avgFrameTime <= 66 ? "" : "warning"}`;
  }
  lastFrameTime = now;

  requestAnimationFrame(updateFps);
}

/**
 * Initialize graph
 */
async function initGraph() {
  if (graph) {
    graph.dispose();
  }

  graph = await createHeroineGraph({
    canvas,
    config: {
      renderer: {
        antialias: false, // Disable for performance with large graphs
      },
    },
  });

  // Listen for simulation events
  graph.on("simulation:tick", (event) => {
    statAlpha.textContent = event.alpha.toFixed(3);
  });

  graph.on("simulation:end", () => {
    statSimulation.textContent = "Stopped";
    isSimulationRunning = false;
    btnSimulation.textContent = "Start Simulation";
  });

  // Update UI
  btnSimulation.disabled = true;
  btnFit.disabled = true;
}

/**
 * Load graph with current settings
 */
async function loadGraph() {
  if (!graph) return;

  const nodeCount = parseInt(nodeCountSelect.value);
  const edgesPerNode = parseInt(edgeRatioSelect.value);

  showLoading(`Generating ${nodeCount.toLocaleString()} nodes...`);

  // Use setTimeout to allow UI to update
  await new Promise((resolve) => setTimeout(resolve, 50));

  showLoading(`Generating graph structure...`);
  await new Promise((resolve) => setTimeout(resolve, 50));

  // Generate graph data
  const data = nodeCount > 100000
    ? generateRandomGraph(nodeCount, edgesPerNode)
    : generateScaleFreeGraph(nodeCount, edgesPerNode);

  showLoading(`Loading ${data.nodes.length.toLocaleString()} nodes...`);
  await new Promise((resolve) => setTimeout(resolve, 50));

  // Load into graph
  await graph.load(data);

  // Update stats
  statNodes.textContent = data.nodes.length.toLocaleString();
  statEdges.textContent = data.edges.length.toLocaleString();
  statSimulation.textContent = "Stopped";
  statAlpha.textContent = "-";

  // Enable controls
  btnSimulation.disabled = false;
  btnFit.disabled = false;

  // Fit to view
  graph.fitToView();

  hideLoading();
}

/**
 * Toggle simulation
 */
function toggleSimulation() {
  if (!graph) return;

  if (isSimulationRunning) {
    graph.stopSimulation();
    statSimulation.textContent = "Stopped";
    btnSimulation.textContent = "Start Simulation";
  } else {
    graph.startSimulation();
    statSimulation.textContent = "Running";
    btnSimulation.textContent = "Stop Simulation";
  }

  isSimulationRunning = !isSimulationRunning;
}

/**
 * Fit graph to view
 */
function fitToView() {
  graph?.fitToView();
}

// Event listeners
btnLoad.addEventListener("click", loadGraph);
btnSimulation.addEventListener("click", toggleSimulation);
btnFit.addEventListener("click", fitToView);

// Initialize
(async () => {
  showLoading("Initializing WebGPU...");
  await initGraph();
  hideLoading();

  // Start FPS counter
  requestAnimationFrame(updateFps);

  // Auto-load initial graph
  await loadGraph();
})();
