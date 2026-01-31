/**
 * HeroineGraph - Interactive Demo
 *
 * Demonstrates all interactive features:
 * - Click to select nodes
 * - Drag to move nodes
 * - Double-click to add nodes
 * - Selection management
 * - Layer controls
 * - Force configuration
 *
 * @module
 */

import {
  createHeroineGraph,
  type GraphInput,
  type HeroineGraph,
  type NodeId,
  type NodeClickEvent,
  type NodeHoverEnterEvent,
  type NodeHoverLeaveEvent,
  type NodeDragEndEvent,
  type SelectionChangeEvent,
  type BackgroundClickEvent,
} from "../../packages/core/mod.ts";

// DOM elements
const canvas = document.getElementById("graph-canvas") as HTMLCanvasElement;
const btnAddNode = document.getElementById("btn-add-node") as HTMLButtonElement;
const btnDeleteNode = document.getElementById("btn-delete-node") as HTMLButtonElement;
const btnClear = document.getElementById("btn-clear") as HTMLButtonElement;
const btnFit = document.getElementById("btn-fit") as HTMLButtonElement;
const btnSimulation = document.getElementById("btn-simulation") as HTMLButtonElement;
const nodeInfo = document.getElementById("node-info") as HTMLDivElement;
const alphaValue = document.getElementById("alpha-value") as HTMLSpanElement;
const repulsionSlider = document.getElementById("repulsion") as HTMLInputElement;
const attractionSlider = document.getElementById("attraction") as HTMLInputElement;
const layerHeatmap = document.getElementById("layer-heatmap") as HTMLInputElement;
const layerContours = document.getElementById("layer-contours") as HTMLInputElement;
const layerLabels = document.getElementById("layer-labels") as HTMLInputElement;
const eventLog = document.getElementById("event-log") as HTMLDivElement;

// State
let graph: HeroineGraph | null = null;
let isSimulationRunning = true;
let selectedNodeId: NodeId | null = null;
let nodeCounter = 0;

// Graph data (mutable for interactive editing)
const nodes: { id: NodeId; label: string; x?: number; y?: number }[] = [];
const edges: { source: NodeId; target: NodeId }[] = [];

/**
 * Log an event
 */
function logEvent(message: string) {
  const entry = document.createElement("div");
  entry.className = "event-log-entry";
  entry.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
  eventLog.insertBefore(entry, eventLog.firstChild);

  // Keep only last 50 entries
  while (eventLog.children.length > 50) {
    eventLog.removeChild(eventLog.lastChild!);
  }
}

/**
 * Update node info panel
 */
function updateNodeInfo(nodeId: NodeId | null) {
  if (nodeId === null) {
    nodeInfo.innerHTML = `<div style="color: #666; font-style: italic;">No node selected</div>`;
    return;
  }

  const node = nodes.find((n) => n.id === nodeId);
  if (!node) return;

  // Count connections
  const connections = edges.filter(
    (e) => e.source === nodeId || e.target === nodeId
  ).length;

  nodeInfo.innerHTML = `
    <div class="node-info-title">${node.label}</div>
    <div class="node-info-row">
      <span class="node-info-label">ID</span>
      <span>${nodeId}</span>
    </div>
    <div class="node-info-row">
      <span class="node-info-label">Connections</span>
      <span>${connections}</span>
    </div>
    <div class="node-info-row">
      <span class="node-info-label">Position</span>
      <span>${node.x?.toFixed(0) ?? "-"}, ${node.y?.toFixed(0) ?? "-"}</span>
    </div>
  `;
}

/**
 * Add a new node
 */
function addNode(x?: number, y?: number) {
  const id = nodeCounter++;
  const label = `Node ${id}`;

  nodes.push({ id, label, x, y });

  // Connect to 1-3 random existing nodes
  if (nodes.length > 1) {
    const connectionCount = Math.min(1 + Math.floor(Math.random() * 3), nodes.length - 1);
    const targets = new Set<number>();

    while (targets.size < connectionCount) {
      const targetIdx = Math.floor(Math.random() * (nodes.length - 1));
      targets.add(targetIdx);
    }

    for (const targetIdx of targets) {
      edges.push({ source: id, target: nodes[targetIdx].id });
    }
  }

  // Reload graph
  reloadGraph();
  logEvent(`Added node ${label}`);
}

/**
 * Delete selected node
 */
function deleteSelectedNode() {
  if (selectedNodeId === null) return;

  // Find and remove node
  const idx = nodes.findIndex((n) => n.id === selectedNodeId);
  if (idx === -1) return;

  const node = nodes[idx];
  nodes.splice(idx, 1);

  // Remove connected edges
  for (let i = edges.length - 1; i >= 0; i--) {
    if (edges[i].source === selectedNodeId || edges[i].target === selectedNodeId) {
      edges.splice(i, 1);
    }
  }

  selectedNodeId = null;
  updateNodeInfo(null);
  reloadGraph();
  logEvent(`Deleted ${node.label}`);
}

/**
 * Clear all nodes and edges
 */
function clearGraph() {
  nodes.length = 0;
  edges.length = 0;
  selectedNodeId = null;
  updateNodeInfo(null);
  reloadGraph();
  logEvent("Cleared graph");
}

/**
 * Reload graph with current data
 */
async function reloadGraph() {
  if (!graph) return;

  const data: GraphInput = {
    nodes: nodes.map((n) => ({
      id: n.id,
      label: n.label,
      x: n.x,
      y: n.y,
      color: n.id === selectedNodeId ? "#ff6b6b" : "#4ecdc4",
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
    })),
  };

  await graph.load(data);

  if (isSimulationRunning) {
    graph.startSimulation();
  }
}

/**
 * Generate initial sample graph
 */
function generateSampleGraph() {
  nodeCounter = 0;
  nodes.length = 0;
  edges.length = 0;

  // Create 20 nodes
  for (let i = 0; i < 20; i++) {
    nodes.push({
      id: nodeCounter++,
      label: `Node ${i}`,
    });
  }

  // Create edges (tree with some cross-links)
  for (let i = 1; i < 20; i++) {
    edges.push({ source: Math.floor(i / 2), target: i });

    // Add some cross-links
    if (Math.random() > 0.7 && i > 3) {
      edges.push({ source: Math.floor(Math.random() * i), target: i });
    }
  }
}

/**
 * Initialize
 */
async function init() {
  // Create graph
  graph = await createHeroineGraph({
    canvas,
    config: {},
  });

  // Event listeners
  graph.on("node:click", (event: NodeClickEvent) => {
    selectedNodeId = event.nodeId;
    updateNodeInfo(event.nodeId);
    logEvent(`Clicked ${nodes.find((n) => n.id === event.nodeId)?.label}`);
  });

  graph.on("node:hoverenter", (event: NodeHoverEnterEvent) => {
    canvas.style.cursor = "pointer";
  });

  graph.on("node:hoverleave", (_event: NodeHoverLeaveEvent) => {
    canvas.style.cursor = "default";
  });

  graph.on("node:dragend", (event: NodeDragEndEvent) => {
    const node = nodes.find((n) => n.id === event.nodeId);
    if (node) {
      node.x = event.position.x;
      node.y = event.position.y;
    }
    logEvent(`Dragged ${node?.label}`);
  });

  graph.on("selection:change", (event: SelectionChangeEvent) => {
    if (event.nodes.length > 0) {
      selectedNodeId = event.nodes[0];
      updateNodeInfo(selectedNodeId);
    } else {
      selectedNodeId = null;
      updateNodeInfo(null);
    }
  });

  graph.on("background:click", (event: BackgroundClickEvent) => {
    // Clear selection on background click
    selectedNodeId = null;
    updateNodeInfo(null);
  });

  graph.on("simulation:tick", (event) => {
    alphaValue.textContent = event.alpha.toFixed(3);
  });

  // Double-click to add node
  canvas.addEventListener("dblclick", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert screen to graph coordinates (simplified)
    addNode(x - rect.width / 2, y - rect.height / 2);
  });

  // Button handlers
  btnAddNode.addEventListener("click", () => addNode());
  btnDeleteNode.addEventListener("click", deleteSelectedNode);
  btnClear.addEventListener("click", clearGraph);
  btnFit.addEventListener("click", () => graph?.fitToView());

  btnSimulation.addEventListener("click", () => {
    isSimulationRunning = !isSimulationRunning;
    if (isSimulationRunning) {
      graph?.startSimulation();
      btnSimulation.textContent = "Simulation On";
      btnSimulation.classList.add("active");
    } else {
      graph?.stopSimulation();
      btnSimulation.textContent = "Simulation Off";
      btnSimulation.classList.remove("active");
    }
  });

  // Force sliders
  repulsionSlider.addEventListener("input", () => {
    const value = parseInt(repulsionSlider.value);
    graph?.setForceConfig({ repulsion: value });
  });

  attractionSlider.addEventListener("input", () => {
    const value = parseInt(attractionSlider.value);
    graph?.setForceConfig({ attraction: value / 100 });
  });

  // Layer toggles
  layerHeatmap.addEventListener("change", () => {
    if (layerHeatmap.checked) {
      graph?.enableHeatmap();
    } else {
      graph?.disableHeatmap();
    }
  });

  layerContours.addEventListener("change", () => {
    graph?.toggleLayer("contours", layerContours.checked);
  });

  layerLabels.addEventListener("change", () => {
    graph?.toggleLayer("labels", layerLabels.checked);
  });

  // Generate and load initial graph
  generateSampleGraph();
  await reloadGraph();
  graph.fitToView();

  logEvent("Initialized");
}

// Start
init().catch(console.error);
