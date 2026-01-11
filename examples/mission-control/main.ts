/**
 * HeroineGraph Mission Control Example
 *
 * A comprehensive demo with interactive controls for all layers and settings.
 *
 * @module
 */

import {
  createHeroineGraph,
  getSupportInfo,
  type GraphInput,
  type HeroineGraph,
} from "../../packages/core/mod.ts";

// ============================================================================
// Types
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
}

// ============================================================================
// Random Data Generation
// ============================================================================

/**
 * Generate a random color in hex format
 */
function randomColor(): string {
  const hue = Math.random() * 360;
  const saturation = 60 + Math.random() * 30;
  const lightness = 50 + Math.random() * 20;
  return hslToHex(hue, saturation, lightness);
}

/**
 * Convert HSL to hex color
 */
function hslToHex(h: number, s: number, l: number): string {
  s /= 100;
  l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

/**
 * Generate random graph data with clusters
 */
function generateRandomGraph(nodeCount: number): GraphInput {
  const nodes: {
    id: string;
    x: number;
    y: number;
    radius: number;
    color: string;
    metadata: Record<string, unknown>;
  }[] = [];
  const edges: { source: string; target: string; width: number; color: string }[] = [];

  // Create clusters for more interesting layouts
  const clusterCount = Math.max(3, Math.floor(Math.sqrt(nodeCount) / 5));
  const nodesPerCluster = Math.ceil(nodeCount / clusterCount);

  // Generate cluster centers
  const clusterCenters: { x: number; y: number; color: string }[] = [];
  for (let i = 0; i < clusterCount; i++) {
    const angle = (i / clusterCount) * Math.PI * 2;
    const radius = 200 + Math.random() * 100;
    clusterCenters.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      color: randomColor(),
    });
  }

  // Generate nodes in clusters
  for (let i = 0; i < nodeCount; i++) {
    const clusterIdx = i % clusterCount;
    const cluster = clusterCenters[clusterIdx];

    // Random offset within cluster
    const spread = 50 + Math.sqrt(nodesPerCluster) * 5;
    const offsetAngle = Math.random() * Math.PI * 2;
    const offsetRadius = Math.random() * spread;

    nodes.push({
      id: `n${i}`,
      x: cluster.x + Math.cos(offsetAngle) * offsetRadius,
      y: cluster.y + Math.sin(offsetAngle) * offsetRadius,
      radius: 3 + Math.random() * 4,
      color: cluster.color,
      metadata: {
        label: `Node ${i}`,
        cluster: clusterIdx,
      },
    });
  }

  // Generate edges - more within clusters, fewer between
  const edgeCount = Math.floor(nodeCount * 1.5);
  const edgeSet = new Set<string>();

  for (let i = 0; i < edgeCount; i++) {
    let sourceIdx: number, targetIdx: number;

    // 70% chance of intra-cluster edge
    if (Math.random() < 0.7) {
      const clusterIdx = Math.floor(Math.random() * clusterCount);
      const clusterStart = clusterIdx * nodesPerCluster;
      const clusterEnd = Math.min((clusterIdx + 1) * nodesPerCluster, nodeCount);
      const clusterSize = clusterEnd - clusterStart;

      if (clusterSize < 2) continue;

      sourceIdx = clusterStart + Math.floor(Math.random() * clusterSize);
      targetIdx = clusterStart + Math.floor(Math.random() * clusterSize);
    } else {
      // Inter-cluster edge
      sourceIdx = Math.floor(Math.random() * nodeCount);
      targetIdx = Math.floor(Math.random() * nodeCount);
    }

    if (sourceIdx === targetIdx) continue;

    const edgeKey = sourceIdx < targetIdx
      ? `${sourceIdx}-${targetIdx}`
      : `${targetIdx}-${sourceIdx}`;

    if (edgeSet.has(edgeKey)) continue;
    edgeSet.add(edgeKey);

    edges.push({
      source: `n${sourceIdx}`,
      target: `n${targetIdx}`,
      width: 0.5 + Math.random() * 1,
      color: "#ffffff22",
    });
  }

  return { nodes, edges };
}

// ============================================================================
// UI Helpers
// ============================================================================

function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}

function $input(id: string): HTMLInputElement {
  return document.getElementById(id) as HTMLInputElement;
}

function $select(id: string): HTMLSelectElement {
  return document.getElementById(id) as HTMLSelectElement;
}

function formatNumber(n: number): string {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return n.toString();
}

// ============================================================================
// Main Application
// ============================================================================

async function main(): Promise<void> {
  const state: AppState = {
    graph: null,
    graphData: null,
    nodeCount: 0,
    edgeCount: 0,
    drawerOpen: false,
    lastFrameTime: performance.now(),
    frameCount: 0,
    fps: 0,
  };

  // Check WebGPU support
  const support = await getSupportInfo();
  if (!support.supported) {
    $("loading-overlay").classList.add("hidden");
    $("error-text").textContent = support.reason || "WebGPU is not supported in this browser.";
    $("error-message").classList.add("visible");
    return;
  }

  // Initialize graph
  const canvas = $("graph-canvas") as HTMLCanvasElement;
  const container = $("graph-container");
  const rect = container.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;

  try {
    state.graph = await createHeroineGraph({ canvas, debug: false });
  } catch (err) {
    console.error("Failed to initialize graph:", err);
    $("loading-overlay").classList.add("hidden");
    $("error-text").textContent = `Initialization failed: ${err}`;
    $("error-message").classList.add("visible");
    return;
  }

  // Hide loading overlay
  $("loading-overlay").classList.add("hidden");

  // Handle resize
  window.addEventListener("resize", () => {
    const r = container.getBoundingClientRect();
    canvas.width = r.width * window.devicePixelRatio;
    canvas.height = r.height * window.devicePixelRatio;
    state.graph?.resize(canvas.width, canvas.height);
  });

  // FPS tracking
  function updateFPS(): void {
    const now = performance.now();
    state.frameCount++;

    if (now - state.lastFrameTime >= 1000) {
      state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
      state.frameCount = 0;
      state.lastFrameTime = now;
      $("stat-fps").textContent = state.fps.toString();
    }

    requestAnimationFrame(updateFPS);
  }
  requestAnimationFrame(updateFPS);

  // ========================================================================
  // Node Buttons
  // ========================================================================

  async function loadNodes(count: number): Promise<void> {
    if (!state.graph) return;

    const data = generateRandomGraph(count);
    await state.graph.load(data);

    state.graphData = data;
    state.nodeCount = data.nodes.length;
    state.edgeCount = data.edges.length;

    $("stat-nodes").textContent = formatNumber(state.nodeCount);
    $("stat-edges").textContent = formatNumber(state.edgeCount);

    // Update labels if enabled
    if ($input("labels-enabled").checked) {
      updateLabels();
    }
  }

  /**
   * Add nodes to the existing graph (instead of replacing)
   */
  async function addNodes(count: number): Promise<void> {
    if (!state.graph) return;

    // Generate new graph data
    const newData = generateRandomGraph(count);

    // If no existing graph, just load it
    if (!state.graphData || state.graphData.nodes.length === 0) {
      await loadNodes(count);
      return;
    }

    // Merge with existing data
    const existingNodeCount = state.graphData.nodes.length;

    // Remap new node IDs to avoid collisions
    const remappedNodes = newData.nodes.map((node, i) => ({
      ...node,
      id: `n${existingNodeCount + i}`,
      metadata: {
        ...node.metadata as Record<string, unknown>,
        label: `Node ${existingNodeCount + i}`,
      },
    }));

    const remappedEdges = newData.edges.map((edge) => ({
      ...edge,
      source: `n${existingNodeCount + parseInt(String(edge.source).slice(1), 10)}`,
      target: `n${existingNodeCount + parseInt(String(edge.target).slice(1), 10)}`,
    }));

    // Combine existing and new data
    const mergedData: GraphInput = {
      nodes: [...state.graphData.nodes, ...remappedNodes],
      edges: [...state.graphData.edges, ...remappedEdges],
    };

    await state.graph.load(mergedData);

    state.graphData = mergedData;
    state.nodeCount = mergedData.nodes.length;
    state.edgeCount = mergedData.edges.length;

    $("stat-nodes").textContent = formatNumber(state.nodeCount);
    $("stat-edges").textContent = formatNumber(state.edgeCount);

    // Update labels if enabled
    if ($input("labels-enabled").checked) {
      updateLabels();
    }

    // Reheat simulation to integrate new nodes
    state.graph.restartSimulation();
  }

  // Helper to update labels from current graph data
  function updateLabels(): void {
    if (!state.graph || !state.graphData) return;

    const labels = state.graphData.nodes.map((node, index) => ({
      nodeId: index,
      text: String((node.metadata as Record<string, unknown>)?.label ?? node.id),
      x: node.x ?? 0,
      y: node.y ?? 0,
      priority: 1 - index / state.graphData!.nodes.length, // Higher priority for earlier nodes
    }));

    state.graph.setLabels(labels);
  }

  // Add node button handlers - ADD nodes to existing graph
  document.querySelectorAll(".node-btn[data-count]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const count = parseInt((btn as HTMLElement).dataset.count || "100", 10);
      addNodes(count);
    });
  });

  // Clear button
  $("clear-btn").addEventListener("click", () => {
    if (!state.graph) return;
    state.graph.load({ nodes: [], edges: [] });
    state.nodeCount = 0;
    state.edgeCount = 0;
    $("stat-nodes").textContent = "0";
    $("stat-edges").textContent = "0";
  });

  // ========================================================================
  // Drawer Toggle
  // ========================================================================

  const drawer = $("drawer");
  const drawerToggle = $("drawer-toggle");

  function toggleDrawer(): void {
    state.drawerOpen = !state.drawerOpen;
    drawer.classList.toggle("open", state.drawerOpen);
    drawerToggle.classList.toggle("open", state.drawerOpen);
  }

  drawerToggle.addEventListener("click", toggleDrawer);

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;

    if (e.code === "Space") {
      e.preventDefault();
      toggleDrawer();
    } else if (e.code === "KeyR") {
      e.preventDefault();
      state.graph?.restartSimulation();
    }
  });

  // ========================================================================
  // Layer Controls
  // ========================================================================

  // Heatmap
  $input("heatmap-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      state.graph?.enableHeatmap({
        colorScale: $select("heatmap-colorscale").value as any,
        radius: parseFloat($input("heatmap-radius").value),
        intensity: parseFloat($input("heatmap-intensity").value),
        opacity: parseFloat($input("heatmap-opacity").value),
      });
    } else {
      state.graph?.disableHeatmap();
    }
  });

  $select("heatmap-colorscale").addEventListener("change", (e) => {
    if (!$input("heatmap-enabled").checked) return;
    state.graph?.setHeatmapConfig({
      colorScale: (e.target as HTMLSelectElement).value as any,
    });
  });

  function setupSlider(
    id: string,
    valueId: string,
    callback: (value: number) => void,
    format: (v: number) => string = (v) => v.toString(),
  ): void {
    const input = $input(id);
    const valueEl = $(valueId);

    input.addEventListener("input", () => {
      const value = parseFloat(input.value);
      valueEl.textContent = format(value);
      callback(value);
    });
  }

  setupSlider(
    "heatmap-radius",
    "heatmap-radius-val",
    (v) => $input("heatmap-enabled").checked && state.graph?.setHeatmapConfig({ radius: v }),
  );

  setupSlider(
    "heatmap-intensity",
    "heatmap-intensity-val",
    (v) => $input("heatmap-enabled").checked && state.graph?.setHeatmapConfig({ intensity: v }),
    (v) => v.toFixed(1),
  );

  setupSlider(
    "heatmap-opacity",
    "heatmap-opacity-val",
    (v) => $input("heatmap-enabled").checked && state.graph?.setHeatmapConfig({ opacity: v }),
    (v) => v.toFixed(2),
  );

  // Contours (requires heatmap for density texture)
  $input("contour-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      // Contours need heatmap's density texture - enable heatmap if not already
      if (!$input("heatmap-enabled").checked) {
        // Enable heatmap with minimal opacity so contours work
        state.graph?.enableHeatmap({
          colorScale: $select("heatmap-colorscale").value as "viridis",
          radius: parseFloat($input("heatmap-radius").value),
          intensity: parseFloat($input("heatmap-intensity").value),
          opacity: 0.0, // Invisible but generates density texture
        });
      }

      const thresholdCount = parseInt($input("contour-thresholds").value, 10);
      const thresholds = Array.from(
        { length: thresholdCount },
        (_, i) => (i + 1) / (thresholdCount + 1),
      );
      state.graph?.enableContour({
        strokeWidth: parseFloat($input("contour-width").value),
        strokeColor: $input("contour-color").value,
        thresholds,
      });
    } else {
      state.graph?.disableContour();
    }
  });

  setupSlider(
    "contour-width",
    "contour-width-val",
    (v) => $input("contour-enabled").checked && state.graph?.setContourConfig({ strokeWidth: v }),
  );

  $input("contour-color").addEventListener("input", (e) => {
    if (!$input("contour-enabled").checked) return;
    state.graph?.setContourConfig({ strokeColor: (e.target as HTMLInputElement).value });
  });

  setupSlider(
    "contour-thresholds",
    "contour-thresholds-val",
    (v) => {
      if (!$input("contour-enabled").checked) return;
      const thresholds = Array.from({ length: v }, (_, i) => (i + 1) / (v + 1));
      state.graph?.setContourConfig({ thresholds });
    },
  );

  // Metaballs
  $input("metaball-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      state.graph?.enableMetaball({
        fillColor: $input("metaball-colorscale").value,
        threshold: parseFloat($input("metaball-threshold").value),
        opacity: parseFloat($input("metaball-opacity").value),
      });
    } else {
      state.graph?.disableMetaball();
    }
  });

  $input("metaball-colorscale").addEventListener("input", (e) => {
    if (!$input("metaball-enabled").checked) return;
    state.graph?.setMetaballConfig({
      fillColor: (e.target as HTMLInputElement).value,
    });
  });

  setupSlider(
    "metaball-threshold",
    "metaball-threshold-val",
    (v) => $input("metaball-enabled").checked && state.graph?.setMetaballConfig({ threshold: v }),
    (v) => v.toFixed(2),
  );

  setupSlider(
    "metaball-opacity",
    "metaball-opacity-val",
    (v) => $input("metaball-enabled").checked && state.graph?.setMetaballConfig({ opacity: v }),
    (v) => v.toFixed(2),
  );

  // Labels
  $input("labels-enabled").addEventListener("change", async (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      await state.graph?.enableLabels({
        fontSize: parseInt($input("labels-fontsize").value, 10),
        fontColor: $input("labels-color").value,
        maxLabels: parseInt($input("labels-max").value, 10),
      });
      // Set label data after enabling the layer
      updateLabels();
    } else {
      state.graph?.disableLabels();
    }
  });

  setupSlider(
    "labels-fontsize",
    "labels-fontsize-val",
    (v) => $input("labels-enabled").checked && state.graph?.setLabelsConfig({ fontSize: v }),
  );

  $input("labels-color").addEventListener("input", (e) => {
    if (!$input("labels-enabled").checked) return;
    state.graph?.setLabelsConfig({ fontColor: (e.target as HTMLInputElement).value });
  });

  setupSlider(
    "labels-max",
    "labels-max-val",
    (v) => $input("labels-enabled").checked && state.graph?.setLabelsConfig({ maxLabels: v }),
  );

  // ========================================================================
  // Edge Flow Controls
  // ========================================================================

  // Flow enable/disable toggle
  $input("flow-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      const preset = $select("flow-preset").value;
      if (preset !== "none") {
        state.graph?.setEdgeFlowPreset(preset as any);
      } else {
        // Enable with current slider values
        state.graph?.setEdgeFlowConfig({
          layer1: {
            enabled: true,
            pulseWidth: parseFloat($input("flow-width").value),
            pulseCount: parseInt($input("flow-count").value, 10),
            speed: parseFloat($input("flow-speed").value),
            waveShape: "sine",
            brightness: parseFloat($input("flow-brightness").value),
            fade: 0.3,
            color: null,
          },
        });
      }
    } else {
      state.graph?.disableEdgeFlow();
    }
  });

  // Flow preset selector
  $select("flow-preset").addEventListener("change", (e) => {
    const preset = (e.target as HTMLSelectElement).value;
    if (preset === "none") {
      state.graph?.disableEdgeFlow();
      $input("flow-enabled").checked = false;
    } else {
      state.graph?.setEdgeFlowPreset(preset as any);
      $input("flow-enabled").checked = true;

      // Update sliders to match preset values
      const config = state.graph?.getEdgeFlowConfig();
      if (config) {
        $input("flow-width").value = config.layer1.pulseWidth.toString();
        $("flow-width-val").textContent = config.layer1.pulseWidth.toFixed(2);
        $input("flow-count").value = config.layer1.pulseCount.toString();
        $("flow-count-val").textContent = config.layer1.pulseCount.toString();
        $input("flow-speed").value = config.layer1.speed.toString();
        $("flow-speed-val").textContent = config.layer1.speed.toFixed(2);
        $input("flow-brightness").value = config.layer1.brightness.toString();
        $("flow-brightness-val").textContent = config.layer1.brightness.toFixed(1);
      }
    }
  });

  // Flow parameter sliders
  function updateFlowFromSliders(): void {
    if (!$input("flow-enabled").checked) return;

    state.graph?.setEdgeFlowConfig({
      layer1: {
        enabled: true,
        pulseWidth: parseFloat($input("flow-width").value),
        pulseCount: parseInt($input("flow-count").value, 10),
        speed: parseFloat($input("flow-speed").value),
        waveShape: "sine",
        brightness: parseFloat($input("flow-brightness").value),
        fade: 0.3,
        color: null,
      },
    });
  }

  setupSlider(
    "flow-width",
    "flow-width-val",
    () => updateFlowFromSliders(),
    (v) => v.toFixed(2),
  );

  setupSlider(
    "flow-count",
    "flow-count-val",
    () => updateFlowFromSliders(),
  );

  setupSlider(
    "flow-speed",
    "flow-speed-val",
    () => updateFlowFromSliders(),
    (v) => v.toFixed(2),
  );

  setupSlider(
    "flow-brightness",
    "flow-brightness-val",
    () => updateFlowFromSliders(),
    (v) => v.toFixed(1),
  );

  // ========================================================================
  // Force Configuration
  // ========================================================================

  // Algorithm selector
  const algorithmSelect = $("force-algorithm") as HTMLSelectElement;
  const algorithmVal = $("force-algorithm-val");
  algorithmSelect.addEventListener("change", () => {
    const type = algorithmSelect.value as "n2" | "barnes-hut" | "force-atlas2" | "density";
    try {
      state.graph?.setForceAlgorithm(type);
      const algorithms = state.graph?.getAvailableAlgorithms() ?? [];
      const selected = algorithms.find((a) => a.id === type);
      algorithmVal.textContent = selected?.name ?? type;
      console.log(`Switched to algorithm: ${selected?.name ?? type}`);
    } catch (e) {
      console.error("Failed to set algorithm:", e);
    }
  });

  // Repulsion strength
  setupSlider(
    "force-repulsion",
    "force-repulsion-val",
    (v) => state.graph?.setForceConfig({ repulsionStrength: -v }),
  );

  // Spring strength
  setupSlider(
    "force-spring",
    "force-spring-val",
    (v) => state.graph?.setForceConfig({ springStrength: v }),
    (v) => v.toFixed(2),
  );

  // Link distance (spring length)
  setupSlider(
    "force-distance",
    "force-distance-val",
    (v) => state.graph?.setForceConfig({ springLength: v }),
  );

  // Center gravity
  setupSlider(
    "force-center",
    "force-center-val",
    (v) => state.graph?.setForceConfig({ centerStrength: v }),
    (v) => v.toFixed(2),
  );

  // Velocity decay (damping)
  setupSlider(
    "force-damping",
    "force-damping-val",
    (v) => state.graph?.setForceConfig({ velocityDecay: v }),
    (v) => v.toFixed(2),
  );

  // ========================================================================
  // Simulation Controls
  // ========================================================================

  $("sim-stop").addEventListener("click", () => state.graph?.stopSimulation());
  $("sim-start").addEventListener("click", () => state.graph?.startSimulation());
  $("sim-restart").addEventListener("click", () => state.graph?.restartSimulation());

  // ========================================================================
  // Load Initial Data
  // ========================================================================

  await loadNodes(1000);
}

// Start the application
main().catch(console.error);
