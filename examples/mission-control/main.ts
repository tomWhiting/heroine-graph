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
  type EdgeInput,
  type GraphInput,
  type HeroineGraph,
  type NodeInput,
} from "../../packages/core/mod.ts";

// ============================================================================
// Types
// ============================================================================

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

interface AppState {
  graph: HeroineGraph | null;
  graphData: GraphInput | null;
  nodeCount: number;
  edgeCount: number;
  drawerOpen: boolean;
  lastFrameTime: number;
  frameCount: number;
  fps: number;
  // Codebase-specific data
  codebaseData: CodebaseData | null;
  codebaseMetrics: Map<number, NodeMetrics> | null;
  // Current edge styling state (tracks GPU state, not original data)
  currentEdgeColors: Float32Array | null;
}

// ============================================================================
// Random Data Generation
// ============================================================================

/**
 * Convert HSL to RGB (0-1 range)
 * @param h Hue (0-1)
 * @param s Saturation (0-1)
 * @param l Lightness (0-1)
 * @returns [r, g, b] each in 0-1 range
 */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };
  return [f(0), f(8), f(4)];
}

// ============================================================================
// Structured Graph Generator - Tech Company Knowledge Graph
// ============================================================================

/** Node type definitions with visual properties */
const NODE_TYPES = {
  person: {
    color: "#4facfe", // Blue
    radius: { min: 4, max: 8 },
    roles: ["Engineer", "Designer", "PM", "Data Scientist", "DevOps", "Manager", "Director", "VP"],
  },
  team: {
    color: "#00f2fe", // Cyan
    radius: { min: 10, max: 14 },
    names: ["Platform", "Frontend", "Backend", "Data", "ML", "Infra", "Security", "Mobile", "Growth", "Core"],
  },
  project: {
    color: "#fa709a", // Pink
    radius: { min: 6, max: 10 },
    prefixes: ["Project", "Initiative", "Feature", "Sprint"],
    statuses: ["active", "planning", "completed", "on-hold"],
  },
  document: {
    color: "#fee140", // Yellow
    radius: { min: 3, max: 5 },
    types: ["RFC", "Spec", "Design Doc", "Runbook", "Postmortem", "README"],
  },
  service: {
    color: "#a18cd1", // Purple
    radius: { min: 7, max: 12 },
    prefixes: ["api-", "svc-", "worker-", "gateway-", "cache-", "db-"],
    tiers: ["tier-1", "tier-2", "tier-3"],
  },
} as const;

/**
 * Generate structured graph data modeling a tech company
 * Distribution roughly: 60% people, 5% teams, 15% projects, 15% documents, 5% services
 */
function generateRandomGraph(nodeCount: number): GraphInput {
  // Pre-allocate arrays with known sizes for better performance
  const counts = {
    team: Math.max(3, Math.floor(nodeCount * 0.05)),
    service: Math.max(5, Math.floor(nodeCount * 0.05)),
    project: Math.floor(nodeCount * 0.15),
    document: Math.floor(nodeCount * 0.15),
    person: 0,
  };
  counts.person = nodeCount - counts.team - counts.service - counts.project - counts.document;

  const nodes: Array<{
    id: string;
    x: number;
    y: number;
    radius: number;
    color: string;
    metadata: Record<string, unknown>;
  }> = [];
  const edges: Array<{ source: string; target: string; width: number; color: string }> = [];
  const edgeSet = new Set<string>();

  // Track nodes by type - essential for type-based filtering/styling
  const nodesByType: Record<string, string[]> = {
    person: [],
    team: [],
    project: [],
    document: [],
    service: [],
  };

  // Layout constants
  const layoutRadius = 150 + Math.sqrt(nodeCount) * 8;
  const teamPositions: { x: number; y: number }[] = [];

  // Pre-compute which projects belong to which team (avoid O(n²) in person loop)
  const teamProjectIndices: number[][] = [];
  for (let t = 0; t < counts.team; t++) {
    const indices: number[] = [];
    for (let p = t; p < counts.project; p += counts.team) {
      indices.push(p);
    }
    teamProjectIndices[t] = indices;
  }

  // Default edge color (neutral gray with transparency)
  const DEFAULT_EDGE_COLOR = "#80808066";

  // Helper to add an edge with deduplication
  const addEdge = (src: string, tgt: string, _typeColor: string, width: number) => {
    const key = src < tgt ? `${src}-${tgt}` : `${tgt}-${src}`;
    if (edgeSet.has(key) || src === tgt) return;
    edgeSet.add(key);
    // Use default gray color for all edges - "Color by Type" button applies type colors
    edges.push({ source: src, target: tgt, width, color: DEFAULT_EDGE_COLOR });
  };

  // Helper to get random radius for a type
  const getRadius = (type: keyof typeof NODE_TYPES) => {
    const r = NODE_TYPES[type].radius;
    return r.min + Math.random() * (r.max - r.min);
  };

  // 1. Generate teams (cluster centers)
  const teamDef = NODE_TYPES.team;
  for (let i = 0; i < counts.team; i++) {
    const angle = (i / counts.team) * Math.PI * 2;
    const dist = layoutRadius * (0.6 + Math.random() * 0.4);
    const x = Math.cos(angle) * dist;
    const y = Math.sin(angle) * dist;
    teamPositions.push({ x, y });

    const id = `team-${i}`;
    nodes.push({
      id,
      x,
      y,
      radius: getRadius("team"),
      color: teamDef.color,
      metadata: {
        type: "team",
        name: teamDef.names[i % teamDef.names.length],
        size: 0, // Will be updated after people are generated
        budget: Math.floor(Math.random() * 10000000) + 500000,
        headcount: 0,
        costCenter: `CC-${1000 + i}`,
      },
    });
    nodesByType.team.push(id);
  }

  // 2. Generate services (near center, interconnected)
  const serviceDef = NODE_TYPES.service;
  for (let i = 0; i < counts.service; i++) {
    const angle = (i / counts.service) * Math.PI * 2 + Math.random() * 0.3;
    const dist = layoutRadius * 0.3 * (0.5 + Math.random() * 0.5);

    const id = `svc-${i}`;
    const tier = serviceDef.tiers[Math.floor(Math.random() * serviceDef.tiers.length)];
    nodes.push({
      id,
      x: Math.cos(angle) * dist,
      y: Math.sin(angle) * dist,
      radius: getRadius("service"),
      color: serviceDef.color,
      metadata: {
        type: "service",
        name: `${serviceDef.prefixes[i % serviceDef.prefixes.length]}${i}`,
        tier,
        uptime: 99 + Math.random() * 0.99,
        latencyP99: Math.floor(Math.random() * 500) + 10,
        requestsPerSec: Math.floor(Math.random() * 10000),
        errorRate: Math.random() * 0.05,
        owner: `team-${i % counts.team}`,
        language: ["Go", "Rust", "Python", "TypeScript", "Java"][Math.floor(Math.random() * 5)],
        lastDeployed: Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000,
      },
    });
    nodesByType.service.push(id);

    // Service dependencies (1-3 per service)
    if (i > 0) {
      const depCount = 1 + Math.floor(Math.random() * 3);
      for (let d = 0; d < depCount; d++) {
        const targetIdx = Math.floor(Math.random() * i);
        addEdge(`svc-${i}`, `svc-${targetIdx}`, "#a18cd166", 1.5);
      }
    }
  }

  // Store project positions for document placement
  const projectPositions: { x: number; y: number }[] = [];

  // 3. Generate projects (distributed across teams)
  const projectDef = NODE_TYPES.project;
  for (let i = 0; i < counts.project; i++) {
    const teamIdx = i % counts.team;
    const teamPos = teamPositions[teamIdx];
    const angle = Math.random() * Math.PI * 2;
    const dist = 30 + Math.random() * 40;
    const x = teamPos.x + Math.cos(angle) * dist;
    const y = teamPos.y + Math.sin(angle) * dist;

    projectPositions.push({ x, y });

    const id = `proj-${i}`;
    const status = projectDef.statuses[Math.floor(Math.random() * projectDef.statuses.length)];
    nodes.push({
      id,
      x,
      y,
      radius: getRadius("project"),
      color: projectDef.color,
      metadata: {
        type: "project",
        name: `${projectDef.prefixes[Math.floor(Math.random() * projectDef.prefixes.length)]} ${String.fromCharCode(65 + (i % 26))}${Math.floor(i / 26) || ""}`,
        status,
        team: `team-${teamIdx}`,
        priority: Math.floor(Math.random() * 5) + 1,
        startDate: Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000,
        dueDate: Date.now() + Math.random() * 180 * 24 * 60 * 60 * 1000,
        completionPercent: status === "completed" ? 100 : Math.floor(Math.random() * 95),
        riskLevel: ["low", "medium", "high", "critical"][Math.floor(Math.random() * 4)],
        storyPoints: Math.floor(Math.random() * 100) + 5,
      },
    });
    nodesByType.project.push(id);

    // Projects use 1-3 services
    const serviceCount = 1 + Math.floor(Math.random() * 3);
    for (let s = 0; s < serviceCount; s++) {
      const svcIdx = Math.floor(Math.random() * counts.service);
      addEdge(id, `svc-${svcIdx}`, "#a18cd144", 1.2);
    }
  }

  // 4. Generate documents (attached to projects)
  const docDef = NODE_TYPES.document;
  for (let i = 0; i < counts.document; i++) {
    const projectIdx = i % counts.project;
    const pos = projectPositions[projectIdx];
    const angle = Math.random() * Math.PI * 2;
    const dist = 15 + Math.random() * 20;

    const id = `doc-${i}`;
    const docType = docDef.types[Math.floor(Math.random() * docDef.types.length)];
    nodes.push({
      id,
      x: pos.x + Math.cos(angle) * dist,
      y: pos.y + Math.sin(angle) * dist,
      radius: getRadius("document"),
      color: docDef.color,
      metadata: {
        type: "document",
        name: `${docType}-${i}`,
        docType,
        project: `proj-${projectIdx}`,
        lastUpdated: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000,
        version: `${Math.floor(Math.random() * 5) + 1}.${Math.floor(Math.random() * 10)}`,
        wordCount: Math.floor(Math.random() * 5000) + 200,
        status: ["draft", "review", "approved", "archived"][Math.floor(Math.random() * 4)],
        confidentiality: ["public", "internal", "confidential", "restricted"][Math.floor(Math.random() * 4)],
      },
    });
    nodesByType.document.push(id);

    // Link document to its project
    addEdge(id, `proj-${projectIdx}`, "#fee14033", 0.6);
  }

  // Track team sizes as we add people
  const teamSizes = new Array(counts.team).fill(0);

  // 5. Generate people (the bulk - distributed across teams)
  const personDef = NODE_TYPES.person;
  for (let i = 0; i < counts.person; i++) {
    const teamIdx = i % counts.team;
    const teamPos = teamPositions[teamIdx];
    teamSizes[teamIdx]++;

    const angle = Math.random() * Math.PI * 2;
    const dist = 20 + Math.random() * 60;
    const role = personDef.roles[Math.floor(Math.random() * personDef.roles.length)];
    const isLead = role.includes("Manager") || role.includes("Director") || role.includes("VP");

    const id = `person-${i}`;
    nodes.push({
      id,
      x: teamPos.x + Math.cos(angle) * dist,
      y: teamPos.y + Math.sin(angle) * dist,
      radius: isLead ? getRadius("person") * 1.3 : getRadius("person"),
      color: personDef.color,
      metadata: {
        type: "person",
        name: `Person ${i}`,
        role,
        team: `team-${teamIdx}`,
        isLead,
        tenure: Math.floor(Math.random() * 8) + 1,
        level: ["IC1", "IC2", "IC3", "IC4", "IC5", "M1", "M2", "D1", "VP"][Math.floor(Math.random() * 9)],
        location: ["SF", "NYC", "London", "Berlin", "Tokyo", "Sydney", "Remote"][Math.floor(Math.random() * 7)],
        startDate: Date.now() - (Math.floor(Math.random() * 8) + 1) * 365 * 24 * 60 * 60 * 1000,
        email: `person${i}@company.com`,
        slackId: `U${String(i).padStart(8, "0")}`,
        reportsTo: isLead ? null : `person-${Math.max(0, i - Math.floor(Math.random() * 10) - 1)}`,
      },
    });
    nodesByType.person.push(id);

    // Person belongs to team
    addEdge(id, `team-${teamIdx}`, "#4facfe44", 1.5);

    // Person works on 1-2 projects (using pre-computed team project indices)
    const teamProjs = teamProjectIndices[teamIdx];
    const projectCount = 1 + Math.floor(Math.random() * 2);
    for (let p = 0; p < projectCount; p++) {
      let projIdx: number;
      if (Math.random() < 0.8 && teamProjs.length > 0) {
        projIdx = teamProjs[Math.floor(Math.random() * teamProjs.length)];
      } else {
        projIdx = Math.floor(Math.random() * counts.project);
      }
      addEdge(id, `proj-${projIdx}`, "#fa709a44", 1.0);
    }

    // Some people author documents (20%)
    if (Math.random() < 0.2) {
      const docIdx = Math.floor(Math.random() * counts.document);
      addEdge(id, `doc-${docIdx}`, "#fee14044", 0.8);
    }

    // Collaboration edges (15%)
    if (Math.random() < 0.15 && i > 0) {
      const collaboratorIdx = Math.floor(Math.random() * i);
      addEdge(id, `person-${collaboratorIdx}`, "#4facfe22", 0.5);
    }
  }

  // Update team sizes in metadata
  for (let i = 0; i < counts.team; i++) {
    nodes[i].metadata.size = teamSizes[i];
    nodes[i].metadata.headcount = teamSizes[i];
  }

  console.log(`Generated graph: ${counts.person} people, ${counts.team} teams, ${counts.project} projects, ${counts.document} documents, ${counts.service} services`);
  console.log(`Total: ${nodes.length} nodes, ${edges.length} edges`);
  console.log(`Nodes by type:`, Object.fromEntries(Object.entries(nodesByType).map(([k, v]) => [k, v.length])));

  return { nodes, edges };
}

// ============================================================================
// Hierarchical Graph Generator
// ============================================================================

/**
 * Generate hierarchical (tree-like) graph data with optional cross-talk.
 *
 * This generator creates a tree structure where each node has a parent (except root),
 * making it ideal for testing hierarchical layout algorithms like Relativity Atlas.
 *
 * @param nodeCount - Total number of nodes to generate
 * @param branchFactor - Average number of children per parent (2-10)
 * @param crossTalk - Percentage of additional random edges (0-100)
 */
function generateHierarchicalGraph(
  nodeCount: number,
  branchFactor: number = 4,
  crossTalk: number = 0,
): GraphInput {
  const nodes: Array<{
    id: string;
    x: number;
    y: number;
    radius: number;
    color: string;
    metadata: Record<string, unknown>;
  }> = [];
  const edges: Array<{ source: string; target: string; width: number; color: string }> = [];
  const edgeSet = new Set<string>();

  // Color palette for depth levels - distinct colors for visual hierarchy
  const depthColors = [
    "#ff6b6b", // Root - red
    "#feca57", // Level 1 - yellow
    "#48dbfb", // Level 2 - cyan
    "#1dd1a1", // Level 3 - green
    "#5f27cd", // Level 4 - purple
    "#ff9ff3", // Level 5 - pink
    "#54a0ff", // Level 6 - blue
    "#00d2d3", // Level 7 - teal
  ];

  const TREE_EDGE_COLOR = "#4facfe44"; // Slightly visible for tree edges
  const CROSS_EDGE_COLOR = "#ff6b6b33"; // Red tint for cross-talk edges

  // Helper to add an edge with deduplication
  const addEdge = (src: string, tgt: string, color: string, width: number): boolean => {
    const key = src < tgt ? `${src}-${tgt}` : `${tgt}-${src}`;
    if (edgeSet.has(key) || src === tgt) return false;
    edgeSet.add(key);
    edges.push({ source: src, target: tgt, width, color });
    return true;
  };

  // Track node depths and parent relationships for cross-talk validation
  const nodeParents: (number | undefined)[] = [];

  // Build tree structure level by level using BFS
  let currentIdx = 0;

  // Create root node at center
  nodes.push({
    id: "node-0",
    x: 0,
    y: 0,
    radius: 15,
    color: depthColors[0],
    metadata: { type: "root", depth: 0, name: "Root" },
  });
  nodeParents.push(undefined);
  currentIdx++;

  // Build tree using BFS-like approach
  let currentLevel = [0];
  let depth = 1;

  while (currentIdx < nodeCount && currentLevel.length > 0) {
    const nextLevel: number[] = [];

    for (const parentIdx of currentLevel) {
      if (currentIdx >= nodeCount) break;

      // ASYMMETRIC branching for organic, real-world-like trees:
      // - Some nodes are leaves (0 children) - ~20% chance
      // - Some are hubs (2x-3x branch factor) - ~10% chance
      // - Most vary between 1 and branchFactor - ~70%
      // This creates uneven, realistic hierarchies like actual codebases/orgs
      const roll = Math.random();
      let numChildren: number;
      if (roll < 0.2) {
        // Leaf node - stop branching here (20% chance)
        numChildren = 0;
      } else if (roll < 0.3) {
        // Hub node - extra children (10% chance)
        numChildren = Math.floor(branchFactor * (1.5 + Math.random() * 1.5));
      } else {
        // Normal variation - 1 to branchFactor children (70% chance)
        numChildren = Math.max(1, Math.floor(1 + Math.random() * branchFactor));
      }
      const parentNode = nodes[parentIdx];

      for (let c = 0; c < numChildren && currentIdx < nodeCount; c++) {
        // Position children in a fan pattern around parent
        const angle = (c / numChildren) * Math.PI * 2 + Math.random() * 0.3;
        const dist = 30 + depth * 20 + Math.random() * 20;

        const x = (parentNode.x ?? 0) + Math.cos(angle) * dist;
        const y = (parentNode.y ?? 0) + Math.sin(angle) * dist;

        const nodeId = `node-${currentIdx}`;
        nodes.push({
          id: nodeId,
          x,
          y,
          radius: Math.max(4, 12 - depth), // Smaller nodes at deeper levels
          color: depthColors[depth % depthColors.length],
          metadata: {
            type: `level-${depth}`,
            depth,
            parent: parentIdx,
            name: `Node ${currentIdx}`,
          },
        });

        nodeParents.push(parentIdx);

        // Tree edge (primary hierarchy)
        addEdge(`node-${parentIdx}`, nodeId, TREE_EDGE_COLOR, 1.5);

        nextLevel.push(currentIdx);
        currentIdx++;
      }
    }

    currentLevel = nextLevel;
    depth++;

    // Safety: max depth to prevent infinite loops with small branch factors
    if (depth > 20) break;
  }

  const treeEdgeCount = edges.length;

  // Add cross-talk edges (random connections between unrelated nodes)
  if (crossTalk > 0 && nodes.length > 2) {
    // Number of cross-talk edges: percentage of tree edges
    // At 100%, we add as many cross-talk edges as tree edges
    const crossTalkCount = Math.floor(treeEdgeCount * (crossTalk / 100));

    let added = 0;
    let attempts = 0;
    const maxAttempts = crossTalkCount * 10;

    while (added < crossTalkCount && attempts < maxAttempts) {
      attempts++;

      // Pick two random nodes
      const a = Math.floor(Math.random() * nodes.length);
      const b = Math.floor(Math.random() * nodes.length);

      // Skip if same node or direct parent-child relationship
      if (a === b) continue;
      const aParent = nodeParents[a];
      const bParent = nodeParents[b];
      if (aParent === b || bParent === a) continue;

      // Add cross-talk edge
      if (addEdge(`node-${a}`, `node-${b}`, CROSS_EDGE_COLOR, 0.8)) {
        added++;
      }
    }

    console.log(`Added ${added} cross-talk edges (${crossTalk}% of ${treeEdgeCount} tree edges)`);
  }

  console.log(`Generated hierarchical graph: ${nodes.length} nodes, ${edges.length} edges, max depth ${depth - 1}`);

  return { nodes, edges };
}

/**
 * Add cross-talk edges to an existing graph in real-time.
 * This simulates how a codebase evolves: starting pure hierarchy, gaining cross-connections over time.
 *
 * @param graphData - Current graph data to modify
 * @param count - Number of cross-talk edges to add
 * @returns Updated graph data with new edges
 */
function addCrossTalkEdges(
  graphData: GraphInput,
  count: number
): { newEdges: Array<{ source: string; target: string; width: number; color: string }>; totalAdded: number } {
  const CROSS_EDGE_COLOR = "#ff6b6b55"; // Red tint for cross-talk edges

  const existingEdges = new Set<string>();
  for (const edge of graphData.edges) {
    const key = edge.source < edge.target ? `${edge.source}-${edge.target}` : `${edge.target}-${edge.source}`;
    existingEdges.add(key);
  }

  const newEdges: Array<{ source: string; target: string; width: number; color: string }> = [];
  const nodeCount = graphData.nodes.length;

  if (nodeCount < 2) return { newEdges, totalAdded: 0 };

  let added = 0;
  let attempts = 0;
  const maxAttempts = count * 20;

  while (added < count && attempts < maxAttempts) {
    attempts++;

    // Pick two random nodes
    const a = Math.floor(Math.random() * nodeCount);
    const b = Math.floor(Math.random() * nodeCount);

    if (a === b) continue;

    const nodeA = graphData.nodes[a];
    const nodeB = graphData.nodes[b];
    const srcId = String(nodeA.id);
    const tgtId = String(nodeB.id);

    // Check if edge already exists
    const key = srcId < tgtId ? `${srcId}-${tgtId}` : `${tgtId}-${srcId}`;
    if (existingEdges.has(key)) continue;

    // Skip direct parent-child relationships (for hierarchical graphs)
    const aParent = (nodeA.metadata as Record<string, unknown>)?.parent;
    const bParent = (nodeB.metadata as Record<string, unknown>)?.parent;
    const aIndex = graphData.nodes.findIndex(n => n.id === nodeA.id);
    const bIndex = graphData.nodes.findIndex(n => n.id === nodeB.id);
    if (aParent === bIndex || bParent === aIndex) continue;

    existingEdges.add(key);
    newEdges.push({
      source: srcId,
      target: tgtId,
      width: 0.8,
      color: CROSS_EDGE_COLOR,
    });
    added++;
  }

  console.log(`Added ${added} cross-talk edges (${attempts} attempts)`);
  return { newEdges, totalAdded: added };
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
    codebaseData: null,
    codebaseMetrics: null,
    currentEdgeColors: null,
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

    // Get graph generator settings from UI controls
    const graphType = ($("graph-type") as HTMLSelectElement)?.value ?? "corporate";
    const branchFactor = parseInt(($("branch-factor") as HTMLInputElement)?.value ?? "4", 10);

    // Generate graph based on selected type
    // Note: cross-talk is now added live after generation, not at generation time
    let data: GraphInput;
    if (graphType === "hierarchical") {
      data = generateHierarchicalGraph(count, branchFactor, 0); // No initial cross-talk
    } else {
      data = generateRandomGraph(count);
    }

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

    // Clear codebase data when loading random data
    state.codebaseData = null;
    state.codebaseMetrics = null;

    // Initialize edge color tracking with the gray default color
    // so opacity slider works correctly from the start
    const trackedColors = new Float32Array(state.edgeCount * 4);
    const defaultR = 0x80 / 255;  // 0.5 gray from #808080
    const defaultG = 0x80 / 255;
    const defaultB = 0x80 / 255;
    const defaultA = 0x66 / 255;  // ~0.4 alpha from #80808066
    for (let i = 0; i < state.edgeCount; i++) {
      trackedColors[i * 4 + 0] = defaultR;
      trackedColors[i * 4 + 1] = defaultG;
      trackedColors[i * 4 + 2] = defaultB;
      trackedColors[i * 4 + 3] = defaultA;
    }
    state.currentEdgeColors = trackedColors;
  }

  /**
   * Load the codebase dataset with real metrics
   */
  async function loadCodebase(): Promise<void> {
    if (!state.graph) return;

    try {
      // Fetch the codebase JSON
      const response = await fetch("./data/codebase.json");
      const codebase: CodebaseData = await response.json();

      // Store for later use with streams
      state.codebaseData = codebase;
      state.codebaseMetrics = new Map();

      // Helper to convert RGBA (0-1) to hex string
      const rgbaToHex = (rgba: number[]): string => {
        const r = Math.round((rgba[0] || 0) * 255);
        const g = Math.round((rgba[1] || 0) * 255);
        const b = Math.round((rgba[2] || 0) * 255);
        return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
      };

      // Convert to GraphInput format - note: don't set colors here, use type-based styling
      const graphData: GraphInput = {
        nodes: codebase.nodes.map((node) => {
          // Store metrics by node index
          state.codebaseMetrics!.set(node.id, node.metrics);

          return {
            id: node.id,
            label: node.label,
            type: node.type, // Type will be used for type-based styling
          };
        }),
        edges: codebase.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          type: edge.type, // Type will be used for type-based styling
        })),
      };

      await state.graph.load(graphData);

      // Apply type-based styling from the codebase data
      // This demonstrates the setNodeTypeStyles() and setEdgeTypeStyles() APIs
      const nodeStyles: Record<string, { color: string; size?: number }> = {};
      for (const [typeName, style] of Object.entries(codebase.typeStyles)) {
        nodeStyles[typeName] = {
          color: rgbaToHex(style.color),
          size: 1.0, // Default size multiplier
        };
      }
      state.graph.setNodeTypeStyles(nodeStyles);

      const edgeStyles: Record<string, { color: string; width?: number; opacity?: number }> = {};
      for (const [typeName, style] of Object.entries(codebase.edgeTypeStyles)) {
        edgeStyles[typeName] = {
          color: rgbaToHex(style.color),
          opacity: style.color[3] || 0.5,
        };
      }
      state.graph.setEdgeTypeStyles(edgeStyles);

      // Track the type-based colors so opacity slider works correctly
      // Compute colors based on edge types and store them
      const edgeCount = graphData.edges.length;
      const trackedColors = new Float32Array(edgeCount * 4);
      for (let i = 0; i < edgeCount; i++) {
        const edge = graphData.edges[i];
        const edgeType = edge.type as string | undefined;
        const typeStyle = edgeType ? codebase.edgeTypeStyles[edgeType] : null;
        if (typeStyle) {
          trackedColors[i * 4 + 0] = typeStyle.color[0] || 0.5;
          trackedColors[i * 4 + 1] = typeStyle.color[1] || 0.5;
          trackedColors[i * 4 + 2] = typeStyle.color[2] || 0.5;
          trackedColors[i * 4 + 3] = typeStyle.color[3] || 0.5;
        } else {
          // Default gray
          trackedColors[i * 4 + 0] = 0.5;
          trackedColors[i * 4 + 1] = 0.5;
          trackedColors[i * 4 + 2] = 0.5;
          trackedColors[i * 4 + 3] = 0.5;
        }
      }
      state.currentEdgeColors = trackedColors;

      state.graphData = graphData;
      state.nodeCount = graphData.nodes.length;
      state.edgeCount = graphData.edges.length;

      $("stat-nodes").textContent = formatNumber(state.nodeCount);
      $("stat-edges").textContent = formatNumber(state.edgeCount);

      // Update labels if enabled
      if ($input("labels-enabled").checked) {
        updateLabels();
      }

      // Note: Codebase streams (errors, warnings, complexity, lines) are available
      // but not auto-created. Use the heatmap data source dropdown or value stream
      // controls to enable them explicitly.

      console.log(`Loaded codebase: ${codebase.name} (${state.nodeCount} files, ${state.edgeCount} dependencies)`);
      console.log("Tip: Use heatmap data source dropdown to visualize codebase metrics (errors, warnings, etc.)");
    } catch (err) {
      console.error("Failed to load codebase:", err);
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

    // Build a map from old IDs to new IDs for remapping edges
    const idMap = new Map<string, string>();
    const existingNodeCount = state.graphData.nodes.length;

    // Remap new node IDs to avoid collisions - use a batch prefix
    const batchId = Date.now();
    const remappedNodes = newData.nodes.map((node, i) => {
      const oldId = String(node.id);
      const newId = `b${batchId}_${oldId}`;
      idMap.set(oldId, newId);
      return {
        ...node,
        id: newId,
        metadata: {
          ...node.metadata as Record<string, unknown>,
          originalId: node.id,
          batchIndex: existingNodeCount + i,
        },
      };
    });

    // Remap edges to use new IDs
    const remappedEdges = newData.edges.map((edge) => ({
      ...edge,
      source: idMap.get(String(edge.source)) ?? edge.source,
      target: idMap.get(String(edge.target)) ?? edge.target,
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
    btn.addEventListener("click", async () => {
      const count = parseInt((btn as HTMLElement).dataset.count || "100", 10);
      console.log(`Adding ${count} nodes...`);
      try {
        await addNodes(count);
        console.log(`Added ${count} nodes successfully`);
      } catch (err) {
        console.error(`Failed to add ${count} nodes:`, err);
      }
    });
  });

  // Generate button handlers - RESTART with new clean graph
  document.querySelectorAll(".gen-btn[data-count]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const count = parseInt((btn as HTMLElement).dataset.count || "100", 10);
      console.log(`Generating ${count} nodes (fresh graph)...`);
      try {
        await loadNodes(count);
        console.log(`Generated ${count} nodes successfully`);
      } catch (err) {
        console.error(`Failed to generate ${count} nodes:`, err);
      }
    });
  });

  // Add cross-talk button handler - add edges incrementally (no reload!)
  $("add-crosstalk-btn").addEventListener("click", async () => {
    if (!state.graph || !state.graphData || state.graphData.nodes.length === 0) {
      console.log("No graph loaded - generate one first");
      return;
    }

    const count = parseInt(($("crosstalk-count") as HTMLInputElement)?.value ?? "50", 10);
    console.log(`Adding ${count} cross-talk edges (incremental)...`);

    try {
      // Generate new cross-talk edges
      const { newEdges, totalAdded } = addCrossTalkEdges(state.graphData, count);

      if (totalAdded === 0) {
        console.log("No cross-talk edges could be added (graph may be fully connected)");
        return;
      }

      // Use incremental addEdges instead of full reload
      const results = await state.graph.addEdges(newEdges);
      const added = results.filter((r) => r !== undefined).length;

      // Track in graphData for future cross-talk generation
      state.graphData = {
        ...state.graphData,
        edges: [...state.graphData.edges, ...newEdges],
      };

      state.edgeCount += added;
      $("stat-edges").textContent = formatNumber(state.edgeCount);

      console.log(`Added ${added} cross-talk edges incrementally`);
    } catch (err) {
      console.error("Failed to add cross-talk edges:", err);
    }
  });

  // Clear button
  $("clear-btn").addEventListener("click", () => {
    if (!state.graph) return;
    state.graph.load({ nodes: [], edges: [] });
    state.nodeCount = 0;
    state.edgeCount = 0;
    state.codebaseData = null;
    state.codebaseMetrics = null;
    state.currentEdgeColors = null;
    $("stat-nodes").textContent = "0";
    $("stat-edges").textContent = "0";
  });

  // Load codebase button
  $("load-codebase-btn").addEventListener("click", async () => {
    console.log("Loading codebase dataset...");
    try {
      await loadCodebase();
    } catch (err) {
      console.error("Failed to load codebase:", err);
    }
  });

  // ========================================================================
  // Incremental Mutation Buttons
  // ========================================================================

  // Add 10 nodes incrementally
  $("mutate-add-nodes").addEventListener("click", async () => {
    if (!state.graph || !state.graphData) return;

    const newNodes: NodeInput[] = [];
    const existingCount = state.nodeCount;
    for (let i = 0; i < 10; i++) {
      const id = `mutated_node_${existingCount + i}_${Date.now()}`;
      const colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96e6a1", "#dda0dd"];
      newNodes.push({
        id,
        color: colors[i % colors.length],
        radius: 4 + Math.random() * 6,
      });
    }

    try {
      const ids = await state.graph.addNodes(newNodes);
      state.nodeCount += ids.length;
      $("stat-nodes").textContent = formatNumber(state.nodeCount);
      console.log(`Incrementally added ${ids.length} nodes`);

      // Also add some edges from new nodes to existing ones
      if (state.graphData.nodes.length > 0) {
        const edgesToAdd: EdgeInput[] = [];
        for (const nodeInput of newNodes) {
          const targetIdx = Math.floor(Math.random() * state.graphData.nodes.length);
          edgesToAdd.push({
            source: nodeInput.id,
            target: state.graphData.nodes[targetIdx].id,
            weight: 0.5,
          });
        }
        const edgeResults = await state.graph.addEdges(edgesToAdd);
        const addedEdges = edgeResults.filter((r) => r !== undefined).length;
        state.edgeCount += addedEdges;
        $("stat-edges").textContent = formatNumber(state.edgeCount);
      }

      // Track in graphData
      state.graphData = {
        ...state.graphData,
        nodes: [...state.graphData.nodes, ...newNodes],
      };
    } catch (err) {
      console.error("Failed to add nodes:", err);
    }
  });

  // Add 20 random edges incrementally
  $("mutate-add-edges").addEventListener("click", async () => {
    if (!state.graph || !state.graphData || state.graphData.nodes.length < 2) return;

    const newEdges: EdgeInput[] = [];
    const nodes = state.graphData.nodes;
    for (let i = 0; i < 20; i++) {
      const srcIdx = Math.floor(Math.random() * nodes.length);
      let tgtIdx = Math.floor(Math.random() * nodes.length);
      if (tgtIdx === srcIdx) tgtIdx = (srcIdx + 1) % nodes.length;
      newEdges.push({
        source: nodes[srcIdx].id,
        target: nodes[tgtIdx].id,
        weight: 0.3 + Math.random() * 0.7,
      });
    }

    try {
      const results = await state.graph.addEdges(newEdges);
      const added = results.filter((r) => r !== undefined).length;
      state.edgeCount += added;
      $("stat-edges").textContent = formatNumber(state.edgeCount);
      console.log(`Incrementally added ${added} edges`);
    } catch (err) {
      console.error("Failed to add edges:", err);
    }
  });

  // Remove 5 random nodes incrementally
  $("mutate-remove-nodes").addEventListener("click", async () => {
    if (!state.graph || !state.graphData || state.graphData.nodes.length < 5) return;

    const nodesToRemove = [];
    const nodesCopy = [...state.graphData.nodes];
    for (let i = 0; i < Math.min(5, nodesCopy.length); i++) {
      const idx = Math.floor(Math.random() * nodesCopy.length);
      nodesToRemove.push(nodesCopy[idx].id);
      nodesCopy.splice(idx, 1);
    }

    try {
      const removed = await state.graph.removeNodes(nodesToRemove);
      state.nodeCount -= removed;
      $("stat-nodes").textContent = formatNumber(state.nodeCount);

      // Update graphData (remove nodes and their edges)
      const removedSet = new Set(nodesToRemove);
      state.graphData = {
        nodes: state.graphData.nodes.filter((n) => !removedSet.has(n.id)),
        edges: state.graphData.edges.filter(
          (e) => !removedSet.has(e.source) && !removedSet.has(e.target),
        ),
      };
      state.edgeCount = state.graphData.edges.length;
      $("stat-edges").textContent = formatNumber(state.edgeCount);

      console.log(`Incrementally removed ${removed} nodes`);
    } catch (err) {
      console.error("Failed to remove nodes:", err);
    }
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
    const value = (e.target as HTMLSelectElement).value;
    const customColorsRow = $("heatmap-custom-colors");

    // Show/hide custom colors row
    if (value === "custom") {
      customColorsRow.style.display = "";
      // Apply custom colors immediately
      applyCustomHeatmapColors();
    } else {
      customColorsRow.style.display = "none";
      if ($input("heatmap-enabled").checked) {
        state.graph?.setHeatmapConfig({
          colorScale: value as any,
        });
      }
    }
  });

  // Custom color inputs
  function applyCustomHeatmapColors(): void {
    if (!state.graph || !$input("heatmap-enabled").checked) return;

    // Helper to convert hex to RGBA (0-1)
    const hexToRgba = (hex: string): [number, number, number, number] => {
      const r = parseInt(hex.slice(1, 3), 16) / 255;
      const g = parseInt(hex.slice(3, 5), 16) / 255;
      const b = parseInt(hex.slice(5, 7), 16) / 255;
      return [r, g, b, 1.0];
    };

    const startColor = hexToRgba($input("heatmap-color-start").value);
    const midColor = hexToRgba($input("heatmap-color-mid").value);
    const endColor = hexToRgba($input("heatmap-color-end").value);

    state.graph.setCustomHeatmapColorScale([
      { position: 0, color: startColor },
      { position: 0.5, color: midColor },
      { position: 1, color: endColor },
    ]);
    console.log("Applied custom heatmap colors");
  }

  $input("heatmap-color-start").addEventListener("input", applyCustomHeatmapColors);
  $input("heatmap-color-mid").addEventListener("input", applyCustomHeatmapColors);
  $input("heatmap-color-end").addEventListener("input", applyCustomHeatmapColors);

  // Heatmap data source (stream → heatmap binding)
  // Creates streams on-demand when user selects a data source
  $select("heatmap-datasource").addEventListener("change", (e) => {
    if (!$input("heatmap-enabled").checked) return;
    const source = (e.target as HTMLSelectElement).value;

    if (source === "density") {
      // Uniform density mode - no stream needed
      state.graph?.setHeatmapDataSource(source);
      console.log("Heatmap using uniform density");
      return;
    }

    // Stream configuration for both codebase and synthetic data
    const streamConfigs: Record<string, { domain: [number, number]; stops: Array<{ position: number; color: [number, number, number, number] }> }> = {
      errors: {
        domain: [0, state.codebaseData ? 5 : 10],
        stops: [
          { position: 0, color: [0, 0, 0, 0] },
          { position: 0.3, color: [0.8, 0.3, 0.1, 0.4] },
          { position: 0.7, color: [1, 0.2, 0.1, 0.7] },
          { position: 1, color: [1, 0.1, 0.05, 1] },
        ],
      },
      warnings: {
        domain: [0, 10],
        stops: [
          { position: 0, color: [0, 0, 0, 0] },
          { position: 0.3, color: [0.8, 0.7, 0.1, 0.3] },
          { position: 0.7, color: [1, 0.8, 0.2, 0.6] },
          { position: 1, color: [1, 0.9, 0.3, 1] },
        ],
      },
      complexity: {
        domain: [0, 30],
        stops: [
          { position: 0, color: [0, 0, 0, 0] },
          { position: 0.3, color: [0.2, 0.5, 0.8, 0.3] },
          { position: 0.6, color: [0.5, 0.3, 0.8, 0.6] },
          { position: 1, color: [0.8, 0.2, 0.9, 1] },
        ],
      },
      lines: {
        domain: [0, state.codebaseData ? 350 : 500],
        stops: [
          { position: 0, color: [0, 0, 0, 0] },
          { position: 0.3, color: [0.2, 0.6, 0.3, 0.3] },
          { position: 0.6, color: [0.3, 0.7, 0.4, 0.6] },
          { position: 1, color: [0.4, 0.9, 0.5, 1] },
        ],
      },
    };

    // Create stream if it doesn't exist
    if (state.graph && !state.graph.hasValueStream(source)) {
      const config = streamConfigs[source];
      if (config) {
        // Define the stream
        state.graph.defineValueStream({
          id: source,
          name: source.charAt(0).toUpperCase() + source.slice(1),
          colorScale: config,
          blendMode: "additive",
          opacity: 1.0,
        });

        // Populate with real codebase metrics or random synthetic data
        const data: Array<{ nodeIndex: number; value: number }> = [];

        if (state.codebaseMetrics) {
          // Use real codebase metrics
          for (const [nodeIndex, metrics] of state.codebaseMetrics) {
            const value = metrics[source as keyof NodeMetrics];
            if (value > 0) {
              data.push({ nodeIndex, value });
            }
          }
          console.log(`Created ${source} stream from codebase metrics (${data.length} values)`);
        } else {
          // Generate random data for synthetic graphs (~30% of nodes)
          const [min, max] = config.domain;
          for (let i = 0; i < state.nodeCount; i++) {
            if (Math.random() < 0.3) {
              const value = min + Math.pow(Math.random(), 2) * (max - min);
              data.push({ nodeIndex: i, value });
            }
          }
          console.log(`Created ${source} stream with ${data.length} random values`);
        }

        state.graph.setStreamValues(source, data);
      }
    }

    state.graph?.setHeatmapDataSource(source);
    console.log(`Heatmap data source: ${source}`);
  });

  // Debounce helper
  function debounce<T extends (...args: Parameters<T>) => void>(
    fn: T,
    delay: number,
  ): (...args: Parameters<T>) => void {
    let timeoutId: number | undefined;
    return (...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => fn(...args), delay) as unknown as number;
    };
  }

  function setupSlider(
    id: string,
    valueId: string,
    callback: (value: number) => void,
    format: (v: number) => string = (v) => v.toString(),
    debounceMs = 16, // ~60fps, immediate feel with slight debounce
  ): void {
    const input = $input(id);
    const valueEl = $(valueId);

    // Debounce the callback to avoid too many GPU updates
    const debouncedCallback = debounce(callback, debounceMs);

    input.addEventListener("input", () => {
      const value = parseFloat(input.value);
      // Update display immediately for responsiveness
      valueEl.textContent = format(value);
      // Debounce the actual update
      debouncedCallback(value);
    });
  }

  // Helper to set edge colors and track them in state (defined early for use by display settings)
  function setEdgeColorsWithTracking(colors: Float32Array): void {
    if (!state.graph) return;
    state.graph.setEdgeColors(colors);
    // Track current colors for opacity slider to use
    state.currentEdgeColors = new Float32Array(colors);
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
      // Contours require heatmap to be enabled for the density texture
      if (!$input("heatmap-enabled").checked) {
        console.warn("Contours require heatmap to be enabled first (for density texture)");
        (e.target as HTMLInputElement).checked = false;
        return;
      }

      const thresholdCount = parseInt($input("contour-thresholds").value, 10);
      const minThreshold = parseFloat($input("contour-min").value);
      const maxThreshold = 0.9;
      const range = maxThreshold - minThreshold;
      const thresholds = Array.from(
        { length: thresholdCount },
        (_, i) => minThreshold + (range * (i + 1)) / (thresholdCount + 1),
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

  // Helper to compute contour thresholds from UI values
  function getContourThresholds(): number[] {
    const count = parseInt($input("contour-thresholds").value, 10);
    const minThreshold = parseFloat($input("contour-min").value);
    const maxThreshold = 0.9;
    const range = maxThreshold - minThreshold;
    return Array.from(
      { length: count },
      (_, i) => minThreshold + (range * (i + 1)) / (count + 1),
    );
  }

  setupSlider(
    "contour-thresholds",
    "contour-thresholds-val",
    () => {
      if (!$input("contour-enabled").checked) return;
      state.graph?.setContourConfig({ thresholds: getContourThresholds() });
    },
  );

  setupSlider(
    "contour-min",
    "contour-min-val",
    () => {
      if (!$input("contour-enabled").checked) return;
      state.graph?.setContourConfig({ thresholds: getContourThresholds() });
    },
    (v) => v.toFixed(2),
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

  // Helper to get flow color (null if using edge color)
  function getFlowColor(): [number, number, number, number] | null {
    if ($input("flow-use-edge-color").checked) return null;
    const hex = $input("flow-color").value;
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b, 1.0];
  }

  // Helper to get layer 1 config from UI
  function getLayer1Config() {
    return {
      enabled: true,
      pulseWidth: parseFloat($input("flow-width").value),
      pulseCount: parseInt($input("flow-count").value, 10),
      speed: parseFloat($input("flow-speed").value),
      waveShape: $select("flow-wave-shape").value as "sine" | "square" | "triangle",
      brightness: parseFloat($input("flow-brightness").value),
      fade: parseFloat($input("flow-fade").value),
      color: getFlowColor(),
    };
  }

  // Helper to get layer 2 config from UI
  // Get Layer 2 flow color (null means use edge color)
  function getFlow2Color(): [number, number, number, number] | null {
    if ($input("flow2-use-edge-color").checked) {
      return null; // Use edge's color
    }
    const hex = $input("flow2-color").value;
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b, 1.0];
  }

  function getLayer2Config() {
    return {
      enabled: $input("flow2-enabled").checked,
      pulseWidth: parseFloat($input("flow2-width").value),
      pulseCount: parseInt($input("flow2-count").value, 10),
      speed: parseFloat($input("flow2-speed").value),
      waveShape: $select("flow2-wave-shape").value as "sine" | "square" | "triangle",
      brightness: parseFloat($input("flow2-brightness").value),
      fade: parseFloat($input("flow2-fade").value),
      color: getFlow2Color(),
    };
  }

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
          layer1: getLayer1Config(),
          layer2: getLayer2Config(), // enabled flag determines if it's active
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
        // Layer 1 controls
        $input("flow-width").value = config.layer1.pulseWidth.toString();
        $("flow-width-val").textContent = config.layer1.pulseWidth.toFixed(2);
        $input("flow-count").value = config.layer1.pulseCount.toString();
        $("flow-count-val").textContent = config.layer1.pulseCount.toString();
        $input("flow-speed").value = config.layer1.speed.toString();
        $("flow-speed-val").textContent = config.layer1.speed.toFixed(2);
        $input("flow-brightness").value = config.layer1.brightness.toString();
        $("flow-brightness-val").textContent = config.layer1.brightness.toFixed(1);
        $input("flow-fade").value = config.layer1.fade.toString();
        $("flow-fade-val").textContent = config.layer1.fade.toFixed(2);
        $select("flow-wave-shape").value = config.layer1.waveShape;

        // Layer 2 controls
        $input("flow2-enabled").checked = config.layer2.enabled;
        if (config.layer2.enabled) {
          $input("flow2-width").value = config.layer2.pulseWidth.toString();
          $("flow2-width-val").textContent = config.layer2.pulseWidth.toFixed(2);
          $input("flow2-count").value = config.layer2.pulseCount.toString();
          $("flow2-count-val").textContent = config.layer2.pulseCount.toString();
          $input("flow2-speed").value = config.layer2.speed.toString();
          $("flow2-speed-val").textContent = config.layer2.speed.toFixed(2);
          $input("flow2-brightness").value = config.layer2.brightness.toString();
          $("flow2-brightness-val").textContent = config.layer2.brightness.toFixed(1);
          $input("flow2-fade").value = config.layer2.fade.toString();
          $("flow2-fade-val").textContent = config.layer2.fade.toFixed(2);
          $select("flow2-wave-shape").value = config.layer2.waveShape;
        }
      }
    }
  });

  // Flow parameter updates
  function updateFlowFromSliders(): void {
    if (!$input("flow-enabled").checked) return;

    state.graph?.setEdgeFlowConfig({
      layer1: getLayer1Config(),
      layer2: getLayer2Config(), // enabled flag determines if it's active
    });
  }

  // Layer 1 sliders
  setupSlider("flow-width", "flow-width-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));
  setupSlider("flow-count", "flow-count-val", () => updateFlowFromSliders());
  setupSlider("flow-speed", "flow-speed-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));
  setupSlider("flow-brightness", "flow-brightness-val", () => updateFlowFromSliders(), (v) => v.toFixed(1));
  setupSlider("flow-fade", "flow-fade-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));

  // Wave shape and color
  $select("flow-wave-shape").addEventListener("change", () => updateFlowFromSliders());
  $input("flow-color").addEventListener("input", () => updateFlowFromSliders());
  $input("flow-use-edge-color").addEventListener("change", () => updateFlowFromSliders());

  // Layer 2 enable/disable
  $input("flow2-enabled").addEventListener("change", () => {
    if ($input("flow-enabled").checked) {
      updateFlowFromSliders();
    }
  });

  // Layer 2 sliders
  setupSlider("flow2-width", "flow2-width-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));
  setupSlider("flow2-count", "flow2-count-val", () => updateFlowFromSliders());
  setupSlider("flow2-speed", "flow2-speed-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));
  setupSlider("flow2-brightness", "flow2-brightness-val", () => updateFlowFromSliders(), (v) => v.toFixed(1));
  setupSlider("flow2-fade", "flow2-fade-val", () => updateFlowFromSliders(), (v) => v.toFixed(2));
  $select("flow2-wave-shape").addEventListener("change", () => updateFlowFromSliders());

  // Flow Layer 2 color controls
  $input("flow2-color").addEventListener("input", () => updateFlowFromSliders());
  $input("flow2-use-edge-color").addEventListener("change", () => updateFlowFromSliders());

  // ========================================================================
  // Curved Edges Controls
  // ========================================================================

  // Curved edges enable/disable toggle
  $input("curved-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      state.graph?.enableCurvedEdges(
        parseInt($input("curved-segments").value, 10),
        parseFloat($input("curved-weight").value),
      );
      // Apply default curvature to all edges
      applyDefaultCurvature();
    } else {
      state.graph?.disableCurvedEdges();
    }
  });

  // Apply default curvature to all edges
  function applyDefaultCurvature(): void {
    if (!state.graph || state.edgeCount === 0) return;
    const curvature = parseFloat($input("curved-curvature").value);
    const curvatures = new Float32Array(state.edgeCount);
    for (let i = 0; i < state.edgeCount; i++) {
      curvatures[i] = curvature;
    }
    state.graph.setEdgeCurvatures(curvatures);
  }

  // Curved edges sliders
  setupSlider(
    "curved-segments",
    "curved-segments-val",
    (v) => {
      if ($input("curved-enabled").checked) {
        state.graph?.setCurvedEdges({ segments: v });
      }
    },
  );

  setupSlider(
    "curved-weight",
    "curved-weight-val",
    (v) => {
      if ($input("curved-enabled").checked) {
        state.graph?.setCurvedEdges({ weight: v });
      }
    },
    (v) => v.toFixed(2),
  );

  setupSlider(
    "curved-curvature",
    "curved-curvature-val",
    () => {
      if ($input("curved-enabled").checked) {
        applyDefaultCurvature();
      }
    },
    (v) => v.toFixed(2),
  );

  // Randomize curvatures button
  $("curved-randomize").addEventListener("click", () => {
    if (!state.graph || state.edgeCount === 0) return;
    const curvatures = new Float32Array(state.edgeCount);
    for (let i = 0; i < state.edgeCount; i++) {
      curvatures[i] = (Math.random() - 0.5) * 0.6; // Range -0.3 to 0.3
    }
    state.graph.setEdgeCurvatures(curvatures);
    if (!$input("curved-enabled").checked) {
      $input("curved-enabled").checked = true;
      state.graph.enableCurvedEdges();
    }
  });

  // ========================================================================
  // Node Borders Controls
  // ========================================================================

  // Border enable/disable toggle
  $input("border-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    if (enabled) {
      state.graph?.enableNodeBorder(
        parseFloat($input("border-width").value),
        ($input("border-color") as HTMLInputElement).value,
      );
    } else {
      state.graph?.disableNodeBorder();
    }
  });

  // Border width slider
  setupSlider(
    "border-width",
    "border-width-val",
    (v) => {
      if ($input("border-enabled").checked) {
        state.graph?.setNodeBorder({ width: v });
      }
    },
    (v) => v.toFixed(1),
  );

  // Border color picker
  ($input("border-color") as HTMLInputElement).addEventListener("input", (e) => {
    if ($input("border-enabled").checked) {
      state.graph?.setNodeBorder({ color: (e.target as HTMLInputElement).value });
    }
  });

  // ========================================================================
  // Force Configuration
  // ========================================================================

  // Algorithm selector
  const algorithmSelect = $("force-algorithm") as HTMLSelectElement;
  const algorithmVal = $("force-algorithm-val");
  const raControls = $("relativity-controls");
  const ttControls = $("tidy-tree-controls");
  const llControls = document.getElementById("linlog-controls");
  const tfdpControls = document.getElementById("t-fdp-controls");
  const commControls = document.getElementById("community-controls");
  const cbControls = document.getElementById("codebase-controls");

  algorithmSelect.addEventListener("change", () => {
    const type = algorithmSelect.value as "n2" | "barnes-hut" | "force-atlas2" | "density" | "relativity-atlas" | "tidy-tree" | "linlog" | "t-fdp" | "community" | "codebase";
    try {
      state.graph?.setForceAlgorithm(type);
      const algorithms = state.graph?.getAvailableAlgorithms() ?? [];
      const selected = algorithms.find((a) => a.id === type);
      algorithmVal.textContent = selected?.name ?? type;
      console.log(`Switched to algorithm: ${selected?.name ?? type}`);

      // Show/hide algorithm-specific controls
      if (raControls) {
        raControls.style.display = type === "relativity-atlas" ? "block" : "none";
      }
      if (ttControls) {
        ttControls.style.display = type === "tidy-tree" ? "block" : "none";
      }
      if (llControls) {
        llControls.style.display = type === "linlog" ? "block" : "none";
      }
      if (tfdpControls) {
        tfdpControls.style.display = type === "t-fdp" ? "block" : "none";
      }
      if (commControls) {
        commControls.style.display = type === "community" ? "block" : "none";
      }
      if (cbControls) {
        cbControls.style.display = type === "codebase" ? "block" : "none";
      }

      // Auto-compute tree layout when switching to tidy-tree
      if (type === "tidy-tree" && state.graph) {
        try {
          state.graph.computeTreeLayout();
          console.log("Auto-computed tree layout on algorithm switch");
        } catch (err) {
          console.warn("Could not auto-compute tree layout:", err);
        }
      }

      // Auto-compute community layout when switching to community
      if (type === "community" && state.graph) {
        try {
          state.graph.computeCommunityLayout();
          console.log("Auto-computed community layout on algorithm switch");
        } catch (err) {
          console.warn("Could not auto-compute community layout:", err);
        }
      }

      // Auto-compute codebase layout when switching to codebase
      if (type === "codebase" && state.graph) {
        try {
          // Map node types to categories if codebase data is available
          let categories: Uint8Array | undefined;
          if (state.codebaseData) {
            const nodeBound = state.graph.nodeCount;
            categories = new Uint8Array(nodeBound);
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
          }
          state.graph.computeCodebaseLayout(categories);
          console.log("Auto-computed codebase layout on algorithm switch");
        } catch (err) {
          console.warn("Could not auto-compute codebase layout:", err);
        }
      }
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

  // Max repulsion distance
  setupSlider(
    "force-max-dist",
    "force-max-dist-val",
    (v) => state.graph?.setForceConfig({ repulsionDistanceMax: v }),
  );

  // Barnes-Hut theta (accuracy)
  setupSlider(
    "force-theta",
    "force-theta-val",
    (v) => state.graph?.setForceConfig({ theta: v }),
    (v) => v.toFixed(2),
  );

  // Max velocity
  setupSlider(
    "force-max-vel",
    "force-max-vel-val",
    (v) => state.graph?.setForceConfig({ maxVelocity: v }),
  );

  // Pin root node (node 0 stays at center)
  $input("pin-root-node").addEventListener("change", (e) => {
    const pinned = (e.target as HTMLInputElement).checked;
    state.graph?.setForceConfig({ pinnedNode: pinned ? 0 : 0xFFFFFFFF });
    console.log(`Pin root node: ${pinned ? "enabled" : "disabled"}`);
  });

  // ========================================================================
  // Relativity Atlas Controls
  // ========================================================================

  // Base mass
  setupSlider(
    "ra-base-mass",
    "ra-base-mass-val",
    (v) => state.graph?.setForceConfig({ relativityBaseMass: v }),
    (v) => v.toFixed(1),
  );

  // Child mass factor
  setupSlider(
    "ra-child-factor",
    "ra-child-factor-val",
    (v) => state.graph?.setForceConfig({ relativityChildMassFactor: v }),
    (v) => v.toFixed(2),
  );

  // Mass exponent
  setupSlider(
    "ra-mass-exp",
    "ra-mass-exp-val",
    (v) => state.graph?.setForceConfig({ relativityMassExponent: v }),
    (v) => v.toFixed(2),
  );

  // Gravity curve selector
  const gravityCurveSelect = $("ra-gravity-curve") as HTMLSelectElement;
  const gravityExpCard = $("ra-gravity-exp-card");

  gravityCurveSelect.addEventListener("change", () => {
    const curve = gravityCurveSelect.value as "linear" | "inverse" | "soft" | "custom";
    state.graph?.setForceConfig({ relativityGravityCurve: curve });

    // Show/hide gravity exponent slider based on curve selection
    if (gravityExpCard) {
      gravityExpCard.style.opacity = curve === "custom" ? "1" : "0.5";
    }
    console.log(`Gravity curve: ${curve}`);
  });

  // Gravity exponent (for custom curve)
  setupSlider(
    "ra-gravity-exp",
    "ra-gravity-exp-val",
    (v) => state.graph?.setForceConfig({ relativityGravityExponent: v }),
    (v) => v.toFixed(2),
  );

  // Max siblings
  setupSlider(
    "ra-max-siblings",
    "ra-max-siblings-val",
    (v) => state.graph?.setForceConfig({ relativityMaxSiblings: Math.floor(v) }),
  );

  // Parent-child multiplier
  setupSlider(
    "ra-parent-child",
    "ra-parent-child-val",
    (v) => state.graph?.setForceConfig({ relativityParentChildMultiplier: v }),
    (v) => v.toFixed(2),
  );

  // Cousin repulsion toggle (2-hop: same grandparent)
  $input("ra-cousin-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    state.graph?.setForceConfig({ relativityCousinRepulsion: enabled });
    console.log(`Cousin repulsion: ${enabled ? "enabled" : "disabled"}`);
  });

  // Cousin strength
  setupSlider(
    "ra-cousin-strength",
    "ra-cousin-strength-val",
    (v) => state.graph?.setForceConfig({ relativityCousinStrength: v }),
    (v) => v.toFixed(2),
  );

  // Phantom zones toggle (mass-based collision boundaries)
  $input("ra-phantom-enabled").addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    state.graph?.setForceConfig({ relativityPhantomZone: enabled });
    console.log(`Phantom zones: ${enabled ? "enabled" : "disabled"}`);
  });

  // Density repulsion (cross-subtree repulsion via density field)
  setupSlider(
    "ra-density-repulsion",
    "ra-density-repulsion-val",
    (v) => state.graph?.setForceConfig({ relativityDensityRepulsion: v }),
    (v) => v.toFixed(2),
  );

  // Phantom multiplier
  setupSlider(
    "ra-phantom-mult",
    "ra-phantom-mult-val",
    (v) => state.graph?.setForceConfig({ relativityPhantomMultiplier: v }),
    (v) => v.toFixed(2),
  );

  // Orbit strength (radial spring pulling children to target orbit radius)
  setupSlider(
    "ra-orbit-strength",
    "ra-orbit-strength-val",
    (v) => state.graph?.setForceConfig({ relativityOrbitStrength: v }),
    (v) => v.toFixed(1),
  );

  // Tangential amplifier (>1 spreads siblings angularly around parent)
  setupSlider(
    "ra-tangential-mult",
    "ra-tangential-mult-val",
    (v) => state.graph?.setForceConfig({ relativityTangentialMultiplier: v }),
    (v) => v.toFixed(1),
  );

  // Orbit radius (base distance from parent, scales with sqrt(sibling count))
  setupSlider(
    "ra-orbit-radius",
    "ra-orbit-radius-val",
    (v) => state.graph?.setForceConfig({ relativityOrbitRadius: v }),
    (v) => v.toFixed(0),
  );

  // ========================================================================
  // LinLog Controls
  // ========================================================================

  // LinLog scaling (kr)
  setupSlider(
    "ll-scaling",
    "ll-scaling-val",
    (v) => state.graph?.setForceConfig({ linlogScaling: v }),
    (v) => v.toFixed(1),
  );

  // LinLog gravity (kg)
  setupSlider(
    "ll-gravity",
    "ll-gravity-val",
    (v) => state.graph?.setForceConfig({ linlogGravity: v }),
    (v) => v.toFixed(1),
  );

  // LinLog edge weight influence
  setupSlider(
    "ll-weight",
    "ll-weight-val",
    (v) => state.graph?.setForceConfig({ linlogEdgeWeightInfluence: v }),
    (v) => v.toFixed(2),
  );

  // LinLog strong gravity toggle
  const llStrongGravity = document.getElementById("ll-strong-gravity") as HTMLInputElement | null;
  llStrongGravity?.addEventListener("change", (e) => {
    const enabled = (e.target as HTMLInputElement).checked;
    state.graph?.setForceConfig({ linlogStrongGravity: enabled });
    console.log(`LinLog strong gravity: ${enabled ? "enabled" : "disabled"}`);
  });

  // ========================================================================
  // t-FDP Controls
  // ========================================================================

  // t-FDP gamma
  setupSlider(
    "tfdp-gamma",
    "tfdp-gamma-val",
    (v) => state.graph?.setForceConfig({ tFdpGamma: v }),
    (v) => v.toFixed(2),
  );

  // t-FDP repulsion scale
  setupSlider(
    "tfdp-repulsion",
    "tfdp-repulsion-val",
    (v) => state.graph?.setForceConfig({ tFdpRepulsionScale: v }),
    (v) => v.toFixed(2),
  );

  // t-FDP alpha (linear spring weight)
  setupSlider(
    "tfdp-alpha",
    "tfdp-alpha-val",
    (v) => state.graph?.setForceConfig({ tFdpAlpha: v }),
    (v) => v.toFixed(2),
  );

  // t-FDP beta (attractive t-force weight)
  setupSlider(
    "tfdp-beta",
    "tfdp-beta-val",
    (v) => state.graph?.setForceConfig({ tFdpBeta: v }),
    (v) => v.toFixed(2),
  );

  // ========================================================================
  // Tidy Tree Controls
  // ========================================================================

  // Level separation
  setupSlider(
    "tt-level-sep",
    "tt-level-sep-val",
    (v) => state.graph?.setForceConfig({ tidyTreeLevelSeparation: v }),
  );

  // Sibling separation
  setupSlider(
    "tt-sibling-sep",
    "tt-sibling-sep-val",
    (v) => state.graph?.setForceConfig({ tidyTreeSiblingSeparation: v }),
    (v) => v.toFixed(2),
  );

  // Subtree separation
  setupSlider(
    "tt-subtree-sep",
    "tt-subtree-sep-val",
    (v) => state.graph?.setForceConfig({ tidyTreeSubtreeSeparation: v }),
    (v) => v.toFixed(2),
  );

  // Stiffness (spring strength toward target)
  setupSlider(
    "tt-stiffness",
    "tt-stiffness-val",
    (v) => state.graph?.setForceConfig({ tidyTreeStiffness: v }),
    (v) => v.toFixed(2),
  );

  // Damping
  setupSlider(
    "tt-damping",
    "tt-damping-val",
    (v) => state.graph?.setForceConfig({ tidyTreeDamping: v }),
    (v) => v.toFixed(2),
  );

  // Coordinate mode selector
  const coordModeSelect = $("tt-coord-mode") as HTMLSelectElement;
  coordModeSelect.addEventListener("change", () => {
    const radial = coordModeSelect.value === "radial";
    state.graph?.setForceConfig({ tidyTreeRadial: radial });
    console.log(`Tidy tree coordinate mode: ${radial ? "radial" : "linear"}`);
  });

  // Recompute button - triggers WASM tree layout and uploads positions to GPU
  $("tt-recompute").addEventListener("click", () => {
    if (!state.graph) return;
    try {
      state.graph.computeTreeLayout(); // auto-detect root
      console.log("Tree layout computed and uploaded to GPU");
    } catch (err) {
      console.error("Failed to compute tree layout:", err);
    }
  });

  // ========================================================================
  // Community Layout Controls
  // ========================================================================

  // Resolution (higher = more communities)
  setupSlider(
    "comm-resolution",
    "comm-resolution-val",
    (v) => state.graph?.setForceConfig({ communityResolution: v }),
    (v) => v.toFixed(2),
  );

  // Cluster spacing
  setupSlider(
    "comm-spacing",
    "comm-spacing-val",
    (v) => state.graph?.setForceConfig({ communitySpacing: v }),
  );

  // Node spacing within community
  setupSlider(
    "comm-node-spacing",
    "comm-node-spacing-val",
    (v) => state.graph?.setForceConfig({ communityNodeSpacing: v }),
  );

  // Spread factor
  setupSlider(
    "comm-spread",
    "comm-spread-val",
    (v) => state.graph?.setForceConfig({ communitySpreadFactor: v }),
    (v) => v.toFixed(2),
  );

  // Stiffness (spring strength toward target)
  setupSlider(
    "comm-stiffness",
    "comm-stiffness-val",
    (v) => state.graph?.setForceConfig({ communityStiffness: v }),
    (v) => v.toFixed(2),
  );

  // Damping
  setupSlider(
    "comm-damping",
    "comm-damping-val",
    (v) => state.graph?.setForceConfig({ communityDamping: v }),
    (v) => v.toFixed(2),
  );

  // Recompute button - triggers Louvain detection and layout upload
  $("comm-recompute").addEventListener("click", () => {
    if (!state.graph) return;
    try {
      state.graph.computeCommunityLayout();
      console.log("Community layout computed and uploaded to GPU");
    } catch (err) {
      console.error("Failed to compute community layout:", err);
    }
  });

  // ========================================================================
  // Codebase Layout Controls
  // ========================================================================

  // Directory padding
  setupSlider(
    "cb-dir-padding",
    "cb-dir-padding-val",
    (v) => state.graph?.setForceConfig({ codebaseDirectoryPadding: v }),
  );

  // File padding
  setupSlider(
    "cb-file-padding",
    "cb-file-padding-val",
    (v) => state.graph?.setForceConfig({ codebaseFilePadding: v }),
  );

  // Spread factor
  setupSlider(
    "cb-spread",
    "cb-spread-val",
    (v) => state.graph?.setForceConfig({ codebaseSpreadFactor: v }),
    (v) => v.toFixed(2),
  );

  // Stiffness (spring strength toward target)
  setupSlider(
    "cb-stiffness",
    "cb-stiffness-val",
    (v) => state.graph?.setForceConfig({ codebaseStiffness: v }),
    (v) => v.toFixed(2),
  );

  // Damping
  setupSlider(
    "cb-damping",
    "cb-damping-val",
    (v) => state.graph?.setForceConfig({ codebaseDamping: v }),
    (v) => v.toFixed(2),
  );

  // Recompute button - triggers circle packing layout and upload
  $("cb-recompute").addEventListener("click", () => {
    if (!state.graph) return;
    try {
      // Build node categories from codebase data if available
      let categories: Uint8Array | undefined;
      if (state.codebaseData) {
        const nodeBound = state.graph.nodeCount;
        categories = new Uint8Array(nodeBound);
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
      }
      state.graph.computeCodebaseLayout(categories);
      console.log("Codebase layout computed and uploaded to GPU");
    } catch (err) {
      console.error("Failed to compute codebase layout:", err);
    }
  });

  // ========================================================================
  // Simulation Controls
  // ========================================================================

  $("sim-stop").addEventListener("click", () => state.graph?.stopSimulation());
  $("sim-start").addEventListener("click", () => state.graph?.startSimulation());
  $("sim-restart").addEventListener("click", () => state.graph?.restartSimulation());

  // ========================================================================
  // Display Settings
  // ========================================================================

  // Theme presets
  const THEMES = {
    dark: {
      background: "#0a0a0f",
      nodeColor: "#4facfe",
      edgeColor: "#808080",
      labelColor: "#ffffff",
    },
    light: {
      background: "#f5f5f5",
      nodeColor: "#3d7dd8",
      edgeColor: "#666666",
      labelColor: "#1a1a1a",
    },
    midnight: {
      background: "#0d1117",
      nodeColor: "#58a6ff",
      edgeColor: "#484f58",
      labelColor: "#c9d1d9",
    },
    contrast: {
      background: "#000000",
      nodeColor: "#00ff00",
      edgeColor: "#00ffff",
      labelColor: "#ffffff",
    },
  };

  // Apply theme preset
  function applyTheme(themeName: keyof typeof THEMES): void {
    const theme = THEMES[themeName];
    if (!theme) return;

    // Update color pickers to match theme
    $input("display-bg-color").value = theme.background;
    $input("display-edge-color").value = theme.edgeColor;

    // Apply background color
    state.graph?.setBackgroundColor(theme.background);

    // Apply label color if labels are enabled
    if ($input("labels-enabled").checked) {
      $input("labels-color").value = theme.labelColor;
      state.graph?.setLabelsConfig({ fontColor: theme.labelColor });
    }

    console.log(`Applied ${themeName} theme`);
  }

  // Theme selector
  $select("display-theme").addEventListener("change", (e) => {
    const themeName = (e.target as HTMLSelectElement).value as keyof typeof THEMES;
    applyTheme(themeName);
  });

  // Background color picker
  $input("display-bg-color").addEventListener("input", (e) => {
    state.graph?.setBackgroundColor((e.target as HTMLInputElement).value);
  });

  // Default edge color picker - applies to all edges
  $input("display-edge-color").addEventListener("input", (e) => {
    if (!state.graph || state.edgeCount === 0) return;
    const hex = (e.target as HTMLInputElement).value;
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const opacity = parseFloat($input("edge-opacity").value);

    const colors = new Float32Array(state.edgeCount * 4);
    for (let i = 0; i < state.edgeCount; i++) {
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = opacity;
    }
    setEdgeColorsWithTracking(colors);
  });

  // ========================================================================
  // Per-Item Styling Controls
  // ========================================================================

  // Type colors mapping (hex to RGB 0-1)
  const TYPE_COLORS: Record<string, [number, number, number]> = {
    person: [0.31, 0.67, 1.0], // #4facfe - Blue
    team: [0.0, 0.95, 1.0], // #00f2fe - Cyan
    project: [0.98, 0.44, 0.60], // #fa709a - Pink
    document: [1.0, 0.88, 0.25], // #fee140 - Yellow
    service: [0.63, 0.55, 0.82], // #a18cd1 - Purple
  };

  // Node size scale slider - scales all node sizes uniformly
  setupSlider(
    "node-size-scale",
    "node-size-scale-val",
    (scale) => {
      if (!state.graph || !state.graphData || state.nodeCount === 0) return;
      const sizes = new Float32Array(state.nodeCount);
      for (let i = 0; i < state.nodeCount; i++) {
        const node = state.graphData.nodes[i];
        const baseSize = (node.radius ?? 5);
        sizes[i] = baseSize * scale;
      }
      state.graph.setNodeSizes(sizes);
    },
    (v) => v.toFixed(1),
  );

  // Color by node type
  $("style-by-type").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.nodeCount === 0) return;

    const colors = new Float32Array(state.nodeCount * 4);
    for (let i = 0; i < state.nodeCount; i++) {
      const nodeType = state.graphData.nodes[i]?.metadata?.type as string;
      const [r, g, b] = TYPE_COLORS[nodeType] ?? [0.5, 0.5, 0.5];
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = 1.0;
    }
    state.graph.setNodeColors(colors);
    console.log("Colored nodes by type");
  });

  // Size by role importance
  $("style-by-role").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.nodeCount === 0) return;

    const sizes = new Float32Array(state.nodeCount);
    for (let i = 0; i < state.nodeCount; i++) {
      const meta = state.graphData.nodes[i]?.metadata;
      const type = meta?.type as string;

      // Size based on type and role
      if (type === "team") {
        sizes[i] = 12; // Teams are large
      } else if (type === "service") {
        const tier = meta?.tier as string;
        sizes[i] = tier === "tier-1" ? 14 : tier === "tier-2" ? 10 : 7;
      } else if (type === "project") {
        const priority = (meta?.priority as number) ?? 3;
        sizes[i] = 6 + (5 - priority) * 1.5; // P1 = 12, P5 = 6
      } else if (type === "person") {
        const isLead = meta?.isLead as boolean;
        sizes[i] = isLead ? 8 : 5;
      } else if (type === "document") {
        sizes[i] = 3;
      } else {
        sizes[i] = 5;
      }
    }
    state.graph.setNodeSizes(sizes);
    console.log("Sized nodes by role importance");
  });

  // Highlight leaders/important nodes
  $("style-highlight-leads").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.nodeCount === 0) return;

    const colors = new Float32Array(state.nodeCount * 4);
    const sizes = new Float32Array(state.nodeCount);

    for (let i = 0; i < state.nodeCount; i++) {
      const meta = state.graphData.nodes[i]?.metadata;
      const type = meta?.type as string;
      const isLead = meta?.isLead as boolean;
      const tier = meta?.tier as string;

      // Highlight important nodes
      const isImportant =
        type === "team" ||
        (type === "person" && isLead) ||
        (type === "service" && tier === "tier-1");

      if (isImportant) {
        // Bright gold/orange for important nodes
        colors[i * 4 + 0] = 1.0;
        colors[i * 4 + 1] = 0.8;
        colors[i * 4 + 2] = 0.2;
        colors[i * 4 + 3] = 1.0;
        sizes[i] = 12;
      } else {
        // Dim gray for others
        colors[i * 4 + 0] = 0.3;
        colors[i * 4 + 1] = 0.3;
        colors[i * 4 + 2] = 0.35;
        colors[i * 4 + 3] = 0.6;
        sizes[i] = 3;
      }
    }
    state.graph.setNodeColors(colors);
    state.graph.setNodeSizes(sizes);
    console.log("Highlighted leaders and important nodes");
  });

  // Random colors (for fun/testing)
  $("style-random-colors").addEventListener("click", () => {
    if (!state.graph || state.nodeCount === 0) return;

    const colors = new Float32Array(state.nodeCount * 4);
    for (let i = 0; i < state.nodeCount; i++) {
      const hue = Math.random();
      const sat = 0.7 + Math.random() * 0.3;
      const light = 0.5 + Math.random() * 0.2;
      const [r, g, b] = hslToRgb(hue, sat, light);
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = 1.0;
    }
    state.graph.setNodeColors(colors);
    console.log(`Randomized colors for ${state.nodeCount} nodes`);
  });

  // Reset styling (reload current graph)
  $("style-reset").addEventListener("click", async () => {
    if (!state.graphData) return;
    await state.graph?.load(state.graphData);
    console.log("Reset styling to defaults");
  });

  // ========================================================================
  // Edge Styling Controls
  // ========================================================================

  // Edge type colors - based on relationship types
  const EDGE_TYPE_COLORS: Record<string, [number, number, number]> = {
    "person-team": [0.31, 0.67, 1.0], // Blue - team membership
    "person-project": [0.98, 0.44, 0.60], // Pink - project work
    "person-document": [1.0, 0.88, 0.25], // Yellow - authorship
    "person-person": [0.31, 0.67, 1.0], // Blue dim - collaboration
    "service-service": [0.63, 0.55, 0.82], // Purple - dependencies
    "project-service": [0.63, 0.55, 0.82], // Purple dim - service usage
    "document-project": [1.0, 0.88, 0.25], // Yellow dim - doc attachment
  };

  // Helper to determine edge type from source/target IDs
  function getEdgeType(source: string, target: string): string {
    const srcType = source.split("-")[0];
    const tgtType = target.split("-")[0];
    // Normalize order (alphabetical)
    const types = [srcType, tgtType].sort();
    return `${types[0]}-${types[1]}`;
  }

  // Color edges by type
  $("edge-color-by-type").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.edgeCount === 0) return;

    const colors = new Float32Array(state.edgeCount * 4);
    for (let i = 0; i < state.edgeCount; i++) {
      const edge = state.graphData.edges[i];
      const edgeType = getEdgeType(String(edge.source), String(edge.target));
      const [r, g, b] = EDGE_TYPE_COLORS[edgeType] ?? [0.5, 0.5, 0.5];
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = parseFloat($input("edge-opacity").value);
    }
    setEdgeColorsWithTracking(colors);
    console.log("Colored edges by type");
  });

  // Width by edge type (importance-based)
  $("edge-width-by-type").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.edgeCount === 0) return;

    const baseScale = parseFloat($input("edge-width-scale").value);
    const widths = new Float32Array(state.edgeCount);

    for (let i = 0; i < state.edgeCount; i++) {
      const edge = state.graphData.edges[i];
      const edgeType = getEdgeType(String(edge.source), String(edge.target));

      // Assign width based on relationship type importance
      let width: number;
      switch (edgeType) {
        case "service-service":
          width = 2.5; // Critical infrastructure dependencies
          break;
        case "person-team":
          width = 2.0; // Team membership
          break;
        case "project-service":
          width = 1.8; // Project dependencies
          break;
        case "person-project":
          width = 1.5; // Work assignments
          break;
        case "document-project":
          width = 1.0; // Documentation links
          break;
        case "person-document":
          width = 0.8; // Authorship
          break;
        case "person-person":
          width = 0.6; // Collaboration (subtle)
          break;
        default:
          width = 1.0;
      }
      widths[i] = width * baseScale;
    }
    state.graph.setEdgeWidths(widths);
    console.log("Set edge widths by type");
  });

  // Highlight service dependencies
  $("edge-highlight-services").addEventListener("click", () => {
    if (!state.graph || !state.graphData || state.edgeCount === 0) return;

    const colors = new Float32Array(state.edgeCount * 4);
    const widths = new Float32Array(state.edgeCount);
    const baseScale = parseFloat($input("edge-width-scale").value);

    for (let i = 0; i < state.edgeCount; i++) {
      const edge = state.graphData.edges[i];
      const edgeType = getEdgeType(String(edge.source), String(edge.target));
      const isServiceEdge = edgeType === "service-service" || edgeType === "project-service";

      if (isServiceEdge) {
        // Bright cyan for service edges
        colors[i * 4 + 0] = 0.0;
        colors[i * 4 + 1] = 1.0;
        colors[i * 4 + 2] = 1.0;
        colors[i * 4 + 3] = 0.9;
        widths[i] = 3.0 * baseScale;
      } else {
        // Dim gray for non-service edges
        colors[i * 4 + 0] = 0.3;
        colors[i * 4 + 1] = 0.3;
        colors[i * 4 + 2] = 0.3;
        colors[i * 4 + 3] = 0.15;
        widths[i] = 0.5 * baseScale;
      }
    }
    setEdgeColorsWithTracking(colors);
    state.graph.setEdgeWidths(widths);
    console.log("Highlighted service dependencies");
  });

  // Edge opacity slider - apply to all edges
  // Uses tracked current colors if available, otherwise falls back to original data
  setupSlider(
    "edge-opacity",
    "edge-opacity-val",
    (v) => {
      if (!state.graph || state.edgeCount === 0) return;
      const colors = new Float32Array(state.edgeCount * 4);

      for (let i = 0; i < state.edgeCount; i++) {
        let r = 0.5, g = 0.5, b = 0.5;

        // Use tracked current colors if available (preserves any styling changes)
        if (state.currentEdgeColors && state.currentEdgeColors.length === state.edgeCount * 4) {
          r = state.currentEdgeColors[i * 4 + 0];
          g = state.currentEdgeColors[i * 4 + 1];
          b = state.currentEdgeColors[i * 4 + 2];
        } else if (state.graphData) {
          // Fall back to original loaded colors
          const existingColor = state.graphData.edges[i]?.color;
          if (existingColor && existingColor.startsWith("#")) {
            const hex = existingColor.slice(1);
            if (hex.length >= 6) {
              r = parseInt(hex.slice(0, 2), 16) / 255;
              g = parseInt(hex.slice(2, 4), 16) / 255;
              b = parseInt(hex.slice(4, 6), 16) / 255;
            }
          }
        }

        colors[i * 4 + 0] = r;
        colors[i * 4 + 1] = g;
        colors[i * 4 + 2] = b;
        colors[i * 4 + 3] = v;
      }
      setEdgeColorsWithTracking(colors);
    },
    (v) => v.toFixed(2),
  );

  // Edge width scale slider - scale all widths
  setupSlider(
    "edge-width-scale",
    "edge-width-scale-val",
    (v) => {
      if (!state.graph || !state.graphData || state.edgeCount === 0) return;
      const widths = new Float32Array(state.edgeCount);
      for (let i = 0; i < state.edgeCount; i++) {
        const edge = state.graphData.edges[i];
        const baseWidth = edge.width ?? 1.0;
        widths[i] = baseWidth * v;
      }
      state.graph.setEdgeWidths(widths);
    },
    (v) => v.toFixed(1),
  );

  // Reset edge styling
  $("edge-reset").addEventListener("click", async () => {
    if (!state.graphData) return;
    // Reload restores all original styling including edges
    await state.graph?.load(state.graphData);
    // Clear tracked edge colors (will fall back to original on next opacity change)
    state.currentEdgeColors = null;
    // Reset slider values
    $input("edge-opacity").value = "0.4";
    $("edge-opacity-val").textContent = "0.40";
    $input("edge-width-scale").value = "1";
    $("edge-width-scale-val").textContent = "1.0";
    console.log("Reset edge styling to defaults");
  });

  // ========================================================================
  // Value Stream Controls
  // ========================================================================

  // Color scale presets for different stream types
  const STREAM_PRESETS = {
    errors: {
      id: "errors",
      name: "Error Count",
      colorScale: {
        domain: [0, 10] as [number, number],
        stops: [
          { position: 0, color: [0, 0, 0, 0] as [number, number, number, number] },
          { position: 0.3, color: [0.8, 0.3, 0.1, 0.4] as [number, number, number, number] },
          { position: 0.7, color: [1, 0.2, 0.1, 0.7] as [number, number, number, number] },
          { position: 1, color: [1, 0.1, 0.05, 1] as [number, number, number, number] },
        ],
      },
    },
    activity: {
      id: "activity",
      name: "Activity Level",
      colorScale: {
        domain: [0, 1] as [number, number],
        stops: [
          { position: 0, color: [0, 0, 0, 0] as [number, number, number, number] },
          { position: 0.5, color: [0.2, 0.5, 0.9, 0.5] as [number, number, number, number] },
          { position: 1, color: [0.3, 0.7, 1, 1] as [number, number, number, number] },
        ],
      },
    },
    importance: {
      id: "importance",
      name: "Importance Score",
      colorScale: {
        domain: [0, 100] as [number, number],
        stops: [
          { position: 0, color: [0, 0, 0, 0] as [number, number, number, number] },
          { position: 0.5, color: [0.5, 0.3, 0.8, 0.5] as [number, number, number, number] },
          { position: 1, color: [0.7, 0.4, 1, 1] as [number, number, number, number] },
        ],
      },
    },
  };

  // Track current active streams
  let activeStreams: string[] = [];

  // Generate random stream values for demo
  function generateStreamData(streamId: string, nodeCount: number): Array<{ nodeIndex: number; value: number }> {
    const data: Array<{ nodeIndex: number; value: number }> = [];
    const preset = STREAM_PRESETS[streamId as keyof typeof STREAM_PRESETS];
    if (!preset) return data;

    const [min, max] = preset.colorScale.domain;
    const range = max - min;

    // Assign values to ~30% of nodes randomly
    for (let i = 0; i < nodeCount; i++) {
      if (Math.random() < 0.3) {
        // Skew toward lower values for more realistic distribution
        const value = min + Math.pow(Math.random(), 2) * range;
        data.push({ nodeIndex: i, value });
      }
    }
    return data;
  }

  // Demo stream selector
  $select("stream-demo-select").addEventListener("change", (e) => {
    const demo = (e.target as HTMLSelectElement).value;
    if (!state.graph || state.nodeCount === 0) return;

    // Clear existing streams
    state.graph.clearAllValueStreams();
    activeStreams = [];

    if (demo === "none") {
      // Reload to reset colors
      if (state.graphData) {
        state.graph.load(state.graphData);
      }
      console.log("Cleared all value streams");
      return;
    }

    const opacity = parseFloat($input("stream-opacity").value);
    const blendMode = $select("stream-blend-mode").value as "additive" | "max" | "multiply" | "replace";

    if (demo === "multi") {
      // Multi-stream demo: errors + activity
      for (const key of ["errors", "activity"] as const) {
        const preset = STREAM_PRESETS[key];
        state.graph.defineValueStream({
          ...preset,
          blendMode,
          opacity,
        });
        const data = generateStreamData(key, state.nodeCount);
        state.graph.setStreamValues(key, data);
        activeStreams.push(key);
      }
      console.log(`Enabled multi-stream demo: errors + activity (${state.nodeCount} nodes)`);
    } else {
      // Single stream demo
      const preset = STREAM_PRESETS[demo as keyof typeof STREAM_PRESETS];
      if (preset) {
        state.graph.defineValueStream({
          ...preset,
          blendMode,
          opacity,
        });
        const data = generateStreamData(demo, state.nodeCount);
        state.graph.setStreamValues(demo, data);
        activeStreams.push(demo);
        console.log(`Enabled ${demo} stream with ${data.length} values`);
      }
    }
  });

  // Stream opacity slider - updates existing streams without regenerating data
  setupSlider(
    "stream-opacity",
    "stream-opacity-val",
    (opacity) => {
      if (!state.graph || activeStreams.length === 0) return;
      // Update opacity on all active streams
      for (const streamId of activeStreams) {
        state.graph.setStreamOpacity(streamId, opacity);
      }
    },
    (v) => v.toFixed(2),
  );

  // Blend mode selector - updates existing streams without regenerating data
  $select("stream-blend-mode").addEventListener("change", (e) => {
    if (!state.graph || activeStreams.length === 0) return;
    const blendMode = (e.target as HTMLSelectElement).value as "additive" | "max" | "multiply" | "replace";
    // Update blend mode on all active streams
    for (const streamId of activeStreams) {
      state.graph.setStreamBlendMode(streamId, blendMode);
    }
    console.log(`Changed blend mode to ${blendMode}`);
  });

  // Randomize values button
  $("stream-randomize").addEventListener("click", () => {
    if (!state.graph || state.nodeCount === 0 || activeStreams.length === 0) {
      console.log("No active streams to randomize");
      return;
    }

    for (const streamId of activeStreams) {
      const data = generateStreamData(streamId, state.nodeCount);
      state.graph.setStreamValues(streamId, data);
    }
    console.log(`Randomized values for ${activeStreams.length} stream(s)`);
  });

  // Clear streams button
  $("stream-clear").addEventListener("click", async () => {
    if (!state.graph) return;

    state.graph.clearAllValueStreams();
    activeStreams = [];
    $select("stream-demo-select").value = "none";

    // Reload to reset colors
    if (state.graphData) {
      await state.graph.load(state.graphData);
    }
    console.log("Cleared all value streams");
  });

  // ========================================================================
  // Graph Generator Controls
  // ========================================================================

  // Cross-talk slider - display value as percentage
  const crossTalkSlider = $("cross-talk") as HTMLInputElement;
  const crossTalkVal = $("cross-talk-val");
  if (crossTalkSlider && crossTalkVal) {
    crossTalkSlider.addEventListener("input", () => {
      crossTalkVal.textContent = `${crossTalkSlider.value}%`;
    });
  }

  // Branch factor slider - display value as integer
  const branchFactorSlider = $("branch-factor") as HTMLInputElement;
  const branchFactorVal = $("branch-factor-val");
  if (branchFactorSlider && branchFactorVal) {
    branchFactorSlider.addEventListener("input", () => {
      branchFactorVal.textContent = branchFactorSlider.value;
    });
  }

  // ========================================================================
  // Load Config from JSON
  // ========================================================================

  /**
   * Helper to set a slider value and its display element.
   * Updates the HTML input and the visible value label.
   */
  function setSlider(
    inputId: string,
    valueId: string,
    value: number,
    format: (v: number) => string = (v) => v.toString(),
  ): void {
    const input = document.getElementById(inputId) as HTMLInputElement | null;
    const display = document.getElementById(valueId) as HTMLElement | null;
    if (input) input.value = String(value);
    if (display) display.textContent = format(value);
  }

  /**
   * Helper to set a select dropdown value.
   */
  function setSelect(selectId: string, value: string): void {
    const el = document.getElementById(selectId) as HTMLSelectElement | null;
    if (el) el.value = value;
  }

  /**
   * Helper to set a checkbox.
   */
  function setCheckbox(checkboxId: string, checked: boolean): void {
    const el = document.getElementById(checkboxId) as HTMLInputElement | null;
    if (el) el.checked = checked;
  }

  /**
   * Helper to set a color input.
   */
  function setColor(colorId: string, value: string): void {
    const el = document.getElementById(colorId) as HTMLInputElement | null;
    if (el) el.value = value;
  }

  /**
   * Convert hex color string to RGBA tuple (0-1 range).
   */
  function hexToRgbaTuple(hex: string): readonly [number, number, number, number] {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b, 1.0] as const;
  }

  /**
   * Load config.json and apply all settings to both the graph API and UI controls.
   * Falls back to defaults if config.json is missing or malformed.
   */
  async function loadConfig(): Promise<void> {
    let config: Record<string, unknown>;
    try {
      const resp = await fetch("/config.json");
      if (!resp.ok) {
        console.warn(`config.json not found (${resp.status}), using defaults`);
        await loadNodes(1000);
        return;
      }
      config = await resp.json();
      console.log("Loaded config.json:", config);
    } catch (err) {
      console.warn("Failed to load config.json, using defaults:", err);
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

    // --- Generator ---
    if (g) {
      if (g.type) setSelect("graph-type", String(g.type));
      if (g.branching != null) {
        setSlider("branch-factor", "branch-factor-val", Number(g.branching));
      }
    }

    // --- Force Algorithm ---
    if (f?.algorithm) {
      const alg = String(f.algorithm) as "n2" | "barnes-hut" | "force-atlas2" | "density" | "relativity-atlas" | "tidy-tree" | "linlog" | "t-fdp" | "community" | "codebase";
      try {
        state.graph?.setForceAlgorithm(alg);
        setSelect("force-algorithm", alg);

        // Update algorithm display name
        const algorithms = state.graph?.getAvailableAlgorithms() ?? [];
        const selected = algorithms.find((a) => a.id === alg);
        $("force-algorithm-val").textContent = selected?.name ?? alg;

        // Show/hide algorithm-specific controls
        const raCtrl = document.getElementById("relativity-controls");
        const ttCtrl = document.getElementById("tidy-tree-controls");
        const llCtrl = document.getElementById("linlog-controls");
        const tfdpCtrl = document.getElementById("t-fdp-controls");
        const commCtrl = document.getElementById("community-controls");
        const cbCtrl = document.getElementById("codebase-controls");
        if (raCtrl) raCtrl.style.display = alg === "relativity-atlas" ? "block" : "none";
        if (ttCtrl) ttCtrl.style.display = alg === "tidy-tree" ? "block" : "none";
        if (llCtrl) llCtrl.style.display = alg === "linlog" ? "block" : "none";
        if (tfdpCtrl) tfdpCtrl.style.display = alg === "t-fdp" ? "block" : "none";
        if (commCtrl) commCtrl.style.display = alg === "community" ? "block" : "none";
        if (cbCtrl) cbCtrl.style.display = alg === "codebase" ? "block" : "none";
      } catch (e) {
        console.error("Failed to set algorithm from config:", e);
      }
    }

    // --- Force Config ---
    if (f) {
      const forceConfig: Record<string, unknown> = {};

      if (f.repulsion != null) {
        forceConfig.repulsionStrength = -Math.abs(Number(f.repulsion));
        setSlider("force-repulsion", "force-repulsion-val", Math.abs(Number(f.repulsion)));
      }
      if (f.springStrength != null) {
        forceConfig.springStrength = Number(f.springStrength);
        setSlider("force-spring", "force-spring-val", Number(f.springStrength), (v) => v.toFixed(2));
      }
      if (f.linkDistance != null) {
        forceConfig.springLength = Number(f.linkDistance);
        setSlider("force-distance", "force-distance-val", Number(f.linkDistance));
      }
      if (f.centerGravity != null) {
        forceConfig.centerStrength = Number(f.centerGravity);
        setSlider("force-center", "force-center-val", Number(f.centerGravity), (v) => v.toFixed(2));
      }
      if (f.damping != null) {
        forceConfig.velocityDecay = Number(f.damping);
        setSlider("force-damping", "force-damping-val", Number(f.damping), (v) => v.toFixed(2));
      }
      if (f.maxRepelDist != null) {
        forceConfig.repulsionDistanceMax = Number(f.maxRepelDist);
        setSlider("force-max-dist", "force-max-dist-val", Number(f.maxRepelDist));
      }
      if (f.theta != null) {
        forceConfig.theta = Number(f.theta);
        setSlider("force-theta", "force-theta-val", Number(f.theta), (v) => v.toFixed(2));
      }
      if (f.maxVelocity != null) {
        forceConfig.maxVelocity = Number(f.maxVelocity);
        setSlider("force-max-vel", "force-max-vel-val", Number(f.maxVelocity));
      }
      if (f.pinRootNode != null) {
        forceConfig.pinnedNode = f.pinRootNode ? 0 : 0xFFFFFFFF;
        setCheckbox("pin-root-node", Boolean(f.pinRootNode));
      }

      // --- Relativity Atlas params ---
      if (ra) {
        if (ra.baseMass != null) {
          forceConfig.relativityBaseMass = Number(ra.baseMass);
          setSlider("ra-base-mass", "ra-base-mass-val", Number(ra.baseMass), (v) => v.toFixed(1));
        }
        if (ra.childFactor != null) {
          forceConfig.relativityChildMassFactor = Number(ra.childFactor);
          setSlider("ra-child-factor", "ra-child-factor-val", Number(ra.childFactor), (v) => v.toFixed(2));
        }
        if (ra.massExponent != null) {
          forceConfig.relativityMassExponent = Number(ra.massExponent);
          setSlider("ra-mass-exp", "ra-mass-exp-val", Number(ra.massExponent), (v) => v.toFixed(2));
        }
        if (ra.gravityCurve != null) {
          forceConfig.relativityGravityCurve = String(ra.gravityCurve);
          setSelect("ra-gravity-curve", String(ra.gravityCurve));
        }
        if (ra.gravityExponent != null) {
          forceConfig.relativityGravityExponent = Number(ra.gravityExponent);
          setSlider("ra-gravity-exp", "ra-gravity-exp-val", Number(ra.gravityExponent), (v) => v.toFixed(2));
        }
        if (ra.maxSiblings != null) {
          forceConfig.relativityMaxSiblings = Number(ra.maxSiblings);
          setSlider("ra-max-siblings", "ra-max-siblings-val", Number(ra.maxSiblings));
        }
        if (ra.parentChildMult != null) {
          forceConfig.relativityParentChildMultiplier = Number(ra.parentChildMult);
          setSlider("ra-parent-child", "ra-parent-child-val", Number(ra.parentChildMult), (v) => v.toFixed(2));
        }
        if (ra.densityRepulsion != null) {
          forceConfig.relativityDensityRepulsion = Number(ra.densityRepulsion);
          setSlider("ra-density-repulsion", "ra-density-repulsion-val", Number(ra.densityRepulsion), (v) => v.toFixed(2));
        }
        if (ra.cousinRepulsion != null) {
          forceConfig.relativityCousinRepulsion = Boolean(ra.cousinRepulsion);
          setCheckbox("ra-cousin-enabled", Boolean(ra.cousinRepulsion));
        }
        if (ra.cousinStrength != null) {
          forceConfig.relativityCousinStrength = Number(ra.cousinStrength);
          setSlider("ra-cousin-strength", "ra-cousin-strength-val", Number(ra.cousinStrength), (v) => v.toFixed(2));
        }
        if (ra.phantomZone != null) {
          forceConfig.relativityPhantomZone = Boolean(ra.phantomZone);
          setCheckbox("ra-phantom-enabled", Boolean(ra.phantomZone));
        }
        if (ra.phantomMultiplier != null) {
          forceConfig.relativityPhantomMultiplier = Number(ra.phantomMultiplier);
          setSlider("ra-phantom-mult", "ra-phantom-mult-val", Number(ra.phantomMultiplier), (v) => v.toFixed(2));
        }
        if (ra.orbitStrength != null) {
          forceConfig.relativityOrbitStrength = Number(ra.orbitStrength);
          setSlider("ra-orbit-strength", "ra-orbit-strength-val", Number(ra.orbitStrength), (v) => v.toFixed(1));
        }
        if (ra.tangentialMultiplier != null) {
          forceConfig.relativityTangentialMultiplier = Number(ra.tangentialMultiplier);
          setSlider("ra-tangential-mult", "ra-tangential-mult-val", Number(ra.tangentialMultiplier), (v) => v.toFixed(1));
        }
        if (ra.orbitRadius != null) {
          forceConfig.relativityOrbitRadius = Number(ra.orbitRadius);
          setSlider("ra-orbit-radius", "ra-orbit-radius-val", Number(ra.orbitRadius), (v) => v.toFixed(0));
        }
      }

      // --- LinLog params ---
      if (ll) {
        if (ll.scaling != null) {
          forceConfig.linlogScaling = Number(ll.scaling);
          setSlider("ll-scaling", "ll-scaling-val", Number(ll.scaling), (v) => v.toFixed(1));
        }
        if (ll.gravity != null) {
          forceConfig.linlogGravity = Number(ll.gravity);
          setSlider("ll-gravity", "ll-gravity-val", Number(ll.gravity), (v) => v.toFixed(1));
        }
        if (ll.edgeWeightInfluence != null) {
          forceConfig.linlogEdgeWeightInfluence = Number(ll.edgeWeightInfluence);
          setSlider("ll-weight", "ll-weight-val", Number(ll.edgeWeightInfluence), (v) => v.toFixed(2));
        }
        if (ll.strongGravity != null) {
          forceConfig.linlogStrongGravity = Boolean(ll.strongGravity);
          setCheckbox("ll-strong-gravity", Boolean(ll.strongGravity));
        }
      }

      // --- t-FDP params ---
      if (tfdp) {
        if (tfdp.gamma != null) {
          forceConfig.tFdpGamma = Number(tfdp.gamma);
          setSlider("tfdp-gamma", "tfdp-gamma-val", Number(tfdp.gamma), (v) => v.toFixed(2));
        }
        if (tfdp.repulsionScale != null) {
          forceConfig.tFdpRepulsionScale = Number(tfdp.repulsionScale);
          setSlider("tfdp-repulsion", "tfdp-repulsion-val", Number(tfdp.repulsionScale), (v) => v.toFixed(2));
        }
        if (tfdp.alpha != null) {
          forceConfig.tFdpAlpha = Number(tfdp.alpha);
          setSlider("tfdp-alpha", "tfdp-alpha-val", Number(tfdp.alpha), (v) => v.toFixed(2));
        }
        if (tfdp.beta != null) {
          forceConfig.tFdpBeta = Number(tfdp.beta);
          setSlider("tfdp-beta", "tfdp-beta-val", Number(tfdp.beta), (v) => v.toFixed(2));
        }
      }

      // --- Tidy Tree params ---
      if (tt) {
        if (tt.levelSeparation != null) {
          forceConfig.tidyTreeLevelSeparation = Number(tt.levelSeparation);
          setSlider("tt-level-sep", "tt-level-sep-val", Number(tt.levelSeparation));
        }
        if (tt.siblingSeparation != null) {
          forceConfig.tidyTreeSiblingSeparation = Number(tt.siblingSeparation);
          setSlider("tt-sibling-sep", "tt-sibling-sep-val", Number(tt.siblingSeparation), (v) => v.toFixed(2));
        }
        if (tt.subtreeSeparation != null) {
          forceConfig.tidyTreeSubtreeSeparation = Number(tt.subtreeSeparation);
          setSlider("tt-subtree-sep", "tt-subtree-sep-val", Number(tt.subtreeSeparation), (v) => v.toFixed(2));
        }
        if (tt.stiffness != null) {
          forceConfig.tidyTreeStiffness = Number(tt.stiffness);
          setSlider("tt-stiffness", "tt-stiffness-val", Number(tt.stiffness), (v) => v.toFixed(2));
        }
        if (tt.damping != null) {
          forceConfig.tidyTreeDamping = Number(tt.damping);
          setSlider("tt-damping", "tt-damping-val", Number(tt.damping), (v) => v.toFixed(2));
        }
        if (tt.coordMode != null) {
          forceConfig.tidyTreeRadial = String(tt.coordMode) === "radial";
          setSelect("tt-coord-mode", String(tt.coordMode));
        }
      }

      // --- Community Layout params ---
      if (comm) {
        if (comm.resolution != null) {
          forceConfig.communityResolution = Number(comm.resolution);
          setSlider("comm-resolution", "comm-resolution-val", Number(comm.resolution), (v) => v.toFixed(2));
        }
        if (comm.spacing != null) {
          forceConfig.communitySpacing = Number(comm.spacing);
          setSlider("comm-spacing", "comm-spacing-val", Number(comm.spacing));
        }
        if (comm.nodeSpacing != null) {
          forceConfig.communityNodeSpacing = Number(comm.nodeSpacing);
          setSlider("comm-node-spacing", "comm-node-spacing-val", Number(comm.nodeSpacing));
        }
        if (comm.spreadFactor != null) {
          forceConfig.communitySpreadFactor = Number(comm.spreadFactor);
          setSlider("comm-spread", "comm-spread-val", Number(comm.spreadFactor), (v) => v.toFixed(2));
        }
        if (comm.stiffness != null) {
          forceConfig.communityStiffness = Number(comm.stiffness);
          setSlider("comm-stiffness", "comm-stiffness-val", Number(comm.stiffness), (v) => v.toFixed(2));
        }
        if (comm.damping != null) {
          forceConfig.communityDamping = Number(comm.damping);
          setSlider("comm-damping", "comm-damping-val", Number(comm.damping), (v) => v.toFixed(2));
        }
      }

      // --- Codebase Layout params ---
      if (cb) {
        if (cb.directoryPadding != null) {
          forceConfig.codebaseDirectoryPadding = Number(cb.directoryPadding);
          setSlider("cb-dir-padding", "cb-dir-padding-val", Number(cb.directoryPadding));
        }
        if (cb.filePadding != null) {
          forceConfig.codebaseFilePadding = Number(cb.filePadding);
          setSlider("cb-file-padding", "cb-file-padding-val", Number(cb.filePadding));
        }
        if (cb.spreadFactor != null) {
          forceConfig.codebaseSpreadFactor = Number(cb.spreadFactor);
          setSlider("cb-spread", "cb-spread-val", Number(cb.spreadFactor), (v) => v.toFixed(2));
        }
        if (cb.stiffness != null) {
          forceConfig.codebaseStiffness = Number(cb.stiffness);
          setSlider("cb-stiffness", "cb-stiffness-val", Number(cb.stiffness), (v) => v.toFixed(2));
        }
        if (cb.damping != null) {
          forceConfig.codebaseDamping = Number(cb.damping);
          setSlider("cb-damping", "cb-damping-val", Number(cb.damping), (v) => v.toFixed(2));
        }
      }

      // Apply all force config at once
      if (Object.keys(forceConfig).length > 0) {
        state.graph?.setForceConfig(forceConfig);
      }
    }

    // --- Load nodes (uses generator settings already applied to UI) ---
    const nodeCount = g?.nodes ? Number(g.nodes) : 1000;
    await loadNodes(nodeCount);

    // --- Display ---
    if (disp) {
      if (disp.theme) setSelect("display-theme", String(disp.theme));
      if (disp.backgroundColor) {
        setColor("display-bg-color", String(disp.backgroundColor));
        state.graph?.setBackgroundColor(String(disp.backgroundColor));
      }
      if (disp.defaultEdgeColor) {
        setColor("display-edge-color", String(disp.defaultEdgeColor));
      }
    }

    // --- Heatmap ---
    if (hm) {
      if (hm.colorScale) setSelect("heatmap-colorscale", String(hm.colorScale));
      if (hm.dataSource) setSelect("heatmap-datasource", String(hm.dataSource));
      if (hm.radius != null) setSlider("heatmap-radius", "heatmap-radius-val", Number(hm.radius));
      if (hm.intensity != null) setSlider("heatmap-intensity", "heatmap-intensity-val", Number(hm.intensity), (v) => v.toFixed(1));
      if (hm.opacity != null) setSlider("heatmap-opacity", "heatmap-opacity-val", Number(hm.opacity), (v) => v.toFixed(2));
      setCheckbox("heatmap-enabled", Boolean(hm.enabled));
      if (hm.enabled) {
        state.graph?.enableHeatmap({
          colorScale: String(hm.colorScale ?? "viridis") as "viridis" | "plasma" | "inferno" | "magma" | "cividis" | "turbo" | "hot" | "cool" | "spectral" | "coolwarm" | "blues" | "greens" | "reds" | "grayscale",
          radius: Number(hm.radius ?? 50),
          intensity: Number(hm.intensity ?? 1.0),
          opacity: Number(hm.opacity ?? 0.8),
        });
      }
    }

    // --- Contours ---
    if (ct) {
      if (ct.strokeWidth != null) setSlider("contour-width", "contour-width-val", Number(ct.strokeWidth));
      if (ct.strokeColor) setColor("contour-color", String(ct.strokeColor));
      if (ct.thresholdCount != null) setSlider("contour-thresholds", "contour-thresholds-val", Number(ct.thresholdCount));
      if (ct.minThreshold != null) setSlider("contour-min", "contour-min-val", Number(ct.minThreshold), (v) => v.toFixed(2));
      setCheckbox("contour-enabled", Boolean(ct.enabled));
      if (ct.enabled) {
        const count = Number(ct.thresholdCount ?? 4);
        const min = Number(ct.minThreshold ?? 0.10);
        const thresholds = Array.from({ length: count }, (_, i) => min + (i * (1 - min)) / count);
        state.graph?.enableContour({
          thresholds,
          strokeWidth: Number(ct.strokeWidth ?? 2),
          strokeColor: String(ct.strokeColor ?? "#ffffff"),
        });
      }
    }

    // --- Metaballs ---
    if (mb) {
      if (mb.threshold != null) setSlider("metaball-threshold", "metaball-threshold-val", Number(mb.threshold), (v) => v.toFixed(2));
      if (mb.opacity != null) setSlider("metaball-opacity", "metaball-opacity-val", Number(mb.opacity), (v) => v.toFixed(2));
      if (mb.fillColor) setColor("metaball-colorscale", String(mb.fillColor));
      setCheckbox("metaball-enabled", Boolean(mb.enabled));
      if (mb.enabled) {
        state.graph?.enableMetaball({
          fillColor: String(mb.fillColor ?? "#4f8cff"),
          threshold: Number(mb.threshold ?? 0.5),
          opacity: Number(mb.opacity ?? 0.6),
        });
      }
    }

    // --- Labels ---
    if (lb) {
      if (lb.fontSize != null) setSlider("labels-fontsize", "labels-fontsize-val", Number(lb.fontSize));
      if (lb.textColor) setColor("labels-color", String(lb.textColor));
      if (lb.maxLabels != null) setSlider("labels-max", "labels-max-val", Number(lb.maxLabels));
      setCheckbox("labels-enabled", Boolean(lb.enabled));
      if (lb.enabled) {
        await state.graph?.enableLabels({
          fontSize: Number(lb.fontSize ?? 14),
          fontColor: String(lb.textColor ?? "#ffffff"),
          maxLabels: Number(lb.maxLabels ?? 100),
        });
      }
    }

    // --- Node Style ---
    if (ns) {
      if (ns.sizeScale != null) {
        setSlider("node-size-scale", "node-size-scale-val", Number(ns.sizeScale), (v) => v.toFixed(2));
      }
      const borders = ns.borders as Record<string, unknown> | undefined;
      if (borders) {
        if (borders.width != null) setSlider("border-width", "border-width-val", Number(borders.width), (v) => v.toFixed(1));
        if (borders.color) setColor("border-color", String(borders.color));
        setCheckbox("border-enabled", Boolean(borders.enabled));
        if (borders.enabled) {
          state.graph?.enableNodeBorder(
            Number(borders.width ?? 2.0),
            String(borders.color ?? "#000000"),
          );
        }
      }
    }

    // --- Edge Style ---
    if (es) {
      if (es.opacity != null) setSlider("edge-opacity", "edge-opacity-val", Number(es.opacity), (v) => v.toFixed(2));
      if (es.widthScale != null) setSlider("edge-width-scale", "edge-width-scale-val", Number(es.widthScale), (v) => v.toFixed(1));
      const curved = es.curved as Record<string, unknown> | undefined;
      if (curved) {
        if (curved.segments != null) setSlider("curved-segments", "curved-segments-val", Number(curved.segments));
        if (curved.weight != null) setSlider("curved-weight", "curved-weight-val", Number(curved.weight), (v) => v.toFixed(2));
        if (curved.curvature != null) setSlider("curved-curvature", "curved-curvature-val", Number(curved.curvature), (v) => v.toFixed(2));
        setCheckbox("curved-enabled", Boolean(curved.enabled));
        if (curved.enabled) {
          state.graph?.enableCurvedEdges(
            Number(curved.segments ?? 19),
            Number(curved.weight ?? 0.80),
          );
        }
      }
    }

    // --- Edge Flow ---
    if (ef) {
      const l1 = ef.layer1 as Record<string, unknown> | undefined;
      const l2 = ef.layer2 as Record<string, unknown> | undefined;

      if (l1) {
        if (l1.waveShape) setSelect("flow-wave-shape", String(l1.waveShape));
        if (l1.pulseWidth != null) setSlider("flow-width", "flow-width-val", Number(l1.pulseWidth), (v) => v.toFixed(2));
        if (l1.pulseCount != null) setSlider("flow-count", "flow-count-val", Number(l1.pulseCount));
        if (l1.speed != null) setSlider("flow-speed", "flow-speed-val", Number(l1.speed), (v) => v.toFixed(2));
        if (l1.brightness != null) setSlider("flow-brightness", "flow-brightness-val", Number(l1.brightness), (v) => v.toFixed(1));
        if (l1.fade != null) setSlider("flow-fade", "flow-fade-val", Number(l1.fade), (v) => v.toFixed(2));
        if (l1.color) setColor("flow-color", String(l1.color));
        setCheckbox("flow-use-edge-color", Boolean(l1.useEdgeColor));
      }
      if (l2) {
        if (l2.waveShape) setSelect("flow2-wave-shape", String(l2.waveShape));
        if (l2.pulseWidth != null) setSlider("flow2-width", "flow2-width-val", Number(l2.pulseWidth), (v) => v.toFixed(2));
        if (l2.pulseCount != null) setSlider("flow2-count", "flow2-count-val", Number(l2.pulseCount));
        if (l2.speed != null) setSlider("flow2-speed", "flow2-speed-val", Number(l2.speed), (v) => v.toFixed(2));
        if (l2.brightness != null) setSlider("flow2-brightness", "flow2-brightness-val", Number(l2.brightness), (v) => v.toFixed(1));
        if (l2.fade != null) setSlider("flow2-fade", "flow2-fade-val", Number(l2.fade), (v) => v.toFixed(2));
        if (l2.color) setColor("flow2-color", String(l2.color));
        setCheckbox("flow2-use-edge-color", Boolean(l2.useEdgeColor));
        setCheckbox("flow2-enabled", Boolean(l2.enabled));
      }

      setCheckbox("flow-enabled", Boolean(ef.enabled));
      if (ef.enabled && l1) {
        state.graph?.setEdgeFlowConfig({
          layer1: {
            enabled: true,
            speed: Number(l1.speed ?? 0.30),
            pulseWidth: Number(l1.pulseWidth ?? 0.10),
            pulseCount: Number(l1.pulseCount ?? 3),
            brightness: Number(l1.brightness ?? 2.0),
            fade: Number(l1.fade ?? 0.30),
            waveShape: (String(l1.waveShape ?? "sine")) as "sine" | "square" | "triangle",
            color: l1.useEdgeColor ? null : hexToRgbaTuple(String(l1.color ?? "#00ffff")),
          },
          layer2: l2 ? {
            enabled: Boolean(l2.enabled),
            speed: Number(l2.speed ?? 0.50),
            pulseWidth: Number(l2.pulseWidth ?? 0.05),
            pulseCount: Number(l2.pulseCount ?? 6),
            brightness: Number(l2.brightness ?? 1.5),
            fade: Number(l2.fade ?? 0.20),
            waveShape: (String(l2.waveShape ?? "square")) as "sine" | "square" | "triangle",
            color: l2.useEdgeColor ? null : hexToRgbaTuple(String(l2.color ?? "#ff6b6b")),
          } : undefined,
        });
      }
    }

    // --- Auto-compute layout for structure-aware algorithms ---
    if (f?.algorithm) {
      const alg = String(f.algorithm);
      if (alg === "community" && state.graph) {
        try {
          state.graph.computeCommunityLayout();
          console.log("Auto-computed community layout from config");
        } catch (err) {
          console.warn("Could not auto-compute community layout:", err);
        }
      }
      if (alg === "codebase" && state.graph) {
        try {
          let categories: Uint8Array | undefined;
          if (state.codebaseData) {
            const nodeBound = state.graph.nodeCount;
            categories = new Uint8Array(nodeBound);
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
          }
          state.graph.computeCodebaseLayout(categories);
          console.log("Auto-computed codebase layout from config");
        } catch (err) {
          console.warn("Could not auto-compute codebase layout:", err);
        }
      }
      if (alg === "tidy-tree" && state.graph) {
        try {
          state.graph.computeTreeLayout();
          console.log("Auto-computed tree layout from config");
        } catch (err) {
          console.warn("Could not auto-compute tree layout:", err);
        }
      }
    }

    console.log("Config applied successfully");
  }

  // ========================================================================
  // Load Initial Data (from config.json or defaults)
  // ========================================================================

  await loadConfig();
}

// Start the application
main().catch(console.error);
