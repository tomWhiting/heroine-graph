/**
 * Graph Data Generators for Storybook Examples
 */

export interface GraphNode {
  id: string;
  x?: number;
  y?: number;
  radius?: number;
  color?: string;
  group?: string;
  label?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  color?: string;
  width?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Generate a random graph with the specified number of nodes
 */
export function generateRandomGraph(nodeCount: number, edgesPerNode = 2): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const edgeCount = Math.floor(nodeCount * edgesPerNode);

  // Generate nodes with rainbow colors
  for (let i = 0; i < nodeCount; i++) {
    const hue = (i / nodeCount) * 360;
    nodes.push({
      id: `n${i}`,
      radius: 3 + Math.random() * 4,
      color: `hsl(${hue}, 70%, 50%)`,
    });
  }

  // Generate random edges
  for (let i = 0; i < edgeCount; i++) {
    const source = Math.floor(Math.random() * nodeCount);
    const offset = Math.floor(Math.random() * Math.min(1000, nodeCount)) - 500;
    const target = Math.max(0, Math.min(nodeCount - 1, source + offset));

    if (source !== target) {
      edges.push({
        source: `n${source}`,
        target: `n${target}`,
        color: '#444',
        width: 0.5,
      });
    }
  }

  return { nodes, edges };
}

/**
 * Generate a random number from standard normal distribution (Box-Muller)
 */
function randomGaussian(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Generate a clustered graph with distinct density regions
 * Great for demonstrating heatmap visualization
 */
export function generateClusteredGraph(clusterCount = 5, nodesPerCluster = 100): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  let nodeId = 0;

  // Cluster centers distributed in a circle
  const clusterCenters: { x: number; y: number }[] = [];
  for (let c = 0; c < clusterCount; c++) {
    const angle = (c / clusterCount) * Math.PI * 2;
    const radius = 200;
    clusterCenters.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    });
  }

  // Generate nodes in each cluster
  for (let c = 0; c < clusterCount; c++) {
    const center = clusterCenters[c];
    const hue = (c / clusterCount) * 360;

    for (let i = 0; i < nodesPerCluster; i++) {
      // Gaussian distribution around cluster center
      const angle = Math.random() * Math.PI * 2;
      const r = Math.abs(randomGaussian() * 50);

      const x = center.x + Math.cos(angle) * r;
      const y = center.y + Math.sin(angle) * r;

      nodes.push({
        id: `node_${nodeId}`,
        x,
        y,
        radius: 3 + Math.random() * 2,
        color: `hsl(${hue}, 70%, 50%)`,
        group: `cluster_${c}`,
      });

      nodeId++;
    }
  }

  // Connect nodes within clusters
  for (let c = 0; c < clusterCount; c++) {
    const startIdx = c * nodesPerCluster;

    for (let i = 0; i < nodesPerCluster; i++) {
      const connections = 2 + Math.floor(Math.random() * 3);
      for (let j = 0; j < connections; j++) {
        const target = startIdx + Math.floor(Math.random() * nodesPerCluster);
        if (startIdx + i !== target) {
          edges.push({
            source: `node_${startIdx + i}`,
            target: `node_${target}`,
            color: '#333',
            width: 0.5,
          });
        }
      }
    }
  }

  // Add inter-cluster connections
  for (let i = 0; i < clusterCount * 3; i++) {
    const sourceCluster = Math.floor(Math.random() * clusterCount);
    const targetCluster = (sourceCluster + 1 + Math.floor(Math.random() * (clusterCount - 1))) % clusterCount;

    const sourceNode = sourceCluster * nodesPerCluster + Math.floor(Math.random() * nodesPerCluster);
    const targetNode = targetCluster * nodesPerCluster + Math.floor(Math.random() * nodesPerCluster);

    edges.push({
      source: `node_${sourceNode}`,
      target: `node_${targetNode}`,
      color: '#666',
      width: 1,
    });
  }

  return { nodes, edges };
}

/**
 * Generate a galaxy-like spiral graph
 */
export function generateSpiralGraph(nodeCount = 1000, arms = 4): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  for (let i = 0; i < nodeCount; i++) {
    const arm = i % arms;
    const progress = i / nodeCount;
    const armAngle = (arm / arms) * Math.PI * 2;
    const spiralAngle = progress * Math.PI * 4 + armAngle;
    const radius = 50 + progress * 200;

    const noise = randomGaussian() * 20;
    const x = Math.cos(spiralAngle) * (radius + noise);
    const y = Math.sin(spiralAngle) * (radius + noise);

    const hue = 200 + progress * 60;
    const brightness = 40 + (1 - progress) * 30;

    nodes.push({
      id: `star_${i}`,
      x,
      y,
      radius: 2 + Math.random() * 3,
      color: `hsl(${hue}, 80%, ${brightness}%)`,
    });

    if (i > arms) {
      edges.push({
        source: `star_${i}`,
        target: `star_${i - arms}`,
        color: '#223',
        width: 0.3,
      });
    }
  }

  return { nodes, edges };
}

/**
 * Generate a social network graph
 */
export function generateSocialNetwork(userCount = 100): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  for (let i = 0; i < userCount; i++) {
    const type = Math.random() > 0.8 ? 'influencer' : 'user';
    nodes.push({
      id: `user_${i}`,
      radius: type === 'influencer' ? 10 : 5,
      color: type === 'influencer' ? '#ff6b6b' : '#4ecdc4',
      label: `User ${i}`,
    });
  }

  for (let i = 0; i < userCount; i++) {
    const numFollowing = 2 + Math.floor(Math.random() * 8);
    for (let j = 0; j < numFollowing; j++) {
      const target = Math.floor(Math.random() * userCount);
      if (target !== i) {
        edges.push({
          source: `user_${i}`,
          target: `user_${target}`,
          color: '#666',
          width: 0.5,
        });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Generate a graph optimized for contour visualization
 * Creates very dense, tight clusters ideal for iso-line rendering
 */
export function generateContourGraph(clusterCount = 3, nodesPerCluster = 200): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  let nodeId = 0;

  // Cluster centers - spread apart for clear separation
  const clusterCenters: { x: number; y: number; spread: number }[] = [];
  for (let c = 0; c < clusterCount; c++) {
    const angle = (c / clusterCount) * Math.PI * 2;
    const radius = 150;
    clusterCenters.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      spread: 20 + Math.random() * 20, // Tight clusters with varying density
    });
  }

  // Generate tightly packed nodes in each cluster
  for (let c = 0; c < clusterCount; c++) {
    const center = clusterCenters[c];
    const hue = (c / clusterCount) * 360;

    for (let i = 0; i < nodesPerCluster; i++) {
      // Very tight Gaussian distribution
      const angle = Math.random() * Math.PI * 2;
      const r = Math.abs(randomGaussian() * center.spread);

      const x = center.x + Math.cos(angle) * r;
      const y = center.y + Math.sin(angle) * r;

      nodes.push({
        id: `node_${nodeId}`,
        x,
        y,
        radius: 2 + Math.random() * 2,
        color: `hsl(${hue}, 70%, 50%)`,
        group: `cluster_${c}`,
      });

      nodeId++;
    }
  }

  // Minimal edges - just enough to form a connected graph
  for (let c = 0; c < clusterCount; c++) {
    const startIdx = c * nodesPerCluster;

    // Connect each node to 1-2 nearby nodes
    for (let i = 0; i < nodesPerCluster; i++) {
      const connections = 1 + Math.floor(Math.random() * 2);
      for (let j = 0; j < connections; j++) {
        const target = startIdx + Math.floor(Math.random() * nodesPerCluster);
        if (startIdx + i !== target) {
          edges.push({
            source: `node_${startIdx + i}`,
            target: `node_${target}`,
            color: '#333',
            width: 0.3,
          });
        }
      }
    }
  }

  // Connect clusters with just a few edges
  for (let i = 0; i < clusterCount; i++) {
    const nextCluster = (i + 1) % clusterCount;
    const sourceNode = i * nodesPerCluster + Math.floor(Math.random() * nodesPerCluster);
    const targetNode = nextCluster * nodesPerCluster + Math.floor(Math.random() * nodesPerCluster);

    edges.push({
      source: `node_${sourceNode}`,
      target: `node_${targetNode}`,
      color: '#666',
      width: 0.5,
    });
  }

  return { nodes, edges };
}

/**
 * Generate a tree structure
 */
export function generateTree(depth = 5, branching = 3): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  let nodeId = 0;

  function addNode(parentId: string | null, currentDepth: number): string {
    const id = `node_${nodeId++}`;
    const hue = (currentDepth / depth) * 120 + 180;

    nodes.push({
      id,
      radius: 8 - currentDepth,
      color: `hsl(${hue}, 70%, 50%)`,
    });

    if (parentId !== null) {
      edges.push({
        source: parentId,
        target: id,
        width: 2 - currentDepth * 0.3,
      });
    }

    if (currentDepth < depth) {
      for (let i = 0; i < branching; i++) {
        addNode(id, currentDepth + 1);
      }
    }

    return id;
  }

  addNode(null, 0);
  return { nodes, edges };
}
