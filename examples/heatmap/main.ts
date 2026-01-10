/**
 * HeroineGraph - Heatmap Layer Example
 *
 * This example demonstrates how to use the heatmap visualization layer
 * to show node density as a colorful overlay.
 *
 * Run with: deno run --allow-read --allow-net examples/heatmap/main.ts
 *
 * @module
 */

// In a real application, you would import from the package:
// import { createHeroineGraph, HeatmapLayer, ... } from '@heroine-graph/core';

// For this example, we import from the local package:
import { getColorScaleNames, getSupportInfo, type GraphInput } from "../../packages/core/mod.ts";

/**
 * Generate a clustered graph with varying density regions
 * Perfect for demonstrating heatmap visualization
 */
export function generateClusteredGraph(
  clusterCount: number,
  nodesPerCluster: number,
): GraphInput {
  const nodes = [];
  const edges = [];
  let nodeId = 0;

  // Cluster centers distributed in a circle
  const clusterCenters = [];
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
      const r = Math.abs(randomGaussian() * 50); // Cluster spread

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

  // Connect nodes within clusters (high density connections)
  for (let c = 0; c < clusterCount; c++) {
    const startIdx = c * nodesPerCluster;
    const endIdx = startIdx + nodesPerCluster;

    for (let i = startIdx; i < endIdx; i++) {
      // Connect to 2-4 random nodes in same cluster
      const connections = 2 + Math.floor(Math.random() * 3);
      for (let j = 0; j < connections; j++) {
        const target = startIdx + Math.floor(Math.random() * nodesPerCluster);
        if (target !== i) {
          edges.push({
            source: `node_${i}`,
            target: `node_${target}`,
            color: "#333",
            width: 0.5,
          });
        }
      }
    }
  }

  // Add some inter-cluster connections
  for (let i = 0; i < clusterCount * 3; i++) {
    const sourceCluster = Math.floor(Math.random() * clusterCount);
    const targetCluster = (sourceCluster + 1 + Math.floor(Math.random() * (clusterCount - 1))) %
      clusterCount;

    const sourceNode = sourceCluster * nodesPerCluster +
      Math.floor(Math.random() * nodesPerCluster);
    const targetNode = targetCluster * nodesPerCluster +
      Math.floor(Math.random() * nodesPerCluster);

    edges.push({
      source: `node_${sourceNode}`,
      target: `node_${targetNode}`,
      color: "#666",
      width: 1,
    });
  }

  return { nodes, edges };
}

/**
 * Generate a random number from standard normal distribution
 */
function randomGaussian(): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Generate a galaxy-like spiral graph
 */
export function generateSpiralGraph(nodeCount: number, arms: number = 3): GraphInput {
  const nodes = [];
  const edges = [];

  for (let i = 0; i < nodeCount; i++) {
    const arm = i % arms;
    const progress = i / nodeCount;
    const armAngle = (arm / arms) * Math.PI * 2;
    const spiralAngle = progress * Math.PI * 4 + armAngle;
    const radius = 50 + progress * 200;

    // Add some noise
    const noise = randomGaussian() * 20;
    const x = Math.cos(spiralAngle) * (radius + noise);
    const y = Math.sin(spiralAngle) * (radius + noise);

    const hue = 200 + progress * 60; // Blue to cyan
    const brightness = 40 + (1 - progress) * 30;

    nodes.push({
      id: `star_${i}`,
      x,
      y,
      radius: 2 + Math.random() * 3,
      color: `hsl(${hue}, 80%, ${brightness}%)`,
      importance: 1 - progress,
    });

    // Connect to nearby nodes
    if (i > arms) {
      edges.push({
        source: `star_${i}`,
        target: `star_${i - arms}`,
        color: "#223",
        width: 0.3,
      });
    }
  }

  return { nodes, edges };
}

/**
 * Main example function
 */
async function main() {
  console.log("==============================================================");
  console.log("           HeroineGraph - Heatmap Layer Example               ");
  console.log("==============================================================");
  console.log();

  // Check support
  const support = await getSupportInfo();
  console.log("Environment Support:");
  console.log(`  WebGPU: ${support.webgpu ? "Yes" : "No"}`);
  console.log(`  WASM:   ${support.wasm ? "Yes" : "No"}`);
  console.log();

  if (!support.supported) {
    console.error(`Not supported: ${support.reason}`);
    console.log("\nTo run this example in a browser:");
    console.log("  1. Run: deno task bundle");
    console.log("  2. Open examples/heatmap/index.html in Chrome 113+ or Edge 113+");
    return;
  }

  // Show available color scales
  const colorScales = getColorScaleNames();
  console.log("Available heatmap color scales:");
  colorScales.forEach((scale) => console.log(`  - ${scale}`));
  console.log();

  // Generate sample data
  console.log("Generating sample graphs for heatmap visualization...");

  const clusteredGraph = generateClusteredGraph(5, 100);
  console.log(
    `  Clustered: ${clusteredGraph.nodes.length} nodes, ${clusteredGraph.edges.length} edges`,
  );

  const spiralGraph = generateSpiralGraph(1000, 4);
  console.log(
    `  Spiral:    ${spiralGraph.nodes.length} nodes, ${spiralGraph.edges.length} edges`,
  );

  console.log();
  console.log("Heatmap Configuration Options:");
  console.log("  - radius:     Gaussian splat radius (default: 50)");
  console.log("  - intensity:  Base intensity per node (default: 0.1)");
  console.log("  - opacity:    Layer opacity 0-1 (default: 0.7)");
  console.log("  - colorScale: Color scheme (viridis, plasma, inferno, etc.)");
  console.log("  - minDensity: Minimum density threshold");
  console.log("  - maxDensity: Maximum density threshold");
  console.log();
  console.log("Example usage in browser:");
  console.log(`
    // Create graph with heatmap layer
    const graph = await createHeroineGraph({ canvas: '#graph' });

    // Add a heatmap layer
    const heatmap = graph.addHeatmapLayer('density', {
      enabled: true,
      radius: 60,
      intensity: 0.15,
      colorScale: 'viridis',
      opacity: 0.8,
    });

    // Load graph data
    await graph.load(clusteredGraph);

    // Toggle heatmap
    heatmap.enabled = false;

    // Change color scale
    heatmap.setColorScale('plasma');

    // Update configuration
    heatmap.setConfig({
      radius: 80,
      intensity: 0.2,
    });
  `);
  console.log();
  console.log("Example data generated successfully.");
  console.log("To visualize, open examples/heatmap/index.html in a browser.");
}

// Run if executed directly
if (import.meta.main) {
  main().catch(console.error);
}

// Export for use in other examples
export { generateClusteredGraph as generateDenseGraph, generateSpiralGraph };
