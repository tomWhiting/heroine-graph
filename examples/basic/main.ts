/**
 * HeroineGraph - Basic Example
 *
 * This example demonstrates how to use HeroineGraph to visualize a graph.
 *
 * Run with: deno run --allow-read --allow-net examples/basic/main.ts
 *
 * @module
 */

// In a real application, you would import from the package:
// import { createHeroineGraph, type GraphInput } from '@heroine-graph/core';

// For this example, we import from the local package:
import { getSupportInfo, type GraphInput } from "../../packages/core/mod.ts";

/**
 * Generate a sample social network graph
 */
function generateSocialNetwork(userCount: number): GraphInput {
  const nodes = [];
  const edges = [];

  // Create users
  for (let i = 0; i < userCount; i++) {
    const type = Math.random() > 0.8 ? "influencer" : "user";
    nodes.push({
      id: `user_${i}`,
      radius: type === "influencer" ? 10 : 5,
      color: type === "influencer" ? "#ff6b6b" : "#4ecdc4",
      metadata: {
        name: `User ${i}`,
        type,
        followers: Math.floor(Math.random() * 10000),
      },
    });
  }

  // Create connections (follow relationships)
  for (let i = 0; i < userCount; i++) {
    // Each user follows 2-10 other users
    const numFollowing = 2 + Math.floor(Math.random() * 8);
    for (let j = 0; j < numFollowing; j++) {
      const target = Math.floor(Math.random() * userCount);
      if (target !== i) {
        edges.push({
          source: `user_${i}`,
          target: `user_${target}`,
          color: "#666",
          width: 0.5,
        });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Generate a hierarchical tree graph
 */
function generateTree(depth: number, branching: number): GraphInput {
  const nodes = [];
  const edges = [];
  let nodeId = 0;

  function addNode(parentId: string | null, currentDepth: number): string {
    const id = `node_${nodeId++}`;
    const hue = (currentDepth / depth) * 120 + 180; // Blue to green gradient

    nodes.push({
      id,
      radius: 8 - currentDepth,
      color: `hsl(${hue}, 70%, 50%)`,
      metadata: { depth: currentDepth },
    });

    if (parentId) {
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

/**
 * Main example function
 */
async function main() {
  console.log("╔═══════════════════════════════════════════════════════════╗");
  console.log("║           HeroineGraph - Basic Example                     ║");
  console.log("╚═══════════════════════════════════════════════════════════╝");
  console.log();

  // Check support
  const support = await getSupportInfo();
  console.log("Environment Support:");
  console.log(`  WebGPU: ${support.webgpu ? "✓" : "✗"}`);
  console.log(`  WASM:   ${support.wasm ? "✓" : "✗"}`);
  console.log();

  if (!support.supported) {
    console.error(`Not supported: ${support.reason}`);
    console.log("\nTo run this example, use a WebGPU-enabled browser.");
    console.log("Open examples/basic/index.html in Chrome 113+ or Edge 113+");
    return;
  }

  // This would run in a browser environment:
  // const graph = await createHeroineGraph({
  //   canvas: '#graph-canvas',
  //   config: {
  //     simulation: {
  //       alphaDecay: 0.02,
  //     },
  //   },
  // });

  // Generate sample data
  console.log("Generating sample graphs...");

  const socialNetwork = generateSocialNetwork(500);
  console.log(
    `  Social Network: ${socialNetwork.nodes.length} nodes, ${socialNetwork.edges.length} edges`,
  );

  const tree = generateTree(5, 3);
  console.log(`  Tree: ${tree.nodes.length} nodes, ${tree.edges.length} edges`);

  // In browser:
  // await graph.load(socialNetwork);
  // graph.startSimulation();

  console.log();
  console.log("Example data generated successfully.");
  console.log("To visualize, open examples/basic/index.html in a browser.");
}

// Run if executed directly
if (import.meta.main) {
  main().catch(console.error);
}

// Export for use in other examples
export { generateSocialNetwork, generateTree };
