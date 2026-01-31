<!--
  Heroine Graph Svelte Example

  Demonstrates the @heroine-graph/svelte wrapper with:
  - Basic graph rendering
  - Event handling
  - Simulation controls
-->
<script lang="ts">
  import { HeroineGraph } from "@heroine-graph/svelte";
  import type { GraphInput, NodeClickEvent, NodeHoverEnterEvent, NodeHoverLeaveEvent } from "@heroine-graph/svelte";
  import type { HeroineGraph as HeroineGraphCore } from "@heroine-graph/core";

  // State
  let graphComponent = $state<{ getGraph: () => HeroineGraphCore | null } | null>(null);
  let selectedNode = $state<number | null>(null);
  let eventLog = $state<string[]>([]);

  // Generate sample graph data
  function generateGraph(nodeCount: number): GraphInput {
    const nodes = Array.from({ length: nodeCount }, (_, i) => ({
      id: i,
      label: `Node ${i}`,
      type: i % 3 === 0 ? "primary" : i % 3 === 1 ? "secondary" : "tertiary",
    }));

    const edges: GraphInput["edges"] = [];
    for (let i = 1; i < nodeCount; i++) {
      edges.push({ source: Math.floor(i / 2), target: i });
      if (Math.random() > 0.7 && i > 3) {
        edges.push({ source: Math.floor(Math.random() * i), target: i });
      }
    }

    return { nodes, edges };
  }

  const graphData = generateGraph(100);

  // Event handlers
  function addLog(message: string) {
    eventLog = [message, ...eventLog.slice(0, 9)];
  }

  function onReady(event: CustomEvent<HeroineGraphCore>) {
    addLog("Graph ready!");
  }

  function onNodeClick(event: CustomEvent<NodeClickEvent>) {
    selectedNode = event.detail.nodeId as number;
    addLog(`Clicked node ${event.detail.nodeId}`);
  }

  function onNodeHoverEnter(event: CustomEvent<NodeHoverEnterEvent>) {
    addLog(`Hover enter: ${event.detail.nodeId}`);
  }

  function onNodeHoverLeave(event: CustomEvent<NodeHoverLeaveEvent>) {
    addLog(`Hover leave: ${event.detail.nodeId}`);
  }

  function onSimulationEnd() {
    addLog("Simulation ended");
  }

  // Control methods
  function fitToView() {
    graphComponent?.getGraph()?.fitToView();
  }

  function restartSimulation() {
    graphComponent?.getGraph()?.restartSimulation();
  }

  function clearSelection() {
    graphComponent?.getGraph()?.clearSelection();
    selectedNode = null;
  }
</script>

<div class="app">
  <!-- Header -->
  <header class="header">
    <h1>Heroine Graph - Svelte</h1>
    {#if selectedNode !== null}
      <span class="selected">Selected: Node {selectedNode}</span>
    {/if}
  </header>

  <!-- Main Content -->
  <div class="content">
    <!-- Graph -->
    <main class="graph-container">
      <HeroineGraph
        bind:this={graphComponent}
        data={graphData}
        on:ready={onReady}
        on:nodeClick={onNodeClick}
        on:nodeHoverEnter={onNodeHoverEnter}
        on:nodeHoverLeave={onNodeHoverLeave}
        on:simulationEnd={onSimulationEnd}
        width="100%"
        height="100%"
      />
    </main>

    <!-- Sidebar -->
    <aside class="sidebar">
      <!-- Controls -->
      <section class="section">
        <h2 class="section-title">Controls</h2>
        <div class="button-group">
          <button class="btn" onclick={fitToView}>Fit to View</button>
          <button class="btn" onclick={restartSimulation}>Restart Simulation</button>
          <button class="btn" onclick={clearSelection}>Clear Selection</button>
        </div>
      </section>

      <!-- Event Log -->
      <section class="section log-section">
        <h2 class="section-title">Event Log</h2>
        <div class="event-log">
          {#if eventLog.length === 0}
            <span class="no-events">No events yet...</span>
          {:else}
            {#each eventLog as log, i}
              <div class="log-entry">{log}</div>
            {/each}
          {/if}
        </div>
      </section>
    </aside>
  </div>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .header {
    padding: 16px 24px;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    gap: 24px;
  }

  .header h1 {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .selected {
    color: #888;
  }

  .content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .graph-container {
    flex: 1;
    position: relative;
  }

  .sidebar {
    width: 280px;
    border-left: 1px solid #333;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow: auto;
  }

  .section-title {
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #888;
  }

  .button-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .btn {
    padding: 8px 12px;
    background: #222;
    border: 1px solid #444;
    border-radius: 4px;
    color: #fff;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .btn:hover {
    background: #333;
  }

  .log-section {
    flex: 1;
    min-height: 0;
  }

  .event-log {
    background: #111;
    border-radius: 6px;
    padding: 12px;
    height: 200px;
    overflow: auto;
    font-size: 0.75rem;
    font-family: monospace;
  }

  .no-events {
    color: #666;
  }

  .log-entry {
    color: #aaa;
    margin-bottom: 4px;
  }
</style>
