/**
 * Heroine Graph React Example
 *
 * Demonstrates the @heroine-graph/react wrapper with:
 * - Basic graph rendering
 * - Event handling
 * - Simulation controls
 */

import { useRef, useState, useCallback } from "react";
import { HeroineGraph, useGraph, useSimulation } from "@heroine-graph/react";
import type { HeroineGraphRef, NodeClickEvent, GraphInput } from "@heroine-graph/react";

// Generate sample graph data
function generateGraph(nodeCount: number): GraphInput {
  const nodes = Array.from({ length: nodeCount }, (_, i) => ({
    id: i,
    label: `Node ${i}`,
    type: i % 3 === 0 ? "primary" : i % 3 === 1 ? "secondary" : "tertiary",
  }));

  const edges: GraphInput["edges"] = [];
  for (let i = 1; i < nodeCount; i++) {
    // Create a tree structure with some cross-links
    edges.push({ source: Math.floor(i / 2), target: i });

    // Add random cross-links
    if (Math.random() > 0.7 && i > 3) {
      edges.push({ source: Math.floor(Math.random() * i), target: i });
    }
  }

  return { nodes, edges };
}

function App() {
  const graphRef = useRef<HeroineGraphRef>(null);
  const [graphData] = useState(() => generateGraph(100));
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [eventLog, setEventLog] = useState<string[]>([]);

  // Use the simulation hook (requires graph from ref)
  const graph = graphRef.current?.getGraph() ?? null;

  const addLog = useCallback((message: string) => {
    setEventLog((prev) => [message, ...prev.slice(0, 9)]);
  }, []);

  const handleNodeClick = useCallback((event: NodeClickEvent) => {
    setSelectedNode(event.nodeId as number);
    addLog(`Clicked node ${event.nodeId}`);
  }, [addLog]);

  const handleReady = useCallback(() => {
    addLog("Graph ready!");
  }, [addLog]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header */}
      <header
        style={{
          padding: "16px 24px",
          borderBottom: "1px solid #333",
          display: "flex",
          alignItems: "center",
          gap: "24px",
        }}
      >
        <h1 style={{ fontSize: "1.25rem", fontWeight: 600 }}>
          Heroine Graph - React
        </h1>
        {selectedNode !== null && (
          <span style={{ color: "#888" }}>
            Selected: Node {selectedNode}
          </span>
        )}
      </header>

      {/* Main Content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Graph */}
        <main style={{ flex: 1, position: "relative" }}>
          <HeroineGraph
            ref={graphRef}
            data={graphData}
            onReady={handleReady}
            onNodeClick={handleNodeClick}
            onNodeHoverEnter={(e) => addLog(`Hover enter: ${e.nodeId}`)}
            onNodeHoverLeave={(e) => addLog(`Hover leave: ${e.nodeId}`)}
            onSimulationEnd={() => addLog("Simulation ended")}
            style={{ width: "100%", height: "100%" }}
          />
        </main>

        {/* Sidebar */}
        <aside
          style={{
            width: "280px",
            borderLeft: "1px solid #333",
            padding: "16px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
            overflow: "auto",
          }}
        >
          {/* Controls */}
          <section>
            <h2
              style={{
                fontSize: "0.875rem",
                fontWeight: 600,
                marginBottom: "12px",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "#888",
              }}
            >
              Controls
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <button
                onClick={() => graphRef.current?.getGraph()?.fitToView()}
                style={buttonStyle}
              >
                Fit to View
              </button>
              <button
                onClick={() => graphRef.current?.getGraph()?.restartSimulation()}
                style={buttonStyle}
              >
                Restart Simulation
              </button>
              <button
                onClick={() => graphRef.current?.getGraph()?.clearSelection()}
                style={buttonStyle}
              >
                Clear Selection
              </button>
            </div>
          </section>

          {/* Event Log */}
          <section style={{ flex: 1, minHeight: 0 }}>
            <h2
              style={{
                fontSize: "0.875rem",
                fontWeight: 600,
                marginBottom: "12px",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "#888",
              }}
            >
              Event Log
            </h2>
            <div
              style={{
                background: "#111",
                borderRadius: "6px",
                padding: "12px",
                height: "200px",
                overflow: "auto",
                fontSize: "0.75rem",
                fontFamily: "monospace",
              }}
            >
              {eventLog.length === 0 ? (
                <span style={{ color: "#666" }}>No events yet...</span>
              ) : (
                eventLog.map((log, i) => (
                  <div key={i} style={{ color: "#aaa", marginBottom: "4px" }}>
                    {log}
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

const buttonStyle: React.CSSProperties = {
  padding: "8px 12px",
  background: "#222",
  border: "1px solid #444",
  borderRadius: "4px",
  color: "#fff",
  cursor: "pointer",
  fontSize: "0.875rem",
};

export default App;
