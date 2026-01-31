# Mission Control & Configuration Fixes

**Date**: 2026-01-12
**Status**: Pre-implementation review

---

## Overview

This document captures all issues identified in the Mission Control demo and proposes a centralized configuration architecture. The goal is:

1. **One thing does one thing** - No silent chaining or side effects
2. **Centralized configuration** - Single source of truth for all settings
3. **Complete UI** - Every feature the core supports should be controllable
4. **Predictable behavior** - What you toggle is what changes, nothing else

---

## Part 1: Issues to Fix

### 1.1 Silent Chaining / Auto-Enable Problems

| Issue | Current Behavior | Expected Behavior |
|-------|------------------|-------------------|
| **Contours → Heatmap** | Enabling contours auto-enables heatmap with opacity=0 | Contours should work independently OR explicitly require heatmap with user confirmation |
| **Heatmap data source → Streams** | Selecting a stream that doesn't exist auto-creates it with random data | Should show error or prompt user to create stream first |
| **Load codebase → Streams** | Loading codebase auto-creates 4 value streams | Should not auto-create; user enables what they want |

**Fix approach**: Remove all auto-enable logic. If a feature depends on another, either make it explicit in UI or decouple the dependency.

---

### 1.2 State Desynchronization

| Issue | Problem | Fix |
|-------|---------|-----|
| **Manual checkbox updates** | Code sets `.checked = value` directly instead of firing change events | Use proper state management or fire events |
| **Edge colors not persisted** | `setEdgeColors()` updates GPU but not `graphData`, so opacity slider reads stale values | Store current colors in state after any color change |
| **Stream clear partial reset** | Clears stream data but heatmap toggle stays ON | Clear should reset related UI state |
| **Preset selection desync** | If `setEdgeFlowPreset()` fails, UI checkbox already changed | Update UI only after API call succeeds |

---

### 1.3 Missing UI Controls

#### Edge Flow (Dual Layer System)
**Core supports** (in `edge_flow.ts`):
- Layer 1: enabled, pulseWidth, pulseCount, speed, waveShape, brightness, fade, color
- Layer 2: same parameters independently
- Wave shapes: square, triangle, sine

**Mission Control currently shows**:
- Preset dropdown
- 4 sliders for layer 1 only (width, count, speed, brightness)

**Missing**:
- [ ] Wave shape selector (square/triangle/sine)
- [ ] Fade slider
- [ ] Color picker for flow
- [ ] Layer 2 toggle
- [ ] Layer 2: all same controls as layer 1

#### Node Styling
**Core supports**:
- `setNodeColors(Float32Array)` - per-node colors
- `setNodeSizes(Float32Array)` - per-node sizes
- `setNodeTypeStyles(map)` - type-based styling

**Mission Control currently shows**:
- Color by type button
- Size by role button
- Highlight leaders button

**Missing**:
- [ ] Node size slider (global size multiplier)
- [ ] Per-type size configuration
- [ ] Color picker for manual coloring

#### Node Borders
**Core supports**:
- `setNodeBorder({ enabled, width, color })`
- `enableNodeBorder()`, `disableNodeBorder()`

**Mission Control shows** (newly added):
- Toggle, width slider, color picker

**Missing**:
- [ ] Per-type border overrides (T057 - types exist but not wired)

#### Curved Edges
**Core supports**:
- `setCurvedEdges({ enabled, segments, weight })`
- `setEdgeCurvatures(Float32Array)` - per-edge curvature

**Mission Control shows** (newly added):
- Toggle, segments, weight, default curvature, randomize button

**Status**: Complete for basic usage

---

### 1.4 Opacity Controls (Clarification)

These are **NOT duplicates** - they control different things:

| Control | What it affects | Location |
|---------|-----------------|----------|
| **Heatmap opacity** | The heatmap visualization layer | Heatmap section |
| **Stream opacity** | The node coloring from value streams | Value Streams section |
| **Edge opacity** | Edge line transparency | Edge Styling section |

**Issue**: Each opacity control should work correctly on its target. Currently:
- Heatmap opacity: Works BUT gets overridden to 0 when contours enable
- Stream opacity: Needs verification
- Edge opacity: Reads stale color data after color changes

**Fix**: Ensure each opacity control affects only its designated target, no cross-contamination.

---

### 1.5 Slider Behavior

**Current**: Sliders use `input` event - fires on every pixel of movement
**Problem**: Causes excessive GPU updates
**Fix**: Use `change` event OR debounce input events (e.g., 50ms)

---

### 1.6 Demo Assumptions vs Real Implementation

| Current Demo Pattern | Better Pattern |
|---------------------|----------------|
| "Color by type" button with hardcoded types | Dropdown to select type, color picker |
| "Size by role" with hardcoded sizes | Size slider + type selector |
| Random data generation | Load from file or manual entry |
| Auto-create streams | Explicit "Create Stream" button |

---

## Part 2: Centralized Configuration Proposal

### Current State: Distributed (38+ DEFAULT constants across 24+ files)

```
packages/core/src/
├── layers/
│   ├── heatmap/config.ts      → DEFAULT_HEATMAP_CONFIG
│   ├── contour/config.ts      → DEFAULT_CONTOUR_CONFIG
│   ├── metaball/config.ts     → DEFAULT_METABALL_CONFIG
│   └── labels/config.ts       → DEFAULT_LABEL_CONFIG
├── simulation/
│   ├── config.ts              → DEFAULT_FORCE_CONFIG, FORCE_PRESETS
│   └── controller.ts          → DEFAULT_SIMULATION_CONFIG
├── renderer/
│   ├── edge_flow.ts           → DEFAULT_EDGE_FLOW_CONFIG, EDGE_FLOW_PRESETS
│   ├── pipelines/edges.ts     → DEFAULT_CURVED_EDGE_CONFIG
│   └── pipelines/nodes.ts     → DEFAULT_NODE_PIPELINE_CONFIG
├── viewport/viewport.ts       → DEFAULT_VIEWPORT_CONFIG
├── api/graph.ts               → DEFAULT_NODE_BORDER_CONFIG (inline!)
└── types.ts                   → Public interfaces only
```

### Proposed: Centralized Configuration

Create a new file: `packages/core/src/config/index.ts`

```typescript
/**
 * Centralized Configuration
 *
 * Single source of truth for all default values and presets.
 * Each subsystem imports from here instead of defining its own defaults.
 */

// =============================================================================
// Graph Defaults
// =============================================================================

export const DEFAULT_GRAPH_CONFIG = {
  defaultNodeRadius: 5,
  defaultNodeColor: [0.4, 0.6, 0.9] as const,
  defaultEdgeWidth: 1,
  defaultEdgeColor: [0.5, 0.5, 0.5] as const,
};

// =============================================================================
// Rendering Defaults
// =============================================================================

export const DEFAULT_NODE_BORDER = {
  enabled: false,
  width: 2.0,
  color: "#333333",
};

export const DEFAULT_CURVED_EDGES = {
  enabled: false,
  segments: 19,
  weight: 0.8,
};

export const DEFAULT_EDGE_FLOW_LAYER = {
  enabled: false,
  pulseWidth: 0.1,
  pulseCount: 1,
  speed: 0.5,
  waveShape: "sine" as const,
  brightness: 1.0,
  fade: 0.0,
  color: null,
};

export const DEFAULT_EDGE_FLOW = {
  layer1: { ...DEFAULT_EDGE_FLOW_LAYER },
  layer2: { ...DEFAULT_EDGE_FLOW_LAYER },
};

// =============================================================================
// Layer Defaults
// =============================================================================

export const DEFAULT_HEATMAP = {
  enabled: false,
  radius: 50,
  intensity: 1.0,
  opacity: 0.7,
  colorScale: "viridis" as const,
  resolution: 512,
  dataSource: "density" as const,
};

export const DEFAULT_CONTOUR = {
  enabled: false,
  thresholds: [0.2, 0.4, 0.6, 0.8],
  lineWidth: 1.5,
  opacity: 0.8,
  color: "#ffffff",
  smoothing: true,
  dataSource: "density" as const,
};

export const DEFAULT_METABALL = {
  enabled: false,
  threshold: 0.5,
  radius: 30,
  opacity: 0.6,
  color: "#4080ff",
  smoothness: 2.0,
  dataSource: "density" as const,
};

export const DEFAULT_LABELS = {
  enabled: false,
  maxLabels: 100,
  minZoom: 0.5,
  fontSize: 12,
  fontFamily: "system-ui",
  color: "#ffffff",
  backgroundColor: "rgba(0,0,0,0.7)",
};

// =============================================================================
// Simulation Defaults
// =============================================================================

export const DEFAULT_FORCE = {
  repulsionStrength: -30,
  springStrength: 0.1,
  springLength: 30,
  centerStrength: 0.01,
  velocityDecay: 0.4,
  maxVelocity: 10,
  // ... all force parameters
};

export const DEFAULT_SIMULATION = {
  autoStart: true,
  alphaMin: 0.001,
  alphaDecay: 0.0228,
  alphaTarget: 0.0,
  ticksPerFrame: 1,
  warmupTicks: 0,
};

// =============================================================================
// Viewport Defaults
// =============================================================================

export const DEFAULT_VIEWPORT = {
  minZoom: 0.1,
  maxZoom: 10,
  zoomSensitivity: 0.002,
  panSensitivity: 1.0,
};

// =============================================================================
// Presets
// =============================================================================

export { FORCE_PRESETS } from "./presets/force";
export { EDGE_FLOW_PRESETS } from "./presets/edge_flow";
export { COLOR_SCALE_PRESETS } from "./presets/color_scales";
```

### Benefits of Centralization

1. **Single import** - `import { DEFAULT_HEATMAP, DEFAULT_FORCE } from "../config"`
2. **Easy to find** - All defaults in one place
3. **No duplication** - Color parsing, data source types defined once
4. **Consistent naming** - `DEFAULT_X` pattern everywhere
5. **Type safety** - Export interfaces alongside defaults
6. **Presets organized** - Separate files for preset collections

---

## Part 3: Implementation Order

### Phase 1: Fix Critical Bugs
1. [ ] Remove silent auto-enable (contours, streams)
2. [ ] Fix state desync (edge colors, checkbox states)
3. [ ] Fix opacity controls to affect correct targets

### Phase 2: Complete Missing UI
1. [ ] Edge flow: wave shape, fade, layer 2 controls
2. [ ] Node styling: size slider
3. [ ] Verify all controls work independently

### Phase 3: Centralize Configuration
1. [ ] Create `config/index.ts` with all defaults
2. [ ] Update all subsystems to import from central config
3. [ ] Remove duplicate DEFAULT_ constants
4. [ ] Consolidate color parsing utilities

### Phase 4: Polish
1. [ ] Debounce sliders
2. [ ] Add proper error handling
3. [ ] Ensure UI updates only after API success
4. [ ] Add loading states for async operations

---

## Part 4: Files to Modify

### Mission Control Demo
- `examples/mission-control/main.ts` - State management, control handlers
- `examples/mission-control/index.html` - Add missing UI controls

### Core Configuration (New)
- `packages/core/src/config/index.ts` - Central defaults
- `packages/core/src/config/presets/force.ts` - Force presets
- `packages/core/src/config/presets/edge_flow.ts` - Edge flow presets
- `packages/core/src/config/presets/color_scales.ts` - Color scale presets

### Core Updates (Import from central config)
- `packages/core/src/layers/heatmap/config.ts`
- `packages/core/src/layers/contour/config.ts`
- `packages/core/src/layers/metaball/config.ts`
- `packages/core/src/layers/labels/config.ts`
- `packages/core/src/simulation/config.ts`
- `packages/core/src/renderer/edge_flow.ts`
- `packages/core/src/renderer/pipelines/edges.ts`
- `packages/core/src/api/graph.ts`

---

## Notes

- Heatmap and Stream opacity are **different controls for different things** - not duplicates
- Contours currently depend on heatmap's density texture - need to decide: decouple OR make dependency explicit
- Edge flow dual-layer system is fully implemented in core, just needs Mission Control UI
- The "assumptions" in demo (random data, hardcoded types) should become proper configurable examples
