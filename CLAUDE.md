# heroine-graph Development Guidelines

## Runtime

**CRITICAL: NEVER use npm, npx, or pnpm. ALWAYS use Deno.**

- Use `deno task <name>` for running tasks
- Use `dx` (Deno equivalent of npx) if you need to run a package binary
- This is a Deno-first project - no exceptions

## Common Commands

```bash
# Development
deno task storybook       # Start Storybook dev server
deno task storybook:clean # Clear Storybook cache
deno task dev             # Watch mode for core package
deno task check           # Type check

# Testing
deno task test            # Run tests
deno task test:coverage   # Run tests with coverage
deno task bench           # Run benchmarks

# Building
deno task build           # Build project
deno task build:wasm      # Build WASM package
deno task bundle          # Bundle for browser

# Code Quality
deno task lint            # Lint code
deno task fmt             # Format code
```

## Technologies

- TypeScript 5.x (Deno 2.x)
- Rust 1.75+ (WASM target)
- WGSL shaders + WebGPU API
- wasm-bindgen
- petgraph (Rust)
- d3-scale/d3-color (TS)

## Project Structure

```text
packages/
  core/         # Main TypeScript library
  wasm/         # Rust WASM bindings
  react/        # React wrapper (planned)
  vue/          # Vue wrapper (planned)
  svelte/       # Svelte wrapper (planned)
stories/        # Storybook stories
tests/          # Test files
```

## Code Style

- Follow Deno standard conventions
- Use `deno fmt` for formatting
- Use `deno lint` for linting
- WGSL shaders: Follow WebGPU best practices

<!-- MANUAL ADDITIONS START -->

## Primary Use Case

**Code repository graph visualization** — mapping repos, directories, files, and symbols (classes, functions, etc.) as hierarchical trees with cross-cutting dependency edges (imports, tests, configs). Primary target: the Meridian project (~35K nodes). heroine-graph repo itself produces ~7.5K nodes with symbols, ~2.7K without.

## Algorithm Design Direction: Gravitational Well / Nested Bubble Model

The user (Tom) has a clear vision for a new layout algorithm built on top of Relativity Atlas infrastructure. Current algorithms all fail for large code trees:
- **ForceAtlas2**: Everything spreads into a uniform circle. No structure visible.
- **Barnes-Hut**: Collapses everything to center, nodes overlap.
- **Tidy Tree**: At 1K+ nodes, creates a solid ring around the edges. Useless at scale.
- **Relativity Atlas**: Closest to working, but jittery, creates grid/cross-hatch artifacts from the 128x128 density field grid and regular angular spacing. Leaf nodes get pulled inward past their parents.

### The Nested Bubble Algorithm (to be built)

**Core concept**: Each node has a computed "well radius" (invisible boundary) based on its subtree. These boundaries prevent unrelated subtrees from overlapping, providing geometric separation rather than relying on force balancing.

**Key principles**:
1. **Bottom-up bubble radius** (Rust, O(N), once per load): Leaves get base radius. Parents get radius = f(children's radii, child count). A dir with 50 files gets a much bigger bubble than one with 3.
2. **Depth-decaying external forces**: `effectiveGravity = baseGravity * decay^depth`. Leaf nodes experience almost no external pull — only their parent's spring. Solves the "leaves pulled inward past parents" problem.
3. **Bubble collision** (GPU): Non-related subtrees' bubbles cannot overlap. Entire subtrees repel as units. Uses smooth falloff (NOT a grid) to avoid cross-hatch artifacts.
4. **Graduated repulsion for free**: Siblings repel directly. Cousins repel via parent bubble collision. No explicit N-hop computation needed.
5. **Simple attraction**: Children attracted to parent via spring. Rest length scales with parent's bubble radius. Tangential spreading for siblings.
6. **Hierarchical convergence**: Instead of 35K nodes fighting for position, ~50 top-level bubbles arrange, then within each bubble sub-bubbles arrange, etc. Much faster convergence.

**Implementation plan** (builds on Relativity Atlas infrastructure):
- Rust: bubble radius computation (bottom-up tree walk in WASM)
- WGSL: bubble collision shader (modified collision shader using wellRadius)
- WGSL: depth-aware gravity shader
- TypeScript: wire into algorithm system, upload per-node wellRadius + depth data
- Rewrite mission-control example as focused code graph visualizer with ~5 tuning knobs

**Density field grid artifact**: The 128x128 density grid in Relativity Atlas causes visible cross-hatch patterns because force gradients have discrete steps at cell boundaries. The bubble approach should use continuous distance functions instead.

### Previous Work Reference
Tom previously forked Cosmos GL (open source version of Cosmograph) and added variable link lengths and node mass strengths per type. That work was partially lost in a git incident. The same concept of different attraction/repulsion constants per link/node type should be carried forward.

## Mission Control Example
Located at `examples/mission-control/`. Currently a 3,600-line kitchen-sink demo showing every feature. Plan is to rewrite as a focused code graph visualizer. The code quality is decent but it's trying to do too much. Has 3 data sources (random corporate graph, hierarchical tree, codebase dataset) and exposes ~50 configuration sliders.

## Known Issues (post branch 004 fixes)
- TypeScript check shows 47 pre-existing errors (WebGPU type index signatures, SharedArrayBuffer). These are NOT regressions.
- Cargo clippy clean, 59/59 Rust tests passing.
- Branch `004-incremental-graph-mutations` has comprehensive fixes for core mutation bugs, GPU shader correctness, framework wrappers, and infrastructure.
<!-- MANUAL ADDITIONS END -->

## Active Technologies
- Rust 1.75+ (edition 2024), TypeScript 5.x for bindings (003-graph-algorithms-wasm)
- N/A (in-memory graph already in GraphEngine) (003-graph-algorithms-wasm)
- TypeScript 5.x (Deno 2.x), WGSL shaders (WebGPU) + WebGPU API, d3-scale/d3-color (TS), existing heroine-graph core (002-visualization-layers)
- N/A (in-memory GPU buffers only) (002-visualization-layers)

## Recent Changes
- 003-graph-algorithms-wasm: Added Rust 1.75+ (edition 2024), TypeScript 5.x for bindings
