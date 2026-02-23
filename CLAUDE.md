# graphmother Development Guidelines

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

**Code repository graph visualization** — mapping repos, directories, files, and symbols (classes, functions, etc.) as hierarchical trees with cross-cutting dependency edges (imports, tests, configs). Primary target: the Meridian project (~35K nodes). graphmother repo itself produces ~7.5K nodes with symbols, ~2.7K without.

## Algorithm Design Direction: Gravitational Well / Nested Bubble Model

The user (Tom) has a clear vision for a new layout algorithm built on top of Relativity Atlas infrastructure. Historically all algorithms failed for large code trees (see Known Issues for the root causes fixed on branch `005-audit-fixes`):
- **ForceAtlas2**: Everything spread into a uniform circle. No structure visible.
- **Barnes-Hut**: Collapsed everything to center, nodes overlapped.
- **Tidy Tree**: At 1K+ nodes, created a solid ring around the edges. Useless at scale.
- **Relativity Atlas**: Closest to working, but jittery, with grid/cross-hatch artifacts from the 128x128 density field grid and regular angular spacing. Leaf nodes got pulled inward past their parents.

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

## Status (post branch 005-audit-fixes)

Suite is fully green: `deno task check` clean (the old "47 pre-existing TS errors" note is obsolete), `deno task test` passing (headless-WebGPU + unit tests), `deno task lint` and `deno fmt --check` clean, `cargo test` 77/77, `cargo clippy -D warnings` clean.

Branch `005-audit-fixes` fixed the root causes behind the historical algorithm failures:
- **Foundations**: WASM NodeId now equals the GPU slot index (SoA/CSR/NodeId spaces no longer diverge after remove+add); simulation and collision dispatch over `nodeHighWater` with dead slots masked via `NODE_FLAG_DEAD`; per-node pinning via `NODE_FLAG_PINNED` respected by `integrate.comp.wgsl`.
- **Barnes-Hut**: GPU radix sort rewritten digit-major with stable ranks (Morton codes were never globally sorted at >=1024 nodes); Karras tree bottom-up aggregation no longer relies on cross-workgroup memory coherence WGSL doesn't provide; split-search loop terminates arithmetically.
- **ForceAtlas2/LinLog**: attraction/repulsion calibration rebalanced (1/d repulsion was ~10x over-strong vs the substituted Hooke springs — the "uniform disc" symptom).
- **t-FDP**: the anti-collapse force floor no longer applies at all distances (it had nullified the t-distribution kernel entirely).
- **Relativity Atlas**: inverse-mass gravity vs depth-decay interaction fixed (leaves were pulled past parents); unbounded F=d attraction tamed; density-field forces de-quantized (bilinear splat/gradient — the cross-hatch and jitter source).
- **Per-edge force races**: non-atomic `forces[i] +=` in attraction shaders lost a degree-proportional fraction of hub attraction (systematic bias against exactly the high-degree parents in code trees).
- **Tidy Tree**: apportion ancestor bug fixed (subtrees overlapped/landed coincident); radial transform and recursion-depth issues fixed.
- **Defaults**: `alphaDecay` is now 0.0228 (d3 convention, ~300 ticks) instead of 0.0002 (~10 min of simmer).
- **Wrappers/examples/publish**: event-name mismatches, vite alias paths, StrictMode double-init, npm publish pipeline, and more.

There is a deterministic test harness under `tests/` (seeded-PRNG fixtures, invariant helpers in `tests/helpers/invariants.ts`, headless WebGPU tests in `tests/gpu/` driving the real pipelines). Physics changes should come with a harness test.

**Toolchain note**: `deno` and `wasm-pack` are required and were previously missing on this machine — install both before building (`deno task build:wasm` needs wasm-pack).
<!-- MANUAL ADDITIONS END -->

## Active Technologies
- Rust 1.75+ (edition 2024), TypeScript 5.x for bindings (003-graph-algorithms-wasm)
- N/A (in-memory graph already in GraphEngine) (003-graph-algorithms-wasm)
- TypeScript 5.x (Deno 2.x), WGSL shaders (WebGPU) + WebGPU API, d3-scale/d3-color (TS), existing graphmother core (002-visualization-layers)
- N/A (in-memory GPU buffers only) (002-visualization-layers)

## Recent Changes
- 003-graph-algorithms-wasm: Added Rust 1.75+ (edition 2024), TypeScript 5.x for bindings
