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
<!-- MANUAL ADDITIONS END -->

## Active Technologies
- Rust 1.75+ (edition 2024), TypeScript 5.x for bindings (003-graph-algorithms-wasm)
- N/A (in-memory graph already in GraphEngine) (003-graph-algorithms-wasm)
- TypeScript 5.x (Deno 2.x), WGSL shaders (WebGPU) + WebGPU API, d3-scale/d3-color (TS), existing heroine-graph core (002-visualization-layers)
- N/A (in-memory GPU buffers only) (002-visualization-layers)

## Recent Changes
- 003-graph-algorithms-wasm: Added Rust 1.75+ (edition 2024), TypeScript 5.x for bindings
