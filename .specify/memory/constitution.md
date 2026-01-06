<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0 (MAJOR - initial constitution ratification)
Modified principles: N/A (initial creation)
Added sections:
  - 10 Core Principles (Consistency through Build With Love)
  - Collaboration Framework
  - Technical Conventions
  - Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ Compatible (no constitution-specific refs)
  - .specify/templates/tasks-template.md: ✅ Compatible (no constitution-specific refs)
Follow-up TODOs: None
==================
-->

# Heroine Graph Constitution

These are our design values. They guide every decision.

## Core Principles

### I. Consistency Is On Us

The system MUST behave consistently. Always. This is not the user's job.

If a user drags a node and it lands somewhere unexpected, that's our fault. If the same
action produces different results in different contexts, that's our fault. If critical
is red, critical is red every single time. Users MUST NOT have to "learn our quirks."

The user defines their model. We render it faithfully. That's the contract.

**Rationale**: A high-performance graph library lives or dies by predictability. When
users are working with millions of nodes, they cannot afford to second-guess whether
their interaction will behave as expected.

### II. Contract, Not Coercion

There is a difference between rules you agree to and rules forced upon you.

**Contract**:
- Rust's type system is strict, but you chose Rust. You sat down at the table.
- WebGPU's shader model has constraints, but you chose GPU programming.
- The API requires positions as Float32Array—you agreed to that interface.

**Coercion**:
- The system says "this node can't be selected" when the user says it should be.
- Reaching into someone's mental model and telling them it's wrong.
- Silently "fixing" user-provided data because we think we know better.

When a user says a node is important, we don't get to disagree. We just render it as
important. We render their intent faithfully. The user defines what things mean. We
make sure those meanings are applied consistently.

### III. Trust Users, Don't Give Them Guns

Users can make bad choices that affect themselves—let them. That's their right.

But they MUST NOT be able to make catastrophic, irreversible mistakes that destroy
everything. The distinction:

- **Their problem**: You loaded 10 million nodes on a laptop and it's slow. Your
  choice, your consequence. Allowed.
- **Our problem**: You accidentally cleared the graph data with a misclick and lost
  your unsaved work. Catastrophic, irreversible. MUST be prevented or recoverable.

Expose power. Guard against annihilation. Create an environment where it's easy to
make the right decision.

### IV. Expose All Controls, Make Defaults Excellent

Every parameter MUST be available. Advanced users need access to everything—force
simulation constants, render passes, buffer configurations, all of it.

But using defaults MUST just work, beautifully. A user who touches nothing MUST have
an excellent experience. A user who customizes everything MUST have access to everything.

No hidden state. No mystery behaviours. If the simulation strength slider is at 0.5,
and the user changes it, it changes. No silent overrides. No "actually we ignore that
setting during initial layout."

This is harder than hiding complexity. We do it anyway.

### V. No Silent Failures

If something is broken, it MUST be visibly broken.

No swallowing WebGPU errors. No "it just didn't render." No mysterious empty canvases.
If the system fails, it fails loudly, with context, with actionable information.

A user MUST NOT wonder "did that work?" They MUST know.

- GPU initialization failures MUST report device capabilities and what's missing
- Shader compilation errors MUST surface with line numbers and context
- Data validation failures MUST identify the problematic nodes/edges

We don't pretend things are fine when they're not. We don't simulate success.

### VI. Automation Over Gatekeeping

Instead of preventing actions, enable users to define consequences.

**Gatekeeping (don't do this)**:
- "You can't add more than 100,000 nodes."
- "Edges can't connect nodes in different groups."
- The system deciding what's allowed.

**Automation (do this)**:
- "When node count exceeds threshold, switch to LOD rendering."
- "When performance drops below 30fps, emit a warning event."
- The user defining what happens in response to events.

Expose the event stream. Let users write handlers against real events. Simple handlers
for people who want simple. Complex handlers for people who need complex.

### VII. Low-Level Primitives Over Opinionated Wrappers

Choose the lower abstraction. Build up, not down.

- WebGPU directly, not through abstraction layers
- Raw buffer management, not magic data binding
- Explicit render passes, not "just call render()"

Opinionated wrappers feel faster at first, then trap you. They've answered questions
you hadn't asked yet. Primitives require more upfront work, then set you free.

This library IS the primitive. Framework wrappers (React, Vue, Svelte) are thin
bindings that don't hide the core API.

### VIII. Circular Dependencies Are Real

Graphs have cycles. That's literally the point.

When a user creates a cyclic graph, we visualize it. When force simulation creates
oscillations, we handle it gracefully. When data flows form loops, we don't crash.

Academic graph theory sometimes treats cycles as edge cases. Real graphs are full of
them. We model reality, not theory.

### IX. Make It Easy to Have Fun

Like Apple: the internals may be complicated, but using it feels effortless.

The goal isn't simplicity—it's ease. A high-performance WebGPU library with Rust/WASM
internals CAN be easy to use if designed with care. We design with care.

Loading a graph and seeing it animate MUST feel magical. Dragging nodes MUST feel
responsive. Zooming into clusters MUST feel smooth. The path of least resistance
MUST be the beautiful path.

### X. Build With Love

We're not building this because someone's paying us. We're building it because we
want it to exist.

Like the chef who cooks because he loves food: we take our time, we get it right,
we care about details others won't notice. Every frame rendered. Every interaction
polished. Every edge case handled.

This doesn't mean slow. It means intentional.

## The Collaboration Framework

This applies to human-AI collaboration, human-human collaboration, and the relationship
between library and user. The structure is the same.

### 1. Baseline (Non-Negotiables)

Safety. Consent. The things that aren't up for discussion.

- In this library: no silent data loss. No pretending things work when they don't.
- In development: no `todo!()` macros shipped. No placeholder implementations.
- In collaboration: stop when redirected. Course-correct when wrong.

These are the walls with signs explaining the forty-foot cliff. Non-negotiable. Ever.

### 2. Goals (What Are We Actually Doing?)

State what each party wants out of this. Agree on it before you start.

- What does success look like for this feature?
- What's the scope? What's explicitly out of scope?
- What are the performance requirements?

If you skip this step, you end up with mismatched expectations and disappointment.

### 3. Contract (You Agreed to Play)

The goals become the contract. You agreed to this specific thing.

- You agreed to implement force-directed layout. You don't get to decide halfway
  through that actually you're building a tree layout.
- You agreed to use WebGPU. You don't get to switch to WebGL because it's easier.
- Stick to the contract. If it needs to change, renegotiate explicitly.

### 4. Presence (Stop When They Tap)

Things change as you go. Be present. Listen. React to what's actually happening.

- When the human redirects, redirect immediately.
- When requirements change, acknowledge and adapt.
- The difference between good collaboration and violation is responsiveness.

### 5. Purpose (The Collaboration Is the Point)

If the goal is just to complete a task, do it alone. If working together, the
working-with is the point. The thinking together. The figuring things out.

The collaboration is the point. The deliverable is a side effect of good collaboration.

## Technical Conventions

### Dependency Management

- Rust dependencies: Use `cargo add` to add dependencies at their latest stable version
- TypeScript/Deno dependencies: Use `deno add` to add dependencies at their latest version
- NEVER pin to outdated versions without explicit justification
- Review changelogs when updating major versions

### Implementation Standards

- NO `todo!()` macros in shipped code
- NO placeholder functions that defer implementation
- NO `unimplemented!()` in any code path that can be reached
- NO "we'll come back to this later" comments without corresponding tracked issues
- Every function MUST be fully implemented or not exist yet
- Specs MUST be complete before implementation begins
- Plans MUST be thorough before tasks are generated
- Tasks MUST be fully enumerated before coding starts

### Quality Gates

- All code MUST compile without warnings
- All tests MUST pass before merge
- Performance benchmarks MUST meet specified targets
- Documentation MUST match implementation

## Governance

This constitution supersedes all other practices. When in conflict, these principles
win.

### Amendment Process

1. Proposed changes MUST be documented with rationale
2. Changes MUST be reviewed against existing principles for conflicts
3. Version MUST be incremented according to semantic versioning:
   - MAJOR: Principle removals or incompatible redefinitions
   - MINOR: New principles or materially expanded guidance
   - PATCH: Clarifications, wording, non-semantic refinements

### Compliance

- All PRs and reviews MUST verify compliance with these principles
- Complexity MUST be justified against Principle VII (Low-Level Primitives)
- Silent failures MUST be treated as bugs per Principle V

### Priority Order

When principles conflict, they are ranked:

1. **Consistency** (non-negotiable)
2. **No silent failures** (trust requires honesty)
3. **Contract, not coercion** (respect user intent)
4. **Trust users / don't give guns** (power with guardrails)
5. **Automation over gatekeeping** (enable, don't restrict)
6. **Expose all controls / excellent defaults** (access and ease)
7. **Easy to have fun** (the feel)
8. **Low-level primitives** (technical choice)
9. **Circular dependencies allowed** (flexibility)
10. **Build with love** (always, but doesn't override safety)

When in doubt, re-read #1.

**Version**: 1.0.0 | **Ratified**: 2026-01-06 | **Last Amended**: 2026-01-06
