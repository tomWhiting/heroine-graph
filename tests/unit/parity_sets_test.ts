/**
 * Unit tests for the ping-pong parity state machine
 * (BindGroupParitySets / ParitySlot in packages/core/src/simulation/pipeline.ts).
 *
 * This is the mechanism GraphMother's per-frame path depends on: every bind
 * group that references a ping-pong buffer is built once per orientation and
 * selected by index, so an off-by-one is invisible — the simulation keeps
 * producing plausible motion while reading the buffer it is writing. graph.ts
 * owns no parity state of its own; it creates slots here and delegates every
 * getter and the frame advance, so these tests cover the shipped logic.
 *
 * No GPU is required: the class only compares buffer identities and never
 * calls a device method, so the buffers are stand-ins and the "bind groups"
 * are plain records of which buffer they were built from.
 */

import { assert, assertEquals, assertStrictEquals, assertThrows } from "jsr:@std/assert@^1";
import { loadPipelineModule } from "../helpers/gpu.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";

const mod = await loadPipelineModule();

/** Stand-in for a GPUBuffer: only its identity matters to the class. */
function fakeBuffer(label: string): GPUBuffer {
  return { label } as unknown as GPUBuffer;
}

/** Field names of SimulationBuffers, so a fake set is structurally complete. */
const BUFFER_FIELDS = [
  "positions",
  "positionsOut",
  "velocities",
  "velocitiesOut",
  "forces",
  "prevForces",
  "edgeSources",
  "edgeTargets",
  "clearUniforms",
  "repulsionUniforms",
  "springUniforms",
  "integrationUniforms",
  "nodeFlags",
  "nodeAlpha",
  "nodeMass",
  "nodeDepth",
  "liveIndices",
  "liveEdgeIndices",
  "edgeBundles",
  "readback",
] as const;

type FakeBuffers =
  & Record<(typeof BUFFER_FIELDS)[number], GPUBuffer>
  & {
    nodeCount: number;
    activeCount: number;
    activeEdgeCount: number;
    bundleCount: number;
    nodeCapacity: number;
  };

/** A fresh fake buffer set; `generation` distinguishes reallocations. */
function fakeBuffers(generation: string): FakeBuffers {
  const set = {
    nodeCount: 4,
    activeCount: 4,
    activeEdgeCount: 0,
    bundleCount: 0,
    nodeCapacity: 4,
  } as FakeBuffers;
  for (const field of BUFFER_FIELDS) set[field] = fakeBuffer(`${generation}.${field}`);
  return set;
}

/** What the "bind group" builder records: the orientation it was handed. */
interface BuiltGroup {
  reads: GPUBuffer;
  writes: GPUBuffer;
}

const buildGroup = (view: { positions: GPUBuffer; positionsOut: GPUBuffer }): BuiltGroup => ({
  reads: view.positions,
  writes: view.positionsOut,
});

/**
 * Wires a slot over a mutable buffers reference, the way graph.ts does
 * (the owner keeps the buffers; the parity sets read them through a closure).
 */
function harness(initial = fakeBuffers("gen0")) {
  const state = { buffers: initial as FakeBuffers | null };
  const sets = new mod.BindGroupParitySets(() => state.buffers);
  const slot = sets.slot<BuiltGroup>("test");
  const rebuild = () => slot.rebuild(buildGroup);
  return { state, sets, slot, rebuild };
}

Deno.test("ParitySlot: an unbuilt slot selects nothing", () => {
  const { slot } = harness();
  assertEquals(slot.built, false);
  assertEquals(slot.current, null);
  assertEquals(slot.label, "test");
});

Deno.test("ParitySlot: the selected variant always reads the live positions buffer", () => {
  const { state, sets, slot, rebuild } = harness();
  rebuild();
  const buffers = state.buffers!;
  const [p0, p1] = [buffers.positions, buffers.positionsOut];

  assertEquals(sets.parity, 0);
  assertStrictEquals(slot.current!.reads, p0);
  assertStrictEquals(slot.current!.writes, p1);

  // One frame: the buffers swap and the index flips together.
  sets.advance();
  assertEquals(sets.parity, 1);
  assertStrictEquals(buffers.positions, p1, "advance must swap the ping-pong buffers");
  assertStrictEquals(slot.current!.reads, p1);
  assertStrictEquals(slot.current!.writes, p0);

  // Two swaps restore the original assignment, and the original variant.
  sets.advance();
  assertEquals(sets.parity, 0);
  assertStrictEquals(buffers.positions, p0);
  assertStrictEquals(slot.current!.reads, p0);
});

Deno.test("ParitySlot: exactly two variants are built, one per orientation", () => {
  const { sets, slot } = harness();
  let builds = 0;
  slot.rebuild((view) => {
    builds++;
    return buildGroup(view);
  });
  assertEquals(builds, 2, "one build per parity, no more");

  const atParity0 = slot.current;
  sets.advance();
  const atParity1 = slot.current;
  sets.advance();
  assertStrictEquals(slot.current, atParity0, "ticking must not rebuild anything");
  assert(atParity1 !== atParity0);
});

Deno.test("ParitySlot: rebuilding at parity 1 places the as-is build at index 1", () => {
  // The case buildForBothParities' `currentParity` parameter exists for:
  // a reallocation that lands while the frame counter is odd.
  const { state, sets, slot, rebuild } = harness();
  rebuild();
  sets.advance();
  assertEquals(sets.parity, 1);

  // Reallocate: fresh buffers, in their as-allocated orientation, while the
  // parity counter stays at 1 (exactly what graph.ts's reallocateNodeBuffers
  // does — it replaces the buffer fields and rebuilds, without resetting parity).
  const grown = fakeBuffers("gen1");
  state.buffers = grown;
  rebuild();
  // Captured before the swap below rotates the fields in place.
  const [q0, q1] = [grown.positions, grown.positionsOut];

  assertStrictEquals(
    slot.current!.reads,
    q0,
    "at parity 1 the as-is build must land at index 1",
  );
  assertStrictEquals(slot.current!.writes, q1);

  // And the next frame keeps agreeing.
  sets.advance();
  assertEquals(sets.parity, 0);
  assertStrictEquals(slot.current!.reads, q1);
});

Deno.test("ParitySlot: a rebuild missed after reallocation throws instead of corrupting", () => {
  const { state, slot, rebuild } = harness();
  rebuild();

  // Reallocation without the matching rebuild — the silent-corruption case:
  // the old bind groups still name destroyed buffers.
  state.buffers = fakeBuffers("gen1");

  const error = assertThrows(
    () => slot.current,
    GraphMotherError,
    "is stale",
  );
  assert(error.message.includes('"test"'), "the failure must name the slot");
});

Deno.test("ParitySlot: a swap without a parity flip throws", () => {
  const { state, slot, rebuild } = harness();
  rebuild();

  // Rotate the buffers behind the state machine's back (the mutation an
  // extra swapSimulationBuffers call, or a dropped parity flip, produces).
  const buffers = state.buffers!;
  const tmp = buffers.positions;
  buffers.positions = buffers.positionsOut;
  buffers.positionsOut = tmp;

  assertThrows(() => slot.current, GraphMotherError, "is stale");
});

Deno.test("ParitySlot: bind groups that outlive their buffers throw", () => {
  const { state, slot, rebuild } = harness();
  rebuild();
  state.buffers = null;

  assertThrows(() => slot.current, GraphMotherError, "outlived its simulation buffers");
});

Deno.test("BindGroupParitySets: clearing drops every slot", () => {
  const { state, sets, slot, rebuild } = harness();
  const second = sets.slot<BuiltGroup>("second");
  rebuild();
  second.rebuild(buildGroup);
  assert(slot.built && second.built);

  sets.clearAll();
  assertEquals(slot.built, false);
  assertEquals(second.built, false);
  assertEquals(slot.current, null);
  assertEquals(second.current, null);

  // A slot rebuilt with no buffers stays empty rather than throwing later.
  state.buffers = null;
  rebuild();
  assertEquals(slot.built, false);
  assertEquals(slot.current, null);
});

Deno.test("BindGroupParitySets: advancing without buffers is a no-op", () => {
  const { state, sets } = harness();
  state.buffers = null;
  sets.advance();
  assertEquals(sets.parity, 0, "parity must not drift while there is nothing to swap");
});

// ---------------------------------------------------------------------------
// Non-ping-pong buffers (the LOD trio: liveIndices, nodeAlpha, nodeMass)
// ---------------------------------------------------------------------------

Deno.test("swappedPingPongView: exchanges the four ping-pong buffers and nothing else", () => {
  const buffers = fakeBuffers("gen0");
  const view = mod.swappedPingPongView(buffers) as unknown as FakeBuffers;

  assertStrictEquals(view.positions, buffers.positionsOut);
  assertStrictEquals(view.positionsOut, buffers.positions);
  assertStrictEquals(view.velocities, buffers.velocitiesOut);
  assertStrictEquals(view.velocitiesOut, buffers.velocities);

  const pingPong = new Set(["positions", "positionsOut", "velocities", "velocitiesOut"]);
  for (const field of BUFFER_FIELDS) {
    if (pingPong.has(field)) continue;
    assertStrictEquals(view[field], buffers[field], `${field} must be shared, not duplicated`);
  }
});

Deno.test("swappedPingPongView: carries the dispatch counts through unchanged", () => {
  // activeCount rides on the buffer struct next to nodeCount, and a parity
  // view that dropped or defaulted it would silently dispatch the repulsion
  // pass over the wrong slice of the active-index list on every other frame.
  const buffers = fakeBuffers("gen0");
  buffers.activeCount = 2;
  const view = mod.swappedPingPongView(buffers) as unknown as FakeBuffers;
  assertEquals(view.nodeCount, buffers.nodeCount);
  assertEquals(view.activeCount, 2);
  assertEquals(view.nodeCapacity, buffers.nodeCapacity);
});

Deno.test("ParitySlot: both variants name the same active-index buffer", () => {
  // The property that makes an LOD transition free: liveIndices does not
  // ping-pong, so overwriting its CONTENTS reaches whichever variant runs next
  // and no bind group is rebuilt. If it were ever swapped or reallocated per
  // parity, a collapse would take effect on alternate frames only.
  const state = { buffers: fakeBuffers("gen0") as FakeBuffers | null };
  const sets = new mod.BindGroupParitySets(() => state.buffers);
  const slot = sets.slot<{ live: GPUBuffer; reads: GPUBuffer }>("repulsion");
  slot.rebuild((view) => ({ live: view.liveIndices, reads: view.positions }));

  const atParity0 = slot.current!;
  sets.advance();
  const atParity1 = slot.current!;

  assertStrictEquals(atParity1.live, atParity0.live, "the active-index buffer must not alternate");
  assertStrictEquals(atParity0.live, state.buffers!.liveIndices);
  // Sanity: the two variants genuinely differ, so the equality above is not
  // passing on one variant used twice.
  assert(atParity1.reads !== atParity0.reads);
});
