/**
 * GPU tests for NODE_FLAG_DEAD / NODE_FLAG_PINNED semantics:
 *
 * - A dead slot exerts no force on anything and never moves — the live
 *   nodes' trajectories are bit-identical regardless of where the dead
 *   slot's (stale) position happens to sit.
 * - A pinned node holds its exact position under strong spring forces
 *   while still repelling neighbors.
 * - Collision resolution never displaces a pinned node (it still pushes
 *   overlapping neighbors away), and dead slots neither move nor push.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createSimHarness,
  DEAD_SLOT_RADIUS,
  GPU_SKIP_MESSAGE,
  NODE_FLAG_DEAD,
  NODE_FLAG_PINNED,
  probeAdapter,
  requestHarnessDevice,
  runCollision,
} from "../helpers/gpu.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

function gpuTest(name: string, fn: (device: GPUDevice) => Promise<void>): void {
  Deno.test({
    name,
    ignore: adapter === null,
    sanitizeResources: false,
    sanitizeOps: false,
    async fn() {
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

gpuTest(
  "GPU flags: dead slot exerts no force and never moves",
  async (device) => {
    // Two live nodes with an edge, plus one dead slot. Run the identical
    // simulation twice with the dead slot parked at different positions —
    // once dead-center between the live pair (where a phantom node would
    // repel both) and once far off to the side. If the dead mask works,
    // the dead slot's position is invisible: live trajectories match bit
    // for bit, and the dead slot itself never moves.
    const run = async (deadX: number, deadY: number) => {
      const harness = await createSimHarness(device, {
        nodeCount: 3,
        positionsX: new Float32Array([-100, deadX, 100]),
        positionsY: new Float32Array([0, deadY, 0]),
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([2]),
        flags: new Uint32Array([0, NODE_FLAG_DEAD, 0]),
      });
      try {
        await harness.tick(50);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    const centered = await run(0, 0);
    const offside = await run(37, 91);

    // Dead slot never moves (position carried through the ping-pong intact)
    assertEquals(centered.x[1], 0, "dead slot moved in x");
    assertEquals(centered.y[1], 0, "dead slot moved in y");
    assertEquals(offside.x[1], 37, "dead slot moved in x");
    assertEquals(offside.y[1], 91, "dead slot moved in y");

    // Dead slot exerts no force: live trajectories are identical no matter
    // where the dead slot sits
    for (const i of [0, 2]) {
      assertEquals(
        centered.x[i],
        offside.x[i],
        `dead slot position leaked into node ${i} (x)`,
      );
      assertEquals(
        centered.y[i],
        offside.y[i],
        `dead slot position leaked into node ${i} (y)`,
      );
    }

    // Sanity: the live pair actually moved (simulation ran)
    assert(
      centered.x[0] !== -100 || centered.x[2] !== 100,
      "live nodes never moved",
    );
  },
);

gpuTest(
  "GPU flags: pinned node stays exactly put under strong forces and still repels",
  async (device) => {
    // Node 0 is pinned at (10, 20) via NODE_FLAG_PINNED. A stretched edge
    // to distant node 2 pulls on it hard every tick; unpinned it would race
    // toward node 2. Node 1 sits right next to the pinned node with no edge:
    // it must be pushed away by the pinned node's repulsion.
    const harness = await createSimHarness(
      device,
      {
        nodeCount: 3,
        positionsX: new Float32Array([10, 13, 510]),
        positionsY: new Float32Array([20, 20, 20]),
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([2]),
        flags: new Uint32Array([NODE_FLAG_PINNED, 0, 0]),
      },
      { centerStrength: 0 },
    );

    try {
      await harness.tick(60);
      const { x, y } = await harness.readPositions();

      // Pinned node held its externally-written position exactly
      assertEquals(x[0], 10, "pinned node drifted in x");
      assertEquals(y[0], 20, "pinned node drifted in y");

      // The pinned node still repels: node 1 started 3 units away and must
      // have been pushed well clear
      const dist1 = Math.hypot(x[1] - 10, y[1] - 20);
      assert(dist1 > 5, `neighbor was not repelled by pinned node: d=${dist1}`);

      // The spring still acts on the free endpoint: node 2 was pulled toward
      // the pinned node
      assert(x[2] < 510, `spring to pinned node never pulled free end: x=${x[2]}`);
    } finally {
      harness.dispose();
    }
  },
);

// ---------------------------------------------------------------------------
// Collision pass (post-integration position resolution)
// ---------------------------------------------------------------------------

for (const variant of ["main", "tiled"] as const) {
  gpuTest(
    `GPU collision (${variant}): pinned node is never displaced but still pushes; dead slots inert`,
    async (device) => {
      // Node 0 pinned at origin, node 1 overlapping it at (3, 0), both
      // radius 5 (min separation 10, overlap 7). Node 2 is a dead slot
      // wedged between them — it must neither move nor push.
      // Expected: node 1 displaced by overlap * 0.5 * strength(0.7) = 2.45
      // (exactly the two-node result), node 0 bit-exact at the origin.
      const result = await runCollision(device, {
        positions: new Float32Array([0, 0, 3, 0, 2, 0]),
        sizes: new Float32Array([5, 5, DEAD_SLOT_RADIUS]),
        flags: new Uint32Array([NODE_FLAG_PINNED, 0, NODE_FLAG_DEAD]),
        variant,
      });

      // Pinned node: bit-exact hold
      assertEquals(result[0], 0, "pinned node displaced in x");
      assertEquals(result[1], 0, "pinned node displaced in y");

      // Dead slot: bit-exact hold
      assertEquals(result[4], 2, "dead slot displaced in x");
      assertEquals(result[5], 0, "dead slot displaced in y");

      // Live node: pushed away by the pinned node's full half-overlap,
      // unaffected by the dead slot
      const expectedX = 3 + (10 - 3) * 0.5 * 0.7;
      assert(
        Math.abs(result[2] - expectedX) < 1e-4,
        `live node not pushed by pinned node: x=${result[2]}, expected ${expectedX}`,
      );
      assertEquals(result[3], 0, "live node displaced in y");
    },
  );
}

gpuTest(
  "GPU collision (grid): pinned node is never displaced but still pushes; dead slots inert",
  async (device) => {
    // Same scenario as the O(n^2) variants, driven through the spatial-hash
    // grid pipeline. The dead slot keeps a live-looking radius so the grid
    // path must exclude it via NODE_FLAG_DEAD alone (not the size sentinel).
    const result = await runCollision(device, {
      positions: new Float32Array([0, 0, 3, 0, 2, 0]),
      sizes: new Float32Array([5, 5, 5]),
      flags: new Uint32Array([NODE_FLAG_PINNED, 0, NODE_FLAG_DEAD]),
      variant: "grid",
      bounds: { minX: -10, minY: -10, maxX: 10, maxY: 10 },
    });

    // Pinned node: bit-exact hold
    assertEquals(result[0], 0, "pinned node displaced in x");
    assertEquals(result[1], 0, "pinned node displaced in y");

    // Dead slot: bit-exact hold despite overlapping both live nodes
    assertEquals(result[4], 2, "dead slot displaced in x");
    assertEquals(result[5], 0, "dead slot displaced in y");

    // Live node: pushed only by the pinned node's half-overlap. If the dead
    // slot leaked into the cell lists it would add its own push and this
    // value would come out ~3.15 higher.
    const expectedX = 3 + (10 - 3) * 0.5 * 0.7;
    assert(
      Math.abs(result[2] - expectedX) < 1e-4,
      `live node not pushed by pinned node alone: x=${result[2]}, expected ${expectedX}`,
    );
    assertEquals(result[3], 0, "live node displaced in y");
  },
);
