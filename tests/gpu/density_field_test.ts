/**
 * GPU tests for findings #4/#10: the density-grid force field must be a
 * CONTINUOUS function of world position.
 *
 * The old implementation truncated each node to its containing cell for
 * both splatting and gradient sampling, making the force field
 * piecewise-constant per cell with jumps at every cell edge — the
 * grid/cross-hatch artifact and a boundary-jitter feedback loop. It also
 * never normalized the gradient by the world-space cell size, so force
 * magnitude depended on grid resolution and bounds extent (repulsion
 * stayed constant as the layout grew, feeding expansion).
 *
 * These tests drive the real DensityFieldAlgorithm plugin (the same
 * density_field.comp.wgsl runs inside Relativity Atlas every frame):
 *
 * - continuity: a probe straddling a cell boundary receives
 *   nearly-identical forces on both sides;
 * - scale normalization: doubling the layout and bounds halves the force
 *   (repulsion weakens as cells grow instead of feeding runaway expansion);
 * - no cross-hatch: over a uniform node lattice, interior forces are near
 *   zero compared to the genuine density falloff at the rim.
 */

import { assert } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
} from "../helpers/gpu.ts";
import { validateForceConfig } from "../../packages/core/src/simulation/config.ts";
import { mulberry32 } from "../fixtures/prng.ts";

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
      const device = await adapter!.requestDevice();
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

async function loadDensityAlgorithm(): Promise<HarnessForceAlgorithm> {
  const mod = await loadModuleInliningWgsl<
    { createDensityFieldAlgorithm(): HarnessForceAlgorithm }
  >(
    new URL(
      "../../packages/core/src/simulation/algorithms/density-field.ts",
      import.meta.url,
    ),
  );
  return mod.createDensityFieldAlgorithm();
}

interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * Run the full density pipeline (clear -> splat -> gradient forces) once
 * over the given positions and read back all per-node forces.
 */
async function runDensityPass(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  positions: Float32Array, // interleaved x,y
  bounds: Bounds,
): Promise<Float32Array> {
  const nodeCount = positions.length / 2;
  const forceConfig = validateForceConfig({});

  const positionsBuf = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positionsBuf, 0, positions.slice().buffer);
  const forcesBuf = device.createBuffer({
    size: nodeCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: nodeCount * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const context = {
    device,
    positions: positionsBuf,
    forces: forcesBuf,
    nodeCount,
    edgeCount: 0,
    forceConfig,
    bounds,
  };

  const pipelines = algorithm.createPipelines({ device });
  const buffers = algorithm.createBuffers(device, nodeCount);
  try {
    const bindGroups = algorithm.createBindGroups(device, pipelines, context, buffers);
    algorithm.updateUniforms(device, buffers, context);

    const encoder = device.createCommandEncoder();
    algorithm.recordRepulsionPass(encoder, pipelines, bindGroups, nodeCount);
    encoder.copyBufferToBuffer(forcesBuf, 0, readback, 0, nodeCount * 8);
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const forces = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return forces;
  } finally {
    buffers.destroy();
    positionsBuf.destroy();
    forcesBuf.destroy();
    readback.destroy();
  }
}

/**
 * A dense blob of background nodes plus one probe node (last index).
 * Bounds [0, 2048]^2 with the default 128-grid puts cell edges at
 * multiples of 16; the blob sits 2 cells left of the probe so the probe
 * lies inside the blob's splat support with a strong +x gradient.
 */
function blobAndProbe(probeX: number, probeY: number, scale = 1): Float32Array {
  const blobCount = 100;
  const rng = mulberry32(7);
  const positions = new Float32Array((blobCount + 1) * 2);
  for (let i = 0; i < blobCount; i++) {
    const angle = 2 * Math.PI * rng();
    const r = 8 * Math.sqrt(rng());
    positions[i * 2] = (992 + r * Math.cos(angle)) * scale;
    positions[i * 2 + 1] = (896 + r * Math.sin(angle)) * scale;
  }
  positions[blobCount * 2] = probeX * scale;
  positions[blobCount * 2 + 1] = probeY * scale;
  return positions;
}

const BOUNDS_2048: Bounds = { minX: 0, minY: 0, maxX: 2048, maxY: 2048 };

gpuTest(
  "GPU density field: force is continuous across cell boundaries (no cross-hatch steps)",
  async (device) => {
    const algorithm = await loadDensityAlgorithm();

    // Probe 0.1 world units either side of the cell edge at x = 1024.
    // The old cell-truncated field changed the entire gradient sample
    // point by one full cell here, flipping the force discontinuously
    // every frame for any node oscillating over the edge.
    const probeIdx = 100;
    const forcesLeft = await runDensityPass(
      device,
      algorithm,
      blobAndProbe(1023.9, 896),
      BOUNDS_2048,
    );
    const forcesRight = await runDensityPass(
      device,
      algorithm,
      blobAndProbe(1024.1, 896),
      BOUNDS_2048,
    );

    const lx = forcesLeft[probeIdx * 2];
    const ly = forcesLeft[probeIdx * 2 + 1];
    const rx = forcesRight[probeIdx * 2];
    const ry = forcesRight[probeIdx * 2 + 1];
    const magLeft = Math.hypot(lx, ly);
    const magRight = Math.hypot(rx, ry);
    console.log(
      `[gpu] density continuity: F(1023.9) = (${lx.toFixed(4)}, ${ly.toFixed(4)}), ` +
        `F(1024.1) = (${rx.toFixed(4)}, ${ry.toFixed(4)})`,
    );

    // The blob is to the probe's left: repulsion must push +x on both sides
    assert(lx > 0, `probe not pushed away from blob: F.x = ${lx}`);
    assert(rx > 0, `probe not pushed away from blob: F.x = ${rx}`);

    // 0.2 world units of movement through a smooth field: forces on the
    // two sides must be nearly identical. The old quantized field jumped
    // by a full cell's gradient difference here.
    const diff = Math.hypot(lx - rx, ly - ry);
    const scale = Math.max(magLeft, magRight);
    assert(
      diff < 0.1 * scale,
      `force steps at cell boundary: |dF| = ${diff} vs |F| = ${scale}`,
    );
  },
);

gpuTest(
  "GPU density field: force weakens as the layout grows (cell-size normalization)",
  async (device) => {
    const algorithm = await loadDensityAlgorithm();

    const probeIdx = 100;
    const forces1 = await runDensityPass(
      device,
      algorithm,
      blobAndProbe(1024.1, 896, 1),
      BOUNDS_2048,
    );
    // Same layout uniformly scaled x2, bounds scaled with it: the
    // normalized grid picture is identical, so the unnormalized gradient
    // was identical too — the old shader applied the SAME force at every
    // scale, sustaining expansion. Normalized by world cell size, the
    // force must halve.
    const forces2 = await runDensityPass(
      device,
      algorithm,
      blobAndProbe(1024.1, 896, 2),
      { minX: 0, minY: 0, maxX: 4096, maxY: 4096 },
    );

    const mag1 = Math.hypot(forces1[probeIdx * 2], forces1[probeIdx * 2 + 1]);
    const mag2 = Math.hypot(forces2[probeIdx * 2], forces2[probeIdx * 2 + 1]);
    console.log(
      `[gpu] density scaling: |F| at scale 1 = ${mag1.toFixed(4)}, at scale 2 = ${mag2.toFixed(4)}`,
    );
    assert(mag1 > 0, "probe receives no force at scale 1");
    const ratio = mag2 / mag1;
    assert(
      Math.abs(ratio - 0.5) < 0.05,
      `force not normalized by cell size: scale-2/scale-1 ratio ${ratio} (expected 0.5)`,
    );
  },
);

gpuTest(
  "GPU density field: uniform lattice interior has near-zero grid-alignment force variance",
  async (device) => {
    const algorithm = await loadDensityAlgorithm();

    // Perfectly uniform node lattice with spacing INCOMMENSURATE with the
    // 16-unit grid cells (11.3 units ~ 0.706 cells), clipped to a disc.
    // A continuous splat + interpolated gradient sees uniform density, so
    // interior forces are ~0 and only the rim (a genuine density edge)
    // pushes outward. The old cell-truncated splat aliased the lattice
    // against the grid (cells alternately caught 0/1/2 nodes), producing
    // strong grid-aligned interior forces — the cross-hatch pattern.
    // (A seeded random cloud is unusable here: its Poisson density noise
    // creates real interior gradients that swamp the artifact signal.)
    const spacing = 11.3;
    const cx = 1024 + 3.7; // offset so the lattice isn't grid-symmetric
    const cy = 1024 + 5.1;
    const discRadius = 600;
    const coords: number[] = [];
    const radii: number[] = [];
    const span = Math.ceil(discRadius / spacing);
    for (let gy = -span; gy <= span; gy++) {
      for (let gx = -span; gx <= span; gx++) {
        const x = cx + gx * spacing;
        const y = cy + gy * spacing;
        const r = Math.hypot(x - cx, y - cy);
        if (r > discRadius) continue;
        coords.push(x, y);
        radii.push(r);
      }
    }
    const nodeCount = radii.length;
    const positions = new Float32Array(coords);

    const forces = await runDensityPass(device, algorithm, positions, BOUNDS_2048);

    let interiorSum = 0;
    let interiorMax = 0;
    let interiorCount = 0;
    let rimSum = 0;
    let rimCount = 0;
    for (let i = 0; i < nodeCount; i++) {
      const mag = Math.hypot(forces[i * 2], forces[i * 2 + 1]);
      assert(Number.isFinite(mag), `non-finite force on node ${i}`);
      if (radii[i] < 0.75 * discRadius) {
        interiorSum += mag;
        interiorCount++;
        if (mag > interiorMax) interiorMax = mag;
      } else if (radii[i] > 0.95 * discRadius) {
        rimSum += mag;
        rimCount++;
      }
    }
    assert(interiorCount > 500 && rimCount > 100, "lattice produced too few sample nodes");

    const interiorMean = interiorSum / interiorCount;
    const rimMean = rimSum / rimCount;
    console.log(
      `[gpu] density lattice (${nodeCount} nodes): interior mean |F| = ` +
        `${interiorMean.toFixed(3)} (max ${interiorMax.toFixed(3)}), ` +
        `rim mean |F| = ${rimMean.toFixed(3)}`,
    );

    // The rim is a real density edge: nodes there must be pushed outward
    assert(rimMean > 0, "rim receives no outward force");
    // Uniform interior: grid-alignment variance must be near zero relative
    // to the genuine signal at the rim
    assert(
      interiorMean < 0.15 * rimMean,
      `grid-aligned interior forces on uniform lattice: interior ${interiorMean} ` +
        `vs rim ${rimMean}`,
    );
  },
);
