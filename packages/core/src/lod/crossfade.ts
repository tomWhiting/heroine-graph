/**
 * Per-Node Render-Alpha Crossfade Scheduler
 *
 * Semantic LOD hides and reveals whole subtrees at once. Popping them in and
 * out on a single frame reads as a glitch, so each node's render alpha is
 * ramped over `transitionMs` instead. Alpha lives in its own f32-per-slot GPU
 * buffer (`SimulationBuffers.nodeAlpha`) that only the node render pipeline
 * reads.
 *
 * Two properties are load-bearing and neither is expressible in a type:
 *
 * 1. **A transition must never reheat the simulation.** At the codebase's
 *    standard reheat a node travels ~25 node-widths over a 150 ms crossfade,
 *    which tears the fade away from the geometry it belongs to. Nothing in
 *    this module touches simulation alpha, and nothing in it may.
 * 2. **Time is an argument, never a reading.** The hot path takes timestamps
 *    from the caller so a fade replays identically in a test.
 *
 * The class is pure CPU state plus one upload: it owns the authoritative
 * Float32Array shadow, derives per-frame values from the supplied clock, and
 * flushes the slots that actually changed as a single contiguous range write —
 * the grow-only scratch plus dirty-upload pattern of
 * `layers/stream_intensity.ts`.
 *
 * @module
 */

/**
 * Fully visible. Every slot holds this until a fade touches it, so a graph
 * with LOD off renders exactly as it did before the buffer existed (the
 * fragment shader's multiply by 1.0 is bit-exactly the identity).
 */
export const NODE_ALPHA_OPAQUE = 1;

/** Fully faded out; the node contributes nothing to the frame. */
export const NODE_ALPHA_TRANSPARENT = 0;

/** One node's in-flight ramp. Removed from the map the instant it lands. */
interface Fade {
  /** Alpha at the moment the fade was scheduled. */
  from: number;
  /** {@link NODE_ALPHA_OPAQUE} or {@link NODE_ALPHA_TRANSPARENT}. */
  to: number;
  /** Clock value at which the ramp started. */
  startMs: number;
  /**
   * `transitionMs` scaled by `|to - from|`. Scaling keeps the ramp *rate*
   * constant, which is what makes a reversal mid-fade look continuous: a node
   * caught at 0.5 on the way down reaches 1.0 in half a transition rather than
   * crawling back over a full one.
   */
  durationMs: number;
}

/**
 * Schedules and drives per-node alpha ramps.
 *
 * One instance per graph. The caller owns the GPU buffer and passes it to
 * {@link flush}; the scheduler never allocates or destroys GPU resources, so
 * an LOD transition can never change buffer identity (which would invalidate
 * the prebuilt ping-pong bind groups).
 */
export class CrossfadeScheduler {
  /** Grow-only shadow of the GPU buffer; only `[0, capacity)` is live. */
  #alpha = new Float32Array(0);
  #capacity = 0;
  #fades = new Map<number, Fade>();
  /** Half-open range of slots changed since the last {@link flush}. */
  #dirtyLo = Infinity;
  #dirtyHi = 0;
  #lastMs = -Infinity;

  /** Number of live slots, as last set by {@link resize}. */
  get capacity(): number {
    return this.#capacity;
  }

  /** Fades still in flight. Zero means every slot has reached its target. */
  get activeCount(): number {
    return this.#fades.size;
  }

  /**
   * Track a change in the node capacity of the GPU buffer.
   *
   * Slots entering the live range are reset to {@link NODE_ALPHA_OPAQUE} and
   * are *not* marked dirty: the caller has just created (or recreated) the
   * buffer at the new capacity with the same value, so re-uploading it would
   * duplicate that write. Fades on slots leaving the live range are dropped.
   */
  resize(capacity: number): void {
    const next = Math.max(0, Math.trunc(capacity));

    if (next > this.#alpha.length) {
      const grown = new Float32Array(next);
      grown.fill(NODE_ALPHA_OPAQUE);
      grown.set(this.#alpha.subarray(0, this.#capacity));
      this.#alpha = grown;
    } else if (next > this.#capacity) {
      // Re-entering slots still hold the previous graph's values.
      this.#alpha.fill(NODE_ALPHA_OPAQUE, this.#capacity, next);
    } else if (next < this.#capacity) {
      for (const slot of this.#fades.keys()) {
        if (slot >= next) this.#fades.delete(slot);
      }
      // An empty or inverted range reads as clean in flush().
      if (this.#dirtyHi > next) this.#dirtyHi = next;
    }

    this.#capacity = next;
  }

  /**
   * Cancel every fade and return every live slot to
   * {@link NODE_ALPHA_OPAQUE}, marking the whole range for upload.
   *
   * This is the "LOD is off" state, and the state a graph load lands in.
   */
  reset(): void {
    this.#fades.clear();
    this.#alpha.fill(NODE_ALPHA_OPAQUE, 0, this.#capacity);
    this.#dirtyLo = 0;
    this.#dirtyHi = this.#capacity;
  }

  /**
   * Current alpha of a slot. Slots outside the live range read as
   * {@link NODE_ALPHA_OPAQUE}, matching what the GPU buffer holds there.
   */
  alphaOf(slot: number): number {
    return slot >= 0 && slot < this.#capacity ? this.#alpha[slot] : NODE_ALPHA_OPAQUE;
  }

  /**
   * Ramp `slots` up to {@link NODE_ALPHA_OPAQUE}.
   *
   * @param slots - Node slots; values outside the live range are ignored
   * @param nowMs - Caller's clock (see module doc: never read here)
   * @param transitionMs - Duration of a full 0→1 ramp; `<= 0` applies at once
   */
  fadeIn(slots: Iterable<number>, nowMs: number, transitionMs: number): void {
    this.#retarget(slots, NODE_ALPHA_OPAQUE, nowMs, transitionMs);
  }

  /**
   * Ramp `slots` down to {@link NODE_ALPHA_TRANSPARENT}.
   *
   * @param slots - Node slots; values outside the live range are ignored
   * @param nowMs - Caller's clock (see module doc: never read here)
   * @param transitionMs - Duration of a full 1→0 ramp; `<= 0` applies at once
   */
  fadeOut(slots: Iterable<number>, nowMs: number, transitionMs: number): void {
    this.#retarget(slots, NODE_ALPHA_TRANSPARENT, nowMs, transitionMs);
  }

  /**
   * Advance every in-flight fade to `nowMs`.
   *
   * @returns whether any slot's alpha changed (i.e. whether {@link flush} has
   *   work to do)
   */
  advance(nowMs: number): boolean {
    return this.#advanceTo(this.#clock(nowMs));
  }

  /**
   * Upload the slots changed since the last flush as one contiguous range
   * write, and clear the dirty range.
   *
   * @returns whether anything was written
   */
  flush(device: GPUDevice, buffer: GPUBuffer): boolean {
    const lo = this.#dirtyLo;
    const hi = this.#dirtyHi;
    if (hi <= lo) return false;

    // dataOffset/size are element counts because the source is a TypedArray,
    // so the dirty range uploads without copying it out first.
    device.queue.writeBuffer(buffer, lo * 4, this.#alpha, lo, hi - lo);
    this.#dirtyLo = Infinity;
    this.#dirtyHi = 0;
    return true;
  }

  /**
   * The caller's clock, clamped to be non-decreasing.
   *
   * A clock that stepped backwards would walk a fade back up its own ramp —
   * precisely the flicker the crossfade exists to remove — so time is clamped
   * rather than trusted. The clamp also makes `now - startMs >= 0` an
   * invariant, which is what lets {@link #advanceTo} skip a lower bound on the
   * interpolation parameter. NaN leaves the clock where it was.
   */
  #clock(nowMs: number): number {
    if (nowMs > this.#lastMs) this.#lastMs = nowMs;
    return this.#lastMs;
  }

  #retarget(
    slots: Iterable<number>,
    target: number,
    nowMs: number,
    transitionMs: number,
  ): void {
    const now = this.#clock(nowMs);
    // Branch off the values as they are *at* `now`, not as of the last
    // advance, so a retarget is independent of whether the caller ticked first.
    this.#advanceTo(now);

    const fullMs = transitionMs > 0 ? transitionMs : 0;

    for (const slot of slots) {
      if (!Number.isInteger(slot) || slot < 0 || slot >= this.#capacity) continue;

      const existing = this.#fades.get(slot);
      // Re-issuing the target a fade is already heading for must not restart
      // it: the LOD controller re-evaluates a band on every crossing, and a
      // restart would stall the ramp indefinitely.
      if (existing ? existing.to === target : this.#alpha[slot] === target) continue;

      const from = this.#alpha[slot];
      const span = Math.abs(target - from);
      if (fullMs === 0 || span === 0) {
        this.#fades.delete(slot);
        this.#write(slot, target);
        continue;
      }
      this.#fades.set(slot, {
        from,
        to: target,
        startMs: now,
        durationMs: fullMs * span,
      });
    }
  }

  #advanceTo(now: number): boolean {
    if (this.#fades.size === 0) return false;

    let changed = false;
    for (const [slot, fade] of this.#fades) {
      const elapsed = now - fade.startMs;
      if (elapsed >= fade.durationMs) {
        this.#fades.delete(slot);
        // Landing writes the endpoint literally, so a fade terminates at
        // exactly 0 or 1 rather than at whatever the ramp rounded to.
        changed = this.#write(slot, fade.to) || changed;
        continue;
      }
      const t = elapsed / fade.durationMs;
      changed = this.#write(slot, fade.from + (fade.to - fade.from) * t) || changed;
    }
    return changed;
  }

  /** Store one slot at f32 precision and widen the dirty range if it moved. */
  #write(slot: number, value: number): boolean {
    const stored = Math.fround(value);
    if (this.#alpha[slot] === stored) return false;

    this.#alpha[slot] = stored;
    if (slot < this.#dirtyLo) this.#dirtyLo = slot;
    if (slot >= this.#dirtyHi) this.#dirtyHi = slot + 1;
    return true;
  }
}
