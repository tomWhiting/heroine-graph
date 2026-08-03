/**
 * Render pause arbitration
 *
 * Presentation is suspended on two independent axes: an explicit host pause
 * (the embedding app knows its surface is occluded, which `document.hidden`
 * cannot tell it — requestAnimationFrame keeps firing for a visible tab whose
 * canvas is covered) and page visibility. They are tracked as separate flags
 * because a visibility event must never lift a host pause: a host that pauses
 * while occluded, then tabs away and back, would otherwise return presenting.
 *
 * @module
 */

/** The presenting half of a render loop, as this gate drives it. */
export interface RenderPauseTarget {
  /** Suspend presentation. */
  pause: () => void;
  /** Resume presentation. */
  resume: () => void;
}

/**
 * Arbitrates host and visibility pause requests into one target state.
 *
 * The target is driven only on effective transitions, so a resume side effect
 * (marking the next frame dirty, say) fires exactly when frames actually
 * resume and not on every redundant call.
 */
export class RenderPauseGate {
  readonly #target: RenderPauseTarget;
  #byHost = false;
  #byVisibility = false;

  constructor(target: RenderPauseTarget) {
    this.#target = target;
  }

  /** Whether presentation is currently suspended, for any reason. */
  get isPaused(): boolean {
    return this.#byHost || this.#byVisibility;
  }

  /** Whether the host asked for the pause, as opposed to page visibility. */
  get isPausedByHost(): boolean {
    return this.#byHost;
  }

  /** Suspend presentation on the host's behalf. No-op when already held. */
  pauseByHost(): void {
    this.#set(true, this.#byVisibility);
  }

  /** Release the host's pause. Visibility may still hold presentation. */
  resumeByHost(): void {
    this.#set(false, this.#byVisibility);
  }

  /** Follow a visibilitychange. Never overrides an outstanding host pause. */
  setHidden(hidden: boolean): void {
    this.#set(this.#byHost, hidden);
  }

  #set(byHost: boolean, byVisibility: boolean): void {
    const was = this.isPaused;
    this.#byHost = byHost;
    this.#byVisibility = byVisibility;
    if (this.isPaused === was) return;
    if (this.isPaused) this.#target.pause();
    else this.#target.resume();
  }
}
