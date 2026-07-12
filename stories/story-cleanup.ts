/**
 * Shared teardown registry for HeroineGraph stories.
 *
 * Storybook's HTML renderer has no per-story teardown hook for play-based
 * initialization, so each story registers its cleanup here and runs any
 * pending cleanups from previously rendered stories before initializing.
 * The registry is module-global, so switching between story files also
 * disposes the previous story's graph, timers, and listeners.
 */

type Cleanup = () => void;

const cleanups: Cleanup[] = [];

/** Register a cleanup to run before the next story initializes. */
export function registerStoryCleanup(cleanup: Cleanup): void {
  cleanups.push(cleanup);
}

/** Dispose everything created by previously rendered stories. */
export function runStoryCleanups(): void {
  while (cleanups.length > 0) {
    const cleanup = cleanups.pop()!;
    try {
      cleanup();
    } catch (err) {
      console.warn("Story cleanup failed:", err);
    }
  }
}
