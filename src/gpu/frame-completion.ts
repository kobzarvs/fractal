export interface GpuCompletionSample {
  /** Observed completed GPU frames per second; not display/presentation FPS. */
  fps: number | null;
  pendingFrames: number;
  completionAgeMs: number | null;
}

const CHECKPOINT_MS = 100;
const WINDOW_MS = 1000;
const MIN_INTERVAL_MS = 250;
const MAX_FENCES = 8;
const MAX_OBSERVATIONS = 32;

/**
 * Observes GPU completion without waiting for it or changing submission cadence.
 * Call recordFrame once after each visible draw (exclude preparation), and sample
 * with fresh performance.now(), not a potentially delayed RAF callback timestamp.
 * Fences confirm cumulative prefixes, so a full queue cannot lose frame coverage.
 */
export class GpuFrameCompletion {
  private readonly gl: WebGL2RenderingContext;
  private readonly fences: (WebGLSync | null)[] = Array(MAX_FENCES).fill(null);
  private readonly coverage = new Float64Array(MAX_FENCES);
  private fenceHead = 0;
  private fenceCount = 0;
  private readonly observedAt = new Float64Array(MAX_OBSERVATIONS);
  private readonly observedCount = new Float64Array(MAX_OBSERVATIONS);
  private observationHead = 0;
  private observationCount = 0;
  private submitted = 0;
  private completed = 0;
  private checkpointCoverage = 0;
  private checkpointAt = -Infinity;
  private firstFrameAt: number | null = null;
  private lastCompletionAt: number | null = null;
  private baselineCoverage: number | null = null;
  private available = true;
  private disposed = false;

  constructor(gl: WebGL2RenderingContext) { this.gl = gl; }

  recordFrame(now: number): void {
    if (!this.available || this.disposed) return;
    this.firstFrameAt ??= now;
    this.submitted++;
    this.checkpoint(now);
  }

  sample(now: number): GpuCompletionSample {
    if (this.available && !this.disposed) {
      try {
        // Poll each oldest checkpoint once at most; never wait for an unfinished one.
        let confirmed = this.completed;
        while (this.fenceCount > 0) {
          const sync = this.fences[this.fenceHead]!;
          const status = this.gl.clientWaitSync(sync, 0, 0);
          if (status === this.gl.TIMEOUT_EXPIRED) break;
          if (status !== this.gl.ALREADY_SIGNALED && status !== this.gl.CONDITION_SATISFIED) {
            this.disable();
            break;
          }
          confirmed = this.coverage[this.fenceHead];
          this.gl.deleteSync(sync);
          this.fences[this.fenceHead] = null;
          this.fenceHead = (this.fenceHead + 1) % MAX_FENCES;
          this.fenceCount--;
        }
        if (this.available && confirmed > this.completed) {
          this.completed = confirmed;
          this.lastCompletionAt = now;
          // A saturated first batch may leave an uncheckpointed tail. Its later
          // catch-up fence must finish the initial baseline, not invent a burst.
          if (this.observationCount === 0 && this.baselineCoverage === null &&
              this.fenceCount === 0 && this.submitted > confirmed) this.baselineCoverage = this.submitted;
          if (this.baselineCoverage === null || confirmed >= this.baselineCoverage) {
            this.baselineCoverage = null;
            this.observe(now);
          }
        }
        // Also cover the final short batch if rendering stopped before its deadline.
        this.checkpoint(now);
      } catch {
        this.disable();
      }
    }

    if (!this.available || this.disposed) return { fps: null, pendingFrames: 0, completionAgeMs: null };
    let fps: number | null = null;
    if (this.firstFrameAt !== null && now - this.firstFrameAt >= WINDOW_MS) {
      this.prune(now);
      const interval = this.observationCount ? now - this.observedAt[this.observationHead] : 0;
      fps = interval >= MIN_INTERVAL_MS
        ? (this.completed - this.observedCount[this.observationHead]) * 1000 / interval
        : 0;
    }
    return {
      fps,
      pendingFrames: this.submitted - this.completed,
      completionAgeMs: this.lastCompletionAt === null ? null : Math.max(0, now - this.lastCompletionAt),
    };
  }

  /** Start a new generation. Deleted checkpoints can never confirm its frames. */
  reset(_now: number): void {
    this.clearFences(this.contextLost());
    this.submitted = this.completed = this.checkpointCoverage = 0;
    this.checkpointAt = -Infinity;
    this.firstFrameAt = this.lastCompletionAt = null;
    this.baselineCoverage = null;
    this.observationHead = this.observationCount = 0;
  }

  dispose(contextLost = false): void {
    this.clearFences(contextLost || this.contextLost());
    this.disposed = true;
    this.available = false;
  }

  private checkpoint(now: number): void {
    if (!this.available || this.disposed || this.fenceCount === MAX_FENCES ||
        this.submitted === this.checkpointCoverage || now - this.checkpointAt < CHECKPOINT_MS) return;
    try {
      if (this.contextLost()) { this.disable(true); return; }
      const sync = this.gl.fenceSync(this.gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      if (sync === null) { this.disable(); return; }
      const index = (this.fenceHead + this.fenceCount) % MAX_FENCES;
      this.fences[index] = sync;
      this.coverage[index] = this.submitted;
      this.fenceCount++;
      this.checkpointCoverage = this.submitted;
      this.checkpointAt = now;
      this.gl.flush();
    } catch {
      this.disable();
    }
  }

  private observe(now: number): void {
    this.prune(now);
    const last = (this.observationHead + this.observationCount - 1) % MAX_OBSERVATIONS;
    if (this.observationCount && this.observedAt[last] === now) {
      this.observedCount[last] = this.completed;
      return;
    }
    if (this.observationCount === MAX_OBSERVATIONS) {
      this.observationHead = (this.observationHead + 1) % MAX_OBSERVATIONS;
      this.observationCount--;
    }
    const index = (this.observationHead + this.observationCount) % MAX_OBSERVATIONS;
    this.observedAt[index] = now;
    this.observedCount[index] = this.completed;
    this.observationCount++;
    // The first confirmed batch is a baseline, not a burst assigned to this instant.
  }

  private prune(now: number): void {
    // Preserve the last observation at/before the window boundary. Infrequent
    // confirmations extend the interval instead of fabricating a recent burst.
    while (this.observationCount > 1) {
      const next = (this.observationHead + 1) % MAX_OBSERVATIONS;
      if (this.observedAt[next] > now - WINDOW_MS) break;
      this.observationHead = next;
      this.observationCount--;
    }
  }

  private contextLost(): boolean {
    try { return this.gl.isContextLost(); } catch { return true; }
  }

  private disable(contextLost = this.contextLost()): void {
    this.available = false;
    this.clearFences(contextLost);
  }

  private clearFences(contextLost: boolean): void {
    for (let index = 0; index < MAX_FENCES; index++) {
      const sync = this.fences[index];
      if (sync !== null && !contextLost) {
        try { this.gl.deleteSync(sync); } catch { /* Telemetry must never stop rendering. */ }
      }
      this.fences[index] = null;
    }
    this.fenceHead = this.fenceCount = 0;
  }
}
