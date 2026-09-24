/** Cadence of submitted renders over the last second, independent of idle rAF callbacks. */
export class FrameRateMeter {
  private frames: number[] = [];

  record(now: number): void {
    this.frames.push(now);
    this.trim(now);
  }

  sample(now: number): number | null {
    this.trim(now);
    if (!this.frames.length) return null;
    if (now - this.frames[this.frames.length - 1]! >= 1000) return 0;
    const elapsed = Math.min(1000, now - this.frames[0]!);
    if (this.frames.length < 2 || elapsed < 250) return null;
    return (this.frames.length - 1) * 1000 / elapsed;
  }

  reset(): void { this.frames.length = 0; }

  private trim(now: number): void {
    let expired = 0;
    // Keep the boundary frame: N frames contain N − 1 measured intervals.
    while (expired + 1 < this.frames.length && this.frames[expired + 1]! <= now - 1000) expired++;
    if (expired) this.frames.splice(0, expired);
  }
}
