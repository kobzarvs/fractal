import test from 'node:test';
import assert from 'node:assert/strict';
import { GpuFrameCompletion } from '../src/gpu/frame-completion.ts';

function gpu() {
  let submitted = 0, completed = 0, fences = 0, waits = 0, flushes = 0, lost = false;
  let allocationFails = false, waitFails = false;
  const live = new Set<{ covered: number }>();
  const gl = {
    SYNC_GPU_COMMANDS_COMPLETE: 1, TIMEOUT_EXPIRED: 2, ALREADY_SIGNALED: 3, CONDITION_SATISFIED: 4, WAIT_FAILED: 5,
    fenceSync(condition: number, flags: number) {
      assert.equal(condition, 1); assert.equal(flags, 0);
      if (allocationFails) return null;
      const sync = { covered: submitted }; live.add(sync); fences++; return sync;
    },
    clientWaitSync(sync: { covered: number }, flags: number, timeout: number) {
      assert.equal(flags, 0); assert.equal(timeout, 0, 'completion polling must never block'); waits++;
      assert.ok(live.has(sync));
      return waitFails ? 5 : completed >= sync.covered ? 3 : 2;
    },
    deleteSync(sync: { covered: number }) { assert.equal(lost, false, 'never delete invalid context handles'); live.delete(sync); },
    flush() { flushes++; }, isContextLost() { return lost; },
  };
  const observer = new GpuFrameCompletion(gl as unknown as WebGL2RenderingContext);
  return { observer, live, submit(now: number) { submitted++; observer.recordFrame(now); },
    finish(count = submitted) { completed = count; },
    failAllocation() { allocationFails = true; }, failWait() { waitFails = true; },
    lose() { lost = true; live.clear(); },
    get submitted() { return submitted; }, get fences() { return fences; }, get waits() { return waits; }, get flushes() { return flushes; } };
}

test('submitted frames never become completed FPS without a signaled fence', () => {
  const g = gpu();
  assert.deepEqual(g.observer.sample(0), { fps: null, pendingFrames: 0, completionAgeMs: null });
  for (let frame = 0; frame < 120; frame++) {
    g.submit(frame * 1000 / 120);
    if (frame === 60) assert.equal(g.observer.sample(500).fps, null);
  }
  assert.deepEqual(g.observer.sample(1000), { fps: 0, pendingFrames: 120, completionAgeMs: null });
  assert.equal(g.observer.sample(5000).fps, 0);
  assert.ok(g.live.size <= 8); assert.ok(g.fences <= 8); assert.equal(g.flushes, g.fences);
  g.observer.dispose(); assert.equal(g.live.size, 0);
});

test('one genuinely completed frame per second cannot be reported as 120 submitted FPS', () => {
  const g = gpu();
  for (let frame = 0; frame < 120; frame++) g.submit(frame * 1000 / 120);
  let observedOne = false;
  for (let second = 1; second <= 120; second++) {
    g.finish(second);
    const sample = g.observer.sample(second * 1000);
    if (sample.fps !== null) {
      assert.ok(sample.fps <= 1.01, `confirmed throughput ${sample.fps} at ${second}s`);
      if (sample.fps > .99) observedOne = true;
    }
  }
  assert.ok(observedOne, 'cumulative checkpoints eventually measure the actual one frame/second rate');
  assert.equal(g.observer.sample(121001).fps, 0, 'an old completion must not keep a stale rate alive');
  assert.equal(g.observer.sample(121001).pendingFrames, 0);
  g.observer.dispose();
});

test('queue saturation stays bounded and a later checkpoint covers the missing submitted prefix', () => {
  const g = gpu();
  for (let frame = 0; frame < 200; frame++) g.submit(frame * 10);
  assert.equal(g.live.size, 8);
  g.finish(1); let sample = g.observer.sample(2000);
  assert.equal(sample.pendingFrames, 199); assert.equal(g.live.size, 8, 'the free slot captures the uncovered prefix');
  g.finish(200); sample = g.observer.sample(2500);
  assert.equal(sample.pendingFrames, 0); assert.equal(g.live.size, 0);
  assert.ok(sample.fps !== null && Number.isFinite(sample.fps));
  g.observer.dispose();
});

test('completion bursts use observation time and preserve the older boundary anchor', () => {
  const g = gpu();
  g.submit(0); g.finish(1); g.observer.sample(0);
  for (let frame = 1; frame <= 100; frame++) g.submit(frame * 10);
  g.observer.sample(4900); // No new completions: do not manufacture completion observations.
  g.finish(); const firstBurst = g.observer.sample(5000);
  // Queue saturation needs one additional marker to cover the remaining prefix.
  const completeBurst = g.observer.sample(5000);
  assert.equal(completeBurst.pendingFrames, 0);
  assert.equal(completeBurst.fps, 20, '100 newly confirmed frames over five observed seconds');
  assert.ok(firstBurst.fps !== null && Number.isFinite(firstBurst.fps));
  assert.equal(g.observer.sample(6001).fps, 0);
  assert.equal(g.observer.sample(6500).completionAgeMs, 1500);
  g.observer.dispose();
});

test('a delayed first batch becomes the baseline rather than a fabricated instantaneous FPS', () => {
  const g = gpu();
  for (let frame = 0; frame < 120; frame++) g.submit(frame * 10);
  g.finish(); const sample = g.observer.sample(5000);
  assert.ok(sample.fps === null || sample.fps === 0);
  const caughtUp = g.observer.sample(5000);
  assert.ok(caughtUp.fps === null || caughtUp.fps === 0, 'same-timestamp fence catch-up cannot divide by zero');
  assert.equal(g.observer.sample(5250).fps, 0, 'all confirmations at the first timestamp share one baseline');
  g.observer.dispose();
});

test('the uncheckpointed tail of a delayed first batch cannot become a later completion spike', () => {
  const g = gpu();
  for (let frame = 0; frame < 120; frame++) g.submit(frame * 10);
  g.finish();
  assert.equal(g.observer.sample(5000).fps, 0);
  const caughtUp = g.observer.sample(5250);
  assert.equal(caughtUp.pendingFrames, 0);
  assert.equal(caughtUp.fps, 0, 'the initial already-completed prefix is one baseline even across later polls');
  assert.equal(g.observer.sample(5500).fps, 0);
  g.observer.dispose();
});

test('sampling an idle short burst eventually checkpoints and confirms its final frames', () => {
  const g = gpu();
  g.submit(0); g.finish(); g.observer.sample(0);
  g.submit(10); g.submit(20); g.finish();
  assert.equal(g.observer.sample(50).pendingFrames, 2);
  assert.equal(g.live.size, 0, 'the checkpoint interval has not elapsed');
  assert.equal(g.observer.sample(100).pendingFrames, 2);
  assert.equal(g.live.size, 1, 'sample must checkpoint the tail even without a new frame');
  assert.equal(g.observer.sample(101).pendingFrames, 0);
  assert.equal(g.observer.sample(1200).fps, 0);
  g.observer.dispose();
});

test('steady confirmed throughput follows GPU completions over the observation window', () => {
  const g = gpu();
  let latest = g.observer.sample(0);
  for (let frame = 0; frame <= 360; frame++) {
    const now = frame * 1000 / 120;
    g.submit(now); g.finish(); latest = g.observer.sample(now);
  }
  assert.ok(latest.fps !== null && latest.fps >= 108 && latest.fps <= 121, `observed ${latest.fps} FPS`);
  assert.ok(g.fences <= 31, 'checkpoints are batched, not allocated every frame');
  assert.equal(g.observer.sample(4200).fps, 0);
  g.observer.dispose();
});

test('reset creates a new measurement generation and cannot consume old GPU completions', () => {
  const g = gpu();
  for (let frame = 0; frame < 50; frame++) g.submit(frame * 10);
  g.observer.reset(500); assert.equal(g.live.size, 0);
  g.finish(50); assert.deepEqual(g.observer.sample(2000), { fps: null, pendingFrames: 0, completionAgeMs: null });
  g.submit(2100); assert.equal(g.observer.sample(3100).fps, 0);
  assert.equal(g.observer.sample(3100).pendingFrames, 1);
  g.finish(); assert.equal(g.observer.sample(3200).pendingFrames, 0);
  g.observer.dispose();
});

test('allocation or wait failure disables telemetry without throwing or reporting invented zero FPS', () => {
  for (const failure of ['allocation', 'wait']) {
    const g = gpu();
    if (failure === 'allocation') g.failAllocation();
    g.submit(0);
    if (failure === 'wait') g.failWait();
    assert.doesNotThrow(() => g.observer.sample(1000));
    assert.deepEqual(g.observer.sample(2000), { fps: null, pendingFrames: 0, completionAgeMs: null }, 'disabled telemetry cannot look like a GPU still waiting');
    assert.equal(g.live.size, 0);
    assert.doesNotThrow(() => g.submit(3000));
    g.observer.dispose();
  }
});

test('context loss disposal never deletes invalid synchronization handles', () => {
  const g = gpu(); g.submit(0); g.lose();
  assert.doesNotThrow(() => g.observer.dispose(true));
  assert.equal(g.observer.sample(2000).fps, null);
  assert.doesNotThrow(() => g.observer.recordFrame(3000));
});
