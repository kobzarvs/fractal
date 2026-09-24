import assert from 'node:assert/strict';
import test from 'node:test';
import { historyTransform, temporalJitter, TemporalAccumulator } from '../src/gpu/temporal.ts';
import type { Camera } from '../src/camera.ts';

const camera: Camera = { x: 0n, y: 0n, bits: 128, logScale: 0 };
const near = (actual: number, expected: number, tolerance = 1e-14) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} differs from ${expected}`);

test('temporal jitter uses the centred original sequence for every AA setting', () => {
  near(temporalJitter(0, 3).x, -1 / 3);
  near(temporalJitter(2, 3).x, 1 / 3);
  for (let samples = 1; samples <= 5; samples++) {
    const offsets = Array.from({ length: samples }, (_, index) => temporalJitter(index, samples));
    near(offsets.reduce((sum, offset) => sum + offset.x, 0), 0);
    near(offsets.reduce((sum, offset) => sum + offset.y, 0), 0);
    if (samples > 1) assert.equal(new Set(offsets.map(offset => offset.y)).size, samples);
  }
});

test('stationary temporal frames use a running mean and settle at the sample budget', () => {
  const accumulator = new TemporalAccumulator();
  assert.equal(accumulator.settling, false);
  const first = accumulator.prepare(camera, 600, 300, 3, 'a');
  assert.equal(first.historyWeight, 0);
  assert.equal(accumulator.settling, true);
  const second = accumulator.prepare(camera, 600, 300, 3, 'a');
  assert.equal(second.historyWeight, 1 / 2);
  assert.equal(accumulator.settling, true);
  const third = accumulator.prepare(camera, 600, 300, 3, 'a');
  near(third.historyWeight, 2 / 3);
  assert.equal(accumulator.settling, false);
});

test('history reprojection retains tiny nonzero camera motion below double range', () => {
  const previous: Camera = { x: 1n << 2048n, y: -(1n << 2048n), bits: 2048, logScale: -1500 };
  const current = { ...previous, x: previous.x + (1n << 546n), y: previous.y + (1n << 545n) };
  const transform = historyTransform(current, previous, 2);
  assert.deepEqual(transform, { scale: 1, x: 1 / 8, y: -1 / 8 });
  const greaterPrecision = { ...current, x: current.x << 64n, y: current.y << 64n, bits: 2112 };
  assert.deepEqual(historyTransform(greaterPrecision, previous, 2), transform);
});

test('motion reprojects history, damps scale changes, then starts a fresh stationary mean', () => {
  const accumulator = new TemporalAccumulator();
  accumulator.prepare(camera, 600, 300, 3, 'a');
  accumulator.prepare(camera, 600, 300, 3, 'a');
  const moved = { ...camera, x: 1n << 124n, logScale: Math.log2(1.125) };
  const motion = accumulator.prepare(moved, 600, 300, 3, 'a');
  assert.equal(motion.moving, true);
  near(motion.historyOffset[0], 1 / 32);
  near(motion.historyWeight, (2 / 3) * Math.exp(-4 * Math.log(1.125)));
  assert.equal(accumulator.settling, true);
  const stopped = accumulator.prepare(moved, 600, 300, 3, 'a');
  assert.equal(stopped.moving, false);
  assert.equal(stopped.historyWeight, 0);
  assert.deepEqual(stopped.jitter, temporalJitter(0, 3));
});

test('camera jumps, changed parameters, resize and explicit reset discard old history', () => {
  for (const change of ['jump', 'zoom', 'parameters', 'resize', 'samples', 'reset']) {
    const accumulator = new TemporalAccumulator();
    accumulator.prepare(camera, 600, 300, 3, 'a');
    accumulator.prepare(camera, 600, 300, 3, 'a');
    if (change === 'reset') accumulator.reset();
    const current = { ...camera,
      x: change === 'jump' ? 1n << 128n : 0n,
      logScale: change === 'zoom' ? 1 : 0,
    };
    const frame = accumulator.prepare(current, change === 'resize' ? 800 : 600, 300,
      change === 'samples' ? 5 : 3, change === 'parameters' ? 'b' : 'a');
    assert.equal(frame.historyWeight, 0, change);
    assert.equal(frame.moving, false, change);
    assert.equal(accumulator.settling, true, change);
  }
});
