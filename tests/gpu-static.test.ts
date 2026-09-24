import test from 'node:test';
import assert from 'node:assert/strict';
import { createRingLayout, ringWindow, missingRingRanges } from '../src/gpu/rings.ts';

test('ring layout covers the canvas at the original 1.5 sampling density', () => {
  const layout = createRingLayout(800, 600, 16384);
  assert.ok(layout);
  assert.ok(layout.bands[0].radius > Math.hypot(800, 600) / 2);
  for (const band of layout.bands) {
    assert.ok(band.angles >= Math.ceil(2 * Math.PI * band.radius * 1.5));
    for (const logScale of [2, -100, -3400]) {
      const [first, last] = ringWindow(band, logScale, 600, 3);
      assert.ok(last - first + 1 <= band.rings, 'ring window must fit its circular storage');
    }
  }
});

test('incremental ring ranges preserve overlaps and fill every missing row once', () => {
  assert.deepEqual(missingRingRanges([10, 20], [10, 20]), []);
  assert.deepEqual(missingRingRanges([12, 22], [10, 20]), [[21, 22]]);
  assert.deepEqual(missingRingRanges([8, 18], [10, 20]), [[8, 9]]);
  assert.deepEqual(missingRingRanges([8, 22], [10, 20]), [[8, 9], [21, 22]]);
  assert.deepEqual(missingRingRanges([50, 60], [10, 20]), [[50, 60]]);
  assert.deepEqual(missingRingRanges([10, 20], null), [[10, 20]]);
});

test('oversized ring cache falls back rather than reducing sampling density', () => {
  assert.equal(createRingLayout(8000, 8000, 4096), null);
});
