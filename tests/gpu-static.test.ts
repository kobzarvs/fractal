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

test('ordinary ring layout keeps its existing atlas positions', () => {
  const layout = createRingLayout(1280, 720, 16384);
  assert.ok(layout);
  assert.deepEqual([layout.width, layout.height], [6931, 1936]);
  assert.deepEqual(layout.bands.map(band => [band.x, band.row, band.angles, band.rings]), [
    [0, 0, 6931, 1154], [0, 1154, 3466, 772], [3466, 1154, 1733, 390],
    [5199, 1154, 867, 199], [6066, 1154, 434, 103], [6500, 1154, 217, 55],
    [6717, 1154, 109, 32], [6826, 1154, 55, 20], [6881, 1154, 28, 14],
    [6909, 1154, 16, 11], [0, 1926, 16, 10],
  ]);
  for (const band of layout.bands) {
    assert.equal(band.columns, band.angles);
    assert.equal(band.strips, 1);
  }
});

test('large canvases preserve angular density and pack physical strips without overlap', () => {
  for (const [width, height, outerAngles] of [
    [1920, 1080, 10391], [2560, 1440, 13851], [3840, 2160, 20772],
  ]) {
    const layout = createRingLayout(width, height, 16384);
    assert.ok(layout, `${width}x${height} should fit the GPU texture dimensions`);
    assert.ok(layout.width <= 16384 && layout.height <= 16384);
    assert.equal(layout.bands[0].angles, outerAngles);
    for (const [index, band] of layout.bands.entries()) {
      assert.equal(band.angles, Math.max(16, Math.ceil(2 * Math.PI * band.radius * 1.5)));
      assert.ok(band.columns > 0 && band.columns <= 16384);
      assert.ok(band.strips >= 1 && band.columns * band.strips >= band.angles);
      assert.ok(band.x >= 0 && band.row >= 0);
      assert.ok(band.x + band.columns <= layout.width);
      assert.ok(band.row + band.rings * band.strips <= layout.height);
      for (const other of layout.bands.slice(0, index)) {
        const overlapX = band.x < other.x + other.columns && other.x < band.x + band.columns;
        const overlapY = band.row < other.row + other.rings * other.strips &&
          other.row < band.row + band.rings * band.strips;
        assert.ok(!overlapX || !overlapY, `physical bands ${index} and ${layout.bands.indexOf(other)} overlap`);
      }
    }
    assert.equal(layout.bands[0].strips, width === 3840 ? 2 : 1);
  }
});

test('ring layout rejects invalid dimensions and a texture too small for physical bands', () => {
  assert.equal(createRingLayout(0, 1080, 16384), null);
  assert.equal(createRingLayout(Number.NaN, 1080, 16384), null);
  assert.equal(createRingLayout(1280, 720, 0), null);
  assert.equal(createRingLayout(8000, 8000, 4096), null);
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
