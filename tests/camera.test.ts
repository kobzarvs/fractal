import test from 'node:test';
import assert from 'node:assert/strict';
import { decimalToFixed, fixedToFE, makeCamera, zoomAt, pan, cameraOffset, setZoom, screenOffset } from '../src/camera.ts';

test('decimal coordinates round symmetrically and preserve deep digits', () => {
  assert.equal(decimalToFixed('-.125', 128), -(1n << 125n));
  assert.equal(decimalToFixed('1.25e-2', 8), 3n);
  assert.equal(decimalToFixed('-1.25e-2', 8), -3n);
  assert.throws(() => decimalToFixed('nope', 128));
  const a = decimalToFixed('-1.12345678901234567890123456789', 576);
  const b = decimalToFixed('-1.12345678901234567890123456788', 576);
  assert.notEqual(a, b);
});
test('floating exponent retains values below double range', () => {
  assert.deepEqual(fixedToFE(1n, 2048), [1, -2048]);
  assert.deepEqual(fixedToFE(-(1n << 1000n), 2048), [-1, -1048]);
});
test('deep pan and zoom anchor stay in fixed point', () => {
  const c = makeCamera(1.5); setZoom(c, 120);
  const before = { ...c };
  zoomAt(c, 0, 0, -1);
  assert.deepEqual(cameraOffset(c, before), { x: [0, 0], y: [0, 0] });
  pan(c, .1, -.2);
  const delta = cameraOffset(c, before);
  assert.notEqual(delta.x[0], 0); assert.ok(delta.x[1] < -390);
  assert.ok(c.bits >= 576);
});
test('DOM vertical coordinates match the shader and zoom preserves the cursor anchor', () => {
  assert.deepEqual(screenOffset(300, 0, 600, 400), [0, -.5]);
  assert.deepEqual(screenOffset(300, 400, 600, 400), [0, .5]);
  const c = makeCamera(1.5); c.logScale = 2;
  const oldY = c.y; zoomAt(c, 0, -.25, -1);
  assert.equal(c.y, oldY - (1n << BigInt(c.bits - 1)));
  const beforePan = c.y; pan(c, 0, .25);
  assert.equal(c.y, beforePan - (1n << BigInt(c.bits - 1)));
});
