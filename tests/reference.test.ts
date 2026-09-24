import assert from 'node:assert/strict';
import test from 'node:test';
import { computeReferenceJs } from '../src/compute/reference-js.ts';
import type { ReferenceRequest, ReferenceResult } from '../src/types.ts';

const SCALE = 1n << 128n;

function request(x: bigint, y: bigint, options: Partial<ReferenceRequest> = {}): ReferenceRequest {
  return {
    id: 17, x: String(x * SCALE), y: String(y * SCALE), bits: 128,
    iterations: 8, fold: 0, celtic: 0, ...options,
  };
}

function orbitAt(result: ReferenceResult, iteration: number): [number, number] {
  const i = iteration * 4;
  return [result.orbit[i] * 2 ** result.orbit[i + 1],
    result.orbit[i + 2] * 2 ** result.orbit[i + 3]];
}

test('origin stays at zero through the requested final orbit sample', async () => {
  const result = await computeReferenceJs(request(0n, 0n, { iterations: 1024 }));
  assert.equal(result.length, 1025);
  assert.equal(result.capacity, 2048);
  assert.equal(result.orbit.length, 2048 * 4);
  assert.deepEqual(orbitAt(result, 0), [0, 0]);
  assert.deepEqual(orbitAt(result, 1024), [0, 0]);
  assert.equal(result.backend, 'js');
});

test('real quadratic orbit records the sample that first crosses the escape radius', async () => {
  const result = await computeReferenceJs(request(2n, 0n));
  assert.equal(result.length, 5);
  assert.deepEqual([0, 1, 2, 3, 4].map(i => orbitAt(result, i)),
    [[0, 0], [2, 0], [6, 0], [38, 0], [1446, 0]]);
});

test('fold changes the sign of a negative cross term', async () => {
  const plain = await computeReferenceJs(request(-1n, 1n));
  const folded = await computeReferenceJs(request(-1n, 1n, { fold: 1 }));
  assert.deepEqual(orbitAt(plain, 2), [-1, -1]);
  assert.deepEqual(orbitAt(folded, 2), [-1, 3]);
});

test('Celtic mode reflects a negative real square and retains its signed square', async () => {
  const plain = await computeReferenceJs(request(1n, 2n));
  const celtic = await computeReferenceJs(request(1n, 2n, { celtic: 1 }));
  assert.deepEqual(orbitAt(plain, 2), [-2, 6]);
  assert.deepEqual(orbitAt(celtic, 2), [4, 6]);
  assert.equal(celtic.realOrbit.length, celtic.capacity * 4);
  assert.equal(celtic.realOrbit[4] * 2 ** celtic.realOrbit[5], -3);
  assert.equal(plain.realOrbit.length, 4);
});

test('fold and Celtic controls round to 1/1024 before use', async () => {
  const result = await computeReferenceJs(request(-1n, 1n, { fold: 0.0006, celtic: 0.0006 }));
  assert.equal(result.fold, 1 / 1024);
  assert.equal(result.celtic, 1 / 1024);
});

test('BLA block for iterations 2 and 3 predicts a nearby two-step orbit', async () => {
  const c = SCALE / 4n;
  const base = await computeReferenceJs({ ...request(0n, 0n), x: String(c), y: String(SCALE / 8n) });
  const delta = SCALE / 1_000_000n;
  const nearby = await computeReferenceJs({ ...request(0n, 0n), x: String(c + delta), y: String(SCALE / 8n) });
  const slot = 1 * 4;
  const bounds = base.blaBounds.subarray(slot, slot + 4);
  assert.ok(bounds[0] > -1000, 'second two-step block is populated');
  const scaleA = 2 ** bounds[2];
  const scaleB = 2 ** bounds[3];
  const predicted = scaleA * (base.blaA[slot] * (orbitAt(nearby, 2)[0] - orbitAt(base, 2)[0])
    + base.blaA[slot + 1] * (orbitAt(nearby, 2)[1] - orbitAt(base, 2)[1]))
    + scaleB * base.blaB[slot] / 1_000_000;
  const actual = orbitAt(nearby, 4)[0] - orbitAt(base, 4)[0];
  // The orbit and BLA matrices are stored as float32 texels.
  assert.ok(Math.abs(predicted - actual) < 2e-8, `${predicted} vs ${actual}`);
});

test('rejects malformed fixed-point input and invalid controls', async () => {
  await assert.rejects(computeReferenceJs({ ...request(0n, 0n), x: '1.5' }), /Invalid reference request/);
  await assert.rejects(computeReferenceJs({ ...request(0n, 0n), bits: 0 }), /Invalid reference request/);
  await assert.rejects(computeReferenceJs({ ...request(0n, 0n), fold: 2 }), /Invalid reference request/);
});

test('cancellation before and after a cooperative yield raises AbortError', async () => {
  await assert.rejects(computeReferenceJs(request(0n, 0n), () => true), { name: 'AbortError' });
  let stopped = false;
  let yields = 0;
  await assert.rejects(computeReferenceJs(request(0n, 0n, { iterations: 1024 }),
    () => stopped, async () => { yields++; stopped = true; }), { name: 'AbortError' });
  assert.equal(yields, 1);
});
