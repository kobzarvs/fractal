import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { WasmCore } from '../src/compute/wasm.ts';
import { computeReferenceJs } from '../src/compute/reference-js.ts';
import type { ReferenceRequest, ReferenceResult } from '../src/types.ts';
import { referenceCases } from './fixtures.ts';

const ARRAYS = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
const noYield = async () => {};

async function core(): Promise<WasmCore> {
  const binary = await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url));
  const { instance } = await WebAssembly.instantiate(binary, {});
  return new WasmCore(instance, false);
}

function assertIdentical(actual: ReferenceResult, expected: ReferenceResult, label: string): void {
  assert.equal(actual.length, expected.length, `${label}: orbit length`);
  assert.equal(actual.capacity, expected.capacity, `${label}: capacity`);
  assert.equal(actual.fold, expected.fold, `${label}: fold quantization`);
  assert.equal(actual.celtic, expected.celtic, `${label}: Celtic quantization`);
  for (const key of ARRAYS) {
    const left = actual[key];
    const right = expected[key];
    assert.equal(left.length, right.length, `${label}: ${key} length`);
    const leftBytes = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
    const rightBytes = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
    for (let byte = 0; byte < leftBytes.length; byte++) {
      if (leftBytes[byte] !== rightBytes[byte]) {
        const texel = byte >>> 2;
        assert.fail(`${label}: ${key} byte ${byte} (float ${texel}): `
          + `WASM ${left[texel]} [${leftBytes.slice(texel * 4, texel * 4 + 4)}], `
          + `JS ${right[texel]} [${rightBytes.slice(texel * 4, texel * 4 + 4)}]`);
      }
    }
  }
}

function adversarialCases(): { name: string; request: ReferenceRequest }[] {
  const bits = 128;
  const one = 1n << BigInt(bits);
  const cases: { name: string; request: ReferenceRequest }[] = [
    { name: 'negative-unit-cross-near-zero', request: {
      id: 100, x: '1', y: '-1', bits, iterations: 1024, fold: 1, celtic: 1,
    } },
    { name: 'opposite-sign-near-half', request: {
      id: 101, x: String(one / 2n + 1n), y: String(-one / 2n + 1n),
      bits, iterations: 1024, fold: 1, celtic: 0.5,
    } },
  ];
  let state = 0x754d5831;
  for (let i = 0; i < 12; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const x = (BigInt(state) - 0x80000000n) << BigInt(bits - 30);
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const y = (BigInt(state) - 0x80000000n) << BigInt(bits - 30);
    cases.push({ name: `seeded-${i}`, request: { id: 102 + i,
      x: String(x), y: String(y), bits, iterations: 1024,
      fold: [0, 0.25, 0.5, 0.75, 1][i % 5],
      celtic: [1, 0.75, 0.5, 0.25, 0][i % 5],
    } });
  }
  return cases;
}

test('SIMD WASM matches the BigInt oracle byte-for-byte', async t => {
  const wasm = await core();
  for (const { name, request } of [...referenceCases, ...adversarialCases()]) {
    await t.test(name, async () => {
      const expected = await computeReferenceJs(request, () => false, noYield);
      const actual = await wasm.compute(request, () => false, noYield);
      assertIdentical(actual, expected, `wasm/${name}`);
    });
  }
});

test('SIMD WASM retains valid result views after memory growth and repeated runs', async () => {
  const wasm = await core();
  const request = referenceCases.at(-1)!.request;
  const first = await wasm.compute(request, () => false, noYield);
  const warmBytes = wasm.memoryBytes;
  const second = await wasm.compute(request, () => false, noYield);
  assert.equal(wasm.memoryBytes, warmBytes, 'same request must reuse allocated memory');
  assertIdentical(second, first, 'wasm/repeat');
  assert.equal(first.orbit[0], 0, 'copied first result stays readable');
});
