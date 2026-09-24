import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { WasmCore } from '../src/compute/wasm.ts';

test('WASM API rejects values that an i32 import would silently coerce', async () => {
  const { instance } = await WebAssembly.instantiate(await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url)), {});
  const core = new WasmCore(instance);
  const valid = { id: 1, x: '0', y: '0', bits: 128, iterations: 16, fold: 1, celtic: 0 };
  for (const invalid of [{ bits: 128.5 }, { iterations: 16.25 }, { fold: NaN }, { celtic: Infinity }, { fold: 1.00001 }, { iterations: 65537 }]) {
    await assert.rejects(core.compute({ ...valid, ...invalid }), /Invalid reference request/);
  }
  assert.equal((await core.compute(valid)).length, 17, 'invalid request must not leave the instance busy');
});
