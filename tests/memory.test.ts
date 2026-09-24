import test from 'node:test';
import assert from 'node:assert/strict';
import { MemoryViews } from '../src/compute/wasm.ts';
import { RequestEpoch } from '../src/compute/protocol.ts';

test('shipped kernel reserves its complete fixed arena before computation', async () => {
  const { readFile } = await import('node:fs/promises');
  const { computeOnlyImports } = await import('../src/compute/wasm-imports.ts');
  const binary = await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url));
  const { instance } = await WebAssembly.instantiate(binary, computeOnlyImports());
  const memory = instance.exports.memory as WebAssembly.Memory;
  const views = new MemoryViews(memory);
  const buffer = memory.buffer, bytes = views.bytes();
  assert.equal(buffer.byteLength, 256 * 1024 * 1024);
  assert.equal(views.mode, 'fixed');
  assert.equal((buffer as ArrayBuffer & { resizable?: boolean }).resizable, false);
  assert.throws(() => memory.grow(1), RangeError);
  bytes[12] = 99;
  assert.equal(memory.buffer, buffer);
  assert.equal(views.bytes(), bytes);
  assert.equal(views.bytes()[12], 99);
});
test('request generations reject old responses even when ids arrive late', () => {
  const epochs = new RequestEpoch(); const first = epochs.next(); const second = epochs.next();
  assert.equal(epochs.isCurrent(first), false); assert.equal(epochs.isCurrent(second), true);
  epochs.cancel(); assert.equal(epochs.isCurrent(second), false);
});
