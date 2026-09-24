import test from 'node:test';
import assert from 'node:assert/strict';
import { MemoryViews } from '../src/compute/wasm.ts';
import { RequestEpoch } from '../src/compute/protocol.ts';

test('fixed-memory views refresh after growth, including grow(0)', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 4 });
  const views = new MemoryViews(memory, false);
  views.bytes()[12] = 99;
  const previous = views.bytes(); memory.grow(1);
  assert.equal(previous.byteLength, 0);
  assert.equal(views.bytes()[12], 99);
  assert.equal(views.bytes().length, 131072);
  memory.grow(0); assert.equal(views.bytes()[12], 99);
});
test('resizable memory is feature-detected and remains usable through growth', () => {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 4 });
  const views = new MemoryViews(memory);
  views.bytes()[0] = 42; memory.grow(1);
  assert.equal(views.bytes().length, 131072); assert.equal(views.bytes()[0], 42);
});
test('request generations reject old responses even when ids arrive late', () => {
  const epochs = new RequestEpoch(); const first = epochs.next(); const second = epochs.next();
  assert.equal(epochs.isCurrent(first), false); assert.equal(epochs.isCurrent(second), true);
  epochs.cancel(); assert.equal(epochs.isCurrent(second), false);
});
