import assert from 'node:assert/strict';
import test from 'node:test';
import { gpuFrameRateDisplay } from '../src/fps-display.ts';

test('a stale worker never leaves the last 120 on the main display', () => {
  const result = gpuFrameRateDisplay({ fps: 120, pendingFrames: 0 }, 'stale');
  assert.equal(result.value, '—');
  assert.match(result.detail, /Нет свежих данных/);
});

test('submission rate is not substituted for missing GPU confirmation', () => {
  const result = gpuFrameRateDisplay({ fps: null, pendingFrames: 120 }, 'active');
  assert.equal(result.value, '—');
  assert.match(result.detail, /подтверждения GPU/);
});

test('a static canvas waits for outstanding confirmations before showing idle zero', () => {
  assert.equal(gpuFrameRateDisplay({ fps: 120, pendingFrames: 2 }, 'idle').value, '—');
  assert.equal(gpuFrameRateDisplay({ fps: 120, pendingFrames: 0 }, 'idle').value, '0');
});

test('GPU rate is explicitly an estimate and suspension never shows an old rate', () => {
  assert.deepEqual(gpuFrameRateDisplay({ fps: 59.92, pendingFrames: 3 }, 'active'),
    { value: '59.9', detail: 'Завершённые GPU кадры · оценка' });
  for (const phase of ['starting', 'failed', 'lost', 'hidden', 'benchmark', 'reference', 'preparing'] as const)
    assert.equal(gpuFrameRateDisplay({ fps: 120, pendingFrames: 0 }, phase).value, '—');
  assert.equal(gpuFrameRateDisplay({ fps: Number.NaN, pendingFrames: 0 }, 'active').value, '—');
});
