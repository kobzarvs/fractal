import test from 'node:test';
import type { TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { WasmRuntimeClient } from '../src/runtime/client.ts';
import type { RuntimeRequest, RuntimeResponse, RuntimeState } from '../src/runtime/protocol.ts';

function fixture(t: TestContext) {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  let worker: FakeWorker;
  class FakeWorker {
    onmessage: ((event: MessageEvent<RuntimeResponse>) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    requests: RuntimeRequest[] = [];
    constructor() { worker = this; }
    postMessage(message: RuntimeRequest) { this.requests.push(message); }
    terminate() {}
    emit(message: RuntimeResponse) { this.onmessage?.({ data: message } as MessageEvent<RuntimeResponse>); }
  }
  const previousWorker = globalThis.Worker;
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  t.after(() => {
    if (previousWorker === undefined) Reflect.deleteProperty(globalThis, 'Worker');
    else globalThis.Worker = previousWorker;
  });
  const canvas = { transferControlToOffscreen: () => ({}) } as HTMLCanvasElement;
  const client = new WasmRuntimeClient(canvas, { width: 320, height: 200, aspect: 1.6,
    settings: { iterations: 16384, aa: 2, fold: 1, celtic: 0, hue: 0 } }, () => {}, () => {});
  void client.ready.catch(() => {});
  t.after(() => client.dispose());
  const state: RuntimeState = { ...client.state, ready: true, pending: false, playing: true, fps: 60, frame: 1 };
  return { client, worker: worker!, state, setNow(value: number) { now = value; } };
}

test('runtime state starts unknown and expires exactly one second after reception', async t => {
  const f = fixture(t);
  assert.equal(f.client.isStateFresh(), false);
  f.setNow(100); f.worker.emit({ type: 'state', state: f.state });
  await f.client.ready;
  assert.equal(f.client.isStateFresh(), true);
  assert.equal(f.client.isStateFresh(1099), true);
  assert.equal(f.client.isStateFresh(1100), false);
  f.setNow(1101);
  assert.equal(f.client.isStateFresh(), false);
});

test('only a subsequent state message renews the freshness window', t => {
  const f = fixture(t);
  f.worker.emit({ type: 'state', state: f.state });
  f.setNow(900); f.worker.emit({ type: 'state', state: { ...f.state, frame: 2 } });
  f.setNow(1899); assert.equal(f.client.isStateFresh(), true);
  f.setNow(1900); assert.equal(f.client.isStateFresh(), false);
  f.worker.emit({ type: 'state', state: { ...f.state, frame: 3 } });
  assert.equal(f.client.isStateFresh(), true);
});

test('outgoing commands and snapshot responses cannot make silent state fresh', async t => {
  const f = fixture(t);
  f.worker.emit({ type: 'state', state: f.state });
  f.setNow(900);
  f.client.send({ type: 'zoom', zoom: 30 });
  const response = f.client.snapshot();
  const request = f.worker.requests.at(-1)!;
  assert.equal(request.type, 'snapshot');
  if (request.type !== 'snapshot') throw new Error('Expected snapshot request');
  const snapshot = { x: '0', y: '0', bits: 128, logScale: -100 };
  f.worker.emit({ type: 'snapshot', id: request.id, snapshot });
  assert.deepEqual(await response, snapshot);
  f.setNow(1000);
  assert.equal(f.client.isStateFresh(), false);
});

for (const failure of ['message', 'event'] as const) {
  test(`a worker ${failure} error invalidates state until another state arrives`, t => {
    const f = fixture(t);
    f.worker.emit({ type: 'state', state: f.state });
    f.setNow(100);
    if (failure === 'message') f.worker.emit({ type: 'error', message: 'worker failed' });
    else f.worker.onerror?.({ message: 'worker failed' } as ErrorEvent);
    assert.equal(f.client.isStateFresh(), false);
    f.client.send({ type: 'zoom', zoom: 31 });
    assert.equal(f.client.isStateFresh(), false);
    f.setNow(200); f.worker.emit({ type: 'state', state: { ...f.state, frame: 2 } });
    assert.equal(f.client.isStateFresh(), true);
  });
}

test('disposing invalidates freshness permanently despite a queued state message', t => {
  const f = fixture(t);
  f.worker.emit({ type: 'state', state: f.state });
  assert.equal(f.client.isStateFresh(), true);
  f.client.dispose();
  assert.equal(f.client.isStateFresh(), false);
  f.setNow(100); f.worker.emit({ type: 'state', state: { ...f.state, frame: 2 } });
  assert.equal(f.client.isStateFresh(), false);
  assert.equal(f.client.state.frame, 1, 'late responses must remain ignored');
});
