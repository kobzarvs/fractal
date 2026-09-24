import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';

// Run the production worker scheduler with a deterministic browser event loop.
// Only the Rust/WebGL boundary is mocked; camera movement marks a real play call.
const workerSource = stripTypeScriptTypes(await readFile(new URL('../src/runtime/worker.ts', import.meta.url), 'utf8'), { mode: 'transform' })
  .replace(/^import .*$/gm, '');
async function harness(preparationSteps = 3) {
  const raf: ((time: number) => void)[] = [], messages: any[] = [], jobs: (() => void)[] = [], events: string[] = [];
  const buffer = new ArrayBuffer(2048), input = new Float64Array(buffer, 0, 64), stats = new Float64Array(buffer, 512, 16);
  let warmups = 0, staticRemaining = 2, now = 0;
  const scope: any = { postMessage: (message: any) => messages.push(structuredClone(message)),
    requestAnimationFrame: (callback: (time: number) => void) => { raf.push(callback); return raf.length; } };
  const core: any = { memory: { buffer }, render_input_ptr: () => 0, render_stats_ptr: () => 512,
    render_text_ptr: () => 1024, render_text_capacity: () => 1024,
    render_camera_command: (operation: number) => {
      if (operation === 4) { stats[12] = input[40]; events.push(input[40] ? 'play' : 'pause'); }
      if (operation === 3) { stats[9] = input[40]; staticRemaining = 2; stats[8] = 1; }
      if (operation === 6) events.push('reset-clock');
      return 0;
    }, render_set_route: () => { stats[12] = 0; staticRemaining = 2; return 0; },
    render_mark_dirty: () => { staticRemaining = 2; stats[8] = 1; }, render_cancel_reference: () => {},
    render_begin_reference: () => { warmups = 0; return 0; }, step: () => 1 };
  class Renderer {
    exports = core; totalReferenceUploadBytes = 0; gpuTimeMs = null;
    refreshReferenceFromCore() { events.push('reference'); }
    cancelPreparation() {}
    resetCompletedFrames() { events.push('reset-gpu-meter'); }
    sampleCompletedFrames() { return { fps: null, pendingFrames: 0, completionAgeMs: null }; }
    renderCamera() {
      if (stats[12]) { stats[9]++; events.push('moving-frame'); }
      else { staticRemaining--; events.push('static-frame'); }
      stats[8] = +(staticRemaining > 0); stats[1]++; stats[15] = 60;
    }
    prepareCamera() { events.push('prepare'); warmups++; stats[6] = +(warmups >= preparationSteps); return warmups >= preparationSteps; }
  }
  class Channel {
    port1: any = { onmessage: null };
    port2 = { postMessage: () => jobs.push(() => this.port1.onmessage()) };
  }
  const canvas = Object.assign(new EventTarget(), { width: 320, height: 200 });
  new Function('self', 'MessageChannel', 'WasmRenderer', 'preloadRenderWasm', workerSource)(scope, Channel, Renderer, async () => {});
  const send = (message: any) => scope.onmessage({ data: message });
  send({ type: 'init', canvas, width: 320, height: 200, aspect: 1.6,
    settings: { aa: 2, iterations: 512, fold: 1, celtic: 0, hue: 0 }, route: { x: '0', y: '0', endZoom: 120 }, zoom: 30 });
  await Promise.resolve(); await Promise.resolve();
  const reference = () => { while (jobs.length) jobs.shift()!(); };
  const frame = () => { now += 16; assert.ok(raf.length); raf.shift()!(now); };
  const latest = () => messages.filter(message => message.type === 'state').at(-1)?.state;
  return { send, frame, reference, latest, events, stats, messages };
}

test('flight waits for the static image and cache preparation, without advancing the camera or reporting FPS', async () => {
  const h = await harness();
  h.send({ type: 'play', enabled: true, endZoom: 120 }); h.reference();
  const zoom = h.stats[9];
  for (let index = 0; index < 4; index++) h.frame();
  assert.deepEqual(h.events.filter(event => event.endsWith('frame') || event === 'prepare'), ['static-frame', 'static-frame', 'prepare', 'prepare']);
  assert.equal(h.stats[9], zoom); assert.equal(h.latest().preparing, true); assert.equal(h.latest().fps, 0);
  h.frame();
  assert.equal(h.latest().preparing, false); assert.equal(h.latest().playing, true);
  assert.deepEqual(h.events.slice(-2), ['reset-clock', 'play']);
  assert.equal(h.stats[9], zoom);
  h.frame(); assert.equal(h.stats[9], zoom + 1);
});

test('pausing during preparation cancels the pending play intent', async () => {
  const h = await harness(); h.reference(); h.send({ type: 'play', enabled: true, endZoom: 120 });
  h.frame(); h.frame(); h.frame(); h.send({ type: 'play', enabled: false, endZoom: 120 });
  for (let index = 0; index < 8; index++) h.frame();
  assert.equal(h.events.includes('play'), false); assert.equal(h.latest().playRequested, false);
});

test('pointer input cancels preparation and visibility suspends all GPU preparation work', async () => {
  const h = await harness(); h.reference(); h.frame(); h.frame();
  h.send({ type: 'suspend', suspended: true });
  h.frame(); h.frame(); assert.equal(h.events.includes('prepare'), false);
  h.send({ type: 'suspend', suspended: false }); h.frame(); h.frame();
  h.send({ type: 'pointer', phase: 'start', x: 1, y: 1, height: 200 });
  const count = h.events.filter(event => event === 'prepare').length;
  for (let index = 0; index < 5; index++) h.frame();
  assert.equal(h.events.filter(event => event === 'prepare').length, count);
  assert.equal(h.latest().preparing, false);
});

test('settings changes pause a requested flight and prepare again before resuming', async () => {
  const h = await harness(1); h.reference(); h.send({ type: 'play', enabled: true, endZoom: 120 });
  for (let index = 0; index < 4; index++) h.frame();
  const before = h.events.length;
  h.send({ type: 'settings', settings: { aa: 5, iterations: 512, fold: 1, celtic: 0, hue: 0 } });
  h.frame(); h.frame(); h.frame();
  assert.deepEqual(h.events.slice(before).filter(event => event.endsWith('frame') || event === 'prepare'), ['static-frame', 'static-frame', 'prepare']);
  assert.equal(h.latest().playing, true);
});
