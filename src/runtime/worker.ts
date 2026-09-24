import { preloadRenderWasm } from '../gpu/render-wasm.ts';
import { WasmRenderer } from '../gpu/wasm-renderer.ts';
import type { CameraSnapshot, RuntimeRequest, RuntimeResponse, RuntimeSettings, RuntimeState } from './protocol.ts';

// Scheduling and browser I/O only. Every camera transform, orbit iteration,
// cache decision and draw-command loop executes in the same WASM instance.
const scope = self as unknown as { onmessage: ((event: MessageEvent<RuntimeRequest>) => void) | null;
  postMessage(message: RuntimeResponse): void; requestAnimationFrame(callback: (time: number) => void): number };
const encoder = new TextEncoder(), decoder = new TextDecoder();
let renderer: WasmRenderer | null = null, canvas: OffscreenCanvas;
let input: Float64Array, stats: Float64Array, textArea: Uint8Array, buffer: ArrayBuffer;
let pending = false, lost = false, suspended = false, dirty = true, playing = false, guided = false;
let preparing = false, prepared = false, playRequested = false, playEndZoom = 120;
let referenceMs = 0, referenceStarted = 0, cpuFrameMs = 0, lastPublication = 0;
let referenceScheduled = false, ready = false, failed = false, referenceDebounce = 0;
let settings: RuntimeSettings;
const queued: RuntimeRequest[] = [];
const referenceQueue = new MessageChannel();

function check(status: number, operation: string): void {
  if (status < 0) throw new Error(`${operation}: WASM ${status}`);
}
function bindMemory(): void {
  const core = renderer!.exports;
  buffer = core.memory.buffer;
  input = new Float64Array(buffer, core.render_input_ptr(), 64);
  stats = new Float64Array(buffer, core.render_stats_ptr(), 16);
  textArea = new Uint8Array(buffer, core.render_text_ptr(), core.render_text_capacity());
}
function args(a = 0, b = 0, c = 0, d = 0, e = 0): void {
  input[40] = a; input[41] = b; input[42] = c; input[43] = d; input[44] = e;
}
function cameraCommand(op: number, a = 0, b = 0, c = 0, d = 0, e = 0): void {
  args(a, b, c, d, e); check(renderer!.exports.render_camera_command(op), 'Камера'); dirty = true;
}
function invalidatePreparation(): void {
  renderer!.cancelPreparation();
  renderer!.resetCompletedFrames();
  prepared = false;
  if (playing) playRequested = true;
  playing = false;
  cameraCommand(4, 0);
  preparing = guided && ready && !pending;
}
function startFlight(): void {
  playRequested = false; preparing = false; playing = true;
  renderer!.resetCompletedFrames();
  cameraCommand(6); cameraCommand(4, 1, playEndZoom);
}
function cancelFlight(): void {
  if (playing) { prepared = false; preparing = guided && ready && !pending; }
  playRequested = false; playing = false;
  renderer!.resetCompletedFrames();
  cameraCommand(4, 0);
}
function encodeCoordinates(x: string, y: string): [number, number] {

  const first = encoder.encodeInto(x, textArea);
  const second = encoder.encodeInto(y, textArea.subarray(first.written));
  if (first.read !== x.length || second.read !== y.length) throw new Error('Координаты не помещаются в память WASM');
  return [first.written, second.written];
}
function snapshot(): CameraSnapshot {
  check(renderer!.exports.render_camera_snapshot(), 'Снимок камеры');
  const xLength = input[48], yLength = input[49];
  return { x: decoder.decode(textArea.subarray(0, xLength)),
    y: decoder.decode(textArea.subarray(xLength, xLength + yLength)), bits: input[50], logScale: input[51] };
}
function publish(now = performance.now()): void {
  if (!renderer || failed) return;
  const gpuFrames = renderer.sampleCompletedFrames();
  const state: RuntimeState = { ready, pending, lost, playing, preparing, playRequested,
    zoom: stats[9], logScale: stats[10], bits: stats[11],
    path: stats[0] === 0 ? 'direct' : stats[0] === 1 ? 'float' : 'fe',
    fps: suspended || pending || preparing || lost || (!playing && !dirty && !stats[8]) ? 0 : stats[15],
    gpuFps: gpuFrames.fps, gpuPendingFrames: gpuFrames.pendingFrames, gpuCompletionAgeMs: gpuFrames.completionAgeMs,
    cpuFrameMs, gpuMs: renderer.gpuTimeMs, referenceMs,
    memoryBytes: buffer.byteLength, frame: stats[1], ringActive: !!stats[6],
    drawCalls: stats[14], uploadBytes: renderer.totalReferenceUploadBytes };
  scope.postMessage({ type: 'state', state }); lastPublication = now;
}
function fail(error: unknown): void {
  renderer?.cancelPreparation();
  pending = false; playing = false; preparing = false; playRequested = false; dirty = false; failed = true;
  scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
}
function scheduleReferenceStep(): void {
  if (referenceScheduled) return;
  referenceScheduled = true; referenceQueue.port2.postMessage(null);
}
function beginReference(): void {
  clearTimeout(referenceDebounce);
  if (!renderer || lost) return;
  pending = true; invalidatePreparation(); preparing = false;
  referenceStarted = performance.now();
  renderer.exports.render_cancel_reference();
  check(renderer.exports.render_begin_reference(), 'Опорная орбита');
  publish(); scheduleReferenceStep();
}
referenceQueue.port1.onmessage = () => {
  referenceScheduled = false;
  if (!renderer || !pending || lost) return;
  try {
    // The numerical loop is inside step(). MessageChannel only yields to input.
    const status = renderer.exports.step(128);
    check(status, 'Расчёт орбиты');
    if (status === 0) { scheduleReferenceStep(); return; }
    renderer.refreshReferenceFromCore();
    referenceMs = performance.now() - referenceStarted;
    pending = false; ready = true; dirty = true;
    preparing = guided; prepared = false;
    cameraCommand(6); publish();
  } catch (error) { fail(error); }
};
function applySettings(next: RuntimeSettings): void {
  settings = next;
  input[3] = next.iterations; input[4] = next.aa; input[5] = next.fold;
  input[6] = next.celtic; input[7] = next.hue; input[17] = 1;
  renderer!.exports.render_mark_dirty();
}
function handle(message: Exclude<RuntimeRequest, { type: 'init' }>): void {
  if (!renderer) { queued.push(message); return; }
  try {
    failed = false;
    switch (message.type) {
      case 'route': {
        cancelFlight(); guided = true;
        const lengths = encodeCoordinates(message.route.x, message.route.y);
        check(renderer.exports.render_set_route(lengths[0], lengths[1], message.route.endZoom, message.zoom), 'Маршрут');
        beginReference(); break;
      }
      case 'overview':
        cancelFlight(); guided = false; cameraCommand(0, message.aspect); beginReference(); break;
      case 'zoom':
        cancelFlight(); cameraCommand(3, message.zoom); invalidatePreparation(); break;
      case 'zoom-at':
        cancelFlight(); guided = false; preparing = false; prepared = false;
        renderer.cancelPreparation();
        cameraCommand(2, message.x, message.y, message.width, message.height, message.delta);
        clearTimeout(referenceDebounce); referenceDebounce = setTimeout(() => {
          try { beginReference(); } catch (error) { fail(error); }
        }, 120) as unknown as number; break;
      case 'pointer':
        cancelFlight(); guided = false; preparing = false; prepared = false;
        renderer.cancelPreparation();
        cameraCommand(message.phase === 'start' ? 8 : message.phase === 'move' ? 9 : 10,
          message.x, message.y, message.height);
        if (message.phase === 'end') beginReference(); break;
      case 'play':
        playEndZoom = message.endZoom;
        if (!message.enabled) cancelFlight();
        else {
          playRequested = true;
          if (!pending && (!guided || prepared)) startFlight();
          else { playing = false; cameraCommand(4, 0); preparing = guided && ready && !pending; }
        }
        break;
      case 'resize':
        canvas.width = message.width; canvas.height = message.height;
        input[0] = message.width; input[1] = message.height;
        renderer.exports.render_mark_dirty(); dirty = true; invalidatePreparation(); break;
      case 'settings': {
        const recompute = settings.iterations !== message.settings.iterations || settings.fold !== message.settings.fold
          || settings.celtic !== message.settings.celtic;
        applySettings(message.settings); dirty = true; invalidatePreparation();
        if (recompute) beginReference(); break;
      }
      case 'suspend':
        suspended = message.suspended; renderer.resetCompletedFrames(); cameraCommand(6); break;
      case 'snapshot':
        scope.postMessage({ type: 'snapshot', id: message.id, snapshot: snapshot() }); break;
      case 'recover-gpu': {
        const extension = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
        if (!extension) throw new Error('WEBGL_lose_context недоступен');
        canvas.addEventListener('webglcontextlost', () => setTimeout(() => extension.restoreContext(), 100), { once: true });
        extension.loseContext(); break;
      }
    }
    publish();
  } catch (error) { fail(error); }
}
function frame(now: number): void {
  if (renderer && !failed && !pending && !lost && !suspended) {
    try {
      if (preparing && !dirty && !stats[8]) {
        input[16] = 1;
        if (renderer.prepareCamera(now)) {
          prepared = true; preparing = false;
          if (playRequested) startFlight();
          publish(now);
        }
      } else if (dirty || playing || stats[8]) {
        input[16] = +(guided && playing);
        const start = performance.now(); renderer.renderCamera(now); cpuFrameMs = performance.now() - start;
        dirty = false; playing = !!stats[12];
      } else if (playRequested) startFlight();
    } catch (error) { fail(error); }
  }
  if (now - lastPublication >= 250) publish(now);
  scope.requestAnimationFrame(frame);
}
async function init(message: Extract<RuntimeRequest, { type: 'init' }>): Promise<void> {
  await preloadRenderWasm();
  canvas = message.canvas; canvas.width = message.width; canvas.height = message.height;
  renderer = new WasmRenderer(canvas, true); bindMemory(); input[0] = message.width; input[1] = message.height;
  applySettings(message.settings);
  if (message.snapshot) {
    const lengths = encodeCoordinates(message.snapshot.x, message.snapshot.y);
    check(renderer.exports.render_set_camera(lengths[0], lengths[1], message.snapshot.bits, message.snapshot.logScale, 0), 'Камера');
    guided = !!message.route;
  } else if (message.route) {
    const lengths = encodeCoordinates(message.route.x, message.route.y);
    check(renderer.exports.render_set_route(lengths[0], lengths[1], message.route.endZoom, message.zoom ?? 0), 'Маршрут');
    guided = true;
  } else cameraCommand(0, message.aspect);
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault(); renderer!.exports.render_cancel_reference();
    clearTimeout(referenceDebounce); lost = true; pending = false; playing = false;
    preparing = false; prepared = false; playRequested = false; publish();
  });
  canvas.addEventListener('webglcontextrestored', () => {
    try {
      const position = snapshot(); renderer!.dispose(); renderer = new WasmRenderer(canvas, true); bindMemory();
      input[0] = canvas.width; input[1] = canvas.height; applySettings(settings);
      const lengths = encodeCoordinates(position.x, position.y);
      check(renderer.exports.render_set_camera(lengths[0], lengths[1], position.bits, position.logScale, 0), 'Камера');
      lost = false; beginReference();
    } catch (error) { fail(error); }
  });
  beginReference();
  for (const command of queued.splice(0)) if (command.type !== 'init') handle(command);
  scope.requestAnimationFrame(frame);
}
scope.onmessage = event => {
  if (event.data.type === 'init') void init(event.data).catch(fail);
  else handle(event.data);
};
