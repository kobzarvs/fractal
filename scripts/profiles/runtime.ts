import { WasmRuntimeClient } from '../../src/runtime/client.ts';
import type { RuntimeState } from '../../src/runtime/protocol.ts';
import { LegacyRenderer } from '../../src/gpu/legacy/renderer.ts';
import { ReferenceClient } from '../../src/compute/client.ts';
import { cameraOffset, decimalToFixed, fixedToNumber, setZoom, type Camera } from '../../src/camera.ts';
import { FrameRateMeter } from '../../src/frame-rate.ts';
import { WESTERN_ARMADA } from '../../src/tours.ts';
import type { RenderView } from '../../src/types.ts';

const query = new URLSearchParams(location.search);
function option(name: string, fallback: number, minimum: number, maximum: number) {
  const value = Number(query.get(name) ?? fallback);
  if (!Number.isFinite(value) || value < minimum || value > maximum) throw new Error(`Invalid ${name}`);
  return value;
}
const options = { width: option('width', 1920, 1, 7680), height: option('height', 1080, 1, 4320),
  depth: option('depth', 30, 0, 115), duration: option('duration', 4000, 1000, 30000),
  aa: option('aa', 2, 1, 5), iterations: option('iter', 16384, 1, 65536) };
const order = (query.get('runs') ?? 'js,wasm,wasm,js').split(',');
if (order.some(value => value !== 'js' && value !== 'wasm') || order.length > 8) throw new Error('Invalid runs');
const settings = { aa: options.aa, iterations: options.iterations, fold: 1, celtic: 0, hue: 0 };
const status = document.querySelector<HTMLElement>('#status')!;
const summary = document.querySelector<HTMLElement>('#summary')!;
const reportNode = document.querySelector<HTMLElement>('#report')!;
const download = document.querySelector<HTMLButtonElement>('#download')!;
const BITS = 576;
interface StateSample { atMs: number; frame: number; fps: number; ringActive: boolean; zoom: number;
  cpuFrameMs: number; gpuMs: number | null; path: string; preparing: boolean; playing: boolean; drawCalls?: number }
function visible() { if (document.hidden) throw new Error('Вкладка скрыта: измерение остановлено.'); }
function nextFrame(): Promise<number> {
  visible();
  return new Promise((resolve, reject) => {
    const handle = requestAnimationFrame(timestamp => { clearTimeout(timeout);
      try { visible(); resolve(timestamp); } catch (error) { reject(error); } });
    const timeout = setTimeout(() => { cancelAnimationFrame(handle); reject(new Error('rAF timeout')); }, 5000);
  });
}
function canvas() {
  const canvas = document.createElement('canvas'); canvas.width = options.width; canvas.height = options.height;
  document.querySelector('#canvas')!.replaceChildren(canvas); return canvas;
}
function summarize(states: StateSample[]) {
  const first = states[0], last = states.at(-1)!;
  let cacheLostTransitions = 0;
  for (let index = 1; index < states.length; index++) if (states[index - 1].ringActive && !states[index].ringActive) cacheLostTransitions++;
  const elapsedMs = last.atMs - first.atMs, frames = last.frame - first.frame;
  return { elapsedMs, frames, fpsMean: frames * 1000 / elapsedMs,
    firstCacheActiveMs: states.find(state => state.ringActive)?.atMs ?? null, cacheLostTransitions,
    cacheActiveSamples: states.filter(state => state.ringActive).length, stateSamples: states.length,
    startZoom: first.zoom, endZoom: last.zoom, lastCacheActive: last.ringActive };
}
async function wasm(run: number) {
  const target = canvas(), startupAt = performance.now(), states: StateSample[] = [];
  let referenceReadyAt: number | null = null, preparedAt = 0, startedAt: number | null = null;
  let resolvePrepared!: () => void, rejectPrepared!: (error: Error) => void;
  let resolveFlight!: () => void, rejectFlight!: (error: Error) => void;
  const prepared = new Promise<void>((resolve, reject) => { resolvePrepared = resolve; rejectPrepared = reject; });
  const flight = new Promise<void>((resolve, reject) => { resolveFlight = resolve; rejectFlight = reject; });
  // An early worker error must reject both phases without an unhandled rejection.
  void prepared.catch(() => {});
  void flight.catch(() => {});
  let requested = false, finished = false, finalState: RuntimeState | undefined;
  const fail = (message: string) => { const error = new Error(message); rejectPrepared(error); rejectFlight(error); };
  const hidden = () => { if (document.hidden) fail('Вкладка скрыта: измерение остановлено.'); };
  document.addEventListener('visibilitychange', hidden);
  const timeout = setTimeout(() => fail('Runtime profile timeout'), 120000);
  const runtime = new WasmRuntimeClient(target, { width: options.width, height: options.height,
    aspect: options.width / options.height, settings, route: WESTERN_ARMADA, zoom: options.depth }, state => {
    if (finished) return;
    const now = performance.now(); finalState = state;
    if (state.ready) referenceReadyAt ??= now;
    if (!requested && state.ready && !state.preparing && !state.pending) { preparedAt = now; resolvePrepared(); }
    if (requested && state.playing && !state.preparing) {
      startedAt ??= now;
      states.push({ atMs: now - startedAt, frame: state.frame, fps: state.fps, ringActive: state.ringActive,
        zoom: state.zoom, cpuFrameMs: state.cpuFrameMs, gpuMs: state.gpuMs, path: state.path,
        preparing: state.preparing, playing: state.playing, drawCalls: state.drawCalls });
      status.textContent = `Запуск ${run + 1}/${order.length}: WASM worker, ${((now - startedAt) / 1000).toFixed(1)} с, ${state.fps.toFixed(1)} FPS, кэш ${state.ringActive ? 'активен' : 'не активен'}`;
      document.title = `WASM ${run + 1}/${order.length} ${state.fps.toFixed(1)} FPS cache=${+state.ringActive}`;
      if (now - startedAt >= options.duration) { finished = true; resolveFlight(); }
    } else {
      status.textContent = `Запуск ${run + 1}/${order.length}: WASM ${state.pending ? 'орбита' : state.preparing ? 'подготовка кэша' : 'готов'}…`;
      document.title = `WASM ${run + 1}/${order.length} ${state.pending ? 'reference' : state.preparing ? 'prepare' : 'ready'}`;
    }
  }, fail);
  try {
    await runtime.ready; await prepared; await nextFrame();
    requested = true; runtime.send({ type: 'play', enabled: true, endZoom: WESTERN_ARMADA.endZoom });
    await flight;
    runtime.send({ type: 'play', enabled: false, endZoom: WESTERN_ARMADA.endZoom });
    return { backend: 'wasm', run, startupToReferenceReadyMs: referenceReadyAt! - startupAt,
      preparationWallMs: preparedAt - referenceReadyAt!, startupToPreparedMs: preparedAt - startupAt,
      referenceMs: finalState!.referenceMs, memoryBytes: finalState!.memoryBytes,
      measuredBy: 'worker state callbacks on main thread; 250ms nominal publication; frame delta includes submitted frames only',
      ...summarize(states), states };
  } finally { clearTimeout(timeout); document.removeEventListener('visibilitychange', hidden); runtime.dispose(); }
}
async function js(run: number) {
  const target = canvas(), renderer = new LegacyRenderer(target), referenceClient = new ReferenceClient();
  const camera: Camera = { x: decimalToFixed(WESTERN_ARMADA.x, BITS), y: decimalToFixed(WESTERN_ARMADA.y, BITS),
    bits: BITS, logScale: Math.log2(3.2) }; setZoom(camera, options.depth);
  const referenceCamera = { ...camera }, states: StateSample[] = [], meter = new FrameRateMeter();
  const makeView = (guided: boolean): RenderView => {
    const offset = cameraOffset(camera, referenceCamera);
    return { center: [fixedToNumber(camera.x, camera.bits), fixedToNumber(camera.y, camera.bits)],
      scale: 2 ** camera.logScale, logScale: camera.logScale, offsetX: offset.x, offsetY: offset.y,
      ...settings, guided, temporal: true, referenceKey: 1, position: camera };
  };
  const startupAt = performance.now();
  try {
    status.textContent = `Запуск ${run + 1}/${order.length}: JS, орбита…`;
    document.title = `JS ${run + 1}/${order.length} reference`;
    const reference = await referenceClient.compute({ x: String(camera.x), y: String(camera.y), bits: BITS,
      iterations: options.iterations, fold: 1, celtic: 0 }, 'js');
    renderer.setReference(reference.result); const referenceMs = reference.result.computeMs; referenceClient.recycle(reference.result);
    const referenceReadyAt = performance.now();
    do { await nextFrame(); renderer.render(makeView(false)); } while (renderer.settling);
    renderer.readPixels(); const preparedAt = performance.now();
    await nextFrame();
    let lastTimestamp: number | null = null, startedAt: number | null = null, lastSampleAt = 0, cpuFrameMs = 0;
    const sample = (now: number, playing: boolean) => states.push({ atMs: now - startedAt!, frame: renderer.stats.frame,
      fps: meter.sample(now) ?? 0, ringActive: renderer.stats.ringActive,
      zoom: (Math.log2(3.2) - camera.logScale) / Math.log2(10), cpuFrameMs,
      gpuMs: renderer.gpuTimeMs, path: renderer.stats.path, preparing: false, playing });
    for (;;) {
      const timestamp = await nextFrame(), now = performance.now();
      if (startedAt === null) { startedAt = now; lastSampleAt = now; sample(now, true); }
      const elapsed = lastTimestamp === null ? 0 : Math.min(.1, (timestamp - lastTimestamp) / 1000); lastTimestamp = timestamp;
      const cpuStarted = performance.now();
      const zoom = (Math.log2(3.2) - camera.logScale) / Math.log2(10);
      setZoom(camera, Math.min(WESTERN_ARMADA.endZoom, zoom + elapsed * .55));
      renderer.render(makeView(true)); cpuFrameMs = performance.now() - cpuStarted; meter.record(now);
      const after = performance.now();
      if (after - lastSampleAt >= 250 || after - startedAt >= options.duration) {
        sample(after, true); lastSampleAt = after;
        status.textContent = `Запуск ${run + 1}/${order.length}: JS, ${((after - startedAt) / 1000).toFixed(1)} с, ${states.at(-1)!.fps.toFixed(1)} FPS, кэш ${renderer.stats.ringActive ? 'активен' : 'не активен'}`;
        document.title = `JS ${run + 1}/${order.length} ${states.at(-1)!.fps.toFixed(1)} FPS cache=${+renderer.stats.ringActive}`;
      }
      if (after - startedAt >= options.duration) break;
    }
    return { backend: 'js', run, startupToReferenceReadyMs: referenceReadyAt - startupAt,
      preparationWallMs: 0, staticPreparationWallMs: preparedAt - referenceReadyAt, startupToPreparedMs: preparedAt - startupAt,
      referenceMs, memoryBytes: reference.memoryBytes, ringCacheColdAtPlay: true,
      measuredBy: 'main-thread 250ms state samples; frame delta includes submitted frames only', ...summarize(states), states };
  } finally { referenceClient.dispose(); renderer.dispose(); target.getContext('webgl2')?.getExtension('WEBGL_lose_context')?.loseContext(); }
}
function probeGpu() {
  const target = document.createElement('canvas'), gl = target.getContext('webgl2', { powerPreference: 'high-performance' });
  if (!gl) return null;
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const result = { source: 'separate main-thread probe',
    vendor: gl.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
    renderer: gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE), gpuTimerSupported: !!gl.getExtension('EXT_disjoint_timer_query_webgl2') };
  gl.getExtension('WEBGL_lose_context')?.loseContext(); return result;
}
async function main() {
  visible(); const gpuDevice = probeGpu(), runs = [];
  for (const [index, backend] of order.entries()) {
    const result = backend === 'wasm' ? await wasm(index) : await js(index); runs.push(result);
    const row = document.createElement('tr');
    for (const value of [`${index + 1} ${backend}`, result.fpsMean.toFixed(2), result.preparationWallMs.toFixed(0),
      result.firstCacheActiveMs?.toFixed(0) ?? '—', result.cacheLostTransitions, result.endZoom.toFixed(3)]) {
      const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell);
    }
    summary.append(row); await nextFrame();
  }
  const report = { kind: 'production-runtime-flight-profile', createdAt: new Date().toISOString(),
    userAgent: navigator.userAgent, devicePixelRatio, gpuDevice, options, order,
    note: 'Submission cadence, not presented/completed GPU FPS. Serial fresh canvases. WASM actual OffscreenCanvas worker prepares before play; JS original renderer begins with cold ring cache. WASM timestamps are receipt times of state publications (nominal 250 ms), so elapsed/frame sampling is quantized and cache transitions can be missed between publications. GPU time null means unavailable. Preparation time is reported separately.', runs };
  const serialized = JSON.stringify(report, null, 2); reportNode.textContent = serialized; download.disabled = false;
  download.onclick = () => { const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'runtime-flight.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); };
  status.textContent = 'Готово';
  document.title = `DONE ${runs.map(run => `${run.backend}=${run.fpsMean.toFixed(1)} loss${run.cacheLostTransitions}`).join(' | ')}`;
  if (query.get('download') === '1') download.click();
}
main().catch(error => { status.textContent = `Ошибка: ${error instanceof Error ? error.message : error}`;
  document.title = status.textContent; console.error(error); });
