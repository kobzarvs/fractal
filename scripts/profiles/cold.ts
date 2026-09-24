import { decimalToFixed, fixedToNumber } from '../../src/camera.ts';
import { loadWasm } from '../../src/compute/wasm.ts';
import { createRenderer, type RendererAdapter, type RendererBackend } from '../../src/gpu/engine.ts';
import { preloadRenderWasm } from '../../src/gpu/render-wasm.ts';
import { WESTERN_ARMADA } from '../../src/tours.ts';
import type { ReferenceResult, RenderView } from '../../src/types.ts';

const query = new URLSearchParams(location.search);
function option(name: string, fallback: number, min: number, max: number) {
  const value = Number(query.get(name) ?? fallback);
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`Invalid ${name}`);
  return value;
}
const options = { width: option('width', 1920, 1, 7680), height: option('height', 1080, 1, 4320),
  aa: option('aa', 2, 1, 5), iterations: option('iter', 16384, 1, 65536),
  frames: option('frames', 90, 2, 1200), rateFPS: option('rateFPS', 60, 1, 1000),
  depths: (query.get('depths') ?? '0,3,10,30').split(',').map(Number) };
if (options.depths.some(depth => !Number.isFinite(depth) || depth < 0 || depth > 120)) throw new Error('Invalid depths');
const status = document.querySelector<HTMLElement>('#status')!;
const reportElement = document.querySelector<HTMLElement>('#report')!;
const summaryElement = document.querySelector<HTMLElement>('#summary')!;
const download = document.querySelector<HTMLButtonElement>('#download')!;
const BITS = 576;
const position = { x: decimalToFixed(WESTERN_ARMADA.x, BITS), y: decimalToFixed(WESTERN_ARMADA.y, BITS), bits: BITS };
const center: [number, number] = [fixedToNumber(position.x, BITS), fixedToNumber(position.y, BITS)];
function view(depth: number, guided: boolean): RenderView {
  const logScale = Math.log2(3.2) - depth * Math.log2(10);
  return { center, logScale, scale: 2 ** logScale, offsetX: [0, 0], offsetY: [0, 0],
    iterations: options.iterations, aa: options.aa, fold: 1, celtic: 0, hue: 0,
    temporal: true, guided, referenceKey: 1, position };
}
function nextFrame(): Promise<number> {
  if (document.hidden) return Promise.reject(new Error('Вкладка скрыта; измерение остановлено.'));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cancelAnimationFrame(handle); reject(new Error('rAF timeout')); }, 5000);
    const handle = requestAnimationFrame(timestamp => { clearTimeout(timer);
      if (document.hidden) reject(new Error('Вкладка скрыта; измерение остановлено.')); else resolve(timestamp); });
  });
}
function statistics(samples: number[]) {
  if (!samples.length) return null;
  const sorted = [...samples].sort((a, b) => a - b), totalMs = samples.reduce((sum, value) => sum + value, 0);
  return { samples: samples.length, totalMs, meanMs: totalMs / samples.length,
    medianMs: sorted[Math.floor(sorted.length / 2)], p95Ms: sorted[Math.ceil(sorted.length * .95) - 1],
    minMs: sorted[0], maxMs: sorted.at(-1) };
}
interface Frame {
  index: number; rendererFrame: number; zoom: number; path: string; ringActive: boolean;
  ringSamples: number; ringsDrawn: number; ringResets: number; cpuMs: number; blockingDrainMs: number;
  gpuMs: number | null; rafAgeMs: number;
}
function gpuDevice(gl: WebGL2RenderingContext) {
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  return { vendor: gl.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
    renderer: gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    version: gl.getParameter(gl.VERSION), shadingLanguage: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE), contextAttributes: gl.getContextAttributes() };
}
function collect(renderer: RendererAdapter, frames: Frame[]) {
  const map = new Map(frames.map(frame => [frame.rendererFrame, frame]));
  for (const timing of renderer.gpuTimings) {
    const frame = map.get(timing.frame);
    if (frame) frame.gpuMs = timing.milliseconds;
  }
}
function summarize(frames: Frame[]) {
  return { frames: frames.length, cpu: statistics(frames.map(frame => frame.cpuMs)),
    blockingDrain: statistics(frames.map(frame => frame.blockingDrainMs)),
    gpu: statistics(frames.flatMap(frame => frame.gpuMs === null ? [] : [frame.gpuMs])),
    ringSamples: frames.reduce((sum, frame) => sum + frame.ringSamples, 0),
    ringsDrawn: frames.reduce((sum, frame) => sum + frame.ringsDrawn, 0),
    ringResets: frames.reduce((sum, frame) => sum + frame.ringResets, 0) };
}
async function run(backend: RendererBackend, depth: number, round: number, reference: ReferenceResult) {
  const canvas = document.createElement('canvas'); canvas.width = options.width; canvas.height = options.height;
  document.querySelector('#canvas')!.replaceChildren(canvas);
  const renderer = createRenderer(canvas, backend), gl = canvas.getContext('webgl2')!;
  const frames: Frame[] = [], device = gpuDevice(gl);
  try {
    renderer.setReference(reference);
    status.textContent = `Глубина ${depth}, запуск ${round + 1}/4, ${backend}: статический кадр до полёта…`;
    const prepStart = performance.now();
    renderer.render(view(depth, false)); renderer.readPixels();
    const staticPreparationMs = performance.now() - prepStart, staticFrame = renderer.stats.frame;
    // Let the prerender query finish without discarding an outstanding query.
    let staticQueryReceived = !renderer.gpuTimerSupported;
    for (let attempt = 0; attempt < 12; attempt++) {
      await nextFrame();
      staticQueryReceived = !renderer.gpuTimerSupported || renderer.gpuTimings.some(item => item.frame === staticFrame);
      if (staticQueryReceived) break;
    }
    if (staticQueryReceived) renderer.clearGpuTimings();
    for (let index = 0; index < options.frames; index++) {
      const timestamp = await nextFrame(); collect(renderer, frames);
      const zoom = depth + index * .55 / options.rateFPS, currentView = view(zoom, true);
      const prior = { ...renderer.stats }, rafAgeMs = performance.now() - timestamp;
      const started = performance.now(); renderer.render(currentView); const cpuMs = performance.now() - started;
      const drainStarted = performance.now(); renderer.readPixels(); const blockingDrainMs = performance.now() - drainStarted;
      frames.push({ index, rendererFrame: renderer.stats.frame, zoom, path: renderer.stats.path,
        ringActive: renderer.stats.ringActive, ringSamples: renderer.stats.ringSamples - prior.ringSamples,
        ringsDrawn: renderer.stats.ringsDrawn - prior.ringsDrawn, ringResets: renderer.stats.ringResets - prior.ringResets,
        cpuMs, blockingDrainMs, gpuMs: null, rafAgeMs });
      if (index % 15 === 0) status.textContent = `Глубина ${depth}, запуск ${round + 1}/4, ${backend}: кадр ${index + 1}/${options.frames}`;
    }
    for (let attempt = 0; attempt < 30; attempt++) {
      await nextFrame(); collect(renderer, frames);
      if (!renderer.gpuTimerSupported || frames.every(frame => frame.gpuMs !== null)) break;
    }
    const firstRingActiveFrame = frames.find(frame => frame.ringActive)?.index ?? null;
    const cold = frames.filter(frame => !frame.ringActive), warm = frames.filter(frame => frame.ringActive);
    return { backend, depth, round, gpuDevice: device, gpuTimerSupported: renderer.gpuTimerSupported,
      staticPreparationMs, staticQueryReceived, firstRingActiveFrame, totals: summarize(frames),
      cold: summarize(cold), warm: summarize(warm), frames };
  } finally { renderer.dispose(); gl.getExtension('WEBGL_lose_context')?.loseContext(); }
}

async function main() {
  if (document.hidden) throw new Error('Вкладка скрыта; измерение остановлено.');
  status.textContent = 'Одна опорная орбита Western armada, 576 бит…';
  await preloadRenderWasm();
  const core = await loadWasm();
  const reference = await core.compute({ id: 1, x: String(position.x), y: String(position.y), bits: BITS,
    iterations: options.iterations, fold: 1, celtic: 0 }, () => false, async () => {});
  const runs = [];
  for (const depth of options.depths) for (const [round, backend] of (['js', 'wasm', 'wasm', 'js'] as const).entries()) {
    const result = await run(backend, depth, round, reference); runs.push(result);
    summaryElement.textContent = JSON.stringify(runs.map(({ frames: _frames, gpuDevice: _gpuDevice, ...summary }) => summary), null, 2);
  }
  const report = { kind: 'cold-flight-gpu-work-profile', createdAt: new Date().toISOString(),
    userAgent: navigator.userAgent, devicePixelRatio, hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null, options,
    note: `GPU workload profile, NOT real FPS: serial fresh contexts, one render per rAF, readPixels drain after every frame; fixed ${options.rateFPS} Hz camera trajectory; static prerender before cold flight; mode0 identical RenderView inputs.`,
    reference: { bits: BITS, length: reference.length, capacity: reference.capacity, computeMs: reference.computeMs }, runs };
  const serialized = JSON.stringify(report, null, 2); reportElement.textContent = serialized;
  download.disabled = false;
  download.onclick = () => { const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'cold-flight.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); };
  status.textContent = 'Готово';
}
main().catch(error => { status.textContent = `Ошибка: ${error instanceof Error ? error.stack : error}`; console.error(error); });
