import { decimalToFixed, fixedToNumber } from './camera.ts';
import { createRingLayout } from './gpu/rings.ts';
import { WasmRenderer, type DirectReferenceResult } from './gpu/wasm-renderer.ts';
import { preloadRenderWasm } from './gpu/render-wasm.ts';
import { WESTERN_ARMADA } from './tours.ts';
import type { RenderView } from './types.ts';

export interface ZoomProfileOptions { width: number; height: number; aa: number; iterations: number }

const BITS = 576;
const DEPTHS = [30, 60, 90, 115] as const;
const FLIGHT_MS = 2000;
const ZOOM_PER_SECOND = .55;
const MAX_FRAME_DELTA_SECONDS = .1;
const MAX_WARMUP_FRAMES = 120;
const FALLBACK_FRAMES = 3;
const NO_YIELD = async () => {};

function statistics(samples: number[]) {
  if (!samples.length) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return { samples: sorted.length,
    medianMs: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * .95) - 1],
    minMs: sorted[0], maxMs: sorted[sorted.length - 1] };
}

function nextFrame(): Promise<number> {
  if (document.hidden) return Promise.reject(new Error('Профиль зума остановлен: вкладка скрыта.'));
  return new Promise<number>((resolve, reject) => {
    const timer = setTimeout(() => {
      cancelAnimationFrame(handle);
      reject(new Error('Профиль зума остановлен: rAF не отдал кадр за 2 секунды.'));
    }, 2000);
    const handle = requestAnimationFrame(timestamp => {
      clearTimeout(timer);
      if (document.hidden) reject(new Error('Профиль зума остановлен: вкладка скрыта.'));
      else resolve(timestamp);
    });
  });
}

function makeView(depth: number, reference: DirectReferenceResult, referenceKey: number,
  position: { x: bigint; y: bigint; bits: number }, options: ZoomProfileOptions,
  guided: boolean): RenderView {
  const logScale = Math.log2(3.2) - depth * Math.log2(10);
  return { center: [fixedToNumber(position.x, position.bits), fixedToNumber(position.y, position.bits)],
    scale: 2 ** logScale, logScale, offsetX: [0, 0], offsetY: [0, 0],
    iterations: options.iterations, fold: 1, celtic: 0, aa: options.aa, hue: 0,
    guided, referenceKey, temporal: true, position };
}

function collectGpuSamples(renderer: WasmRenderer, frameIds: Set<number>,
  collected: Set<number>, samples: number[]): void {
  for (const sample of renderer.gpuTimings) {
    if (frameIds.has(sample.frame) && !collected.has(sample.frame)) {
      collected.add(sample.frame);
      samples.push(sample.milliseconds);
    }
  }
}

async function fixedFallback(renderer: WasmRenderer, depth: number, reference: DirectReferenceResult,
  referenceKey: number, position: { x: bigint; y: bigint; bits: number }, options: ZoomProfileOptions) {
  const view = makeView(depth, reference, referenceKey, position, options, false);
  renderer.render(view);
  renderer.readPixels(); // Shader compile and texture setup stay outside the timed samples.
  renderer.clearGpuTimings();
  const gpuTimes: number[] = [];
  for (let index = 0; index < FALLBACK_FRAMES; index++) {
    renderer.render(view);
    const frameId = renderer.stats.frame;
    renderer.readPixels(); // Synchronise after, not inside, the hardware query.
    let elapsed: number | undefined;
    if (renderer.gpuTimerSupported) {
      const deadline = performance.now() + 1500;
      do {
        await nextFrame(); // Query availability can lag synchronous readPixels.
        elapsed = renderer.gpuTimings.find(sample => sample.frame === frameId)?.milliseconds;
      } while (elapsed === undefined && performance.now() < deadline);
    }
    if (elapsed !== undefined) gpuTimes.push(elapsed);
  }
  return { frames: FALLBACK_FRAMES, gpu: statistics(gpuTimes), gpuSamplesReceived: gpuTimes.length };
}

async function profileFlight(renderer: WasmRenderer, depth: number, reference: DirectReferenceResult,
  referenceKey: number, position: { x: bigint; y: bigint; bits: number }, options: ZoomProfileOptions) {
  // A synchronous readPixels warmup drain can finish inside a rendering
  // opportunity whose pending rAF timestamp predates the drain. Consume it
  // without rendering, then reset the camera clock before measured callbacks.
  await nextFrame();
  renderer.clearGpuTimings();
  const beforeFlight = { ...renderer.stats };
  const frameIds = new Set<number>(), collected = new Set<number>(), gpuTimes: number[] = [];
  const core = renderer.exports, input = renderer.renderInput;
  input[16] = 1; input[17] = 1;
  input[40] = depth; core.render_camera_command(3);
  input[40] = 1; input[41] = depth + 2; core.render_camera_command(4);
  core.render_camera_command(6);
  let firstTimestamp: number | null = null;
  let frames = 0, ringActiveFrames = 0, lastElapsed = 0;
  const cpuTimes: number[] = [];
  do {
    const timestamp = await nextFrame();
    firstTimestamp ??= timestamp;
    lastElapsed = timestamp - firstTimestamp;
    const started = performance.now();
    renderer.renderCamera(timestamp);
    cpuTimes.push(performance.now() - started);
    frameIds.add(renderer.stats.frame);
    frames++;
    if (renderer.stats.ringActive) ringActiveFrames++;
    collectGpuSamples(renderer, frameIds, collected, gpuTimes);
  } while (lastElapsed < FLIGHT_MS || frames < 2);
  renderer.readPixels(); // Drain finished GPU queries after the measured flight.
  collectGpuSamples(renderer, frameIds, collected, gpuTimes);
  return { targetDurationMs: FLIGHT_MS, elapsedMs: lastElapsed,
    startZoomPower10: depth, endZoomPower10: renderer.renderStats[9], zoomRateLog10PerSecond: ZOOM_PER_SECOND,
    maxFrameDeltaSeconds: MAX_FRAME_DELTA_SECONDS,
    frames, intervals: frames - 1, fps: (frames - 1) * 1000 / lastElapsed,
    ringActiveFrames, totalFrames: frames, ringActiveFraction: ringActiveFrames / frames,
    ringResets: renderer.stats.ringResets - beforeFlight.ringResets,
    ringSamples: renderer.stats.ringSamples - beforeFlight.ringSamples,
    ringFrames: renderer.stats.ringFrames - beforeFlight.ringFrames,
    referenceUploads: renderer.stats.referenceUploads - beforeFlight.referenceUploads,
    cpu: statistics(cpuTimes), gpu: statistics(gpuTimes), gpuSamplesReceived: gpuTimes.length };
}

/** Diagnostic only: one WASM reference and one full-quality renderer at the caller's canvas resolution. */
export async function runZoomProfile(options: ZoomProfileOptions, progress: (message: string) => void) {
  if (!Number.isSafeInteger(options.width) || options.width < 1
    || !Number.isSafeInteger(options.height) || options.height < 1
    || !Number.isSafeInteger(options.aa) || options.aa < 1 || options.aa > 5
    || !Number.isSafeInteger(options.iterations) || options.iterations < 1 || options.iterations > 65_536) {
    throw new Error('Некорректные параметры профиля зума.');
  }
  if (document.hidden) throw new Error('Профиль зума требует видимой вкладки.');

  await preloadRenderWasm();
  const canvas = document.createElement('canvas');
  canvas.width = options.width;
  canvas.height = options.height;
  const renderer = new WasmRenderer(canvas);
  try {
    const gl = canvas.getContext('webgl2')!;
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const layout = createRingLayout(options.width, options.height, maxTextureSize);
    const cacheLayout = layout ? { available: true, width: layout.width, height: layout.height,
      texels: layout.width * layout.height, bands: layout.bands.length } : { available: false,
      width: null, height: null, texels: null, bands: 0 };

    progress('Профиль зума: расчёт одной опорной орбиты Western armada…');
    const position = { x: decimalToFixed(WESTERN_ARMADA.x, BITS),
      y: decimalToFixed(WESTERN_ARMADA.y, BITS), bits: BITS };
    renderer.renderInput[4] = options.aa;
    const reference = await renderer.computeReferenceDirect({ id: 1, x: String(position.x), y: String(position.y),
      bits: BITS, iterations: options.iterations, fold: 1, celtic: 0 }, () => false, NO_YIELD);

    const depths = [];
    for (const [index, depth] of DEPTHS.entries()) {
      let coldFlight = null;
      if (depth === 115) {
        progress('Профиль зума: глубина 10^115, холодный полёт до прогрева кэша…');
        coldFlight = await profileFlight(renderer, depth, reference, reference.id + index + 1,
          position, options);
      }
      progress(`Профиль зума: глубина 10^${depth}, прогрев кэша…`);
      // A distinct view key starts each depth with a fresh ring cache without
      // uploading or recomputing the reference orbit.
      const referenceKey = reference.id + 100 + index;
      const input = renderer.renderInput, core = renderer.exports;
      input[16] = 1; input[17] = 1;
      input[40] = depth; core.render_camera_command(3);
      input[40] = 1; input[41] = depth + 2; core.render_camera_command(4);
      core.render_camera_command(6);
      const beforeWarmup = { ...renderer.stats };
      let warmupFrames = 0, warmedRingActive = false;
      if (layout) {
        for (; warmupFrames < MAX_WARMUP_FRAMES; warmupFrames++) {
          await nextFrame();
          renderer.renderCamera(0);
          if (renderer.stats.ringActive) { warmedRingActive = true; warmupFrames++; break; }
        }
      }
      const warmup = { frames: warmupFrames, ringActive: warmedRingActive,
        ringResets: renderer.stats.ringResets - beforeWarmup.ringResets,
        ringSamples: renderer.stats.ringSamples - beforeWarmup.ringSamples,
        skippedNoLayout: !layout };

      // rAF submission does not guarantee that GPU warmup draws have finished.
      // Drain them outside the measured flight so their queue cannot bias FPS.
      renderer.readPixels();

      progress(`Профиль зума: глубина 10^${depth}, полёт 2 с…`);
      const flight = await profileFlight(renderer, depth, reference, referenceKey,
        position, options);

      progress(`Профиль зума: глубина 10^${depth}, три кадра без guided…`);
      const fullScreen = await fixedFallback(renderer, depth, reference, referenceKey,
        position, options);
      depths.push({ depth, path: renderer.stats.path, coldFlight, warmup, flight, fullScreen });
    }
    return { kind: 'burning-ship-zoom-profile', createdAt: new Date().toISOString(),
      userAgent: navigator.userAgent, options: { ...options },
      reference: { bits: BITS, iterations: options.iterations, length: reference.length,
        computeMs: reference.computeMs, uploads: renderer.stats.referenceUploads },
      gpuTimerSupported: renderer.gpuTimerSupported, maxTextureSize, cacheLayout,
      warmupGpuDrained: true, clockBarrierAfterGpuDrain: true,
      depths };
  } finally { renderer.dispose(); }
}
