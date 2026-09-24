import { cameraOffset, decimalToFixed, fixedToNumber, setZoom } from './camera.ts';
import type { Camera } from './camera.ts';
import { computeReferenceJs } from './compute/reference-js.ts';
import { ReferenceClient } from './compute/client.ts';
import { probeCancellation } from './compute/cancellation-probe.ts';
import { loadWasm } from './compute/wasm.ts';
import type { WasmCore } from './compute/wasm.ts';
import { createRenderer } from './gpu/engine.ts';
import { preloadRenderWasm } from './gpu/render-wasm.ts';
import { WasmRenderer } from './gpu/wasm-renderer.ts';
import type { RendererAdapter, RendererBackend } from './gpu/engine.ts';
import { WESTERN_ARMADA } from './tours.ts';
import type { ReferenceRequest, ReferenceResult, RenderView } from './types.ts';

const WARMUPS = 2, RUNS = 9, WIDTH = 320, HEIGHT = 200, AA = 2, ITERATIONS = 16384;
const ARRAY_NAMES = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
const noYield = async () => {};
const pause = (milliseconds = 0) => new Promise<void>(resolve => setTimeout(resolve, milliseconds));
/** Consume the current rendering opportunity after a synchronous GPU drain.
 * readPixels can block an already-started browser frame; its pending rAF still
 * carries that old start timestamp. The measured callback must be the next one. */
function animationClockBarrier(): Promise<void> {
  if (document.hidden) return Promise.reject(new Error('Вкладка скрылась перед FPS замером.'));
  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      cancelAnimationFrame(handle); reject(new Error('rAF не отдал кадр после прогрева за 3 секунды.'));
    }, 3000);
    const handle = requestAnimationFrame(() => {
      clearTimeout(timeout);
      if (document.hidden) reject(new Error('Вкладка скрылась перед FPS замером.'));
      else resolve();
    });
  });
}

const summary = (samples: number[]) => {
  const sorted = [...samples].sort((a, b) => a - b);
  const totalMs = samples.reduce((total, sample) => total + sample, 0);
  return { runs: samples.length, totalMs, meanMs: totalMs / samples.length, medianMs: sorted[Math.floor(sorted.length / 2)],
    p95Ms: sorted[Math.ceil(sorted.length * .95) - 1], minMs: sorted[0], maxMs: sorted[sorted.length - 1], samplesMs: samples };
};

function byteDiff(actual: ReferenceResult, expected: ReferenceResult) {
  let mismatchedBytes = 0;
  const arrays = ARRAY_NAMES.map(name => {
    const a = new Uint8Array(actual[name].buffer, actual[name].byteOffset, actual[name].byteLength);
    const b = new Uint8Array(expected[name].buffer, expected[name].byteOffset, expected[name].byteLength);
    let different = Math.abs(a.length - b.length), firstMismatch: number | null = a.length === b.length ? null : Math.min(a.length, b.length);
    for (let index = 0; index < Math.min(a.length, b.length); index++) if (a[index] !== b[index]) {
      different++; firstMismatch ??= index;
    }
    mismatchedBytes += different;
    return { name, bytes: a.byteLength, mismatchedBytes: different, firstMismatch };
  });
  const metadataEqual = actual.length === expected.length && actual.capacity === expected.capacity
    && actual.bits === expected.bits && actual.iterations === expected.iterations
    && actual.fold === expected.fold && actual.celtic === expected.celtic;
  return { equal: metadataEqual && mismatchedBytes === 0, metadataEqual, mismatchedBytes, arrays };
}

function pixelDiff(a: Uint8Array, b: Uint8Array) {
  if (a.length !== b.length) throw new Error(`Pixel diff: разные размеры RGBA (${a.length} и ${b.length}).`);
  let mismatchedChannels = 0, mismatchedPixels = 0, maxAbsoluteDifference = 0, sum = 0;
  for (let pixel = 0; pixel < a.length; pixel += 4) {
    let different = false;
    for (let channel = 0; channel < 4; channel++) {
      const delta = Math.abs(a[pixel + channel] - b[pixel + channel]);
      if (delta) { mismatchedChannels++; different = true; }
      maxAbsoluteDifference = Math.max(maxAbsoluteDifference, delta); sum += delta;
    }
    if (different) mismatchedPixels++;
  }
  return { equal: mismatchedChannels === 0, mismatchedChannels, mismatchedPixels,
    maxAbsoluteDifference, meanAbsoluteDifference: sum / a.length };
}

function imageContent(pixels: Uint8Array) {
  const colours = new Set<number>();
  let minRGB = 255, maxRGB = 0, nonOpaquePixels = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (colours.size < 256) colours.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
    minRGB = Math.min(minRGB, pixels[i], pixels[i + 1], pixels[i + 2]);
    maxRGB = Math.max(maxRGB, pixels[i], pixels[i + 1], pixels[i + 2]);
    if (pixels[i + 3] !== 255) nonOpaquePixels++;
  }
  return { nontrivialRGB: colours.size > 1, distinctRGBAtLeast: colours.size, minRGB, maxRGB, nonOpaquePixels };
}

function referenceRequest(id: number, x: string, y: string, bits: number, iterations: number): ReferenceRequest {
  return { id, x: String(decimalToFixed(x, bits)), y: String(decimalToFixed(y, bits)), bits, iterations, fold: 1, celtic: 0 };
}

async function cpuCase(name: string, request: ReferenceRequest, core: WasmCore, progress: (message: string) => void) {
  const runners = {
    js: () => computeReferenceJs(request, () => false, noYield),
    wasm: () => core.compute(request, () => false, noYield),
  };
  const samples = { js: [] as number[], wasm: [] as number[] };
  const latest: Partial<Record<keyof typeof runners, ReferenceResult>> = {};
  for (let round = 0; round < WARMUPS; round++) for (const variant of ['js', 'wasm'] as const) {
    progress(`CPU ${name}: прогрев ${round + 1}/${WARMUPS}, ${variant}`);
    await pause(); latest[variant] = await runners[variant]();
  }
  const memoryBefore = core.memoryBytes;
  let memoryStable = true;
  // No timer/rAF/scheduler yields occur inside either numerical kernel. The
  // event-loop yield and byte comparisons below are outside each timed sample.
  for (let round = 0; round < RUNS; round++) {
    const variants = round % 2 ? ['wasm', 'js'] as const : ['js', 'wasm'] as const;
    for (const variant of variants) {
      progress(`CPU ${name}: ${round + 1}/${RUNS}, ${variant}`); await pause();
      const started = performance.now();
      latest[variant] = await runners[variant]();
      samples[variant].push(performance.now() - started);
      memoryStable &&= core.memoryBytes === memoryBefore;
    }
  }
  const js = latest.js!, wasm = latest.wasm!;
  const jsTime = summary(samples.js), wasmTime = summary(samples.wasm);
  const report = {
    name, bits: request.bits, iterations: request.iterations, length: js.length, capacity: js.capacity,
    js: jsTime,
    wasm: { ...wasmTime, memoryMode: core.views.mode, memoryBytes: core.memoryBytes, memoryBytesStable: memoryStable },
    wasmSpeedupVsJs: jsTime.medianMs / wasmTime.medianMs,
    exactArrays: byteDiff(wasm, js),
  };
  return { report, js, wasm };
}

async function workerCase(expected: ReferenceResult, local: { js: { medianMs: number }; wasm: { medianMs: number } },
  progress: (message: string) => void) {
  const client = new ReferenceClient();
  const request = referenceRequest(7100, WESTERN_ARMADA.x, WESTERN_ARMADA.y, 576, ITERATIONS);
  const variants = [];
  try {
    for (const backend of ['wasm', 'js'] as const) {
      const wall: number[] = [], computation: number[] = [];
      let memoryBytes = 0, memoryMode = '', memoryBytesStable = true, allArraysEqual = true;
      for (let round = -WARMUPS; round < RUNS; round++) {
        progress(`Worker ${backend}: ${round < 0 ? 'прогрев' : `${round + 1}/${RUNS}`}`); await pause();
        const started = performance.now();
        const result = await client.compute(request, backend);
        const wallMs = performance.now() - started;
        const comparison = byteDiff(result.result, expected);
        allArraysEqual &&= comparison.equal;
        if (round >= 0) {
          wall.push(wallMs); computation.push(result.result.computeMs);
          memoryBytesStable &&= result.memoryBytes === memoryBytes;
        } else memoryBytes = result.memoryBytes;
        memoryMode = result.memoryMode;
        client.recycle(result.result);
      }
      const wallTime = summary(wall), computationTime = summary(computation);
      const localMedian = local[backend].medianMs;
      variants.push({ backend, wall: wallTime, computation: computationTime, localMedianMs: localMedian,
        workerWallOverLocalRatio: wallTime.medianMs / localMedian,
        workerComputationOverLocalRatio: computationTime.medianMs / localMedian,
        memoryBytes, memoryMode, memoryBytesStable, allArraysEqual });
    }
    progress('Worker: отмена устаревшего запроса и повторное использование буферов');
    const expensive = referenceRequest(7101, '0', '0', 128, 65536);
    const first = client.compute(expensive, 'wasm').then(
      () => ({ rejected: false, errorName: null as string | null }),
      (error: unknown) => ({ rejected: true, errorName: error instanceof Error ? error.name : String(error) }),
    );
    // Queue the replacement immediately: the expensive job must never replace
    // the camera's newer reference even if its worker response arrives late.
    const second = await client.compute(request, 'wasm');
    const cancelled = await first;
    const replacementArrays = byteDiff(second.result, expected);
    const replacementId = second.result.id;
    const stableBytes = second.memoryBytes;
    client.recycle(second.result);
    let recycledRunsExact = true, memoryAfterRaceStable = true;
    const recycledComputeMs: number[] = [];
    for (let round = 0; round < 3; round++) {
      const next = await client.compute(request, 'wasm');
      recycledRunsExact &&= byteDiff(next.result, expected).equal;
      memoryAfterRaceStable &&= next.memoryBytes === stableBytes;
      recycledComputeMs.push(next.result.computeMs);
      client.recycle(next.result);
    }
    return { variants, cancellation: { firstRejectedWithAbortError: cancelled.rejected && cancelled.errorName === 'AbortError',
      firstErrorName: cancelled.errorName, replacementId, replacementArrays, recycledRunsExact,
      memoryAfterRaceStable, stableMemoryBytes: stableBytes, recycledComputeMs },
      timingNote: 'wall includes messaging; computation includes the worker cooperative scheduler; local core has no scheduler yields' };
  } finally { client.dispose(); }
}

function makeView(zoom: number | null, referenceKey: number): RenderView {
  const logScale = Math.log2(3.2) - (zoom ?? 0) * Math.log2(10);
  return {
    center: zoom === null ? [-.45, -.45] : [Number(WESTERN_ARMADA.x), Number(WESTERN_ARMADA.y)],
    scale: 2 ** logScale, logScale, offsetX: [0, 0], offsetY: [0, 0], iterations: ITERATIONS,
    fold: 1, celtic: 0, aa: AA, hue: 0, guided: false, referenceKey,
  };
}

function harness(backend: RendererBackend, width = WIDTH, height = HEIGHT) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  return { canvas, renderer: createRenderer(canvas, backend) };
}

async function gpuQuery(renderer: RendererAdapter, frame: number): Promise<number | null> {
  if (!renderer.gpuTimerSupported) { await pause(); return null; }
  const deadline = performance.now() + 5000;
  do {
    await pause(8);
    const sample = renderer.gpuTimings.find(item => item.frame === frame);
    if (sample) return sample.milliseconds;
  } while (performance.now() < deadline);
  return null; // Disjoint/timed-out queries cannot serve as GPU measurements.
}

async function gpuCase(renderers: Record<RendererBackend, RendererAdapter>, zoom: number | null,
  references: Record<RendererBackend, ReferenceResult>,
  progress: (message: string) => void) {
  const name = zoom === null ? 'overview' : `western-armada-10^${zoom}`;
  for (const backend of ['js', 'wasm'] as const) renderers[backend].setReference(references[backend]);
  for (let round = 0; round < WARMUPS; round++) {
    for (const backend of ['js', 'wasm'] as const) {
      const renderer = renderers[backend];
      progress(`GPU ${name}: прогрев ${round + 1}/${WARMUPS}, ${backend}`);
      renderer.render(makeView(zoom, references[backend].id));
      renderer.readPixels(); // Shader compilation and upload stay outside timed samples.
      await pause();
    }
  }
  for (const backend of ['js', 'wasm'] as const) renderers[backend].clearGpuTimings();
  const samples = { js: [] as number[], wasm: [] as number[] };
  for (let round = 0; round < RUNS; round++) {
    for (const backend of round % 2 ? ['wasm', 'js'] as const : ['js', 'wasm'] as const) {
      const renderer = renderers[backend];
      progress(`GPU ${name}: ${round + 1}/${RUNS}, ${backend}`);
      renderer.render(makeView(zoom, references[backend].id));
      const elapsed = await gpuQuery(renderer, renderer.stats.frame);
      if (elapsed !== null) samples[backend].push(elapsed);
    }
  }
  const images = {} as Record<RendererBackend, Uint8Array>;
  for (const backend of ['js', 'wasm'] as const) {
    renderers[backend].render(makeView(zoom, references[backend].id));
    images[backend] = renderers[backend].readPixels();
  }
  const jsPixels = images.js, wasmPixels = images.wasm;
  const jsVsWasmPixels = pixelDiff(jsPixels, wasmPixels);
  const content = { js: imageContent(jsPixels), wasm: imageContent(wasmPixels) };
  const gpuTime = { js: samples.js.length === RUNS ? summary(samples.js) : null,
    wasm: samples.wasm.length === RUNS ? summary(samples.wasm) : null };
  await pause();
  return {
    name, zoomPower10: zoom, mode: 'full-screen-spatial-aa', guided: false,
    width: WIDTH, height: HEIGHT, aaSamples: AA, iterations: ITERATIONS,
    path: { js: renderers.js.stats.path, wasm: renderers.wasm.stats.path },
    logScale: makeView(zoom, references.wasm.id).logScale,
    jsVsWasmPixels, content,
    gpuTimerSupported: { js: renderers.js.gpuTimerSupported, wasm: renderers.wasm.gpuTimerSupported },
    gpuTime, gpuSamplesReceived: { js: samples.js.length, wasm: samples.wasm.length },
    gpuSpeedupVsJs: jsVsWasmPixels.equal && gpuTime.js && gpuTime.wasm
      ? gpuTime.js.medianMs / gpuTime.wasm.medianMs : null,
  };
}

const ANIMATION_START_ZOOM = 30, ANIMATION_END_ZOOM = 30.55, ANIMATION_DURATION_MS = 1000;
interface AnimationSample { frames: number; intervals: number; elapsedMs: number; fps: number;
  endZoomPower10: number; ringActiveFrames: number; ringResets: number; ringSamples: number;
  warmupFrames: number; warmupRingActive: boolean; cpuFrame: ReturnType<typeof summary>;
  maxRafGapMs: number; clampedRafTimeMs: number; expectedEndZoomPower10: number;
  warmupDrainMs: number; firstFrameRafAgeMs: number; firstFrameTimestampMs: number; firstFrameCallbackNowMs: number;
  firstTimestampBeforeDrainMs: number;
  fallbackFrames: Array<{ frame: number; elapsedMs: number; rafGapMs: number; cpuMs: number }> }

interface AnimationDriver {
  start(zoom: number): void;
  warmup(): void;
  beginMeasurement(): void;
  frame(timestamp: number): void;
  stop(): void;
  zoom(): number;
}

/** Exercise each engine's full camera/planner path. Both run serially under the
 * same rAF clock here; the application's worker scheduling is not GPU time. */
function animationDriver(renderer: RendererAdapter, reference: ReferenceResult): AnimationDriver {
  if (renderer instanceof WasmRenderer) {
    const core = renderer.exports, input = renderer.renderInput, stats = renderer.renderStats;
    const command = (op: number, first = 0, second = 0) => {
      input[40] = first; input[41] = second;
      const status = core.render_camera_command(op);
      if (status < 0) throw new Error(`WASM flight command ${op}: ${status}`);
    };
    input[3] = ITERATIONS; input[4] = AA; input[5] = 1; input[6] = 0; input[7] = 0;
    input[16] = 1; input[17] = 1; input[18] = reference.id;
    return {
      start(zoom) { command(3, zoom); command(4, 1, zoom + 1); input[16] = 1; },
      warmup() { renderer.renderCamera(0); },
      beginMeasurement() { command(6); },
      frame(timestamp) { renderer.renderCamera(timestamp); },
      stop() { command(4, 0); },
      zoom: () => stats[9],
    };
  }
  const initial: Camera = { x: decimalToFixed(WESTERN_ARMADA.x, reference.bits),
    y: decimalToFixed(WESTERN_ARMADA.y, reference.bits), bits: reference.bits, logScale: Math.log2(3.2) };
  const camera = { ...initial };
  let currentZoom = 0, previousTimestamp: number | null = null;
  function renderCamera() {
    const offset = cameraOffset(camera, initial);
    renderer.render({ center: [fixedToNumber(camera.x, camera.bits), fixedToNumber(camera.y, camera.bits)],
      scale: 2 ** camera.logScale, logScale: camera.logScale, offsetX: offset.x, offsetY: offset.y,
      iterations: ITERATIONS, fold: 1, celtic: 0, aa: AA, hue: 0,
      guided: true, temporal: true, referenceKey: reference.id, position: camera });
  }
  return {
    start(zoom) { currentZoom = zoom; setZoom(camera, zoom); previousTimestamp = null; },
    warmup: renderCamera,
    beginMeasurement() { previousTimestamp = null; },
    frame(timestamp) {
      const elapsed = previousTimestamp === null ? 0 : Math.max(0, Math.min(.1, (timestamp - previousTimestamp) / 1000));
      previousTimestamp = timestamp;
      // Match the application camera recurrence, including the inverse zoom
      // conversion, instead of giving JS a cheaper precomputed RenderView.
      currentZoom = (Math.log2(3.2) - camera.logScale) / Math.log2(10) + elapsed * .55;
      setZoom(camera, currentZoom); renderCamera();
    },
    stop() {},
    zoom: () => currentZoom,
  };
}

async function animationPass(renderer: RendererAdapter, driver: AnimationDriver, startZoom: number): Promise<AnimationSample> {
  driver.start(startZoom);
  let warmupFrames = 0;
  for (; warmupFrames < 120; warmupFrames++) {
    driver.warmup();
    if (renderer.stats.ringActive) { warmupFrames++; break; }
    await pause();
  }
  const drainStarted = performance.now();
  renderer.readPixels(); // Drain warmup GPU work before timing rAF render calls.
  const drainFinished = performance.now(), warmupDrainMs = drainFinished - drainStarted;
  const warmupRingActive = renderer.stats.ringActive;
  const before = { ...renderer.stats };
  // Apply the same clock barrier to JS and WASM. No rendering or camera motion
  // occurs here, and the warmed ring cache remains intact.
  await animationClockBarrier();
  driver.beginMeasurement(); // Reset only the clock, retaining the warmed cache.
  return new Promise<AnimationSample>((resolve, reject) => {
    let started: number | null = null, previousTimestamp: number | null = null;
    let maxRafGapMs = 0, clampedRafTimeMs = 0, firstFrameRafAgeMs = 0, firstFrameTimestampMs = 0, firstFrameCallbackNowMs = 0;
    const fallbackFrames: AnimationSample['fallbackFrames'] = [];
    let frames = 0, ringActiveFrames = 0, handle = 0, finished = false;
    const cpuFrames: number[] = [];
    const fail = (error: Error) => {
      if (finished) return;
      finished = true; cancelAnimationFrame(handle); clearTimeout(watchdog); driver.stop(); reject(error);
    };
    const watchdog = setTimeout(() => fail(new Error('rAF не отдал кадры за 3 секунды.')), 3000);
    const frame = (timestamp: number) => {
      if (document.hidden) return fail(new Error('Вкладка скрылась во время FPS замера.'));
      if (started === null) { firstFrameTimestampMs = timestamp; firstFrameCallbackNowMs = performance.now();
        firstFrameRafAgeMs = firstFrameCallbackNowMs - timestamp; }
      started ??= timestamp;
      const elapsed = timestamp - started;
      const rafGapMs = previousTimestamp === null ? 0 : timestamp - previousTimestamp;
      previousTimestamp = timestamp; maxRafGapMs = Math.max(maxRafGapMs, rafGapMs);
      clampedRafTimeMs += Math.max(0, rafGapMs - 100);
      try {
        const cpuStart = performance.now();
        driver.frame(timestamp);
        const cpuMs = performance.now() - cpuStart; cpuFrames.push(cpuMs);
        frames++;
        if (renderer.stats.ringActive) ringActiveFrames++;
        else fallbackFrames.push({ frame: frames, elapsedMs: elapsed, rafGapMs, cpuMs });
      } catch (error) { return fail(error instanceof Error ? error : new Error(String(error))); }
      if (elapsed >= ANIMATION_DURATION_MS && frames > 1) {
        try { renderer.readPixels(); } // Keep the next ABBA pass free of queued GPU work.
        catch (error) { return fail(error instanceof Error ? error : new Error(String(error))); }
        finished = true; clearTimeout(watchdog);
        const endZoomPower10 = driver.zoom(); driver.stop();
        resolve({ frames, intervals: frames - 1, elapsedMs: elapsed, fps: (frames - 1) * 1000 / elapsed,
          endZoomPower10, ringActiveFrames, ringResets: renderer.stats.ringResets - before.ringResets,
          ringSamples: renderer.stats.ringSamples - before.ringSamples,
          warmupFrames, warmupRingActive, cpuFrame: summary(cpuFrames), maxRafGapMs, clampedRafTimeMs,
          expectedEndZoomPower10: startZoom + .55 * elapsed / 1000, fallbackFrames, warmupDrainMs,
          firstFrameRafAgeMs, firstFrameTimestampMs, firstFrameCallbackNowMs,
          firstTimestampBeforeDrainMs: Math.max(0, drainFinished - firstFrameTimestampMs) });
      } else handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
  });
}

async function animationCase(renderers: Record<RendererBackend, RendererAdapter>,
  references: Record<RendererBackend, ReferenceResult>, width: number, height: number,
  progress: (message: string) => void) {
  const scene = { width, height, aaSamples: AA, iterations: ITERATIONS, referenceBits: 576,
    zoomDeltaPower10: ANIMATION_END_ZOOM - ANIMATION_START_ZOOM,
    zoomRateLog10PerSecond: .55, durationMs: ANIMATION_DURATION_MS, guided: true, clockBarrierAfterGpuDrain: true };
  if (document.hidden) return { supported: false, reason: 'Вкладка скрыта', scenes: [], scene };
  try {
    renderers.js.setReference(references.js);
    if (!(renderers.wasm instanceof WasmRenderer)) throw new Error('WASM flight requires the Rust render runtime.');
    // Production setup: same instance computes the reference and uploads its
    // linear-memory buffers to GPU. No worker result copies or RenderView loop.
    await renderers.wasm.computeReferenceDirect(referenceRequest(references.wasm.id,
      WESTERN_ARMADA.x, WESTERN_ARMADA.y, references.wasm.bits, ITERATIONS));
    const drivers = { js: animationDriver(renderers.js, references.js), wasm: animationDriver(renderers.wasm, references.wasm) };
    const scenes = [];
    for (const startZoom of [30, 115]) {
      const samples = { js: [] as AnimationSample[], wasm: [] as AnimationSample[] };
      // ABBA order reduces drift from thermal state and browser scheduling.
      for (const [index, backend] of (['js', 'wasm', 'wasm', 'js'] as const).entries()) {
        progress(`FPS 10^${startZoom}: ${index + 1}/4, ${backend}`);
        samples[backend].push(await animationPass(renderers[backend], drivers[backend], startZoom));
      }
      const summarize = (runs: AnimationSample[]) => ({ runs: runs.length,
        medianFps: (runs[0].fps + runs[1].fps) / 2, samples: runs,
        cpuFrame: summary(runs.flatMap(run => run.cpuFrame.samplesMs)),
        ringActiveFraction: runs.reduce((n, run) => n + run.ringActiveFrames / run.frames, 0) / runs.length });
      scenes.push({ startZoomPower10: startZoom, endZoomPower10: startZoom + scene.zoomDeltaPower10,
        js: summarize(samples.js), wasm: summarize(samples.wasm) });
    }
    return { supported: true, reason: null, scenes, scene,
      note: 'Serial ABBA, warmed caches retained. A shared rAF clock barrier after GPU drain excludes stale pre-drain frame timestamps. JS computes its camera and RenderView; WASM render_frame(mode=1) computes camera, cache and draw dispatch entirely in Rust. CPU frame time includes camera/planning/WebGL submission and excludes GPU completion/rAF wait. Both benchmark drivers run on the same thread; production WASM uses an OffscreenCanvas worker. Guided cache policies differ; FPS is refresh-rate limited.' };
  } catch (error) {
    return { supported: false, reason: error instanceof Error ? error.message : String(error), scenes: [], scene };
  }
}

async function ringCase(renderers: Record<RendererBackend, RendererAdapter>,
  references: Record<RendererBackend, ReferenceResult>,
  progress: (message: string) => void) {
  const images = {} as Record<RendererBackend, Uint8Array>;
  const variants = [];
  for (const backend of ['js', 'wasm'] as const) {
    const renderer = renderers[backend], reference = references[backend];
    renderer.setReference(reference);
    const view = { ...makeView(30, reference.id), guided: true };
    const initialRows = renderer.stats.ringsDrawn;
    let frames = 0;
    do {
      progress(`Кольцевой кэш: ${backend}, заполнение ${frames + 1}`);
      renderer.render(view); frames++;
      await pause();
      if (frames >= 128 && !renderer.stats.ringActive) throw new Error('Кольцевой кэш не заполнился за 128 кадров при 320×200.');
    } while (!renderer.stats.ringActive);
    images[backend] = renderer.readPixels();
    const fullRows = renderer.stats.ringsDrawn - initialRows;
    const beforeReuse = renderer.stats.ringsDrawn;
    renderer.render(view);
    const repeatedViewNewRows = renderer.stats.ringsDrawn - beforeReuse;
    const repeatPixels = pixelDiff(images[backend], renderer.readPixels());
    const beforeZoom = renderer.stats.ringsDrawn, samplesBeforeZoom = renderer.stats.ringSamples;
    const zoomed = { ...view, logScale: view.logScale - .02, scale: view.scale * 2 ** -.02 };
    renderer.render(zoomed);
    variants.push({ backend, initialFillFrames: frames, initialRows: fullRows,
      repeatedViewNewRows, zeroWorkOnRepeatedView: repeatedViewNewRows === 0,
      repeatPixels,
      zoomNewRows: renderer.stats.ringsDrawn - beforeZoom,
      zoomNewSamples: renderer.stats.ringSamples - samplesBeforeZoom,
      zoomReusedCache: renderer.stats.ringActive,
    });
    await pause();
  }
  return { enabled: true, zoomPower10: 30, mode: 'guided-original-vs-segmented',
    jsVsWasmPixels: pixelDiff(images.js, images.wasm),
    content: { js: imageContent(images.js), wasm: imageContent(images.wasm) }, variants,
    qualityNote: 'Original JS uses adaptive-density continuous rings and original assembly; WASM uses segmented 1.5-density rings. Cross-renderer pixels are informational.' };
}

function temporalView(reference: ReferenceResult): RenderView {
  return { ...makeView(30, reference.id), aa: 3, temporal: true,
    position: { x: decimalToFixed(WESTERN_ARMADA.x, reference.bits), y: decimalToFixed(WESTERN_ARMADA.y, reference.bits), bits: reference.bits } };
}
async function settleTemporal(renderer: RendererAdapter, view: RenderView) {
  renderer.resetTemporal();
  let frames = 0;
  do {
    renderer.render(view); frames++; await pause();
    if (frames > 32) throw new Error('Temporal AA не завершил накопление за 32 неподвижных кадра.');
  } while (renderer.settling);
  return frames;
}
async function temporalCase(renderers: Record<RendererBackend, RendererAdapter>,
  references: Record<RendererBackend, ReferenceResult>,
  progress: (message: string) => void) {
  progress('Temporal AA: сравнение полностью накопленных кадров');
  const images = {} as Record<RendererBackend, Uint8Array>;
  const variants = [];
  for (const backend of ['js', 'wasm'] as const) {
    const renderer = renderers[backend], reference = references[backend];
    renderer.setReference(reference);
    const frames = await settleTemporal(renderer, temporalView(reference));
    images[backend] = renderer.readPixels();
    const resetFrames = await settleTemporal(renderer, temporalView(reference));
    variants.push({ backend, frames, resetFrames, settledAtRequestedSamples: frames === 3 && resetFrames === 3,
      repeatAfterReset: pixelDiff(images[backend], renderer.readPixels()) });
  }
  return { supported: true, enabled: true, aaSamples: 3, zoomPower10: 30, variants,
    jsVsWasmPixels: pixelDiff(images.js, images.wasm),
    content: { js: imageContent(images.js), wasm: imageContent(images.wasm) },
    qualityNote: 'Within-renderer reset is a correctness check; cross-renderer temporal pixels are informational.' };
}

function contextEvent(canvas: HTMLCanvasElement, type: 'webglcontextlost' | 'webglcontextrestored') {
  return new Promise<void>((resolve, reject) => {
    const listener = (event: Event) => {
      if (type === 'webglcontextlost') event.preventDefault();
      clearTimeout(timeout); canvas.removeEventListener(type, listener); resolve();
    };
    const timeout = setTimeout(() => { canvas.removeEventListener(type, listener); reject(new Error(`${type}: событие не пришло за 2 секунды.`)); }, 2000);
    canvas.addEventListener(type, listener);
  });
}

async function contextRestoreCase(backend: RendererBackend, reference: ReferenceResult,
  progress: (message: string) => void) {
  const test = harness(backend); let renderer = test.renderer;
  const gl = test.canvas.getContext('webgl2')!;
  const extension = gl.getExtension('WEBGL_lose_context');
  try {
    if (!extension) return { supported: false, restored: null, pixels: null, temporalSupported: false, temporalPixels: null, temporalFrames: null, reason: 'WEBGL_lose_context недоступен' };
    progress(`GPU ${backend}: проверка потери и восстановления контекста`);
    renderer.setReference(reference);
    const view = makeView(10, reference.id);
    renderer.render(view); const before = renderer.readPixels();
    const temporalSupported = 'resetTemporal' in renderer;
    const beforeFrames = temporalSupported ? await settleTemporal(renderer, temporalView(reference)) : null;
    const beforeTemporal = temporalSupported ? renderer.readPixels() : null;
    const lost = contextEvent(test.canvas, 'webglcontextlost');
    extension.loseContext(); await lost;
    const restored = contextEvent(test.canvas, 'webglcontextrestored');
    await pause(50); extension.restoreContext(); await restored;
    // Match the app lifecycle: retire stale GPU handles after restoration.
    renderer.dispose();
    renderer = createRenderer(test.canvas, backend);
    renderer.setReference(reference); renderer.render(view);
    const pixels = pixelDiff(before, renderer.readPixels());
    const afterFrames = temporalSupported ? await settleTemporal(renderer, temporalView(reference)) : null;
    return { supported: true, restored: true, pixels, temporalSupported, temporalPixels: beforeTemporal ? pixelDiff(beforeTemporal, renderer.readPixels()) : null,
      temporalFrames: { before: beforeFrames, after: afterFrames }, reason: null };
  } catch (error) {
    // A lifecycle failure is one independent test result. Preserve all earlier
    // CPU/GPU evidence so a late restore failure cannot hide pixel mismatches.
    return { supported: extension !== null, restored: false, pixels: null,
      temporalSupported: 'resetTemporal' in renderer, temporalPixels: null, temporalFrames: null,
      reason: error instanceof Error ? error.message : String(error) };
  } finally { renderer.dispose(); }
}

/** Runs on a detached disposable canvas. CPU kernels, shader work and pixel
 * correctness are measured independently; rAF/event-loop cadence is never GPU time. */
export async function runBenchmark(progress: (message: string) => void,
  options: { flightWidth?: number; flightHeight?: number } = {}) {
  const flightWidth = options.flightWidth ?? WIDTH, flightHeight = options.flightHeight ?? HEIGHT;
  if (!Number.isSafeInteger(flightWidth) || flightWidth < 1
    || !Number.isSafeInteger(flightHeight) || flightHeight < 1) {
    throw new Error('Размер canvas для FPS сравнения должен быть положительным целым числом.');
  }
  progress('Загрузка WASM SIMD для проверки…'); await pause();
  const [wasm] = await Promise.all([loadWasm(), preloadRenderWasm()]);
  const deep = await cpuCase('western-armada-576-16384', referenceRequest(7001, WESTERN_ARMADA.x, WESTERN_ARMADA.y, 576, ITERATIONS), wasm, progress);
  const interior = await cpuCase('interior-origin-128-1024', referenceRequest(7002, '0', '0', 128, 1024), wasm, progress);
  const worker = await workerCase(deep.js, deep.report, progress);
  progress('Raw Worker: проверка отмены до начала GPU измерений…');
  await pause();
  const cancellationProbe = await probeCancellation();
  const staticHarness = { js: harness('js'), wasm: harness('wasm') };
  const renderers = { js: staticHarness.js.renderer, wasm: staticHarness.wasm.renderer };
  const references = { js: deep.js, wasm: deep.wasm };
  const gl = staticHarness.wasm.canvas.getContext('webgl2')!;
  const debugRenderer = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuDevice = {
    vendor: String(gl.getParameter(debugRenderer ? debugRenderer.UNMASKED_VENDOR_WEBGL : gl.VENDOR)),
    renderer: String(gl.getParameter(debugRenderer ? debugRenderer.UNMASKED_RENDERER_WEBGL : gl.RENDERER)),
    identification: debugRenderer ? 'unmasked' : 'masked',
  };
  try {
    const gpu = [];
    for (const zoom of [null, 10, 30, 90, 120]) gpu.push(await gpuCase(renderers, zoom, references, progress));
    const rings = await ringCase(renderers, references, progress);
    const temporal = await temporalCase(renderers, references, progress);
    const flightHarness = { js: harness('js', flightWidth, flightHeight),
      wasm: harness('wasm', flightWidth, flightHeight) };
    let animationFps;
    try {
      animationFps = await animationCase({ js: flightHarness.js.renderer, wasm: flightHarness.wasm.renderer },
        references, flightWidth, flightHeight, progress);
    } finally { flightHarness.js.renderer.dispose(); flightHarness.wasm.renderer.dispose(); }
    const contextRestore = { js: await contextRestoreCase('js', deep.js, progress),
      wasm: await contextRestoreCase('wasm', deep.wasm, progress) };
    const cpu = [deep.report, interior.report];
    const memoryBytesStable = cpu.every(item => item.wasm.memoryBytesStable)
      && worker.variants.every(item => item.memoryBytesStable) && worker.cancellation.memoryAfterRaceStable;
    const workerChecksPassed = worker.variants.every(item => item.allArraysEqual)
      && worker.cancellation.firstRejectedWithAbortError && worker.cancellation.replacementArrays.equal && worker.cancellation.recycledRunsExact;
    const rawCancellationChecksPassed = cancellationProbe.checks.passed;
    const exactReferenceArrays = cpu.every(item => item.exactArrays.equal);
    const exactGpuPixels = gpu.every(item => item.jsVsWasmPixels.equal);
    const nontrivialGpuImages = gpu.every(item => item.content.js.nontrivialRGB && item.content.wasm.nontrivialRGB);
    const ringChecksPassed = rings.variants.every(item => item.zeroWorkOnRepeatedView && item.repeatPixels.equal
      && item.zoomNewRows > 0 && item.zoomReusedCache);
    const temporalChecksPassed = temporal.supported && temporal.variants.every(item =>
      item.settledAtRequestedSamples && item.repeatAfterReset.equal)
      && temporal.content.js.nontrivialRGB && temporal.content.wasm.nontrivialRGB;
    const contextRestorePassed = Object.values(contextRestore).every(item => !item.supported
      || (item.restored === true && item.pixels?.equal === true
        && (!item.temporalSupported || (item.temporalPixels?.equal === true
          && item.temporalFrames?.before === 3 && item.temporalFrames?.after === 3))));
    progress('Проверки завершены');
    return {
      kind: 'burning-ship-browser-benchmark', createdAt: new Date().toISOString(), userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency, devicePixelRatio, gpuDevice,
      resizableMemory: { wasm: wasm.views.mode === 'resizable',
        wasmBufferResizable: (wasm.views.bytes().buffer as ArrayBuffer & { resizable?: boolean }).resizable ?? false },
      settings: { width: WIDTH, height: HEIGHT, aaSamples: AA, iterations: ITERATIONS, referenceBits: 576,
        flightWidth, flightHeight,
        warmups: WARMUPS, measuredRuns: RUNS,
        cpuTiming: 'performance.now; no scheduler yields inside compute', gpuTiming: 'EXT_disjoint_timer_query_webgl2; null when unavailable',
        pixelFormat: 'RGBA8', canvasAttached: false },
      memoryBytesStable, checks: { exactReferenceArrays, exactGpuPixels, fullScreenPixelsEqual: exactGpuPixels,
        nontrivialGpuImages, ringChecksPassed, temporalChecksPassed, contextRestorePassed, workerChecksPassed, rawCancellationChecksPassed,
        passed: memoryBytesStable && exactReferenceArrays && exactGpuPixels && nontrivialGpuImages && ringChecksPassed
          && temporalChecksPassed && contextRestorePassed && workerChecksPassed && rawCancellationChecksPassed },
      cpu, worker, cancellationProbe, gpu, rings, temporal, animationFps, contextRestore,
    };
  } finally { renderers.js.dispose(); renderers.wasm.dispose(); }
}
