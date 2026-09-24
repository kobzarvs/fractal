import { decimalToFixed } from './camera.ts';
import { computeReferenceJs } from './compute/reference-js.ts';
import { ReferenceClient } from './compute/client.ts';
import { probeCancellation } from './compute/cancellation-probe.ts';
import { loadWasm } from './compute/wasm.ts';
import type { WasmCore } from './compute/wasm.ts';
import { FractalRenderer } from './gpu/renderer.ts';
import { WESTERN_ARMADA } from './tours.ts';
import type { ReferenceRequest, ReferenceResult, RenderView } from './types.ts';

const WARMUPS = 2, RUNS = 9, WIDTH = 320, HEIGHT = 200, AA = 2, ITERATIONS = 16384;
const ARRAY_NAMES = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
const noYield = async () => {};
const pause = (milliseconds = 0) => new Promise<void>(resolve => setTimeout(resolve, milliseconds));
const summary = (samples: number[]) => {
  const sorted = [...samples].sort((a, b) => a - b);
  return { runs: samples.length, medianMs: sorted[Math.floor(sorted.length / 2)],
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

async function cpuCase(name: string, request: ReferenceRequest, cores: { scalar: WasmCore; simd: WasmCore }, progress: (message: string) => void) {
  const runners = {
    js: () => computeReferenceJs(request, () => false, noYield),
    scalar: () => cores.scalar.compute(request, () => false, noYield),
    simd: () => cores.simd.compute(request, () => false, noYield),
  };
  const samples = { js: [] as number[], scalar: [] as number[], simd: [] as number[] };
  const latest: Partial<Record<keyof typeof runners, ReferenceResult>> = {};
  for (let round = 0; round < WARMUPS; round++) for (const variant of ['js', 'scalar', 'simd'] as const) {
    progress(`CPU ${name}: прогрев ${round + 1}/${WARMUPS}, ${variant}`);
    await pause(); latest[variant] = await runners[variant]();
  }
  const memoryBefore = { scalar: cores.scalar.memoryBytes, simd: cores.simd.memoryBytes };
  const memoryStable = { scalar: true, simd: true };
  // No timer/rAF/scheduler yields occur inside either numerical kernel. The
  // event-loop yield and byte comparisons below are outside each timed sample.
  for (let round = 0; round < RUNS; round++) {
    const variants = round % 2 ? ['simd', 'scalar', 'js'] as const : ['js', 'scalar', 'simd'] as const;
    for (const variant of variants) {
      progress(`CPU ${name}: ${round + 1}/${RUNS}, ${variant}`); await pause();
      const started = performance.now();
      latest[variant] = await runners[variant]();
      samples[variant].push(performance.now() - started);
      memoryStable.scalar &&= cores.scalar.memoryBytes === memoryBefore.scalar;
      memoryStable.simd &&= cores.simd.memoryBytes === memoryBefore.simd;
    }
  }
  const js = latest.js!, scalar = latest.scalar!, simd = latest.simd!;
  const jsTime = summary(samples.js), scalarTime = summary(samples.scalar), simdTime = summary(samples.simd);
  const report = {
    name, bits: request.bits, iterations: request.iterations, length: js.length, capacity: js.capacity,
    js: jsTime, scalar: { ...scalarTime, memoryMode: cores.scalar.views.mode, memoryBytes: cores.scalar.memoryBytes, memoryBytesStable: memoryStable.scalar },
    simd: { ...simdTime, memoryMode: cores.simd.views.mode, memoryBytes: cores.simd.memoryBytes, memoryBytesStable: memoryStable.simd },
    scalarSpeedupVsJs: jsTime.medianMs / scalarTime.medianMs,
    simdSpeedupVsJs: jsTime.medianMs / simdTime.medianMs,
    simdSpeedupVsScalar: scalarTime.medianMs / simdTime.medianMs,
    exactArrays: { scalar: byteDiff(scalar, js), simd: byteDiff(simd, js) },
  };
  return { report, js, wasm: scalar };
}

async function workerCase(expected: ReferenceResult, local: { scalar: { medianMs: number }; simd: { medianMs: number } },
  progress: (message: string) => void) {
  const client = new ReferenceClient();
  const request = referenceRequest(7100, WESTERN_ARMADA.x, WESTERN_ARMADA.y, 576, ITERATIONS);
  const variants = [];
  try {
    for (const backend of ['wasm', 'simd'] as const) {
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
      const localMedian = backend === 'wasm' ? local.scalar.medianMs : local.simd.medianMs;
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

function makeView(zoom: number | null, optimized: boolean, referenceKey: number): RenderView {
  const logScale = Math.log2(3.2) - (zoom ?? 0) * Math.log2(10);
  return {
    center: zoom === null ? [-.45, -.45] : [Number(WESTERN_ARMADA.x), Number(WESTERN_ARMADA.y)],
    scale: 2 ** logScale, logScale, offsetX: [0, 0], offsetY: [0, 0], iterations: ITERATIONS,
    fold: 1, celtic: 0, aa: AA, hue: 0, optimized, guided: false, referenceKey,
  };
}

function harness() {
  const canvas = document.createElement('canvas');
  canvas.width = WIDTH; canvas.height = HEIGHT;
  return { canvas, renderer: new FractalRenderer(canvas) };
}

async function gpuQuery(renderer: FractalRenderer, frame: number): Promise<number | null> {
  if (!renderer.gpuTimerSupported) { await pause(); return null; }
  const deadline = performance.now() + 5000;
  do {
    await pause(8);
    const sample = renderer.gpuTimings.find(item => item.frame === frame);
    if (sample) return sample.milliseconds;
  } while (performance.now() < deadline);
  return null; // Disjoint/timed-out queries cannot serve as GPU measurements.
}

async function gpuCase(renderer: FractalRenderer, zoom: number | null, js: ReferenceResult, wasm: ReferenceResult,
  progress: (message: string) => void) {
  const name = zoom === null ? 'overview' : `western-armada-10^${zoom}`;
  renderer.setReference(wasm);
  for (let round = 0; round < WARMUPS; round++) for (const optimized of [false, true]) {
    progress(`GPU ${name}: прогрев ${round + 1}/${WARMUPS}, ${optimized ? 'optimized' : 'baseline'}`);
    renderer.render(makeView(zoom, optimized, wasm.id));
    renderer.readPixels(); // Finish warmup/compilation before beginning the measured set.
    await pause();
  }
  renderer.clearGpuTimings();
  const samples = { baseline: [] as number[], optimized: [] as number[] };
  for (let round = 0; round < RUNS; round++) for (const optimized of round % 2 ? [true, false] : [false, true]) {
    const variant = optimized ? 'optimized' : 'baseline';
    progress(`GPU ${name}: ${round + 1}/${RUNS}, ${variant}`);
    renderer.render(makeView(zoom, optimized, wasm.id));
    const elapsed = await gpuQuery(renderer, renderer.stats.frame);
    if (elapsed !== null) samples[variant].push(elapsed);
  }
  renderer.render(makeView(zoom, false, wasm.id)); const baselinePixels = renderer.readPixels();
  renderer.render(makeView(zoom, true, wasm.id)); const optimizedPixels = renderer.readPixels();
  renderer.setReference(js);
  renderer.render(makeView(zoom, true, js.id)); const jsPixels = renderer.readPixels();
  const baselineVsOptimized = pixelDiff(baselinePixels, optimizedPixels);
  const jsVsWasmPixels = pixelDiff(jsPixels, optimizedPixels);
  const content = imageContent(optimizedPixels);
  const performanceComparisonAccepted = baselineVsOptimized.equal && jsVsWasmPixels.equal && content.nontrivialRGB;
  const baselineGpu = samples.baseline.length === RUNS ? summary(samples.baseline) : null;
  const optimizedGpu = samples.optimized.length === RUNS ? summary(samples.optimized) : null;
  await pause();
  return {
    name, zoomPower10: zoom, path: renderer.stats.path, logScale: makeView(zoom, true, wasm.id).logScale,
    baselineVsOptimized, jsVsWasmPixels, content, performanceComparisonAccepted, gpuTimerSupported: renderer.gpuTimerSupported,
    baselineGpu, optimizedGpu, gpuSamplesReceived: { baseline: samples.baseline.length, optimized: samples.optimized.length },
    gpuSpeedup: performanceComparisonAccepted && baselineGpu && optimizedGpu ? baselineGpu.medianMs / optimizedGpu.medianMs : null,
  };
}

async function ringCase(renderer: FractalRenderer, reference: ReferenceResult, progress: (message: string) => void) {
  renderer.setReference(reference);
  const images: Uint8Array[] = [];
  const variants = [];
  for (const optimized of [false, true]) {
    const view = { ...makeView(30, optimized, reference.id), guided: true };
    const initialRows = renderer.stats.ringsDrawn;
    let frames = 0;
    do {
      progress(`Кольцевой кэш: ${optimized ? 'optimized' : 'baseline'}, заполнение ${frames + 1}`);
      renderer.render(view); frames++;
      await pause();
      if (frames >= 128 && !renderer.stats.ringActive) throw new Error('Кольцевой кэш не заполнился за 128 кадров при 320×200.');
    } while (!renderer.stats.ringActive);
    images.push(renderer.readPixels());
    const fullRows = renderer.stats.ringsDrawn - initialRows;
    const beforeReuse = renderer.stats.ringsDrawn;
    renderer.render(view);
    const repeatedViewNewRows = renderer.stats.ringsDrawn - beforeReuse;
    const beforeZoom = renderer.stats.ringsDrawn, samplesBeforeZoom = renderer.stats.ringSamples;
    const zoomed = { ...view, logScale: view.logScale - .02, scale: view.scale * 2 ** -.02 };
    renderer.render(zoomed);
    variants.push({ optimized, initialFillFrames: frames, initialRows: fullRows,
      repeatedViewNewRows, zeroWorkOnRepeatedView: repeatedViewNewRows === 0,
      zoomNewRows: renderer.stats.ringsDrawn - beforeZoom,
      zoomNewSamples: renderer.stats.ringSamples - samplesBeforeZoom,
      zoomReusedCache: renderer.stats.ringActive,
    });
    await pause();
  }
  return { enabled: true, zoomPower10: 30, baselineVsOptimized: pixelDiff(images[0], images[1]),
    content: imageContent(images[1]), variants };
}

type TemporalRenderer = FractalRenderer & { readonly settling: boolean; resetTemporal(): void };
function temporalView(reference: ReferenceResult, optimized: boolean): RenderView {
  return { ...makeView(30, optimized, reference.id), aa: 3, temporal: true,
    position: { x: decimalToFixed(WESTERN_ARMADA.x, reference.bits), y: decimalToFixed(WESTERN_ARMADA.y, reference.bits), bits: reference.bits } };
}
async function settleTemporal(renderer: FractalRenderer, view: RenderView) {
  const temporal = renderer as TemporalRenderer;
  if (typeof temporal.resetTemporal !== 'function') throw new Error('Temporal AA: renderer.resetTemporal ещё недоступен.');
  temporal.resetTemporal();
  let frames = 0;
  do {
    temporal.render(view); frames++; await pause();
    if (frames > 32) throw new Error('Temporal AA не завершил накопление за 32 неподвижных кадра.');
  } while (temporal.settling);
  return frames;
}
async function temporalCase(renderer: FractalRenderer, reference: ReferenceResult, progress: (message: string) => void) {
  if (!('resetTemporal' in renderer)) return { supported: false, reason: 'Temporal API pending', enabled: false, aaSamples: 3,
    zoomPower10: 30, baselineFrames: null, optimizedFrames: null, resetFrames: null, settledAtRequestedSamples: false,
    baselineVsOptimized: null, repeatAfterReset: null, content: null };
  progress('Temporal AA: сравнение полностью накопленных кадров');
  renderer.setReference(reference);
  const baselineFrames = await settleTemporal(renderer, temporalView(reference, false));
  const baseline = renderer.readPixels();
  const optimizedFrames = await settleTemporal(renderer, temporalView(reference, true));
  const optimized = renderer.readPixels();
  const resetFrames = await settleTemporal(renderer, temporalView(reference, true));
  return { supported: true, reason: null, enabled: true, aaSamples: 3, zoomPower10: 30, baselineFrames, optimizedFrames, resetFrames,
    settledAtRequestedSamples: baselineFrames === 3 && optimizedFrames === 3 && resetFrames === 3,
    baselineVsOptimized: pixelDiff(baseline, optimized),
    repeatAfterReset: pixelDiff(optimized, renderer.readPixels()), content: imageContent(optimized) };
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

async function contextRestoreCase(reference: ReferenceResult, progress: (message: string) => void) {
  const test = harness(); let renderer = test.renderer;
  const gl = test.canvas.getContext('webgl2')!;
  const extension = gl.getExtension('WEBGL_lose_context');
  try {
    if (!extension) return { supported: false, restored: null, pixels: null, temporalSupported: false, temporalPixels: null, temporalFrames: null, reason: 'WEBGL_lose_context недоступен' };
    progress('GPU: проверка потери и восстановления контекста');
    renderer.setReference(reference);
    const view = makeView(10, true, reference.id);
    renderer.render(view); const before = renderer.readPixels();
    const temporalSupported = 'resetTemporal' in renderer;
    const beforeFrames = temporalSupported ? await settleTemporal(renderer, temporalView(reference, true)) : null;
    const beforeTemporal = temporalSupported ? renderer.readPixels() : null;
    const lost = contextEvent(test.canvas, 'webglcontextlost');
    extension.loseContext(); await lost;
    const restored = contextEvent(test.canvas, 'webglcontextrestored');
    await pause(50); extension.restoreContext(); await restored;
    // Match the app lifecycle: retire stale GPU handles after restoration.
    renderer.dispose();
    renderer = new FractalRenderer(test.canvas);
    renderer.setReference(reference); renderer.render(view);
    const pixels = pixelDiff(before, renderer.readPixels());
    const afterFrames = temporalSupported ? await settleTemporal(renderer, temporalView(reference, true)) : null;
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
export async function runBenchmark(progress: (message: string) => void) {
  progress('Загрузка scalar/SIMD WASM для проверки…'); await pause();
  const [scalar, simd] = await Promise.all([loadWasm('scalar'), loadWasm('simd')]);
  const cores = { scalar, simd };
  const deep = await cpuCase('western-armada-576-16384', referenceRequest(7001, WESTERN_ARMADA.x, WESTERN_ARMADA.y, 576, ITERATIONS), cores, progress);
  const interior = await cpuCase('interior-origin-128-1024', referenceRequest(7002, '0', '0', 128, 1024), cores, progress);
  const worker = await workerCase(deep.js, deep.report, progress);
  progress('Raw Worker: проверка отмены до начала GPU измерений…');
  await pause();
  const cancellationProbe = await probeCancellation();
  const { canvas, renderer } = harness();
  const gl = canvas.getContext('webgl2')!;
  const debugRenderer = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuDevice = {
    vendor: String(gl.getParameter(debugRenderer ? debugRenderer.UNMASKED_VENDOR_WEBGL : gl.VENDOR)),
    renderer: String(gl.getParameter(debugRenderer ? debugRenderer.UNMASKED_RENDERER_WEBGL : gl.RENDERER)),
    identification: debugRenderer ? 'unmasked' : 'masked',
  };
  try {
    const gpu = [];
    for (const zoom of [null, 10, 30, 90, 120]) gpu.push(await gpuCase(renderer, zoom, deep.js, deep.wasm, progress));
    const rings = await ringCase(renderer, deep.wasm, progress);
    const temporal = await temporalCase(renderer, deep.wasm, progress);
    const contextRestore = await contextRestoreCase(deep.wasm, progress);
    const cpu = [deep.report, interior.report];
    const memoryBytesStable = cpu.every(item => item.scalar.memoryBytesStable && item.simd.memoryBytesStable)
      && worker.variants.every(item => item.memoryBytesStable) && worker.cancellation.memoryAfterRaceStable;
    const workerChecksPassed = worker.variants.every(item => item.allArraysEqual)
      && worker.cancellation.firstRejectedWithAbortError && worker.cancellation.replacementArrays.equal && worker.cancellation.recycledRunsExact;
    const rawCancellationChecksPassed = cancellationProbe.checks.passed;
    const exactReferenceArrays = cpu.every(item => item.exactArrays.scalar.equal && item.exactArrays.simd.equal);
    const exactGpuPixels = gpu.every(item => item.baselineVsOptimized.equal && item.jsVsWasmPixels.equal);
    const nontrivialGpuImages = gpu.every(item => item.content.nontrivialRGB);
    const ringChecksPassed = rings.baselineVsOptimized.equal && rings.content.nontrivialRGB
      && rings.variants.every(item => item.zeroWorkOnRepeatedView && item.zoomNewRows > 0 && item.zoomReusedCache);
    const temporalChecksPassed = temporal.supported && temporal.settledAtRequestedSamples && temporal.baselineVsOptimized?.equal === true
      && temporal.repeatAfterReset?.equal === true && temporal.content?.nontrivialRGB === true;
    const contextRestorePassed = !contextRestore.supported || (contextRestore.restored === true && contextRestore.pixels?.equal === true
      && (!contextRestore.temporalSupported || (contextRestore.temporalPixels?.equal === true
        && contextRestore.temporalFrames?.before === 3 && contextRestore.temporalFrames?.after === 3)));
    progress('Проверки завершены');
    return {
      kind: 'burning-ship-browser-benchmark', createdAt: new Date().toISOString(), userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency, devicePixelRatio, gpuDevice,
      resizableMemory: { scalar: scalar.views.mode === 'resizable', simd: simd.views.mode === 'resizable',
        scalarBufferResizable: (scalar.views.bytes().buffer as ArrayBuffer & { resizable?: boolean }).resizable ?? false,
        simdBufferResizable: (simd.views.bytes().buffer as ArrayBuffer & { resizable?: boolean }).resizable ?? false },
      settings: { width: WIDTH, height: HEIGHT, aaSamples: AA, iterations: ITERATIONS, referenceBits: 576,
        warmups: WARMUPS, measuredRuns: RUNS, gpuVariantsAlternated: true,
        cpuTiming: 'performance.now; no scheduler yields inside compute', gpuTiming: 'EXT_disjoint_timer_query_webgl2; null when unavailable',
        pixelFormat: 'RGBA8', canvasAttached: false },
      memoryBytesStable, checks: { exactReferenceArrays, exactGpuPixels, nontrivialGpuImages, ringChecksPassed, temporalChecksPassed, contextRestorePassed, workerChecksPassed, rawCancellationChecksPassed,
        passed: memoryBytesStable && exactReferenceArrays && exactGpuPixels && nontrivialGpuImages && ringChecksPassed && temporalChecksPassed && contextRestorePassed && workerChecksPassed && rawCancellationChecksPassed },
      cpu, worker, cancellationProbe, gpu, rings, temporal, contextRestore,
    };
  } finally { renderer.dispose(); }
}
