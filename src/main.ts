import './style.css';
import { makeCamera, decimalToFixed, fixedToNumber, precisionBits, setZoom, zoomAt, pan, cameraOffset, screenOffset } from './camera.ts';
import type { Camera } from './camera.ts';
import type { RenderView } from './types.ts';
import { TOURS, WESTERN_ARMADA } from './tours.ts';
import { ReferenceClient } from './compute/client.ts';
import type { Backend } from './compute/protocol.ts';
import { createRenderer } from './gpu/engine.ts';
import type { RendererAdapter } from './gpu/engine.ts';
import { FrameRateMeter } from './frame-rate.ts';
import { WasmRuntimeClient } from './runtime/client.ts';
import type { CameraSnapshot, RuntimeSettings, RuntimeState } from './runtime/protocol.ts';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <canvas id="fractal" aria-label="Интерактивный фрактал Burning Ship"></canvas>
  <aside class="performance" aria-label="Частота кадров">
    <div class="fps-line"><span id="fps-value">—</span><span class="fps-unit">FPS</span></div>
    <div id="fps-engine">WASM SIMD + GPU</div><p id="fps-state">Загрузка…</p>
  </aside>
  <header class="brand"><span class="mark">∿</span><div><h1>Burning Ship</h1><p>ИНТЕРАКТИВНЫЙ ФРАКТАЛ</p></div></header>
  <section class="controls" aria-label="Настройки фрактала">
    <div class="eyebrow">ЭКСПЕДИЦИЯ</div>
    <label>Маршрут<select id="route">${TOURS.map(t => `<option value="${t.id}">${t.name}</option>`).join('')}</select></label>
    <div class="buttons"><button id="play" class="primary">Начать полёт</button><button id="reset" title="Весь фрактал">Обзор</button></div>
    <label class="zoom-label">Увеличение <output id="zoom-value">10⁰</output><input id="zoom" type="range" min="0" max="120" step="0.05" value="0"></label>
    <div class="grid"><label>Итерации<select id="iterations"><option>1024</option><option>4096</option><option>8192</option><option selected>16384</option><option>32768</option><option>65536</option></select></label>
    <label>Сглаживание<select id="aa"><option value="1">1 сэмпл</option><option value="2" selected>2 сэмпла</option><option value="3">3 сэмпла</option><option value="4">4 сэмпла</option><option value="5">5 сэмплов</option></select></label></div>
    <label>Движок<select id="backend"><option value="wasm">WASM SIMD + GPU</option><option value="js">JavaScript · исходные шейдеры</option></select></label>
    <p class="hint">Колесо — масштаб · перетаскивание — перемещение</p>
    <button id="benchmark" class="secondary">Проверить точность и скорость</button>
  </section>
  <footer class="telemetry"><span id="status" role="status">Загрузка…</span><span id="metrics"></span></footer>
  <dialog id="report"><div class="report-head"><h2>Точность и производительность</h2><button id="close-report" aria-label="Закрыть отчёт">×</button></div><p id="bench-status" role="status"></p><div id="bench-summary"></div><details id="bench-details"><summary>Подробные измерения (JSON)</summary><pre id="bench-output"></pre></details><button id="download-report" hidden>Скачать JSON</button></dialog>
`;
const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let canvas = el<HTMLCanvasElement>('fractal');
const status = el('status'), metrics = el('metrics');
let wasmRuntime: WasmRuntimeClient | null = null, engineVersion = 0;
let pixelWidth = 0, pixelHeight = 0;
const fpsValue = el('fps-value'), fpsEngine = el('fps-engine'), fpsState = el('fps-state');
const frameRate = new FrameRateMeter();
const iterationsInput = el<HTMLSelectElement>('iterations'), routeInput = el<HTMLSelectElement>('route');
const zoomInput = el<HTMLInputElement>('zoom'), backendInput = el<HTMLSelectElement>('backend');
const aaInput = el<HTMLSelectElement>('aa'), playButton = el<HTMLButtonElement>('play');
const params = new URLSearchParams(location.search);
if (params.has('diagnostics')) {
  const recovery = document.createElement('button'); recovery.textContent = 'Проверить восстановление GPU'; recovery.className = 'secondary';
  recovery.onclick = () => {
    if (wasmRuntime) { wasmRuntime.send({ type: 'recover-gpu' }); return; }
    const extension = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
    if (!extension) { tell('WEBGL_lose_context недоступен'); return; }
    canvas.addEventListener('webglcontextlost', () => setTimeout(() => extension.restoreContext(), 100), { once: true });
    extension.loseContext();
  };
  document.querySelector('.controls')!.append(recovery);
  const profile = document.createElement('button'); profile.textContent = 'Измерить глубокий полёт WASM'; profile.className = 'secondary';
  profile.id = 'zoom-profile'; profile.onclick = () => void profileZoom();
  document.querySelector('.controls')!.append(profile);
}
let tour = TOURS.find(t => t.id === params.get('route')) ?? WESTERN_ARMADA;
routeInput.value = tour.id; zoomInput.max = String(tour.endZoom);
const requestedIterations = Number(params.get('iter'));
if ([1024, 4096, 8192, 16384, 32768, 65536].includes(requestedIterations)) iterationsInput.value = String(requestedIterations);
let camera: Camera = { x: 0n, y: 0n, bits: 128, logScale: 0 }, referenceCamera: Camera | null = null;
let renderer: RendererAdapter, client: ReferenceClient | null = null;
let referenceKey = 0, requestVersion = 0, referencePending = false, dirty = true, playing = false;
let lost = false, rendererFailed = false, engineStarting = false, benchmarking = false, lastFrame = performance.now(), lastMetrics = 0;
let renderedSinceMetrics = false;
let referenceMs = 0, memoryBytes = 0, memoryMode = '', actualBackend = '', routeLocked = false, panTimer = 0;
const zoom = () => wasmRuntime ? wasmRuntime.state.zoom : (Math.log2(3.2) - camera.logScale) / Math.log2(10);
const runtimeSettings = (): RuntimeSettings => ({ iterations: Number(iterationsInput.value), aa: Number(aaInput.value), fold: 1, celtic: 0, hue: 0 });
function tell(message: string, error = false) { status.textContent = message; status.classList.toggle('error', error); }
function stop() { playing = false; playButton.textContent = 'Продолжить полёт'; wasmRuntime?.send({ type: 'play', enabled: false, endZoom: tour.endZoom }); }
function markDirty() { dirty = true; zoomInput.value = String(Math.max(0, zoom())); el('zoom-value').textContent = `10^${Math.max(0, zoom()).toFixed(1)}`; }
function chooseTour(targetZoom = 0) {
  stop(); routeLocked = true;
  if (wasmRuntime) { wasmRuntime.send({ type: 'route', route: tour, zoom: targetZoom }); return; }
  const bits = precisionBits(Math.log2(3.2) - tour.endZoom * Math.log2(10));
  camera = { x: decimalToFixed(tour.x, bits), y: decimalToFixed(tour.y, bits), bits, logScale: Math.log2(3.2) };
  setZoom(camera, targetZoom); referenceCamera = null; markDirty(); void requestReference();
}
async function requestReference() {
  if (wasmRuntime) return;
  const referenceClient = client ??= new ReferenceClient();
  frameRate.reset();
  const version = ++requestVersion, snapshot = { ...camera }; referencePending = true;
  tell('Вычисление опорной орбиты…');
  try {
    const computation = await referenceClient.compute({ x: String(snapshot.x), y: String(snapshot.y), bits: snapshot.bits,
      iterations: Number(iterationsInput.value), fold: 1, celtic: 0 }, backendInput.value as Backend);
    if (version !== requestVersion || lost) { referenceClient.recycle(computation.result); return; }
    renderer.setReference(computation.result);
    referenceKey = computation.result.id; referenceCamera = snapshot; referenceMs = computation.result.computeMs;
    memoryBytes = computation.memoryBytes; memoryMode = computation.memoryMode;
    actualBackend = computation.result.backend === 'js' ? 'JS' : 'WASM SIMD';
    referenceClient.recycle(computation.result); tell('Готово'); markDirty();
  } catch (error) {
    if (version === requestVersion && !(error instanceof Error && error.name === 'AbortError')) {
      stop(); tell(error instanceof Error ? error.message : String(error), true);
    }
  } finally { if (version === requestVersion) referencePending = false; }
}
function view(): RenderView {
  const offset = referenceCamera ? cameraOffset(camera, referenceCamera) : { x: [0, 0], y: [0, 0] };
  return { center: [fixedToNumber(camera.x, camera.bits), fixedToNumber(camera.y, camera.bits)],
    scale: 2 ** camera.logScale, logScale: camera.logScale, offsetX: offset.x as [number, number], offsetY: offset.y as [number, number],
    iterations: Number(iterationsInput.value), fold: 1, celtic: 0, aa: Number(aaInput.value), hue: 0,
    guided: playing && routeLocked,
    referenceKey, temporal: true, position: { x: camera.x, y: camera.y, bits: camera.bits } };
}
function resize() {
  const width = Math.max(1, Math.round(innerWidth * devicePixelRatio)), height = Math.max(1, Math.round(innerHeight * devicePixelRatio));
  if (width !== pixelWidth || height !== pixelHeight) {
    pixelWidth = width; pixelHeight = height;
    if (wasmRuntime) wasmRuntime.send({ type: 'resize', width, height });
    else { canvas.width = width; canvas.height = height; frameRate.reset(); markDirty(); }
  }
}
function updateFrameRate(now: number) {
  if (wasmRuntime) {
    const state = wasmRuntime.state;
    fpsValue.textContent = state.pending || state.preparing || state.lost || benchmarking || rendererFailed ? '—' : state.fps.toFixed(1).replace(/\.0$/, '');
    fpsEngine.textContent = 'WASM SIMD + GPU';
    fpsState.textContent = engineStarting ? 'Загрузка движка…' : rendererFailed ? 'Ошибка движка · можно переключить' : state.lost ? 'Восстановление GPU…' : state.pending ? 'Расчёт опорной орбиты…'
      : benchmarking ? 'Идёт сравнение движков…' : state.preparing ? 'Подготовка кэша…' : !state.playing && !state.fps ? 'Кадр готов · рендер приостановлен' : 'Кадры за последнюю секунду';
    return;
  }
  const suspended = lost || rendererFailed || benchmarking || document.hidden || referencePending;
  const active = renderedSinceMetrics || playing || drag !== null || dirty || renderer?.settling;
  const fps = suspended ? null : active ? frameRate.sample(now) : 0;
  fpsValue.textContent = fps === null ? '—' : fps.toFixed(1).replace(/\.0$/, '');
  fpsEngine.textContent = backendInput.value === 'wasm' ? 'WASM SIMD + GPU' : 'JS · исходные шейдеры';
  fpsState.textContent = engineStarting ? 'Загрузка движка…' : lost ? 'Восстановление GPU…' : rendererFailed ? 'Ошибка движка · можно переключить'
    : benchmarking ? 'Идёт сравнение движков…'
    : document.hidden ? 'Вкладка скрыта' : referencePending ? 'Расчёт опорной орбиты…'
    : !active ? 'Кадр готов · рендер приостановлен'
    : fps === null ? 'Измерение FPS…' : 'Кадры за последнюю секунду';
  if (suspended || !active) frameRate.reset();
  renderedSinceMetrics = false;
}
function animate(now: number) {
  const elapsed = Math.min(.1, (now - lastFrame) / 1000); lastFrame = now;
  if (!wasmRuntime && !lost && !rendererFailed && !benchmarking && !document.hidden) {
    if (playing && !referencePending) {
      setZoom(camera, Math.min(tour.endZoom, zoom() + elapsed * .55)); markDirty();
      if (zoom() >= tour.endZoom - 1e-6) stop();
    }
    if ((dirty || renderer.settling) && (camera.logScale >= -8 || (!referencePending && referenceCamera))) {
      try { renderer.render(view()); frameRate.record(now); renderedSinceMetrics = true; dirty = false; }
      catch (error) { stop(); dirty = false; tell(error instanceof Error ? error.message : String(error), true); }
    }
    if (now - lastMetrics > 250) {
      const gpu = renderer.gpuTimeMs;
      metrics.textContent = `${renderer.stats.path.toUpperCase()} · ${canvas.width}×${canvas.height} · Орбита ${actualBackend} ${referenceMs.toFixed(1)} мс · GPU ${gpu === null ? 'н/д' : gpu.toFixed(2) + ' мс'}${memoryBytes ? ` · ${(memoryBytes / 1048576).toFixed(1)} MiB / ${memoryMode}` : ''}${params.has('diagnostics') && playing ? ` · Кэш ${renderer.stats.ringActive ? 'активен' : 'не активен'}` : ''}`;
    }
  }
  if (now - lastMetrics > 250) { updateFrameRate(now); lastMetrics = now; }
  requestAnimationFrame(animate);
}
playButton.onclick = () => {
  if (playing) { stop(); markDirty(); return; }
  if (!routeLocked || zoom() >= tour.endZoom) chooseTour();
  frameRate.reset();
  playing = true; playButton.textContent = 'Пауза';
  wasmRuntime?.send({ type: 'play', enabled: true, endZoom: tour.endZoom });
};
el('reset').onclick = () => {
  stop(); routeLocked = false;
  if (wasmRuntime) { wasmRuntime.send({ type: 'overview', aspect: innerWidth / innerHeight }); return; }
  camera = makeCamera(innerWidth / innerHeight); referenceCamera = null; markDirty(); void requestReference();
};
routeInput.onchange = () => { tour = TOURS.find(t => t.id === routeInput.value)!; zoomInput.max = String(tour.endZoom); chooseTour(); };
zoomInput.oninput = () => {
  stop();
  if (!routeLocked) chooseTour(Number(zoomInput.value));
  else if (wasmRuntime) wasmRuntime.send({ type: 'zoom', zoom: Number(zoomInput.value) });
  else { setZoom(camera, Number(zoomInput.value)); markDirty(); }
};
iterationsInput.onchange = () => {
  stop();
  if (wasmRuntime) { wasmRuntime.send({ type: 'settings', settings: runtimeSettings() }); return; }
  referenceCamera = null; void requestReference();
};
backendInput.onchange = () => { void startEngine(); };
aaInput.onchange = () => { if (wasmRuntime) wasmRuntime.send({ type: 'settings', settings: runtimeSettings() }); else markDirty(); };
let drag: { x: number; y: number } | null = null;
function attachCanvasHandlers() {
const attachedCanvas = canvas, attachedVersion = engineVersion;
const currentCanvas = () => attachedCanvas === canvas && attachedVersion === engineVersion;
canvas.addEventListener('wheel', event => {
  event.preventDefault(); stop(); routeLocked = false;
  const rect = canvas.getBoundingClientRect();
  if (wasmRuntime) {
    wasmRuntime.send({ type: 'zoom-at', x: event.clientX - rect.left, y: event.clientY - rect.top,
      width: rect.width, height: rect.height, delta: event.deltaY }); return;
  }
  const [x, y] = screenOffset(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height);
  zoomAt(camera, x, y, Math.max(-2, Math.min(2, event.deltaY * .003)));
  markDirty(); window.clearTimeout(panTimer); panTimer = window.setTimeout(() => void requestReference(), 120);
}, { passive: false });
canvas.onpointerdown = event => {
  if (event.button !== 0) return; stop(); routeLocked = false; drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId);
  wasmRuntime?.send({ type: 'pointer', phase: 'start', x: event.clientX, y: event.clientY, height: canvas.clientHeight });
};
canvas.onpointermove = event => {
  if (!drag) return;
  if (wasmRuntime) { wasmRuntime.send({ type: 'pointer', phase: 'move', x: event.clientX, y: event.clientY, height: canvas.clientHeight }); return; }
  pan(camera, (event.clientX - drag.x) / canvas.clientHeight, (event.clientY - drag.y) / canvas.clientHeight);
  drag = { x: event.clientX, y: event.clientY }; markDirty();
};
canvas.onpointerup = event => { if (drag) {
  drag = null;
  if (wasmRuntime) wasmRuntime.send({ type: 'pointer', phase: 'end', x: event.clientX, y: event.clientY, height: canvas.clientHeight });
  else void requestReference();
} };
canvas.onpointercancel = event => { drag = null;
  if (wasmRuntime) wasmRuntime.send({ type: 'pointer', phase: 'end', x: event.clientX, y: event.clientY, height: canvas.clientHeight });
  else void requestReference();
};
canvas.addEventListener('webglcontextlost', event => { if (!currentCanvas() || wasmRuntime) return; event.preventDefault(); lost = true; stop(); requestVersion++; client?.cancel(); referencePending = false; tell('GPU-контекст потерян. Ожидание восстановления…', true); });
canvas.addEventListener('webglcontextrestored', () => {
  if (!currentCanvas() || wasmRuntime) return;
  lost = false;
  try { renderer.dispose(); renderer = createRenderer(canvas, backendInput.value as Backend); rendererFailed = false; referenceCamera = null; void requestReference(); }
  catch (error) { rendererFailed = true; tell(String(error), true); }
});
}
function acceptRuntimeState(state: RuntimeState) {
  rendererFailed = false;
  referencePending = state.pending; lost = state.lost; playing = state.playing || state.playRequested;
  playButton.textContent = playing ? 'Пауза' : 'Продолжить полёт';
  zoomInput.value = String(state.zoom); el('zoom-value').textContent = `10^${state.zoom.toFixed(1)}`;
  tell(state.lost ? 'GPU-контекст потерян. Ожидание восстановления…' : state.pending ? 'Вычисление опорной орбиты…' : state.preparing ? 'Подготовка кэша…' : 'Готово', state.lost);
  metrics.textContent = `${state.path.toUpperCase()} · ${pixelWidth}×${pixelHeight} · Орбита WASM SIMD ${state.referenceMs.toFixed(1)} мс · CPU кадра ${state.cpuFrameMs.toFixed(2)} мс · GPU ${state.gpuMs === null ? 'н/д' : state.gpuMs.toFixed(2) + ' мс'} · ${(state.memoryBytes / 1048576).toFixed(1)} MiB${params.has('diagnostics') && state.playing ? ` · Кэш ${state.ringActive ? 'активен' : 'не активен'}` : ''}`;
}
async function startEngine(initial = false) {
  const version = ++engineVersion;
  engineStarting = true;
  stop(); rendererFailed = true; referencePending = true; requestVersion++; client?.cancel();
  try {
    let snapshot: CameraSnapshot | undefined;
    if (!initial) {
      if (wasmRuntime) {
        try { snapshot = await wasmRuntime.snapshot(); }
        catch { routeLocked = false; /* A failed worker must not prevent switching back to original JS. */ }
      } else snapshot = { x: String(camera.x), y: String(camera.y), bits: camera.bits, logScale: camera.logScale };
    }
    if (version !== engineVersion) return;
    wasmRuntime?.dispose(); wasmRuntime = null; renderer?.dispose(); drag = null;
    if (!initial) {
      const replacement = document.createElement('canvas'); replacement.id = 'fractal';
      replacement.setAttribute('aria-label', 'Интерактивный фрактал Burning Ship');
      replacement.width = pixelWidth; replacement.height = pixelHeight;
      canvas.replaceWith(replacement); canvas = replacement;
    }
    attachCanvasHandlers(); lost = false; frameRate.reset(); referenceCamera = null;
    if (backendInput.value === 'wasm') {
      client?.dispose(); client = null;
      if (initial) routeLocked = params.has('zoom');
      const runtime = new WasmRuntimeClient(canvas, { width: pixelWidth, height: pixelHeight,
        aspect: innerWidth / innerHeight, settings: runtimeSettings(), snapshot,
        route: routeLocked ? tour : undefined, zoom: Number(params.get('zoom')) || 0 },
        state => { if (version === engineVersion) acceptRuntimeState(state); },
        message => { if (version === engineVersion) { rendererFailed = true; tell(message, true); } });
      wasmRuntime = runtime;
      await runtime.ready;
      if (version !== engineVersion) return;
      rendererFailed = false;
    } else {
      camera = snapshot ? { x: BigInt(snapshot.x), y: BigInt(snapshot.y), bits: snapshot.bits, logScale: snapshot.logScale }
        : makeCamera(innerWidth / innerHeight);
      renderer = createRenderer(canvas, 'js'); rendererFailed = false;
      markDirty(); void requestReference();
    }
  } catch (error) { if (version === engineVersion) { rendererFailed = true; referencePending = false; tell(error instanceof Error ? error.message : String(error), true); } }
  finally { if (version === engineVersion) engineStarting = false; }
}
addEventListener('resize', resize);
document.addEventListener('visibilitychange', () => { wasmRuntime?.send({ type: 'suspend', suspended: document.hidden || benchmarking }); frameRate.reset(); lastFrame = performance.now(); updateFrameRate(lastFrame); });
addEventListener('pagehide', () => { wasmRuntime?.dispose(); client?.dispose(); renderer?.dispose(); }, { once: true });
addEventListener('pageshow', event => { if ((event as PageTransitionEvent).persisted) location.reload(); });
el('close-report').onclick = () => el<HTMLDialogElement>('report').close();
function reportDownload(serialized: string) {
  const download = el<HTMLButtonElement>('download-report'); download.hidden = false;
  download.onclick = () => {
    const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = 'fractal-benchmark.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
}
async function profileZoom() {
  if (benchmarking) return;
  stop(); benchmarking = true; frameRate.reset(); updateFrameRate(performance.now());
  wasmRuntime?.send({ type: 'suspend', suspended: true });
  const button = el<HTMLButtonElement>('zoom-profile'); button.disabled = true;
  el<HTMLDialogElement>('report').showModal(); el('bench-output').textContent = ''; el('bench-summary').textContent = '';
  el<HTMLDetailsElement>('bench-details').open = true; el('download-report').hidden = true;
  try {
    const { runZoomProfile } = await import('./zoom-profile.ts');
    const report = await runZoomProfile({ width: pixelWidth, height: pixelHeight,
      aa: Number(aaInput.value), iterations: Number(iterationsInput.value) }, message => { el('bench-status').textContent = message; });
    const serialized = JSON.stringify(report, null, 2); el('bench-output').textContent = serialized;
    el('bench-status').textContent = 'Измерение глубокого полёта WASM завершено'; reportDownload(serialized);
  } catch (error) { el('bench-status').textContent = 'Ошибка измерения'; el('bench-output').textContent = String(error); }
  finally { benchmarking = false; button.disabled = false; wasmRuntime?.send({ type: 'suspend', suspended: document.hidden }); if (!wasmRuntime) markDirty(); }
}
el('benchmark').onclick = async () => {
  if (benchmarking) return; stop(); benchmarking = true;
  wasmRuntime?.send({ type: 'suspend', suspended: true });
  frameRate.reset(); updateFrameRate(performance.now());
  const button = el<HTMLButtonElement>('benchmark'); button.disabled = true;
  el<HTMLDialogElement>('report').showModal(); el('bench-output').textContent = ''; el('bench-summary').textContent = '';
  el<HTMLDetailsElement>('bench-details').open = false; el('download-report').hidden = true;
  try {
    const { runBenchmark } = await import('./benchmark.ts');
    const { renderBenchmarkSummary } = await import('./benchmark-summary.ts');
    const report = await runBenchmark(message => { el('bench-status').textContent = message; },
      { flightWidth: pixelWidth, flightHeight: pixelHeight });
    const serialized = JSON.stringify(report, null, 2); el('bench-output').textContent = serialized;
    el('bench-summary').innerHTML = renderBenchmarkSummary(report);
    el('bench-status').textContent = report.checks.passed ? 'Проверка завершена · все проверки пройдены' : 'Проверка завершена · обнаружены расхождения';
    reportDownload(serialized);
  } catch (error) { el('bench-status').textContent = 'Ошибка проверки'; el('bench-output').textContent = String(error); el<HTMLDetailsElement>('bench-details').open = true; }
  finally { benchmarking = false; button.disabled = false; wasmRuntime?.send({ type: 'suspend', suspended: document.hidden }); if (!wasmRuntime) markDirty(); }
};
try {
  rendererFailed = true; resize(); void startEngine(true);
  requestAnimationFrame(animate);
} catch (error) { tell(error instanceof Error ? error.message : String(error), true); }
