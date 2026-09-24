import './style.css';
import { makeCamera, decimalToFixed, fixedToNumber, precisionBits, setZoom, zoomAt, pan, cameraOffset, screenOffset } from './camera.ts';
import type { Camera } from './camera.ts';
import type { RenderView } from './types.ts';
import { TOURS, WESTERN_ARMADA } from './tours.ts';
import { ReferenceClient } from './compute/client.ts';
import type { Backend } from './compute/protocol.ts';
import { FractalRenderer } from './gpu/renderer.ts';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <canvas id="fractal" aria-label="Интерактивный фрактал Burning Ship"></canvas>
  <header class="brand"><span class="mark">∿</span><div><h1>Burning Ship</h1><p>RUST / WASM + GPU</p></div></header>
  <section class="controls" aria-label="Настройки фрактала">
    <div class="eyebrow">ЭКСПЕДИЦИЯ</div>
    <label>Маршрут<select id="route">${TOURS.map(t => `<option value="${t.id}">${t.name}</option>`).join('')}</select></label>
    <div class="buttons"><button id="play" class="primary">Начать полёт</button><button id="reset" title="Весь фрактал">Обзор</button></div>
    <label class="zoom-label">Увеличение <output id="zoom-value">10⁰</output><input id="zoom" type="range" min="0" max="120" step="0.05" value="0"></label>
    <div class="grid"><label>Итерации<select id="iterations"><option>1024</option><option>4096</option><option>8192</option><option selected>16384</option><option>32768</option><option>65536</option></select></label>
    <label>Сглаживание<select id="aa"><option value="1">1 сэмпл</option><option value="2" selected>2 сэмпла</option><option value="3">3 сэмпла</option><option value="4">4 сэмпла</option><option value="5">5 сэмплов</option></select></label></div>
    <label>Движок<select id="backend"><option value="wasm">WASM SIMD + GPU</option><option value="js">JavaScript + GPU</option></select></label>
    <p class="hint">Колесо — масштаб · перетаскивание — перемещение</p>
    <button id="benchmark" class="secondary">Проверить точность и скорость</button>
  </section>
  <footer class="telemetry"><span id="status" role="status">Загрузка…</span><span id="metrics"></span></footer>
  <dialog id="report"><div class="report-head"><h2>Точность и производительность</h2><button id="close-report" aria-label="Закрыть отчёт">×</button></div><p id="bench-status" role="status"></p><pre id="bench-output"></pre><button id="download-report" hidden>Скачать JSON</button></dialog>
`;
const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>('fractal'), status = el('status'), metrics = el('metrics');
const iterationsInput = el<HTMLSelectElement>('iterations'), routeInput = el<HTMLSelectElement>('route');
const zoomInput = el<HTMLInputElement>('zoom'), backendInput = el<HTMLSelectElement>('backend');
const aaInput = el<HTMLSelectElement>('aa'), playButton = el<HTMLButtonElement>('play');
const params = new URLSearchParams(location.search);
if (params.has('diagnostics')) {
  const recovery = document.createElement('button'); recovery.textContent = 'Проверить восстановление GPU'; recovery.className = 'secondary';
  recovery.onclick = () => {
    const extension = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
    if (!extension) { tell('WEBGL_lose_context недоступен'); return; }
    canvas.addEventListener('webglcontextlost', () => setTimeout(() => extension.restoreContext(), 100), { once: true });
    extension.loseContext();
  };
  document.querySelector('.controls')!.append(recovery);
}
let tour = TOURS.find(t => t.id === params.get('route')) ?? WESTERN_ARMADA;
routeInput.value = tour.id; zoomInput.max = String(tour.endZoom);
const requestedIterations = Number(params.get('iter'));
if ([1024, 4096, 8192, 16384, 32768, 65536].includes(requestedIterations)) iterationsInput.value = String(requestedIterations);
let camera = makeCamera(innerWidth / innerHeight), referenceCamera: Camera | null = null;
let renderer: FractalRenderer, client = new ReferenceClient();
let referenceKey = 0, requestVersion = 0, referencePending = false, dirty = true, playing = false;
let lost = false, benchmarking = false, lastFrame = performance.now(), lastMetrics = 0;
let referenceMs = 0, memoryBytes = 0, memoryMode = '', actualBackend = '', routeLocked = false, panTimer = 0;
const zoom = () => (Math.log2(3.2) - camera.logScale) / Math.log2(10);
function tell(message: string, error = false) { status.textContent = message; status.classList.toggle('error', error); }
function stop() { playing = false; playButton.textContent = 'Продолжить полёт'; }
function markDirty() { dirty = true; zoomInput.value = String(Math.max(0, zoom())); el('zoom-value').textContent = `10^${Math.max(0, zoom()).toFixed(1)}`; }
function chooseTour(targetZoom = 0) {
  stop(); routeLocked = true;
  const bits = precisionBits(Math.log2(3.2) - tour.endZoom * Math.log2(10));
  camera = { x: decimalToFixed(tour.x, bits), y: decimalToFixed(tour.y, bits), bits, logScale: Math.log2(3.2) };
  setZoom(camera, targetZoom); referenceCamera = null; markDirty(); void requestReference();
}
async function requestReference() {
  const version = ++requestVersion, snapshot = { ...camera }; referencePending = true;
  tell('Вычисление опорной орбиты…');
  try {
    const computation = await client.compute({ x: String(snapshot.x), y: String(snapshot.y), bits: snapshot.bits,
      iterations: Number(iterationsInput.value), fold: 1, celtic: 0 }, backendInput.value as Backend);
    if (version !== requestVersion || lost) { client.recycle(computation.result); return; }
    renderer.setReference(computation.result);
    referenceKey = computation.result.id; referenceCamera = snapshot; referenceMs = computation.result.computeMs;
    memoryBytes = computation.memoryBytes; memoryMode = computation.memoryMode;
    actualBackend = computation.result.backend === 'js' ? 'JS' : 'WASM SIMD';
    client.recycle(computation.result); tell('Готово'); markDirty();
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
  if (width !== canvas.width || height !== canvas.height) { canvas.width = width; canvas.height = height; markDirty(); }
}
function animate(now: number) {
  const elapsed = Math.min(.1, (now - lastFrame) / 1000); lastFrame = now;
  if (!lost && !benchmarking && !document.hidden) {
    if (playing && !referencePending) {
      setZoom(camera, Math.min(tour.endZoom, zoom() + elapsed * .55)); markDirty();
      if (zoom() >= tour.endZoom - 1e-6) stop();
    }
    if ((dirty || renderer.settling) && (camera.logScale >= -8 || (!referencePending && referenceCamera))) {
      try { renderer.render(view()); dirty = false; }
      catch (error) { stop(); dirty = false; tell(error instanceof Error ? error.message : String(error), true); }
    }
    if (now - lastMetrics > 250) {
      const gpu = renderer.gpuTimeMs;
      metrics.textContent = `${renderer.stats.path.toUpperCase()} · ${canvas.width}×${canvas.height} · ${actualBackend} ${referenceMs.toFixed(1)} мс · GPU ${gpu === null ? 'н/д' : gpu.toFixed(2) + ' мс'}${memoryBytes ? ` · ${(memoryBytes / 1048576).toFixed(1)} MiB / ${memoryMode}` : ''}`;
      lastMetrics = now;
    }
  }
  requestAnimationFrame(animate);
}
playButton.onclick = () => {
  if (playing) { stop(); markDirty(); return; }
  if (!routeLocked || zoom() >= tour.endZoom) chooseTour();
  playing = true; playButton.textContent = 'Пауза';
};
el('reset').onclick = () => { stop(); routeLocked = false; camera = makeCamera(innerWidth / innerHeight); referenceCamera = null; markDirty(); void requestReference(); };
routeInput.onchange = () => { tour = TOURS.find(t => t.id === routeInput.value)!; zoomInput.max = String(tour.endZoom); chooseTour(); };
zoomInput.oninput = () => { stop(); if (!routeLocked) chooseTour(Number(zoomInput.value)); else { setZoom(camera, Number(zoomInput.value)); markDirty(); } };
iterationsInput.onchange = () => { stop(); referenceCamera = null; void requestReference(); };
backendInput.onchange = () => { stop(); void requestReference(); };
aaInput.onchange = markDirty;
canvas.addEventListener('wheel', event => {
  event.preventDefault(); stop(); routeLocked = false;
  const rect = canvas.getBoundingClientRect();
  const [x, y] = screenOffset(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height);
  zoomAt(camera, x, y, Math.max(-2, Math.min(2, event.deltaY * .003)));
  markDirty(); window.clearTimeout(panTimer); panTimer = window.setTimeout(() => void requestReference(), 120);
}, { passive: false });
let drag: { x: number; y: number } | null = null;
canvas.onpointerdown = event => { if (event.button !== 0) return; stop(); routeLocked = false; drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); };
canvas.onpointermove = event => {
  if (!drag) return;
  pan(camera, (event.clientX - drag.x) / canvas.clientHeight, (event.clientY - drag.y) / canvas.clientHeight);
  drag = { x: event.clientX, y: event.clientY }; markDirty();
};
canvas.onpointerup = () => { if (drag) { drag = null; void requestReference(); } };
canvas.onpointercancel = () => { drag = null; void requestReference(); };
canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); lost = true; stop(); requestVersion++; client.cancel(); referencePending = false; tell('GPU-контекст потерян. Ожидание восстановления…', true); });
canvas.addEventListener('webglcontextrestored', () => {
  try { renderer.dispose(); renderer = new FractalRenderer(canvas); lost = false; referenceCamera = null; void requestReference(); }
  catch (error) { tell(String(error), true); }
});
addEventListener('resize', resize);
addEventListener('pagehide', () => { client.dispose(); renderer?.dispose(); }, { once: true });
addEventListener('pageshow', event => { if ((event as PageTransitionEvent).persisted) location.reload(); });
el('close-report').onclick = () => el<HTMLDialogElement>('report').close();
el('benchmark').onclick = async () => {
  if (benchmarking) return; stop(); benchmarking = true;
  const button = el<HTMLButtonElement>('benchmark'); button.disabled = true;
  el<HTMLDialogElement>('report').showModal(); el('bench-output').textContent = '';
  try {
    const { runBenchmark } = await import('./benchmark.ts');
    const report = await runBenchmark(message => { el('bench-status').textContent = message; });
    const serialized = JSON.stringify(report, null, 2); el('bench-output').textContent = serialized;
    el('bench-status').textContent = 'Проверка завершена';
    const download = el<HTMLButtonElement>('download-report'); download.hidden = false;
    download.onclick = () => {
      const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
      const a = document.createElement('a'); a.href = url; a.download = 'fractal-benchmark.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  } catch (error) { el('bench-status').textContent = 'Ошибка проверки'; el('bench-output').textContent = String(error); }
  finally { benchmarking = false; button.disabled = false; markDirty(); }
};
try {
  renderer = new FractalRenderer(canvas); resize();
  if (params.has('zoom')) chooseTour(Math.max(0, Math.min(tour.endZoom, Number(params.get('zoom')) || 0)));
  else void requestReference();
  requestAnimationFrame(animate);
} catch (error) { tell(error instanceof Error ? error.message : String(error), true); }
