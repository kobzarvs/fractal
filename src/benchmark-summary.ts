/** Browser report fields used by the visible two-engine comparison. */
export interface BenchmarkSummaryInput {
  settings: { width: number; height: number; flightWidth: number; flightHeight: number;
    aaSamples: number; iterations: number; referenceBits: number };
  worker: { variants: Array<{ backend: 'js' | 'wasm'; wall: { medianMs: number };
    computation: { medianMs: number }; allArraysEqual: boolean }> };
  checks: { exactReferenceArrays: boolean; exactGpuPixels: boolean };
  gpu: Array<{ zoomPower10: number | null; jsVsWasmPixels: { equal: boolean; mismatchedPixels: number };
    gpuTime: { js: { medianMs: number } | null; wasm: { medianMs: number } | null };
    gpuTimerSupported: { js: boolean; wasm: boolean } }>;
  animationFps?: { supported: boolean; reason?: string | null;
    scenes: Array<{ startZoomPower10: number;
      js: { medianFps: number; runs: number; ringActiveFraction: number; cpuFrame?: { medianMs: number; p95Ms: number; meanMs?: number; totalMs?: number } };
      wasm: { medianFps: number; runs: number; ringActiveFraction: number; cpuFrame?: { medianMs: number; p95Ms: number; meanMs?: number; totalMs?: number } } }> };
}

const decimal = new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const oneDecimal = new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const integer = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });
const valid = (value: number): boolean => Number.isFinite(value) && value >= 0;
const ms = (value: number | undefined): string => value !== undefined && valid(value) ? `${decimal.format(value)} мс` : '—';
const count = (value: number): string => valid(value) ? integer.format(value) : '—';
const frameRate = (value: number): string => valid(value) ? `${oneDecimal.format(value)} кадр/с` : '—';
const percent = (value: number): string => valid(value) && value <= 1 ? `${oneDecimal.format(value * 100)}%` : '—';
const ratio = (numerator: number | undefined, denominator: number | undefined): string =>
  numerator !== undefined && denominator !== undefined && valid(numerator)
    && valid(denominator) && denominator > 0 && Number.isFinite(numerator / denominator)
    ? `${decimal.format(numerator / denominator)}×` : '—';

/** Static HTML plus formatted numbers only; report text is never injected. */
export function renderBenchmarkSummary(report: BenchmarkSummaryInput): string {
  const js = report.worker.variants.find(item => item.backend === 'js');
  const wasm = report.worker.variants.find(item => item.backend === 'wasm');
  const referenceEqual = report.checks.exactReferenceArrays && !!js?.allArraysEqual && !!wasm?.allArraysEqual;
  const pixelsEqual = report.gpu.length > 0 && report.checks.exactGpuPixels
    && report.gpu.every(item => item.jsVsWasmPixels.equal);
  const deep = report.gpu.filter(item => item.zoomPower10 !== null)
    .sort((a, b) => (b.zoomPower10 ?? 0) - (a.zoomPower10 ?? 0))[0];
  const jsGpu = deep?.gpuTime.js?.medianMs, wasmGpu = deep?.gpuTime.wasm?.medianMs;
  const gpuRatio = deep?.jsVsWasmPixels.equal ? ratio(jsGpu, wasmGpu) : '—';
  const gpuNote = deep?.gpuTimerSupported.js && deep.gpuTimerSupported.wasm
    ? 'Аппаратное время GPU; ожидание query в замер не входит.'
    : 'GPU-таймер недоступен хотя бы для одного движка.';
  const fpsRows = report.animationFps?.supported
    ? report.animationFps.scenes.map(scene => `<tr><th>10<sup>${count(scene.startZoomPower10)}</sup></th>`
      + `<td>${frameRate(scene.js.medianFps)}</td><td>${frameRate(scene.wasm.medianFps)}</td>`
      + `<td>${percent(scene.js.ringActiveFraction)} / ${percent(scene.wasm.ringActiveFraction)}</td></tr>`).join('')
    : '';
  const cpuFrameRows = report.animationFps?.supported
    ? report.animationFps.scenes.filter(scene => scene.js.cpuFrame && scene.wasm.cpuFrame)
      .map(scene => `<tr><th>10<sup>${count(scene.startZoomPower10)}</sup></th>`
        + `<td>${ms(scene.js.cpuFrame?.meanMs)} / ${ms(scene.js.cpuFrame?.medianMs)} / ${ms(scene.js.cpuFrame?.p95Ms)}</td>`
        + `<td>${ms(scene.wasm.cpuFrame?.meanMs)} / ${ms(scene.wasm.cpuFrame?.medianMs)} / ${ms(scene.wasm.cpuFrame?.p95Ms)}</td>`
        + `<td>${ratio(scene.js.cpuFrame?.meanMs, scene.wasm.cpuFrame?.meanMs)}</td></tr>`).join('')
    : '';
  const cpuFrameTable = cpuFrameRows
    ? `<table><caption>CPU на кадр: среднее / медиана / p95</caption><thead><tr><th>Глубина</th><th>JS original</th>`
      + `<th>WASM</th><th>JS/WASM</th></tr></thead><tbody>${cpuFrameRows}</tbody></table>`
      + `<p>Среднее — сумма CPU-времени, делённая на число кадров. Медиана 0 означает время ниже точности таймера, а не бесплатный кадр. Включены камера, планирование кэша и вызовы WebGL. Ожидание GPU и rAF в CPU-время не входит. `
      + `В WASM замеряется полный Rust render runtime; движки запускаются последовательно в порядке JS–WASM–WASM–JS.</p>`
    : '';
  const fpsTable = fpsRows
    ? `<p>Полёт: ${count(report.settings.flightWidth)}×${count(report.settings.flightHeight)}, AA ${count(report.settings.aaSamples)}; `
      + `две серии на движок, скорость камеры 0,55 log10/с.</p>`
      + `<table><caption>Отправка кадров</caption><thead><tr><th>Глубина</th><th>JS original</th>`
      + `<th>WASM</th><th>Кэш активен JS / WASM</th></tr></thead><tbody>${fpsRows}</tbody></table>`
      + '<p>Завершение GPU и показ на экране этим тестом не измеряются.</p>'
    : '<p>Частота отправки кадров в полёте не измерена.</p>';

  return `<section class="benchmark-summary" aria-label="Сравнение движков">
    <h3>JS original + GPU и WASM SIMD + GPU</h3>
    <p>Опорная орбита Western armada: ${count(report.settings.referenceBits)} бит, ${count(report.settings.iterations)} итераций.</p>
    <table><caption>Расчёт в worker, медиана</caption><thead><tr><th>Движок</th><th>Расчёт</th><th>Полный ответ</th></tr></thead>
      <tbody><tr><th>Исходный JS</th><td>${ms(js?.computation.medianMs)}</td><td>${ms(js?.wall.medianMs)}</td></tr>
      <tr><th>WASM SIMD</th><td>${ms(wasm?.computation.medianMs)}</td><td>${ms(wasm?.wall.medianMs)}</td></tr></tbody></table>
    <p>Отношение JS/WASM: расчёт ${ratio(js?.computation.medianMs, wasm?.computation.medianMs)}, `
      + `полный ответ ${ratio(js?.wall.medianMs, wasm?.wall.medianMs)}; больше 1 означает ускорение WASM.</p>
    <p>Опорные массивы ${referenceEqual ? 'совпадают побайтово' : 'различаются'}.</p>
    <table><caption>Полноэкранный GPU, ${count(report.settings.width)}×${count(report.settings.height)}, `
      + `AA ${count(report.settings.aaSamples)}, guided выключен, глубина 10<sup>${count(deep?.zoomPower10 ?? 0)}</sup></caption>
      <thead><tr><th>Renderer</th><th>Медиана GPU</th></tr></thead><tbody>
      <tr><th>JS original</th><td>${ms(jsGpu)}</td></tr><tr><th>WASM optimized</th><td>${ms(wasmGpu)}</td></tr></tbody></table>
    <p>${gpuNote} Пиксели полноэкранных кадров ${pixelsEqual ? 'совпадают' : 'различаются'}; `
      + `отношение GPU JS/WASM ${gpuRatio}.</p>
    ${fpsTable}
    ${cpuFrameTable}
    <p>Частота отправки кадров ограничена расписанием rAF; расчёт опорной орбиты во время полёта не повторяется. `
      + `В полёте исходный JS и WASM по-разному сглаживают кольцевой кэш: JS применяет исходную `
      + `адаптивную плотность, WASM — сегментированный кэш плотности 1,5. Этот замер сравнивает `
      + `режимы как есть; одинаковые параметры качества и пиксели проверяются в полноэкранном GPU-тесте выше.</p>
  </section>`;
}
