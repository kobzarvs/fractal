import assert from 'node:assert/strict';
import test from 'node:test';
import { renderBenchmarkSummary } from '../src/benchmark-summary.ts';
import type { BenchmarkSummaryInput } from '../src/benchmark-summary.ts';

const report: BenchmarkSummaryInput = {
  settings: { width: 320, height: 200, flightWidth: 1334, flightHeight: 1240,
    aaSamples: 2, iterations: 16384, referenceBits: 576 },
  worker: { variants: [
    { backend: 'wasm', wall: { medianMs: 32 }, computation: { medianMs: 20 }, allArraysEqual: true },
    { backend: 'js', wall: { medianMs: 40 }, computation: { medianMs: 30 }, allArraysEqual: true },
  ] },
  checks: { exactReferenceArrays: true, exactGpuPixels: true },
  gpu: [{ name: '<script>alert(1)</script>', zoomPower10: 120,
    jsVsWasmPixels: { equal: true, mismatchedPixels: 0 },
    gpuTime: { js: { medianMs: 8.4 }, wasm: { medianMs: 4.2 } },
    gpuTimerSupported: { js: true, wasm: true } }],
  animationFps: { supported: true, scenes: [
    { startZoomPower10: 30, js: { medianFps: 59.8, runs: 2, ringActiveFraction: 1, cpuFrame: { medianMs: .42, p95Ms: .73, meanMs: .48, totalMs: 48 } },
      wasm: { medianFps: 59.9, runs: 2, ringActiveFraction: .5, cpuFrame: { medianMs: .21, p95Ms: .34, meanMs: .24, totalMs: 24 } } },
    { startZoomPower10: 115, js: { medianFps: 31, runs: 2, ringActiveFraction: .9 },
      wasm: { medianFps: 42, runs: 2, ringActiveFraction: .8 } },
  ] },
};

test('summary compares worker computation and full response for the two engines', () => {
  const html = renderBenchmarkSummary(report);
  assert.match(html, /JS original \+ GPU/);
  assert.match(html, /WASM SIMD \+ GPU/);
  assert.match(html, /30,00 мс/);
  assert.match(html, /20,00 мс/);
  assert.match(html, /40,00 мс/);
  assert.match(html, /32,00 мс/);
  assert.match(html, /1,50×/);
  assert.match(html, /1,25×/);
  assert.match(html, /Пиксели полноэкранных кадров совпадают/);
  assert.match(html, /8,40 мс/);
  assert.match(html, /4,20 мс/);
  assert.match(html, /1334×1240|1 334×1 240/);
  assert.match(html, /59,8 FPS/);
  assert.match(html, /CPU на кадр: среднее \/ медиана \/ p95/);
  assert.match(html, /0,48 мс \/ 0,42 мс \/ 0,73 мс/);
  assert.match(html, /0,24 мс \/ 0,21 мс \/ 0,34 мс/);
  assert.match(html, /полный Rust render runtime/);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /по-разному сглаживают кольцевой кэш/);
});

test('summary states missing GPU timing and visible pixel differences', () => {
  const html = renderBenchmarkSummary({ ...report,
    checks: { exactReferenceArrays: true, exactGpuPixels: false },
    gpu: [{ zoomPower10: 120, jsVsWasmPixels: { equal: false, mismatchedPixels: 10 },
      gpuTime: { js: null, wasm: null }, gpuTimerSupported: { js: false, wasm: false } }],
    animationFps: { supported: false, reason: 'rAF is throttled', scenes: [] },
  });
  assert.match(html, /Пиксели полноэкранных кадров различаются/);
  assert.match(html, /GPU-таймер недоступен/i);
  assert.match(html, /FPS полёта не измерен/i);
});

test('CPU comparison uses the nonzero aggregate mean when timer quantization makes the median zero', () => {
  const scene = { startZoomPower10: 30,
    js: { medianFps: 60, runs: 2, ringActiveFraction: 1, cpuFrame: { meanMs: .3, medianMs: .3, p95Ms: .5 } },
    wasm: { medianFps: 60, runs: 2, ringActiveFraction: 1, cpuFrame: { meanMs: .06, medianMs: 0, p95Ms: .1 } } };
  const html = renderBenchmarkSummary({ ...report, animationFps: { supported: true, scenes: [scene] } });
  assert.match(html, /0,06 мс \/ 0,00 мс \/ 0,10 мс/);
  assert.match(html, /5,00×/);
  assert.match(html, /Медиана 0 означает время ниже точности таймера/);
});
