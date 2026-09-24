import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { WasmCore } from '../src/compute/wasm.ts';
import { computeReferenceJs } from '../src/compute/reference-js.ts';
import { referenceCases } from '../tests/fixtures.ts';

const noYield = async () => {};
const arrayNames = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'];
const runsOption = process.argv.find(arg => arg.startsWith('--runs='));
const runs = runsOption ? Number(runsOption.slice(7)) : 9;
if (!Number.isSafeInteger(runs) || runs < 9) throw new Error('--runs must be an integer >= 9');
const caseOption = process.argv.find(arg => arg.startsWith('--case='));
const cases = caseOption
  ? referenceCases.filter(item => item.name === caseOption.slice(7))
  : referenceCases;
if (cases.length === 0) throw new Error(`Unknown fixture: ${caseOption}`);

async function loadCore(variant) {
  const binary = await readFile(new URL(`../public/wasm/core-${variant}.wasm`, import.meta.url));
  const { instance } = await WebAssembly.instantiate(binary, {});
  return new WasmCore(instance, false);
}

function assertSame(actual, expected, label) {
  if (actual.length !== expected.length || actual.capacity !== expected.capacity) {
    throw new Error(`${label}: orbit length/capacity differs from JS`);
  }
  for (const name of arrayNames) {
    const a = actual[name];
    const b = expected[name];
    if (a.byteLength !== b.byteLength) throw new Error(`${label}: ${name} length differs from JS`);
    const aa = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const bb = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < aa.length; i++) {
      if (aa[i] !== bb[i]) throw new Error(`${label}: ${name} differs from JS at byte ${i}`);
    }
  }
}

function summary(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    runs: sorted.length,
    medianMs: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
    minMs: sorted[0],
    maxMs: sorted.at(-1),
  };
}

async function measure(run, memoryBytes) {
  // Two warmups let WASM reserve its final memory size before the measured runs.
  await run();
  await run();
  const stableBytes = memoryBytes?.();
  const samples = [];
  for (let i = 0; i < runs; i++) {
    const started = performance.now();
    await run();
    samples.push(performance.now() - started);
    if (memoryBytes && memoryBytes() !== stableBytes) {
      throw new Error(`WASM memory grew during a measured run (${stableBytes} -> ${memoryBytes()})`);
    }
  }
  return { ...summary(samples), ...(memoryBytes ? { stableMemoryBytes: stableBytes } : {}) };
}

async function main() {
  const scalar = await loadCore('scalar');
  const simd = await loadCore('simd');
  const output = {
    kind: 'cpu-reference',
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    warmups: 2,
    cases: [],
  };
  for (const { name, request } of cases) {
    const expected = await computeReferenceJs(request, () => false, noYield);
    const scalarResult = await scalar.compute(request, () => false, noYield);
    const simdResult = await simd.compute(request, () => false, noYield);
    assertSame(scalarResult, expected, `scalar/${name}`);
    assertSame(simdResult, expected, `simd/${name}`);
    const js = await measure(() => computeReferenceJs(request, () => false, noYield));
    const scalarTime = await measure(() => scalar.compute(request, () => false, noYield),
      () => scalar.memoryBytes);
    const simdTime = await measure(() => simd.compute(request, () => false, noYield),
      () => simd.memoryBytes);
    output.cases.push({ name, bits: request.bits, iterations: request.iterations,
      length: expected.length, js, scalar: scalarTime, simd: simdTime,
      scalarSpeedup: js.medianMs / scalarTime.medianMs,
      simdSpeedup: js.medianMs / simdTime.medianMs });
  }
  process.stdout.write(`${JSON.stringify(output)}\n`);
}

main().catch(error => {
  process.stdout.write(`${JSON.stringify({ error: String(error) })}\n`);
  process.exitCode = 1;
});
