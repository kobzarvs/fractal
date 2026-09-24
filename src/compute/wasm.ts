import type { ReferenceRequest, ReferenceResult } from '../types.ts';
import { abortError } from './protocol.ts';
import { computeOnlyImports } from './wasm-imports.ts';

export class MemoryViews {
  readonly memory: WebAssembly.Memory;
  readonly mode: 'resizable' | 'fixed';
  private readonly bytesView: Uint8Array;
  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;
    this.mode = 'fixed';
    this.bytesView = new Uint8Array(memory.buffer);
  }
  bytes(): Uint8Array { return this.bytesView; }
  floats(pointer: number, length: number): Float32Array { return new Float32Array(this.bytes().buffer, pointer, length); }
}
interface CoreExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  input_ptr: () => number; input_capacity: () => number;
  begin: (x: number, y: number, bits: number, iterations: number, fold: number, celtic: number) => number;
  step: (steps: number) => number; result_ptr: (kind: number) => number; result_len: (kind: number) => number;
  orbit_length: () => number; capacity: () => number; last_error: () => number;
}
export class WasmCore {
  readonly views: MemoryViews;
  private core: CoreExports;
  private busy = false;
  constructor(instance: WebAssembly.Instance) {
    this.core = instance.exports as CoreExports; this.views = new MemoryViews(this.core.memory);
  }
  get memoryBytes(): number { return this.views.bytes().byteLength; }
  async compute(request: ReferenceRequest, cancelled = () => false, yieldControl = async () => {}, acquire?: (bytes: number) => ArrayBuffer): Promise<ReferenceResult> {
    if (!request || !Number.isSafeInteger(request.id)
      || typeof request.x !== 'string' || !/^[+-]?\d+$/.test(request.x)
      || typeof request.y !== 'string' || !/^[+-]?\d+$/.test(request.y)
      || !Number.isSafeInteger(request.bits) || request.bits < 1 || request.bits > 4096
      || !Number.isSafeInteger(request.iterations) || request.iterations < 1 || request.iterations > 65536
      || !Number.isFinite(request.fold) || request.fold < 0 || request.fold > 1
      || !Number.isFinite(request.celtic) || request.celtic < 0 || request.celtic > 1) throw new Error('Invalid reference request');
    if (this.busy) throw new Error('WASM instance is already computing');
    this.busy = true;
    try {
      if (cancelled()) throw abortError();
      const started = performance.now(), x = new TextEncoder().encode(request.x), y = new TextEncoder().encode(request.y);
      const c = this.core;
      if (x.length + y.length > c.input_capacity()) throw new Error('Координаты слишком длинные');
      const pointer = c.input_ptr(), input = this.views.bytes();
      input.set(x, pointer); input.set(y, pointer + x.length);
      if (c.begin(x.length, y.length, request.bits, request.iterations, Math.round(request.fold * 1024), Math.round(request.celtic * 1024)) < 0) throw new Error(`WASM begin: ${c.last_error()}`);
      let status = 0;
      while (status === 0) {
        if (cancelled()) throw abortError();
        status = c.step(128);
        if (status < 0) throw new Error(`WASM step: ${c.last_error()}`);
        if (status === 0) await yieldControl();
      }
      if (cancelled()) throw abortError();
      const arrays = Array.from({ length: 5 }, (_, kind) => {
        const source = this.views.floats(c.result_ptr(kind), c.result_len(kind));
        const output = new Float32Array(acquire ? acquire(source.byteLength) : new ArrayBuffer(source.byteLength));
        output.set(source); return output;
      });
      return { ...request, fold: Math.round(request.fold * 1024) / 1024, celtic: Math.round(request.celtic * 1024) / 1024,
        length: c.orbit_length(), capacity: c.capacity(), orbit: arrays[0], realOrbit: arrays[1],
        blaA: arrays[2], blaB: arrays[3], blaBounds: arrays[4], computeMs: performance.now() - started, backend: 'wasm' };
    } finally { this.busy = false; }
  }
}
export async function loadWasm(): Promise<WasmCore> {
  const response = await fetch(`${import.meta.env.BASE_URL}wasm/core-simd.wasm`);
  if (!response.ok) throw new Error(`WASM: HTTP ${response.status}. Выполните npm run build:wasm.`);
  // Streaming compilation avoids buffering when the server sends application/wasm.
  const fallback = response.clone();
  let instance: WebAssembly.Instance;
  try { ({ instance } = await WebAssembly.instantiateStreaming(response, computeOnlyImports())); }
  catch {
    try { ({ instance } = await WebAssembly.instantiate(await fallback.arrayBuffer(), computeOnlyImports())); }
    catch (error) {
      if (error instanceof WebAssembly.CompileError) throw new Error('Не удалось загрузить WASM SIMD. Выберите JavaScript или пересоберите ядро.');
      throw error;
    }
  }
  return new WasmCore(instance);
}
