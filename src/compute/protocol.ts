import type { ReferenceRequest, ReferenceResult } from '../types.ts';
export type Backend = 'auto' | 'wasm' | 'simd' | 'js';
export type ToWorker = { type: 'compute'; request: ReferenceRequest; backend: Backend }
  | { type: 'cancel' } | { type: 'recycle'; buffers: ArrayBuffer[] };
export type FromWorker = { type: 'result'; result: ReferenceResult; memoryBytes: number; memoryMode: string; variant: 'scalar' | 'simd' | 'js' }
  | { type: 'error'; id: number; error: string };
export class RequestEpoch {
  private epoch = 0;
  next(): number { return ++this.epoch; }
  cancel(): void { ++this.epoch; }
  isCurrent(epoch: number): boolean { return epoch === this.epoch; }
}
export function abortError(): Error { const error = new Error('Расчёт отменён'); error.name = 'AbortError'; return error; }
export const resultBuffers = (r: ReferenceResult): ArrayBuffer[] => [r.orbit, r.realOrbit, r.blaA, r.blaB, r.blaBounds].map(a => a.buffer as ArrayBuffer);
