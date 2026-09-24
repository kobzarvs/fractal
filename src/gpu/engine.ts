import type { ReferenceResult, RenderView } from '../types';
import { WasmRenderer } from './wasm-renderer';
export { preloadRenderWasm } from './render-wasm';
import type { GpuTiming, RendererStats } from './renderer';
import { LegacyRenderer } from './legacy/renderer';

export type RendererBackend = 'js' | 'wasm';

/** Shared controls and measurements for the two independently owned renderers. */
export interface RendererAdapter {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  readonly stats: RendererStats;
  readonly settling: boolean;
  readonly gpuTimerSupported: boolean;
  readonly gpuTimeMs: number | null;
  readonly gpuTimings: readonly GpuTiming[];
  setReference(reference: ReferenceResult): void;
  render(view: RenderView): void;
  readPixels(): Uint8Array;
  resetTemporal(): void;
  pollGpuTimers(): void;
  clearGpuTimings(): void;
  dispose(): void;
}

export function createRenderer(canvas: HTMLCanvasElement | OffscreenCanvas, backend: RendererBackend): RendererAdapter {
  if (backend === 'js') return new LegacyRenderer(canvas);
  if (backend === 'wasm') return new WasmRenderer(canvas);
  throw new Error(`Неизвестный движок рендеринга: ${String(backend)}.`);
}
