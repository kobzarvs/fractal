export interface ReferenceRequest {
  id: number; x: string; y: string; bits: number; iterations: number;
  fold: number; celtic: number;
}
export interface ReferenceResult {
  id: number; length: number; capacity: number; iterations: number; bits: number;
  fold: number; celtic: number;
  orbit: Float32Array; realOrbit: Float32Array;
  blaA: Float32Array; blaB: Float32Array; blaBounds: Float32Array;
  computeMs: number; backend: 'wasm' | 'js';
}
export interface RenderView {
  center: [number, number]; scale: number; logScale: number;
  offsetX: [number, number]; offsetY: [number, number];
  iterations: number; fold: number; celtic: number; aa: number; hue: number;
  optimized: boolean; guided: boolean; referenceKey: number;
  temporal?: boolean;
  position?: { x: bigint; y: bigint; bits: number };
}
export interface Tour { id: string; name: string; x: string; y: string; endZoom: number; iterations: number }
