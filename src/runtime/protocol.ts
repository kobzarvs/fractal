export interface RuntimeSettings { iterations: number; aa: number; fold: number; celtic: number; hue: number }
export interface RuntimeRoute { x: string; y: string; endZoom: number }
export interface CameraSnapshot { x: string; y: string; bits: number; logScale: number }
export interface RuntimeState {
  ready: boolean; pending: boolean; lost: boolean; playing: boolean;
  preparing: boolean; playRequested: boolean;
  zoom: number; logScale: number; bits: number; path: string;
  fps: number; cpuFrameMs: number; gpuMs: number | null;
  referenceMs: number; memoryBytes: number; frame: number;
  ringActive: boolean; drawCalls: number; uploadBytes: number;
}
export type RuntimeCommand =
  | { type: 'route'; route: RuntimeRoute; zoom: number }
  | { type: 'overview'; aspect: number }
  | { type: 'zoom'; zoom: number }
  | { type: 'zoom-at'; x: number; y: number; width: number; height: number; delta: number }
  | { type: 'pointer'; x: number; y: number; height: number; phase: 'start' | 'move' | 'end' }
  | { type: 'play'; enabled: boolean; endZoom: number }
  | { type: 'resize'; width: number; height: number }
  | { type: 'settings'; settings: RuntimeSettings }
  | { type: 'suspend'; suspended: boolean }
  | { type: 'recover-gpu' };
export type RuntimeRequest = RuntimeCommand
  | { type: 'init'; canvas: OffscreenCanvas; width: number; height: number; aspect: number;
      settings: RuntimeSettings; route?: RuntimeRoute; zoom?: number; snapshot?: CameraSnapshot }
  | { type: 'snapshot'; id: number };
export type RuntimeResponse =
  | { type: 'state'; state: RuntimeState }
  | { type: 'snapshot'; id: number; snapshot: CameraSnapshot }
  | { type: 'error'; message: string };
