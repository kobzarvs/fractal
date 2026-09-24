import type { CameraSnapshot, RuntimeCommand, RuntimeRequest, RuntimeResponse, RuntimeState } from './protocol.ts';

/** UI transport only. Camera and renderer calculations live in the WASM worker. */
export class WasmRuntimeClient {
  readonly ready: Promise<void>;
  state: RuntimeState = { ready: false, pending: true, lost: false, playing: false,
    zoom: 0, logScale: 0, bits: 128, path: 'direct', fps: 0, cpuFrameMs: 0, gpuMs: null,
    referenceMs: 0, memoryBytes: 0, frame: 0, ringActive: false, drawCalls: 0, uploadBytes: 0 };
  private readonly worker: Worker;
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private sequence = 0;
  private disposed = false;
  private failure: Error | null = null;
  private readonly pendingSnapshots = new Map<number, { resolve: (snapshot: CameraSnapshot) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();

  constructor(canvas: HTMLCanvasElement,
    initial: Omit<Extract<RuntimeRequest, { type: 'init' }>, 'type' | 'canvas'>,
    onState: (state: RuntimeState) => void, onError: (message: string) => void) {
    if (typeof canvas.transferControlToOffscreen !== 'function') {
      throw new Error('WASM-рендерер требует OffscreenCanvas. В этом браузере доступен исходный JS-режим.');
    }
    this.ready = new Promise((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject; });
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<RuntimeResponse>) => {
      if (this.disposed) return;
      const message = event.data;
      if (message.type === 'state') {
        this.failure = null;
        this.state = message.state;
        if (message.state.ready) this.resolveReady();
        onState(message.state);
      } else if (message.type === 'snapshot') {
        const pending = this.pendingSnapshots.get(message.id);
        if (pending) { clearTimeout(pending.timer); pending.resolve(message.snapshot); }
        this.pendingSnapshots.delete(message.id);
      } else {
        const error = new Error(message.message);
        this.rejectPending(error); onError(message.message);
      }
    };
    this.worker.onerror = event => {
      const message = event.message || 'Ошибка WASM render worker';
      this.rejectPending(new Error(message)); onError(message);
    };
    try {
      const offscreen = canvas.transferControlToOffscreen();
      this.worker.postMessage({ ...initial, type: 'init', canvas: offscreen } satisfies RuntimeRequest, [offscreen]);
    } catch (error) { this.worker.terminate(); throw error; }
  }

  send(command: RuntimeCommand): void { if (!this.disposed) this.worker.postMessage(command); }

  snapshot(): Promise<CameraSnapshot> {
    if (this.disposed) return Promise.reject(new Error('WASM render worker завершён.'));
    if (this.failure) return Promise.reject(this.failure);
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingSnapshots.delete(id); reject(new Error('WASM worker не ответил на запрос камеры.'));
      }, 10000);
      this.pendingSnapshots.set(id, { resolve, reject, timer });
      this.worker.postMessage({ type: 'snapshot', id } satisfies RuntimeRequest);
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.worker.terminate();
    this.rejectPending(new Error('WASM render worker завершён.'));
  }

  private rejectPending(error: Error): void {
    this.failure = error; this.rejectReady(error);
    for (const pending of this.pendingSnapshots.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pendingSnapshots.clear();
  }
}
