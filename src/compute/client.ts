import type { ReferenceRequest, ReferenceResult } from '../types.ts';
import type { Backend, FromWorker, ToWorker } from './protocol.ts';
import { abortError, resultBuffers } from './protocol.ts';
export interface Computation { result: ReferenceResult; memoryBytes: number; memoryMode: string }
export class ReferenceClient {
  private worker: Worker;
  private sequence = 0;
  private active: { id: number; resolve: (value: Computation) => void; reject: (error: Error) => void } | null = null;
  constructor() { this.worker = this.createWorker(); }
  private createWorker(): Worker {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<FromWorker>) => {
      const message = event.data, id = message.type === 'result' ? message.result.id : message.id;
      if (!this.active || id !== this.active.id) {
        if (message.type === 'result') this.recycle(message.result);
        return;
      }
      const active = this.active; this.active = null;
      if (message.type === 'error') active.reject(new Error(message.error));
      else active.resolve(message);
    };
    worker.onerror = event => {
      this.active?.reject(new Error(event.message || 'Ошибка worker')); this.active = null;
      worker.terminate(); this.worker = this.createWorker();
    };
    return worker;
  }
  private post(message: ToWorker, transfers: Transferable[] = []) { this.worker.postMessage(message, transfers); }
  compute(request: Omit<ReferenceRequest, 'id'>, backend: Backend): Promise<Computation> {
    this.active?.reject(abortError());
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.active = { id, resolve, reject }; this.post({ type: 'compute', request: { ...request, id }, backend });
    });
  }
  recycle(result: ReferenceResult) { const buffers = resultBuffers(result); this.post({ type: 'recycle', buffers }, buffers); }
  cancel() { this.active?.reject(abortError()); this.active = null; this.post({ type: 'cancel' }); }
  dispose() { this.active?.reject(abortError()); this.active = null; this.worker.terminate(); }
}
