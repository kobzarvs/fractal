/// <reference lib="webworker" />
import { computeReferenceJs } from './reference-js.ts';
import { loadWasm } from './wasm.ts';
import type { WasmCore } from './wasm.ts';
import { RequestEpoch, resultBuffers } from './protocol.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const epoch = new RequestEpoch();
let corePromise: Promise<WasmCore> | undefined;
const pool: ArrayBuffer[] = [];
let pending: Extract<ToWorker, { type: 'compute' }> | null = null, draining = false;
// MessageChannel yields a task without nested setTimeout's 4 ms clamp. Check the
// budget after each bounded kernel chunk so cancellation stays responsive.
const channel = new MessageChannel();
let resume: (() => void) | undefined, sliceStarted = performance.now();
channel.port1.onmessage = () => { const callback = resume; resume = undefined; sliceStarted = performance.now(); callback?.(); };
const yieldTask = () => new Promise<void>(resolve => { resume = resolve; channel.port2.postMessage(null); });
const yieldControl = () => performance.now() - sliceStarted < 4 ? Promise.resolve() : yieldTask();
function send(message: FromWorker, transfer: Transferable[] = []) { scope.postMessage(message, transfer); }
function acquire(bytes: number): ArrayBuffer {
  const i = pool.findIndex(buffer => buffer.byteLength === bytes);
  return i < 0 ? new ArrayBuffer(bytes) : pool.splice(i, 1)[0];
}
async function getCore(): Promise<WasmCore> {
  corePromise ??= loadWasm();
  try { return await corePromise; } catch (error) { corePromise = undefined; throw error; }
}
async function drain() {
  if (draining) return;
  draining = true;
  try {
    while (pending) {
      const job = pending; pending = null;
      sliceStarted = performance.now();
      const current = epoch.next(), cancelled = () => !epoch.isCurrent(current);
      try {
        const core = job.backend === 'wasm' ? await getCore() : undefined;
        const result = core
          ? await core.compute(job.request, cancelled, yieldControl, acquire)
          : await computeReferenceJs(job.request, cancelled, yieldControl);
        // A replacement can arrive during the final slice. Give its message a
        // task boundary before publishing, even when computation finished fast.
        await yieldTask();
        if (!cancelled()) send({ type: 'result', result, memoryBytes: core?.memoryBytes ?? 0, memoryMode: core?.views.mode ?? 'JS BigInt' }, resultBuffers(result));
      } catch (error) {
        if (!cancelled() && !(error instanceof Error && error.name === 'AbortError')) send({ type: 'error', id: job.request.id, error: error instanceof Error ? error.message : String(error) });
      }
    }
  } finally { draining = false; }
}
scope.onmessage = (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  if (message.type === 'recycle') {
    for (const buffer of message.buffers) if (buffer.byteLength && pool.length < 10) pool.push(buffer);
    return;
  }
  epoch.cancel();
  if (message.type === 'cancel') pending = null;
  else { pending = message; void drain(); }
};
