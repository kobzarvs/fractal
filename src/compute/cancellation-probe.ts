import type { ReferenceRequest, ReferenceResult } from '../types.ts';
import { computeReferenceJs } from './reference-js.ts';
import { resultBuffers } from './protocol.ts';
import type { Backend, FromWorker, ToWorker } from './protocol.ts';

const SIGNAL_DELAY_MS = 5;
const FULL_ITERATIONS = 4096;
const SHORT_ITERATIONS = 8;
const BITS = 4096;
const NEGATIVE_ONE_FIXED = String(-(1n << BigInt(BITS)));
const WATCHDOG_MS = 30000; // A hung-worker safeguard, never an acceptance threshold.
const ARRAYS = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
type ResultMessage = Extract<FromWorker, { type: 'result' }>;
interface Receipt { id: number; type: FromWorker['type']; receivedMs: number; computeMs: number | null }
interface Arrival { message: ResultMessage; receivedAt: number }

function origin(id: number, iterations: number): ReferenceRequest {
  return { id, x: '0', y: '0', bits: BITS, iterations, fold: 1, celtic: 0 };
}

function exactOrigin(actual: ReferenceResult, expected: ReferenceResult): boolean {
  if (actual.length !== expected.length || actual.capacity !== expected.capacity || actual.bits !== expected.bits
      || actual.iterations !== expected.iterations || actual.fold !== expected.fold || actual.celtic !== expected.celtic) return false;
  return ARRAYS.every(name => {
    const a = new Uint8Array(actual[name].buffer, actual[name].byteOffset, actual[name].byteLength);
    const b = new Uint8Array(expected[name].buffer, expected[name].byteOffset, expected[name].byteLength);
    return a.length === b.length && a.every((value, index) => value === b[index]);
  });
}

function timings(samples: { wallMs: number; computeMs: number }[]) {
  const walls = samples.map(sample => sample.wallMs).sort((a, b) => a - b);
  return { samples, minWallMs: walls[0], medianWallMs: walls[Math.floor(walls.length / 2)], maxWallMs: walls[walls.length - 1] };
}

/** Raw production-worker test: unlike ReferenceClient, no client-side id filter
 * can hide a stale publication. Run in a foreground browser with no competing
 * benchmark; the measured signal delay and full-job times expose inconclusive
 * trials when the main thread was too busy to send cancellation in time. */
export async function probeCancellation() {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  const started = performance.now(), receipts: Receipt[] = [];
  const waiting = new Map<number, { resolve: (value: Arrival) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  let sequence = 9000, failure: Error | null = null;
  const post = (message: ToWorker, transfers: Transferable[] = []) => worker.postMessage(message, transfers);
  const recycle = (result: ReferenceResult) => { const buffers = resultBuffers(result); post({ type: 'recycle', buffers }, buffers); };
  const fail = (error: Error) => {
    failure = error;
    for (const pending of waiting.values()) { clearTimeout(pending.timer); pending.reject(error); }
    waiting.clear();
  };
  worker.onerror = event => fail(new Error(event.message || 'Cancellation probe: worker failed'));
  worker.onmessageerror = () => fail(new Error('Cancellation probe: worker message could not be decoded'));
  worker.onmessage = (event: MessageEvent<FromWorker>) => {
    const message = event.data, receivedAt = performance.now();
    const id = message.type === 'result' ? message.result.id : message.id;
    receipts.push({ id, type: message.type, receivedMs: receivedAt - started, computeMs: message.type === 'result' ? message.result.computeMs : null });
    if (message.type === 'error') { fail(new Error(`Cancellation probe request ${id}: ${message.error}`)); return; }
    const pending = waiting.get(id);
    if (!pending) { recycle(message.result); return; }
    clearTimeout(pending.timer); waiting.delete(id); pending.resolve({ message, receivedAt });
  };
  const waitFor = (id: number): Promise<Arrival> => new Promise((resolve, reject) => {
    if (failure) { reject(failure); return; }
    const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Cancellation probe request ${id} timed out`)); }, WATCHDOG_MS);
    waiting.set(id, { resolve, reject, timer });
  });
  const compute = (id: number, iterations: number, backend: Backend) => {
    const request = origin(id, iterations);
    // (-1, -1) has a bounded nonzero orbit: unlike the origin, it exercises
    // 4096-bit multiplication long enough for the 5 ms signal to arrive early.
    if (iterations === FULL_ITERATIONS) request.x = request.y = NEGATIVE_ONE_FIXED;
    post({ type: 'compute', request, backend });
  };
  const measure = async (iterations: number, backend: Backend) => {
    const id = ++sequence, arrival = waitFor(id), sentAt = performance.now();
    compute(id, iterations, backend);
    const { message, receivedAt } = await arrival;
    const sample = { wallMs: receivedAt - sentAt, computeMs: message.result.computeMs };
    recycle(message.result); return sample;
  };

  try {
    const expected = await computeReferenceJs(origin(0, SHORT_ITERATIONS), () => false, async () => {});
    const variants = [];
    // Both implementations share the same cooperative scheduler. WASM is the
    // application default; JS also reproduces starvation independently of WASM.
    for (const backend of ['wasm', 'js'] as const) {
      await measure(FULL_ITERATIONS, backend); // Load module and allocate final buffers before timing.
      const fullSamples = [], shortSamples = [];
      for (let run = 0; run < 3; run++) {
        fullSamples.push(await measure(FULL_ITERATIONS, backend));
        shortSamples.push(await measure(SHORT_ITERATIONS, backend));
      }
      const fullBaseline = timings(fullSamples), shortBaseline = timings(shortSamples);
      const trial = async (cancel: boolean) => {
        const obsoleteId = ++sequence, latestId = ++sequence, firstReceipt = receipts.length;
        const latestArrival = waitFor(latestId), sentAt = performance.now();
        compute(obsoleteId, FULL_ITERATIONS, backend);
        await new Promise<void>(resolve => setTimeout(resolve, SIGNAL_DELAY_MS));
        const signalledAt = performance.now();
        if (cancel) post({ type: 'cancel' });
        // For explicit cancellation the following short request is a FIFO health
        // barrier. It ensures the cancelled job cannot publish later unseen.
        compute(latestId, SHORT_ITERATIONS, backend);
        const { message, receivedAt } = await latestArrival;
        const finalHealthExact = exactOrigin(message.result, expected);
        recycle(message.result);
        const received = receipts.slice(firstReceipt);
        const old = received.filter(receipt => receipt.id === obsoleteId && receipt.type === 'result');
        const signalDelayMs = signalledAt - sentAt, wallAfterSignalMs = receivedAt - signalledAt;
        const oldResultBeforeSignal = old.some(receipt => receipt.receivedMs <= signalledAt - started);
        const timingConclusive = !oldResultBeforeSignal && signalDelayMs < fullBaseline.minWallMs;
        const estimatedWithoutCancellationMs = Math.max(0, fullBaseline.medianWallMs - signalDelayMs) + shortBaseline.medianWallMs;
        const measuredFullJobBoundMs = fullBaseline.maxWallMs + shortBaseline.maxWallMs;
        return { obsoleteId, latestId, signal: cancel ? 'cancel-then-health-request' : 'replacement-request',
          requestedSignalDelayMs: SIGNAL_DELAY_MS, signalDelayMs, received,
          receivedResultIds: received.filter(receipt => receipt.type === 'result').map(receipt => receipt.id),
          obsoleteResultPublished: old.length > 0, oldResultBeforeSignal, timingConclusive,
          obsoleteResultSuppressed: old.length === 0, finalHealthExact, wallAfterSignalMs,
          wallAfterSignalOverFullBaseline: wallAfterSignalMs / fullBaseline.medianWallMs,
          estimatedWithoutCancellationMs, fasterThanEstimatedUncancelledWork: wallAfterSignalMs < estimatedWithoutCancellationMs,
          measuredFullJobBoundMs, withinMeasuredFullJobBound: wallAfterSignalMs <= measuredFullJobBoundMs };
      };
      variants.push({ backend, fullBaseline, shortBaseline, replacement: await trial(false), cancellation: await trial(true) });
    }
    const trials = variants.flatMap(variant => [variant.replacement, variant.cancellation]);
    const checks = { timingConclusive: trials.every(trial => trial.timingConclusive),
      obsoleteResultsSuppressed: trials.every(trial => trial.obsoleteResultSuppressed),
      finalHealthExact: trials.every(trial => trial.finalHealthExact) };
    return { kind: 'raw-worker-cancellation-probe', bits: BITS, fullIterations: FULL_ITERATIONS, shortIterations: SHORT_ITERATIONS,
      coordinates: { format: 'decimal', full: { x: '-1', y: '-1' }, health: { x: '0', y: '0' } },
      variants, checks: { ...checks, passed: checks.timingConclusive && checks.obsoleteResultsSuppressed && checks.finalHealthExact },
      timingNote: 'Latency ratios and bounds use this worker\'s three measured full/short jobs; no fixed millisecond limit is an acceptance criterion. A late signal makes the trial inconclusive, not evidence of cancellation failure.' };
  } finally {
    worker.terminate();
    for (const pending of waiting.values()) clearTimeout(pending.timer);
    waiting.clear();
  }
}
