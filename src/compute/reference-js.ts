import type { ReferenceRequest, ReferenceResult } from '../types.ts';

// Independent, readable port of the reference worker in
// https://newton-fractal.pages.dev/assets/reference-worker-xqHYK19O.js.
// Keep its floating-point operation order: these arrays are the numerical
// oracle for the Rust/WASM and GPU paths.
const TEXTURE_WIDTH = 1024;
const BLA_LEVELS = 10;
const EMPTY_BOUND = -(2 ** 100);

type Matrix = [number, number, number, number];
type Level = {
  stride: number; offset: number; A: Matrix; B: Matrix;
  radius: number; cRadius: number;
};

function validRequest(req: ReferenceRequest): boolean {
  return req !== null && typeof req === 'object'
    && Number.isSafeInteger(req.id)
    && typeof req.x === 'string' && /^[+-]?\d+$/.test(req.x)
    && typeof req.y === 'string' && /^[+-]?\d+$/.test(req.y)
    && Number.isSafeInteger(req.bits) && req.bits > 0
    && Number.isSafeInteger(req.iterations) && req.iterations > 0
    && typeof req.fold === 'number' && Number.isFinite(req.fold) && req.fold >= 0 && req.fold <= 1
    && typeof req.celtic === 'number' && Number.isFinite(req.celtic) && req.celtic >= 0 && req.celtic <= 1;
}

function quantizeControl(value: number): number {
  return Math.round(Math.min(1, Math.max(0, value)) * TEXTURE_WIDTH) / TEXTURE_WIDTH;
}

function toMantissaExponent(value: bigint, bits: number): [number, number] {
  if (value === 0n) return [0, 0];
  const magnitude = value < 0n ? -value : value;
  const length = magnitude.toString(2).length;
  const drop = Math.max(0, length - 53);
  const rounded = drop > 0
    ? (magnitude + (1n << BigInt(drop - 1))) >> BigInt(drop)
    : magnitude;
  let mantissa = Number(rounded) / 2 ** (length - drop - 1);
  let exponent = length - 1 - bits;
  if (mantissa >= 2) {
    mantissa *= 0.5;
    exponent++;
  }
  return [value < 0n ? -mantissa : mantissa, exponent];
}

function matrixNorm(matrix: Matrix): number {
  return Math.max(Math.abs(matrix[0]) + Math.abs(matrix[1]),
    Math.abs(matrix[2]) + Math.abs(matrix[3]));
}

function multiply(left: Matrix, right: Matrix): Matrix {
  return [
    left[0] * right[0] + left[1] * right[2],
    left[0] * right[1] + left[1] * right[3],
    left[2] * right[0] + left[3] * right[2],
    left[2] * right[1] + left[3] * right[3],
  ];
}

function matrixExponent(matrix: Matrix): number {
  const maximum = Math.max(...matrix.map(Math.abs));
  return maximum > 0 ? Math.floor(Math.log2(maximum)) : 0;
}

function newLevel(stride: number, offset: number): Level {
  return { stride, offset, A: [1, 0, 0, 1], B: [0, 0, 0, 0],
    radius: Infinity, cRadius: Infinity };
}

/** Cancellation rejects with `AbortError`; `yieldControl` runs after every 128 computed steps. */
export async function computeReferenceJs(
  request: ReferenceRequest,
  cancelled: () => boolean = () => false,
  yieldControl: () => Promise<void> = () => new Promise(resolve => setTimeout(resolve, 0)),
): Promise<ReferenceResult> {
  if (!validRequest(request)) throw new Error('Invalid reference request');
  const started = performance.now();
  const { id, bits, iterations } = request;
  const fold = quantizeControl(request.fold);
  const celtic = quantizeControl(request.celtic);
  const foldNumerator = BigInt(TEXTURE_WIDTH - 2 * Math.round(fold * TEXTURE_WIDTH));
  const celticNumerator = BigInt(TEXTURE_WIDTH - 2 * Math.round(celtic * TEXTURE_WIDTH));
  const foldMultiplier = 1 - 2 * fold;
  const celticMultiplier = 1 - 2 * celtic;
  const workingBits = bits + 32;
  const shift = BigInt(workingBits);
  const cx = BigInt(request.x) << 32n;
  const cy = BigInt(request.y) << 32n;
  const capacity = Math.ceil((iterations + 1) / TEXTURE_WIDTH) * TEXTURE_WIDTH;
  const orbit = new Float32Array(capacity * 4);
  const realOrbit = new Float32Array(celtic ? capacity * 4 : 4);
  const blaA = new Float32Array(capacity * 4);
  const blaB = new Float32Array(capacity * 4);
  const blaBounds = new Float32Array(capacity * 4).fill(EMPTY_BOUND);
  const levels = Array.from({ length: BLA_LEVELS }, (_, level) =>
    newLevel(2 << level, capacity - (capacity >> level)));
  let zx = 0n;
  let zy = 0n;
  let length = 0;

  for (let iteration = 0; iteration <= iterations; iteration++) {
    if (cancelled()) throw new DOMException('Reference computation cancelled', 'AbortError');
    const [mx, ex] = toMantissaExponent(zx, workingBits);
    const [my, ey] = toMantissaExponent(zy, workingBits);
    orbit.set([mx, ex, my, ey], iteration * 4);
    const realSquare = zx * zx - zy * zy;
    if (celtic) {
      const [mr, er] = toMantissaExponent(realSquare, 2 * workingBits);
      realOrbit.set([mr, er, 0, 0], iteration * 4);
    }
    length = iteration + 1;

    const x = mx * 2 ** ex;
    const y = my * 2 ** ey;
    if (x * x + y * y > 65536) break;

    const radius = Math.min(Math.abs(x), Math.abs(y), 1e-7 * Math.hypot(x, y),
      celtic ? Math.min(Math.abs(x - y), Math.abs(x + y)) / 2 : Infinity);
    const sign = Math.sign(x) * Math.sign(y);
    const crossDerivative = sign < 0 ? foldMultiplier : sign > 0 ? 1 : 1 - fold;
    const realDerivative = realSquare < 0n ? celticMultiplier : realSquare > 0n ? 1 : 1 - celtic;
    const step: Matrix = [2 * realDerivative * x, -2 * realDerivative * y,
      2 * crossDerivative * y, 2 * crossDerivative * x];

    for (const level of levels) {
      level.radius = Math.min(level.radius, radius / (4 * matrixNorm(level.A)));
      const bNorm = matrixNorm(level.B);
      if (bNorm > 0) level.cRadius = Math.min(level.cRadius, radius / (4 * bNorm));
      level.A = multiply(step, level.A);
      level.B = multiply(step, level.B);
      level.B[0] += 1;
      level.B[3] += 1;
      if ((iteration + 1) % level.stride === 0) {
        const index = (level.offset + Math.floor(iteration / level.stride)) * 4;
        if (Number.isFinite(matrixNorm(level.A)) && Number.isFinite(matrixNorm(level.B))
          && level.radius > 0 && level.cRadius > 0) {
          const exponentA = matrixExponent(level.A);
          const exponentB = matrixExponent(level.B);
          blaA.set(level.A.map(value => value * 2 ** -exponentA), index);
          blaB.set(level.B.map(value => value * 2 ** -exponentB), index);
          blaBounds.set([Math.log2(level.radius), Math.log2(level.cRadius),
            exponentA, exponentB], index);
        }
        level.A = [1, 0, 0, 1];
        level.B = [0, 0, 0, 0];
        level.radius = Infinity;
        level.cRadius = Infinity;
      }
    }

    const realTerm = realSquare < 0n && celtic
      ? (realSquare * celticNumerator >> shift + 10n)
      : realSquare >> shift;
    const cross = zx * zy;
    const imaginaryTerm = cross < 0n
      ? (2n * cross * foldNumerator >> shift + 10n)
      : (2n * cross >> shift);
    zx = realTerm + cx;
    zy = imaginaryTerm + cy;
    if ((iteration & 127) === 127) await yieldControl();
  }

  return { id, length, capacity, iterations, bits, fold, celtic, orbit,
    realOrbit, blaA, blaB, blaBounds, computeMs: performance.now() - started, backend: 'js' };
}
