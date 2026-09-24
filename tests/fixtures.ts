import type { ReferenceRequest } from '../src/types.ts';
import { WESTERN_ARMADA } from '../src/tours.ts';

// Test-only parser. Decimal input is rounded half up in magnitude, then signed,
// matching the camera parser used by the source application.
function fixed(decimal: string, bits: number): string {
  const match = /^([+-]?)(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(decimal);
  if (!match || !(match[2] || match[3])) throw new Error(`Invalid fixture decimal: ${decimal}`);
  const exponent = Number(match[4] ?? 0) - (match[3]?.length ?? 0);
  if (!Number.isSafeInteger(exponent) || Math.abs(exponent) > 100_000) {
    throw new Error(`Invalid fixture exponent: ${decimal}`);
  }
  let scaled = BigInt((match[2] || '0') + (match[3] || '')) << BigInt(bits);
  if (exponent >= 0) scaled *= 10n ** BigInt(exponent);
  else {
    const divisor = 10n ** BigInt(-exponent);
    scaled = (scaled + divisor / 2n) / divisor;
  }
  return String(match[1] === '-' ? -scaled : scaled);
}

function caseAt(name: string, id: number, x: string, y: string, bits: number,
  iterations: number, fold: number, celtic: number): { name: string; request: ReferenceRequest } {
  return { name, request: { id, x: fixed(x, bits), y: fixed(y, bits),
    bits, iterations, fold, celtic } };
}

export const referenceCases: { name: string; request: ReferenceRequest }[] = [
  caseAt('origin-128-1024', 1, '0', '0', 128, 1024, 0, 0),
  caseAt('negative-cross-128-1024', 2, '-1', '1', 128, 1024, 0.75, 0),
  caseAt('celtic-near-axis-256-1024', 3,
    '0.500000000000000000000000000000000001', '-0.499999999999999999999999999999999999',
    256, 1024, 0.25, 0.875),
  caseAt('western-armada-576-16384', 4, WESTERN_ARMADA.x, WESTERN_ARMADA.y,
    576, 16384, 1, 0),
];
