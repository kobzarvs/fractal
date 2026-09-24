export interface Camera { x: bigint; y: bigint; bits: number; logScale: number }
export const precisionBits = (logScale: number) => Math.ceil(Math.max(128, 128 - logScale) / 64) * 64;
// DOM y grows downwards; shipOffset flips WebGL's bottom-up UV to this convention.
export const screenOffset = (x: number, y: number, width: number, height: number): [number, number] => [(x - width / 2) / height, y / height - .5];

export function decimalToFixed(value: string, bits: number): bigint {
  const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value.trim());
  if (!m || !(m[2] || m[3])) throw new Error('Некорректная координата');
  const power = Number(m[4] || 0) - (m[3]?.length || 0);
  if (!Number.isSafeInteger(power) || Math.abs(power) > 10000) throw new Error('Координата вне диапазона');
  let n = BigInt((m[2] || '0') + (m[3] || '')) << BigInt(bits);
  if (power >= 0) n *= 10n ** BigInt(power);
  else { const divisor = 10n ** BigInt(-power); n = (n + divisor / 2n) / divisor; }
  return m[1] === '-' ? -n : n;
}

export function fixedToFE(value: bigint, bits: number): [number, number] {
  if (value === 0n) return [0, 0];
  const n = value < 0n ? -value : value, length = n.toString(2).length;
  const shift = Math.max(0, length - 53);
  const rounded = shift ? (n + (1n << BigInt(shift - 1))) >> BigInt(shift) : n;
  let mantissa = Number(rounded) / 2 ** (length - shift - 1), exponent = length - 1 - bits;
  if (mantissa >= 2) { mantissa *= .5; exponent++; }
  return [value < 0n ? -mantissa : mantissa, exponent];
}
export const fixedToNumber = (n: bigint, bits: number) => { const [m, e] = fixedToFE(n, bits); return m * 2 ** e; };
export function fixedScale(logScale: number, bits: number): bigint {
  const exponent = Math.floor(logScale), n = BigInt(Math.round(2 ** (logScale - exponent + 52)));
  const shift = exponent + bits - 52;
  return shift >= 0 ? n << BigInt(shift) : n >> BigInt(-shift);
}
function multiply(n: bigint, factor: number): bigint {
  if (!factor || n === 0n) return 0n;
  const exponent = Math.floor(Math.log2(Math.abs(factor)));
  const value = n * BigInt(Math.round(factor / 2 ** exponent * 2 ** 52));
  return exponent >= 52 ? value << BigInt(exponent - 52) : value >> BigInt(52 - exponent);
}
export function makeCamera(aspect: number): Camera {
  const logScale = Math.log2(Math.max(3.2, 3.5 / aspect)), bits = precisionBits(logScale);
  return { x: decimalToFixed('-.45', bits), y: decimalToFixed('-.45', bits), bits, logScale };
}
export function pan(camera: Camera, x: number, y: number): void {
  const scale = fixedScale(camera.logScale, camera.bits);
  camera.x -= multiply(scale, x); camera.y -= multiply(scale, y);
}
export function zoomAt(camera: Camera, x: number, y: number, deltaLog2: number): void {
  const next = Math.max(-3900, Math.min(Math.log2(100), camera.logScale + deltaLog2));
  const bits = Math.max(camera.bits, precisionBits(next));
  camera.x <<= BigInt(bits - camera.bits); camera.y <<= BigInt(bits - camera.bits); camera.bits = bits;
  const difference = fixedScale(camera.logScale, bits) - fixedScale(next, bits);
  camera.x += multiply(difference, x); camera.y += multiply(difference, y); camera.logScale = next;
}
export function setZoom(camera: Camera, zoom: number): void {
  zoomAt(camera, 0, 0, Math.log2(3.2) - zoom * Math.log2(10) - camera.logScale);
}
export function cameraOffset(camera: Camera, reference: Camera) {
  const bits = Math.max(camera.bits, reference.bits);
  return {
    x: fixedToFE((camera.x << BigInt(bits - camera.bits)) - (reference.x << BigInt(bits - reference.bits)), bits),
    y: fixedToFE((camera.y << BigInt(bits - camera.bits)) - (reference.y << BigInt(bits - reference.bits)), bits),
  };
}
