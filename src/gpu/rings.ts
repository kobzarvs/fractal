// Polar ring layout adapted from https://newton-fractal.pages.dev/ (rings.ts).
// Retain 1.5x angular/radial sampling. If it does not fit, use the full renderer.
export interface RingBand {
  x: number; row: number; angles: number; step: number; rings: number;
  radius: number; outwards: number; inwards: number;
  window: [number, number] | null;
}
export interface RingLayout { bands: RingBand[]; width: number; height: number }
const DENSITY = 1.5;
const MAX_TEXELS = 24_000_000;

export function createRingLayout(width: number, height: number, maxTextureSize: number): RingLayout | null {
  const radii = [Math.hypot(width, height) / 2 + 1];
  while (radii.length < 16 && radii[radii.length - 1] > 1) radii.push(radii[radii.length - 1] / 2);
  let x = 0, row = 0, rowHeight = 0, textureWidth = 0;
  const bands = radii.map((radius, index): RingBand => {
    const angles = Math.max(16, Math.ceil(2 * Math.PI * radius * DENSITY));
    const step = 2 * Math.PI / angles / Math.LN2;
    const outwards = index > 0 ? 0.5 : 0;
    const inwards = index < radii.length - 1 ? 0.5 : 0;
    const rings = Math.ceil((1 + outwards + inwards) / step) + 7;
    textureWidth ||= angles;
    if (x + angles > textureWidth) { row += rowHeight; x = 0; rowHeight = 0; }
    const band = { x, row, angles, step, rings, radius, outwards, inwards, window: null };
    x += angles;
    rowHeight = Math.max(rowHeight, rings);
    return band;
  });
  const textureHeight = row + rowHeight;
  if (textureWidth > maxTextureSize || textureHeight > maxTextureSize || textureWidth * textureHeight > MAX_TEXELS) return null;
  return { bands, width: textureWidth, height: textureHeight };
}

export function ringWindow(band: RingBand, logScale: number, height: number, origin: number): [number, number] {
  const outer = logScale + Math.log2(band.radius / height) + band.outwards;
  const span = 1 + band.outwards + band.inwards;
  return [Math.floor((origin - outer) / band.step) - 2, Math.ceil((origin - outer + span) / band.step) + 2];
}

export function missingRingRanges(wanted: [number, number], previous: [number, number] | null): [number, number][] {
  if (!previous || previous[1] < wanted[0] || previous[0] > wanted[1]) return [wanted];
  const missing: [number, number][] = [];
  if (wanted[0] < previous[0]) missing.push([wanted[0], previous[0] - 1]);
  if (wanted[1] > previous[1]) missing.push([previous[1] + 1, wanted[1]]);
  return missing;
}

export function positiveModulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}
