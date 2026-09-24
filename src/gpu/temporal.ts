// Temporal accumulation and reprojection retain qg/Yg/Xg from the original
// newton-fractal renderer. Pure camera/sample state is independently testable.
import { cameraOffset } from '../camera.ts';
import type { Camera } from '../camera.ts';

export interface TemporalFrame {
  jitter: { x: number; y: number };
  historyOffset: [number, number];
  historyScale: number;
  historyWeight: number;
  moving: boolean;
}

export function temporalJitter(index: number, samples: number): { x: number; y: number } {
  let sum = 0;
  for (let sample = 0; sample < samples; sample++) sum += ((sample + 0.5) * 0.61803398875) % 1;
  return {
    x: (index + 0.5) / samples - 0.5,
    y: (((index + 0.5) * 0.61803398875) % 1) - sum / samples,
  };
}

export function historyTransform(camera: Camera, previous: Camera, aspect: number) {
  const delta = cameraOffset(camera, previous);
  // Scale the FE exponent before conversion, keeping subnormal-range world deltas
  // visible at deep zoom. Converting either camera to Number first loses them.
  const scaled = ([mantissa, exponent]: [number, number]) =>
    mantissa === 0 ? 0 : mantissa * 2 ** (exponent - previous.logScale);
  return {
    scale: 2 ** (camera.logScale - previous.logScale),
    x: scaled(delta.x) / aspect,
    y: -scaled(delta.y),
  };
}

export class TemporalAccumulator {
  private previous: Camera | null = null;
  private samples = 1;
  private historySamples = 0;
  private stationarySamples = 0;
  private sampleIndex = 0;
  private width = 0;
  private height = 0;
  private key = '';

  get settling(): boolean { return this.samples > 1 && this.stationarySamples < this.samples; }

  reset(): void {
    this.previous = null;
    this.samples = 1;
    this.historySamples = this.stationarySamples = this.sampleIndex = 0;
  }

  prepare(camera: Camera, width: number, height: number, samples: number, key: string): TemporalFrame {
    if (width !== this.width || height !== this.height || samples !== this.samples || key !== this.key) this.reset();
    this.width = width;
    this.height = height;
    this.samples = samples;
    this.key = key;
    let transform = this.previous ? historyTransform(camera, this.previous, width / height) : null;
    if (transform && (!Number.isFinite(transform.x + transform.y) ||
        transform.scale < 0.8 || transform.scale > 1.25 || Math.hypot(transform.x, transform.y) > 0.25)) {
      this.reset();
      this.samples = samples;
      transform = null;
    }
    const moving = transform !== null && (transform.scale !== 1 || transform.x !== 0 || transform.y !== 0);
    if (moving) this.stationarySamples = 0;
    else if (this.stationarySamples === 0) this.historySamples = this.sampleIndex = 0;
    const frame: TemporalFrame = {
      jitter: temporalJitter(this.sampleIndex % samples, samples),
      historyOffset: [transform?.x ?? 0, transform?.y ?? 0],
      historyScale: transform?.scale ?? 1,
      historyWeight: this.previous && this.historySamples > 0
        ? Math.min(this.historySamples / (this.historySamples + 1), 1 - 1 / samples) *
          (moving ? Math.exp(-4 * Math.abs(Math.log(transform!.scale))) : 1)
        : 0,
      moving,
    };
    this.previous = { ...camera };
    this.historySamples = Math.min(samples, this.historySamples + 1);
    if (!moving) this.stationarySamples++;
    this.sampleIndex++;
    return frame;
  }
}

export const temporalFragment = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragmentColour;
uniform sampler2D currentMap;
uniform sampler2D historyMap;
uniform vec2 texelSize;
uniform vec2 historyOffset;
uniform float historyScale;
uniform float historyWeight;
uniform float moving;

vec4 sampleCurrent(vec2 uv) {
    return texture(currentMap, clamp(uv, 0.5 * texelSize, 1.0 - 0.5 * texelSize));
}
void main() {
    vec4 current = sampleCurrent(vUv);
    vec2 oldUv = (vUv - 0.5) * historyScale + 0.5 + historyOffset;
    vec2 margin = texelSize * 0.5;
    if (historyWeight <= 0.0 || (moving > 0.5 &&
        (any(lessThan(oldUv, margin)) || any(greaterThan(oldUv, 1.0 - margin))))) {
        fragmentColour = current;
        return;
    }
    vec4 history = texture(historyMap, oldUv);
    if (moving > 0.5) {
        vec4 low = current, high = current;
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                vec4 neighbour = sampleCurrent(vUv + vec2(float(x), float(y)) * texelSize);
                low = min(low, neighbour);
                high = max(high, neighbour);
            }
        }
        history = clamp(history, low, high);
    }
    fragmentColour = mix(current, history, historyWeight);
}
`;

export const temporalCompositeFragment = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragmentColour;
uniform sampler2D frameMap;
void main() { fragmentColour = texture(frameMap, vUv); }
`;
