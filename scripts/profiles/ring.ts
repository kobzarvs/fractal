import { ringFragment, contiguousRingFragment, vertexShader } from '../../src/gpu/shaders.ts';
// Snapshot of the pre-optimization assembler, retained as an independent oracle.
const baselineFragment = `#version 300 es
#define SHIP_RING_BANDS 16
precision highp float;
precision highp int;

// Assembles a frame from the ring map built in rings.ts. Band j covers pixel
// radii (outer/2^(j+1), outer/2^j] with angles proportional to its radius, so
// every pixel spans one to two texels per axis. A 2x2 box of bilinear taps
// integrates that footprint.
in vec2 vUv;
out vec4 fragmentColour;
uniform highp sampler2D ringMap;
uniform float aspect;
uniform float viewHeight; // display pixels per view height
uniform int bandCount;
uniform vec4 bandLayout[SHIP_RING_BANDS]; // first row, angles, log2 step, rings
uniform vec4 bandPlace[SHIP_RING_BANDS]; // ring index at one view height (mod rings), outer radius, first column, strip columns

vec3 ringTexel(ivec2 origin, ivec2 texel, int columns, int rings) {
    // Logical angles continue across physical strips stacked vertically.
    ivec2 physical = ivec2(texel.x % columns, texel.y + (texel.x / columns) * rings);
    return texelFetch(ringMap, origin + physical, 0).rgb;
}

vec3 ringSample(vec4 band, float column, float columns, vec2 at) {
    vec2 base = floor(at), f = at - base;
    // Wrap angles and rings with integers: for an exact multiple, float mod can
    // return the divisor itself and read the next band's first ring.
    ivec2 size = ivec2(band.yw);
    ivec2 a = ivec2(mod(base, band.yw)) % size, b = (a + 1) % size;
    ivec2 origin = ivec2(int(column), int(band.x));
    return mix(
        mix(ringTexel(origin, a, int(columns), size.y), ringTexel(origin, ivec2(b.x, a.y), int(columns), size.y), f.x),
        mix(ringTexel(origin, ivec2(a.x, b.y), int(columns), size.y), ringTexel(origin, b, int(columns), size.y), f.x),
        f.y);
}

// Log2 width on each side of a band boundary where neighbours cross-fade
// (BLEND in rings.ts). Half an octave blends like trilinear mipmapping: only a
// band's centre is unmixed, so no radius switches between sample lattices.
const float BLEND = 0.5;

vec3 bandColour(int index, vec2 p, float radius) {
    vec4 band = bandLayout[index];
    float angles = band.y / 6.283185307179586;
    // Texel centres sit on the sampled angles and rings.
    vec2 at = vec2(atan(p.y, p.x) * angles,
                   bandPlace[index].x - log2(radius / viewHeight) / band.z);
    float quarter = angles / radius * 0.25;
    float column = bandPlace[index].z;
    float columns = bandPlace[index].w;
    return 0.25 * (ringSample(band, column, columns, at + vec2(-quarter, -quarter)) +
                   ringSample(band, column, columns, at + vec2(quarter, -quarter)) +
                   ringSample(band, column, columns, at + vec2(-quarter, quarter)) +
                   ringSample(band, column, columns, at + vec2(quarter, quarter)));
}

vec3 assembleRing(vec2 uv) {
    vec2 p = vec2((uv.x - 0.5) * aspect, 0.5 - uv.y);
    float innermost = bandPlace[bandCount - 1].y * 0.5;
    float radius = max(length(p) * viewHeight, innermost);
    float u = log2(bandPlace[0].y / radius);
    int index = clamp(int(floor(u)), 0, bandCount - 1);
    float d = u - float(index);
    vec3 colour = bandColour(index, p, radius);
    if (d > 1.0 - BLEND && index + 1 < bandCount)
        colour = mix(colour, bandColour(index + 1, p, radius), (d - 1.0 + BLEND) / (2.0 * BLEND));
    else if (d < BLEND && index > 0)
        colour = mix(bandColour(index - 1, p, radius), colour, (d + BLEND) / (2.0 * BLEND));
    return colour;
}

uniform vec2 res;
uniform float aa;
void main() {
    vec3 colour = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        if (float(i) >= aa) break;
        vec2 offset = aa < 1.5 ? vec2(0.0) :
            vec2((float(i) + 0.5) / aa - 0.5, fract((float(i) + 0.5) * 0.61803398875) - 0.5);
        colour += assembleRing(vUv + offset / res);
    }
    fragmentColour = vec4(colour / aa, 1.0);
}
`;
import { createRingLayout, positiveModulo } from '../../src/gpu/rings.ts';

const query = new URLSearchParams(location.search);
const width = Number(query.get('width') ?? 1920), height = Number(query.get('height') ?? 1080);
if (![width, height].every(value => Number.isSafeInteger(value) && value > 0 && value <= 7680)) throw new Error('Invalid dimensions');
const status = document.querySelector<HTMLElement>('#status')!;
const summary = document.querySelector<HTMLElement>('#summary')!;
window.addEventListener('error', event => { status.textContent = `Ошибка: ${event.message}`; });
const reportNode = document.querySelector<HTMLElement>('#report')!;
const download = document.querySelector<HTMLButtonElement>('#download')!;
const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!;
canvas.width = width; canvas.height = height;
const context = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false,
  premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
if (!context) throw new Error('WebGL2 unavailable');
const gl: WebGL2RenderingContext = context;
const hardwareMaximum = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
const maxima = (query.get('maxima') ?? '16384,8192').split(',').map(Number);
if (maxima.some(value => !Number.isSafeInteger(value) || value < 1)) throw new Error('Invalid maxima');
let layout: NonNullable<ReturnType<typeof createRingLayout>>;
let candidateName: 'contiguous' | 'generic';
interface TimerExtension { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }
const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExtension | null;
const texture = gl.createTexture()!, framebuffer = gl.createFramebuffer()!, vao = gl.createVertexArray()!;
gl.bindVertexArray(vao); gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.disable(gl.BLEND); gl.disable(gl.DITHER);

function nextFrame(): Promise<number> {
  if (document.hidden) return Promise.reject(new Error('Hidden tab; benchmark stopped'));
  return new Promise((resolve, reject) => {
    const handle = requestAnimationFrame(timestamp => { clearTimeout(timeout);
      if (document.hidden) reject(new Error('Hidden tab; benchmark stopped')); else resolve(timestamp); });
    const timeout = setTimeout(() => { cancelAnimationFrame(handle); reject(new Error('rAF timeout')); }, 5000);
  });
}
function compile(type: number, source: string) {
  const shader = gl.createShader(type)!; gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? 'Shader compile failed');
  return shader;
}
function program(fragment: string) {
  const vertex = compile(gl.VERTEX_SHADER, vertexShader), shader = compile(gl.FRAGMENT_SHADER, fragment);
  const program = gl.createProgram()!; gl.attachShader(program, vertex); gl.attachShader(program, shader); gl.linkProgram(program);
  gl.deleteShader(vertex); gl.deleteShader(shader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'Link failed');
  return program;
}
const programs = { original: program(baselineFragment), contiguous: program(contiguousRingFragment), generic: program(ringFragment) };
type Variant = keyof typeof programs;
const uniformNames = ['ringMap', 'aspect', 'viewHeight', 'res', 'aa', 'bandCount', 'bandLayout[0]', 'bandPlace[0]'] as const;
const locations = Object.fromEntries(Object.entries(programs).map(([name, handle]) =>
  [name, Object.fromEntries(uniformNames.map(uniform => [uniform, gl.getUniformLocation(handle, uniform)]))])) as
  Record<Variant, Record<typeof uniformNames[number], WebGLUniformLocation | null>>;
const bandLayout = new Float32Array(16 * 4), bandPlace = new Float32Array(16 * 4);

function draw(variant: Variant, aa: number, shift: number) {
  const uniforms = locations[variant]; gl.useProgram(programs[variant]);
  gl.uniform1i(uniforms.ringMap, 0); gl.uniform1f(uniforms.aspect, width / height);
  gl.uniform1f(uniforms.viewHeight, height); gl.uniform2f(uniforms.res, width, height);
  gl.uniform1f(uniforms.aa, aa); gl.uniform1i(uniforms.bandCount, layout!.bands.length);
  for (const [index, band] of layout!.bands.entries()) bandPlace.set([
    positiveModulo(shift / band.step, band.rings), band.radius, band.x, band.columns], index * 4);
  gl.uniform4fv(uniforms['bandLayout[0]'], bandLayout); gl.uniform4fv(uniforms['bandPlace[0]'], bandPlace);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
function statistics(samples: number[]) {
  if (!samples.length) return null;
  const sorted = [...samples].sort((a, b) => a - b), totalMs = samples.reduce((sum, value) => sum + value, 0);
  return { samples: samples.length, totalMs, meanMs: totalMs / samples.length,
    medianMs: (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2,
    minMs: sorted[0], maxMs: sorted.at(-1) };
}
function gpuDevice() {
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  return { vendor: gl.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
    renderer: gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    version: gl.getParameter(gl.VERSION), maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) };
}
const expected = new Uint8Array(width * height * 4), actual = new Uint8Array(expected.length), drain = new Uint8Array(4);
function drainGpu() { gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, drain); }
function comparePixels() {
  let channels = 0, pixels = 0, maxDifference = 0, firstMismatch: number | null = null;
  for (let index = 0; index < expected.length; index += 4) {
    let different = false;
    for (let channel = 0; channel < 4; channel++) {
      const delta = Math.abs(expected[index + channel] - actual[index + channel]);
      if (delta) { channels++; different = true; firstMismatch ??= index + channel; maxDifference = Math.max(maxDifference, delta); }
    }
    if (different) pixels++;
  }
  return { equal: channels === 0, mismatchedPixels: pixels, mismatchedChannels: channels, maxDifference, firstMismatch };
}
async function sample(variant: Variant, aa: number, shift: number) {
  await nextFrame();
  const query = timer ? gl.createQuery()! : null;
  if (query && timer) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
  const started = performance.now(); draw(variant, aa, shift); const cpuMs = performance.now() - started;
  if (query && timer) gl.endQuery(timer.TIME_ELAPSED_EXT);
  gl.flush(); const drainStarted = performance.now(); drainGpu(); const blockingDrainMs = performance.now() - drainStarted;
  let gpuMs: number | null = null;
  if (query && timer) {
    for (let attempt = 0; attempt < 45; attempt++) {
      await nextFrame();
      if (gl.getParameter(timer.GPU_DISJOINT_EXT)) break;
      if (gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) { gpuMs = Number(gl.getQueryParameter(query, gl.QUERY_RESULT)) / 1e6; break; }
    }
    gl.deleteQuery(query);
  }
  return { variant, gpuMs, cpuMs, blockingDrainMs };
}
async function sha256(source: string) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source));
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
}
async function runLayout(maximum: number) {
  const current = createRingLayout(width, height, Math.min(hardwareMaximum, maximum));
  if (!current) throw new Error(`Atlas does not fit MAX_TEXTURE_SIZE=${maximum}`);
  layout = current; candidateName = layout.bands.every(band => band.strips === 1) ? 'contiguous' : 'generic';
  bandLayout.fill(0);
  for (const [index, band] of layout.bands.entries()) bandLayout.set([band.row, band.angles, band.step, band.rings], index * 4);
  status.textContent = 'GPU заполняет детерминированную RGBA8 текстуру…';
  gl.bindTexture(gl.TEXTURE_2D, texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, layout!.width, layout!.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Atlas framebuffer incomplete');
  const filler = program(`#version 300 es
precision highp float;
precision highp int;
out vec4 fragmentColour;
void main() {
  uvec2 p = uvec2(gl_FragCoord.xy);
  uint k = (p.x * 1664525u + p.y * 1013904223u) ^ ((p.x >> 3u) * (p.y + 17u));
  fragmentColour = vec4(vec3(uvec3(k, k >> 8u, k >> 16u) & 255u) / 255.0, 1.0);
}`);
  gl.viewport(0, 0, layout!.width, layout!.height); gl.useProgram(filler); gl.drawArrays(gl.TRIANGLES, 0, 3);
  drainGpu(); gl.deleteProgram(filler); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, width, height);
  const step = layout!.bands[0].step, rings = layout!.bands[0].rings;
  const shifts = [0, step * .5, step, -step, (rings - 1) * step, rings * step];
  const cases = [];
  for (const aa of [1, 2, 5]) for (const [shiftIndex, shift] of shifts.entries()) {
    status.textContent = `AA ${aa}, сдвиг ${shiftIndex + 1}/${shifts.length}: точность и ABBA × 2…`;
    document.title = `RING ${candidateName} AA${aa} ${shiftIndex + 1}/${shifts.length}`;
    for (const variant of ['original', candidateName] as const) {
      draw(variant, aa, shift); draw(variant, aa, shift); drainGpu();
    }
    draw('original', aa, shift); gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, expected);
    draw(candidateName, aa, shift); gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, actual);
    const pixels = comparePixels(), samples = [];
    for (const variant of ['original', candidateName, candidateName, 'original', 'original', candidateName, candidateName, 'original'] as const)
      samples.push(await sample(variant, aa, shift));
    const original = statistics(samples.filter(sample => sample.variant === 'original').flatMap(sample => sample.gpuMs === null ? [] : [sample.gpuMs]));
    const candidate = statistics(samples.filter(sample => sample.variant === candidateName).flatMap(sample => sample.gpuMs === null ? [] : [sample.gpuMs]));
    const originalDrain = statistics(samples.filter(sample => sample.variant === 'original').map(sample => sample.blockingDrainMs));
    const candidateDrain = statistics(samples.filter(sample => sample.variant === candidateName).map(sample => sample.blockingDrainMs));
    const result = { aa, shiftIndex, shift, pixels, original, candidate, originalDrain, candidateDrain,
      gpuRatioCandidateOverOriginal: candidate && original ? candidate.medianMs / original.medianMs : null, samples };
    cases.push(result);
    const row = document.createElement('tr');
    for (const value of [candidateName, maximum, aa, shiftIndex, pixels.equal ? 'exact' : `${pixels.mismatchedPixels} pixels`,
      original?.medianMs.toFixed(3) ?? 'N/A', candidate?.medianMs.toFixed(3) ?? 'N/A',
      originalDrain?.medianMs.toFixed(2), candidateDrain?.medianMs.toFixed(2)]) {
      const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell);
    }
    summary.append(row);
  }
  return { maximum, variant: candidateName, atlas: { width: layout.width, height: layout.height,
    bytes: layout.width * layout.height * 4, strips: layout.bands.map(band => band.strips) }, cases };
}
async function main() {
  const layouts = [];
  for (const maximum of maxima) layouts.push(await runLayout(maximum));
  const report = { kind: 'production-ring-shader-microbenchmark', createdAt: new Date().toISOString(),
    userAgent: navigator.userAgent, gpuDevice: gpuDevice(), gpuTimerSupported: !!timer,
    options: { width, height, maxima, aa: [1, 2, 5], timingOrder: 'ABBAABBA', drainEachFrame: true },
    baseline: 'ringFragment snapshot before contiguous optimization', originalSourceSha256: await sha256(baselineFragment),
    contiguousSourceSha256: await sha256(contiguousRingFragment), genericSourceSha256: await sha256(ringFragment),
    note: 'Actual production exports against the pre-optimization shader. Same deterministic atlas/uniforms; exact RGBA8 comparison. GPU queries may be unavailable. Blocking drain is readPixels wall time, not hardware GPU time or FPS.',
    exactPixels: layouts.every(layout => layout.cases.every(item => item.pixels.equal)), layouts };
  const serialized = JSON.stringify(report, null, 2); reportNode.textContent = serialized;
  download.disabled = false;
  download.onclick = () => { const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'ring-shader.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); };
  status.textContent = `Готово; пиксели ${report.exactPixels ? 'совпадают' : 'НЕ совпадают'}`;
  document.title = `RING DONE exact=${+report.exactPixels} layouts=${layouts.length}`;
  if (query.get('download') === '1') download.click();
}
main().catch(error => { status.textContent = `Ошибка: ${error instanceof Error ? error.stack : error}`;
  document.title = `RING ERROR ${error instanceof Error ? error.message : error}`; console.error(error); });
