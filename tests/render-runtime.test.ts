import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';
import { cameraOffset, fixedToNumber, makeCamera, pan, setZoom, zoomAt } from '../src/camera.ts';
import type { Camera } from '../src/camera.ts';
import type { RenderView } from '../src/types.ts';

// The previous JS planner is an independent migration oracle, never a selectable
// application engine. Capture its actual WebGL submissions without a real GPU.
const source = await readFile(new URL('../src/gpu/renderer.ts', import.meta.url), 'utf8');
const compiled = stripTypeScriptTypes(source, { mode: 'transform' })
  .replaceAll("'./shaders'", JSON.stringify(new URL('../src/gpu/shaders.ts', import.meta.url).href))
  .replaceAll("'./rings'", JSON.stringify(new URL('../src/gpu/rings.ts', import.meta.url).href))
  .replaceAll("'./temporal'", JSON.stringify(new URL('../src/gpu/temporal.ts', import.meta.url).href));
const { FractalRenderer } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const binary = await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url));
type Draw = { kind: number; path: number; flags?: number; viewport: number[]; values: Record<string, number[]> };
function fakeCanvas(width: number, height: number, maximum = 16384) {
  const calls: Draw[] = [];
  let current: any, viewport: number[] = [], framebuffer: any;
  let enumId = 1;
  const enums: Record<string, number> = { NO_ERROR: 0 };
  const gl: any = new Proxy({
    getShaderPrecisionFormat: () => ({ precision: 23 }), getExtension: () => null,
    createVertexArray: () => ({}), createShader: (type: number) => ({ type, source: '' }), shaderSource: (shader: any, text: string) => { shader.source = text; },
    createProgram: () => ({ shaders: [], values: {} }), attachShader: (program: any, shader: any) => program.shaders.push(shader),
    getShaderParameter: () => true, getProgramParameter: () => true,
    getUniformLocation: (program: any, name: string) => ({ program, name }),
    useProgram: (program: any) => { current = program; },
    createTexture: () => ({}), createFramebuffer: () => ({}),
    bindFramebuffer: (_: number, fb: any) => { framebuffer = fb; },
    checkFramebufferStatus: () => gl.FRAMEBUFFER_COMPLETE,
    getParameter: (name: number) => name === gl.MAX_TEXTURE_SIZE ? maximum : 0,
    getError: () => 0, isContextLost: () => false,
    uniform1i: (loc: any, value: number) => { loc.program.values[loc.name] = [value]; },
    viewport: (...v: number[]) => { viewport = v; },
    drawArrays: () => {
      const fragment = current.shaders.find((shader: any) => shader.type === gl.FRAGMENT_SHADER).source;
      const path = Number(fragment.match(/#define SHIP_PATH (\d)/)?.[1] ?? 0);
      const kind = fragment.includes('#define SHIP_RING 1') ? 1 : fragment.includes('uniform highp sampler2D ringMap') ? 2
        : fragment.includes('uniform sampler2D currentMap') ? 4 : fragment.includes('uniform sampler2D frameMap') ? 5 : framebuffer ? 3 : 0;
      calls.push({ kind, path, viewport: viewport.slice(), values: structuredClone(current.values) });
    },
  }, { get(target: any, property: string) {
    if (property in target) return target[property];
    if (/^[A-Z][A-Z_0-9]*$/.test(property)) return enums[property] ??= enumId++;
    if (/^uniform[1-4]f$/.test(property)) return (loc: any, ...values: number[]) => { loc.program.values[loc.name] = values.map(Math.fround); };
    return () => {};
  } });
  return { canvas: { width, height, getContext: () => gl, addEventListener() {}, removeEventListener() {} }, calls };
}
const SHIP = { center: [0, 2], scale: [2, 1], fold: [3, 1], celtic: [4, 1], hue: [5, 1], aspect: [6, 1], res: [7, 2], aa: [9, 1], jitter: [10, 2], iterations: [12, 1], referenceLength: [13, 1], offsetX: [14, 2], offsetY: [16, 2], viewScale: [18, 2], referenceSize: [20, 2], ringBlock: [22, 4] };
async function runtime(width: number, height: number, maximum = 16384) {
  let core: any;
  const calls: Draw[] = [];
  const layouts: number[][] = [];
  const { instance } = await WebAssembly.instantiate(binary, { gpu: {
    ring_target: (w: number, h: number) => { layouts.push([w, h]); return 1; }, temporal_targets: () => 1, reference_texture: () => 0,
    draw: (ptr: number) => {
      const header = new Int32Array(core.memory.buffer, ptr, 8), raw = new Float32Array(core.memory.buffer, ptr + 32, 40);
      const values: Record<string, number[]> = {};
      for (const [name, [start, count]] of Object.entries(SHIP)) values[name] = Array.from(raw.slice(start, start + count));
      values.historyOffset = Array.from(raw.slice(26, 28)); values.historyScale = [raw[28]]; values.historyWeight = [raw[29]]; values.moving = [raw[30]]; values.texelSize = Array.from(raw.slice(31, 33));
      values.viewHeight = [height]; values.bandCount = [raw[33]];
      const bands = new Float32Array(core.memory.buffer, ptr + 192, 64), places = new Float32Array(core.memory.buffer, ptr + 448, 64);
      for (let i = 0; i < raw[33]; i++) { values[`bandLayout[${i}]`] = Array.from(bands.slice(i * 4, i * 4 + 4)); values[`bandPlace[${i}]`] = Array.from(places.slice(i * 4, i * 4 + 4)); }
      calls.push({ kind: header[0], path: header[1], flags: header[2], viewport: Array.from(header.slice(3, 7)), values }); return 0;
    },
  } });
  core = instance.exports;
  core.render_mark_dirty();
  const input = new Float64Array(core.memory.buffer, core.render_input_ptr(), 64);
  input[0] = width; input[1] = height; input[2] = maximum;
  function render(view: RenderView) {
    input.set([view.iterations, view.aa, view.fold, view.celtic, view.hue, view.logScale, view.scale, ...view.center, ...view.offsetX, ...view.offsetY, +view.guided, +!!view.temporal, view.referenceKey, 1, 16385, 17408, 1, 1, 0, 16384, +!!view.position], 3);
    if (view.position) {
      const x = new TextEncoder().encode(String(view.position.x)), y = new TextEncoder().encode(String(view.position.y));
      const text = new Uint8Array(core.memory.buffer, core.render_text_ptr(), core.render_text_capacity()); text.set(x); text.set(y, x.length);
      assert.equal(core.render_set_position(x.length, y.length, view.position.bits, view.logScale), 0);
    }
    assert.equal(core.render_frame(0, 1000), 0);
  }
  return { core, input, calls, layouts, render };
}
function checkDraws(wasm: Draw[], js: Draw[], label: string) {
  assert.equal(wasm.length, js.length, `${label} draw count`);
  for (let i = 0; i < js.length; i++) {
    const actual = wasm[i], expected = js[i];
    assert.equal(actual.kind, expected.kind, `${label} draw ${i} kind`);
    if ([0, 1, 3].includes(actual.kind)) assert.equal(actual.path, expected.path, `${label} draw ${i} path`);
    assert.deepEqual(actual.viewport, expected.viewport, `${label} draw ${i} viewport`);
    for (const [name, value] of Object.entries(expected.values)) {
      if (['referenceOrbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds', 'ringMap', 'currentMap', 'historyMap', 'frameMap'].includes(name)) continue;
      assert.deepEqual(actual.values[name], value, `${label} draw ${i} uniform ${name}`);
    }
  }
  wasm.length = js.length = 0;
}
function reference() {
  const arrays = Array.from({ length: 5 }, () => new Float32Array(17408 * 4));
  return { id: 1, capacity: 17408, length: 16385, iterations: 16384, bits: 576, fold: 1, celtic: 0, orbit: arrays[0], realOrbit: arrays[1], blaA: arrays[2], blaB: arrays[3], blaBounds: arrays[4], backend: 'js', computeMs: 0 };
}
function view(camera: Camera, initial: Camera, guided: boolean): RenderView {
  const offset = cameraOffset(camera, initial);
  return { center: [fixedToNumber(camera.x, camera.bits), fixedToNumber(camera.y, camera.bits)], scale: 2 ** camera.logScale, logScale: camera.logScale,
    offsetX: offset.x, offsetY: offset.y, iterations: 16384, aa: 2, fold: 1, celtic: 0, hue: 0, guided, referenceKey: 1, temporal: !guided, position: camera };
}
test('WASM marks only contiguous ring atlases for direct texel addressing', async () => {
  for (const [width, height, maximum, contiguous] of [[1920, 1080, 16384, true], [3840, 2160, 16384, false], [1280, 720, 4096, false]] as const) {
    const wasm = await runtime(width, height, maximum);
    try {
      const camera = makeCamera(width / height); setZoom(camera, 30);
      const frameView = view(camera, camera, true);
      for (let frame = 0; frame < 60 && !wasm.calls.some(call => call.kind === 2); frame++) wasm.render(frameView);
      const assembly = wasm.calls.find(call => call.kind === 2);
      assert.ok(assembly, `${width}x${height} cache must finish filling`);
      assert.equal(!!(assembly.flags! & 2), contiguous, `${width}x${height} atlas addressing`);
      assert.equal(assembly.flags! & 1, 0, 'assembly must not inherit the ring-write scissor flag');
      for (const draw of wasm.calls) if (draw.kind !== 2) assert.equal(draw.flags! & 2, 0);
    } finally { wasm.core.render_dispose(); }
  }
});
test('WASM ring planner submits the same commands and f32 uniforms as the former JS planner', async () => {
  for (const [width, height, maximum] of [[320, 200, 16384], [1920, 1080, 16384], [3840, 2160, 16384], [1280, 720, 4096]]) {
    const fake = fakeCanvas(width, height, maximum), js = new FractalRenderer(fake.canvas), wasm = await runtime(width, height, maximum);
    js.setReference(reference());
    const camera = makeCamera(width / height); setZoom(camera, 115); const initial = { ...camera };
    for (let frame = 0; frame < 160; frame++) {
      if (frame > 85) setZoom(camera, 115 + (frame - 85) * 0.0031);
      if (frame === 130) setZoom(camera, 110);
      const frameView = view(camera, initial, true);
      js.render(frameView); wasm.render(frameView);
      checkDraws(wasm.calls, fake.calls, `${width}x${height} frame ${frame}`);
    }
    js.dispose();
  }
});
test('WASM temporal planner matches JS jitter and reprojection during motion and settling', async () => {
  const width = 320, height = 200, fake = fakeCanvas(width, height), js = new FractalRenderer(fake.canvas), wasm = await runtime(width, height);
  js.setReference(reference());
  const camera = makeCamera(width / height); setZoom(camera, 115); const initial = { ...camera };
  for (let frame = 0; frame < 80; frame++) {
    if (frame < 25) zoomAt(camera, 0.17, -0.13, -0.0031);
    if (frame >= 30 && frame < 55) pan(camera, 0.00051, -0.00024);
    const frameView = view(camera, initial, false); frameView.aa = 5;
    js.render(frameView); wasm.render(frameView);
    checkDraws(wasm.calls, fake.calls, `temporal frame ${frame}`);
  }
  js.dispose();
});

test('WASM direct/float/FE paths retain boundary uniforms and reject invalid input', async () => {
  const width = 333, height = 217, fake = fakeCanvas(width, height), js = new FractalRenderer(fake.canvas), wasm = await runtime(width, height);
  js.setReference(reference());
  const camera = makeCamera(width / height), initial = { ...camera };
  for (const logScale of [-7.999999999999, -8, -8.000000000001, -79.999999999999, -80, -80.000000000001, -3900]) {
    camera.logScale = logScale;
    const frameView = view(camera, initial, false); frameView.temporal = false;
    js.render(frameView); wasm.render(frameView);
    checkDraws(wasm.calls, fake.calls, `scale ${logScale}`);
  }
  wasm.input[3] = 0; assert.equal(wasm.core.render_frame(0, 1000), -20);
  wasm.input[3] = 16384; wasm.input[5] = Number.NaN; assert.equal(wasm.core.render_frame(0, 1000), -20);
  js.dispose();
});

test('clock reset preserves a warmed Rust-owned flight cache and exact one-second trajectory', async () => {
  const wasm = await runtime(1920, 1080);
  const camera = makeCamera(1920 / 1080); setZoom(camera, 30);
  const initial = { ...camera };
  wasm.render(view(camera, initial, true)); // Seed the external reference metadata.
  assert.equal(wasm.core.render_camera_command(5), 0); // Snapshot reference camera.
  const input = wasm.input;
  const stats = new Float64Array(wasm.core.memory.buffer, wasm.core.render_stats_ptr(), 16);
  input[16] = 1; input[17] = 1; input[40] = 1; input[41] = 31;
  assert.equal(wasm.core.render_camera_command(4), 0);
  for (let frame = 0; frame < 120 && !stats[6]; frame++) assert.equal(wasm.core.render_frame(1, 0), 0);
  assert.equal(stats[6], 1);
  const resets = stats[4], samples = stats[3];
  assert.equal(wasm.core.render_camera_command(6), 0);
  wasm.calls.length = 0;
  assert.equal(wasm.core.render_frame(1, 50_000), 0);
  assert.equal(stats[6], 1); assert.equal(stats[4], resets); assert.equal(stats[3], samples);
  assert.deepEqual(wasm.calls.map(call => call.kind), [2], 'the first measured frame only assembles the existing cache');
  for (let frame = 1; frame <= 400; frame++) {
    assert.equal(wasm.core.render_frame(1, 50_000 + frame * 2.5), 0);
    assert.equal(stats[6], 1, `guided frame ${frame}`);
  }
  assert.ok(Math.abs(stats[9] - 30.55) < 1e-10, `end zoom ${stats[9]}`);
  assert.equal(stats[4], resets);
});

test('an oversized zoom jump keeps the bounded fullscreen fallback instead of rebuilding the whole atlas', async () => {
  const wasm = await runtime(3840, 2160);
  const camera = makeCamera(3840 / 2160); setZoom(camera, 30);
  const initial = { ...camera };
  try {
    const stats = new Float64Array(wasm.core.memory.buffer, wasm.core.render_stats_ptr(), 16);
    for (let frame = 0; frame < 120 && !stats[6]; frame++) wasm.render(view(camera, initial, true));
    assert.equal(stats[6], 1);
    wasm.calls.length = 0;
    setZoom(camera, 60);
    wasm.render(view(camera, initial, true));
    assert.equal(stats[6], 0);
    assert.deepEqual(wasm.calls.map(call => call.kind), [0], 'large jumps must preserve the original full-frame fallback');
  } finally { wasm.core.render_dispose(); }
});
