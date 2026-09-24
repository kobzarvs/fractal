import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';

// Execute the production bridge in Node with real WASM and a bounded WebGL mock.
// Transform parameter properties and resolve browser module URLs, as in the
// independent planner-oracle tests; no bridge implementation is duplicated.
const moduleUrl = (source: string) => `data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source, { mode: 'transform' })).toString('base64')}`;
const loaderSource = await readFile(new URL('../src/gpu/render-wasm.ts', import.meta.url), 'utf8');
const loaderUrl = moduleUrl(loaderSource.replace('import.meta.env.BASE_URL', "''"));
const loader = await import(loaderUrl);
let rendererSource = await readFile(new URL('../src/gpu/wasm-renderer.ts', import.meta.url), 'utf8');
for (const name of ['shaders', 'temporal']) rendererSource = rendererSource.replaceAll(`'./${name}'`, JSON.stringify(new URL(`../src/gpu/${name}.ts`, import.meta.url).href));
rendererSource = rendererSource.replaceAll("'./render-wasm'", JSON.stringify(loaderUrl));
const { WasmRenderer } = await import(moduleUrl(rendererSource));
const binary = await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url));
const fetchBefore = globalThis.fetch;
globalThis.fetch = async () => new Response(binary, { headers: { 'Content-Type': 'application/wasm' } });
try { await loader.preloadRenderWasm(); } finally { globalThis.fetch = fetchBefore; }

type Upload = { source: Float32Array; offset: number; width: number; height: number; sub: boolean };
function canvas(width = 320, height = 200, maximum = 16384) {
  const live = new Set<object>(), uploads: Upload[] = [], deleted: object[] = [];
  const errors: number[] = [], enums: Record<string, number> = { NO_ERROR: 0 };
  let enumId = 1, allocationError = 0, incomplete = false, draws = 0, waitCalls = 0, syncResult = 'ALREADY_SIGNALED';
  const shaderSources = new Map<object, string>(), programShaders = new Map<object, object[]>(), drawSources: string[] = [];
  let currentProgram: object;
  const create = () => { const handle = {}; live.add(handle); return handle; };
  const remove = (handle: object) => { live.delete(handle); deleted.push(handle); };
  const noops = new Set(['bindVertexArray', 'disable', 'enable', 'bindTexture', 'activeTexture', 'texParameteri',
    'bindFramebuffer', 'framebufferTexture2D', 'shaderSource', 'compileShader', 'attachShader', 'linkProgram',
    'useProgram', 'uniform1i', 'uniform1f', 'uniform2fv', 'uniform4fv', 'viewport', 'scissor', 'readPixels', 'flush']);
  const gl: any = new Proxy({
    getShaderPrecisionFormat: () => ({ precision: 23 }), getExtension: () => null,
    getParameter: () => maximum, getError: () => errors.shift() ?? 0, isContextLost: () => false,
    createVertexArray: create, createTexture: create, createFramebuffer: create, createProgram: create, createShader: create,
    deleteVertexArray: remove, deleteTexture: remove, deleteFramebuffer: remove, deleteProgram: remove, deleteShader: remove,
    fenceSync: create, deleteSync: remove,
    clientWaitSync: (_sync: object, flags: number, timeout: number) => {
      assert.equal(flags, 0); assert.equal(timeout, 0, 'GPU completion is polled without blocking'); waitCalls++; return gl[syncResult];
    },
    getShaderParameter: () => true, getProgramParameter: () => true, getUniformLocation: () => null,
    checkFramebufferStatus: () => incomplete ? gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT : gl.FRAMEBUFFER_COMPLETE,
    shaderSource: (shader: object, source: string) => shaderSources.set(shader, source),
    attachShader: (program: object, shader: object) => programShaders.set(program, [...programShaders.get(program) ?? [], shader]),
    useProgram: (program: object) => { currentProgram = program; },
    drawArrays: () => {
      draws++;
      drawSources.push((programShaders.get(currentProgram) ?? []).map(shader => shaderSources.get(shader)!).find(source => source.includes('fragmentColour'))!);
    },
    texImage2D: (...a: any[]) => {
      if (a[2] === gl.RGBA8) { if (allocationError) { errors.push(allocationError); allocationError = 0; } }
      else uploads.push({ source: a[8], offset: a[9] ?? 0, width: a[3], height: a[4], sub: false });
    },
    texSubImage2D: (...a: any[]) => uploads.push({ source: a[8], offset: a[9] ?? 0, width: a[4], height: a[5], sub: true }),
  }, { get(target: any, property: string) {
    if (property in target) return target[property];
    if (/^[A-Z][A-Z_0-9]*$/.test(property)) return enums[property] ??= enumId++;
    if (noops.has(property)) return () => {};
    throw new Error(`Unexpected WebGL call: ${property}`);
  } });
  const result = Object.assign(new EventTarget(), { width, height, getContext: () => gl });
  return { canvas: result, gl, live, uploads, deleted, drawSources, get draws() { return draws; },
    get waitCalls() { return waitCalls; }, syncResult(value: string) { syncResult = value; },
    failAllocation(error: number) { allocationError = error; }, incomplete() { incomplete = true; } };
}
const request = { id: 1, x: '0', y: '0', bits: 128, iterations: 512, fold: 1, celtic: 0 };
function view(guided = false, temporal = false) {
  return { center: [0, 0], scale: 2 ** -300, logScale: -300, offsetX: [0, 0], offsetY: [0, 0],
    iterations: 512, fold: 1, celtic: 0, aa: temporal ? 3 : 2, hue: 0, guided, temporal,
    referenceKey: 1, position: { x: 0n, y: 0n, bits: 128 } };
}

test('WASM cache preparation completes or reports optional cache unavailability without moving the camera', async () => {
  for (const [name, width, height, maximum] of [['ready', 320, 200, 16384], ['oom', 320, 200, 16384],
    ['no-layout', 3024, 1964, 8192]] as const) {
    const fake = canvas(width, height, maximum), renderer = new WasmRenderer(fake.canvas);
    try {
      await renderer.computeReferenceDirect(request);
      renderer.renderInput[4] = 2;
      renderer.renderInput[16] = 1;
      renderer.renderInput[40] = 30; renderer.exports.render_camera_command(3);
      renderer.exports.render_camera_snapshot();
      const logScale = renderer.renderInput[51], frame = renderer.stats.frame;
      if (name === 'oom') fake.failAllocation(fake.gl.OUT_OF_MEMORY);
      let ready = false;
      for (let index = 0; index < 100 && !ready; index++) ready = renderer.prepareCamera(index * 16);
      assert.equal(ready, true, `${name}: the scheduler must not wait indefinitely`);
      assert.equal(renderer.stats.ringActive, name === 'ready');
      renderer.exports.render_camera_snapshot();
      assert.equal(renderer.renderInput[51], logScale);
      assert.equal(renderer.stats.frame, frame, 'preparation is not an animation frame');
    } finally { renderer.dispose(); }
    assert.equal(fake.live.size, 0);
  }
});

test('preparation waits for a GPU fence without extra draws and can cancel an unfinished fence', async () => {
  const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
  try {
    await renderer.computeReferenceDirect(request); renderer.renderInput[4] = 2; renderer.renderInput[16] = 1;
    renderer.renderInput[40] = 30; renderer.exports.render_camera_command(3);
    fake.syncResult('TIMEOUT_EXPIRED');
    for (let index = 0; index < 100 && !renderer.stats.ringActive; index++) assert.equal(renderer.prepareCamera(index * 16), false);
    assert.equal(renderer.stats.ringActive, true);
    const draws = fake.draws, waits = fake.waitCalls;
    for (let index = 0; index < 3; index++) assert.equal(renderer.prepareCamera(2000 + index * 16), false);
    assert.equal(fake.draws, draws); assert.equal(fake.waitCalls - waits, 3);
    fake.syncResult('CONDITION_SATISFIED'); assert.equal(renderer.prepareCamera(2100), true);
    assert.equal(fake.draws, draws);
    assert.equal(renderer.prepareCamera(2200), false);
    const liveWithFence = fake.live.size;
    renderer.cancelPreparation(); assert.equal(fake.live.size, liveWithFence - 1);
    assert.equal(renderer.prepareCamera(2300), false);
  } finally { renderer.dispose(); }
  assert.equal(fake.live.size, 0, 'dispose releases an unfinished fence');
});

test('the WebGL bridge selects contiguous addressing only for an unsplit WASM atlas', async () => {
  for (const [width, height, maximum, contiguous] of [[1920, 1080, 16384, true], [3840, 2160, 16384, false], [1280, 720, 4096, false]] as const) {
    const fake = canvas(width, height, maximum), renderer = new WasmRenderer(fake.canvas);
    try {
      await renderer.computeReferenceDirect(request);
      for (let frame = 0; frame < 60 && !renderer.stats.ringActive; frame++) renderer.render(view(true));
      assert.equal(renderer.stats.ringActive, true);
      const source = fake.drawSources.at(-1)!;
      assert.ok(source.includes('uniform highp sampler2D ringMap'), 'the actual draw must use ring assembly');
      assert.equal(source.includes('#define SHIP_RING_CONTIGUOUS'), contiguous, `${width}x${height} with MAX_TEXTURE_SIZE=${maximum}`);
    } finally { renderer.dispose(); }
    assert.equal(fake.live.size, 0);
  }
});

test('WebGL uploads use permanent WASM views and reuse equal-size texture storage', async () => {
  const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
  const buffer = renderer.views.buffer, floats = renderer.views.floats, input = renderer.renderInput, stats = renderer.renderStats;
  try {
    const reference = await renderer.computeReferenceDirect(request);
    assert.equal(buffer.byteLength, 256 * 1024 * 1024);
    assert.equal(fake.uploads.length, 5);
    for (const [kind, upload] of fake.uploads.entries()) {
      assert.equal(upload.source, floats);
      if (kind !== 1) assert.equal(upload.offset, renderer.exports.result_ptr(kind) / 4);
      assert.equal(upload.width * upload.height, kind === 1 ? 1 : reference.capacity);
    }
    const bytes = reference.capacity * 16 * 4 + 16;
    assert.equal(renderer.totalReferenceUploadBytes, bytes);
    assert.equal(renderer.intermediateReferenceCopyBytes, 0);
    renderer.refreshReferenceFromCore();
    assert.equal(renderer.totalReferenceUploadBytes, bytes * 2);
    assert.ok(fake.uploads.slice(5).every(upload => upload.sub && upload.source === floats));
    renderer.render(view()); renderer.render(view(true)); renderer.render(view(false, true));
    renderer.refreshViews(); renderer.views.refresh();
    assert.equal(renderer.views.buffer, buffer); assert.equal(renderer.views.floats, floats);
    assert.equal(renderer.renderInput, input); assert.equal(renderer.renderStats, stats);
  } finally { renderer.dispose(); }
  assert.equal(fake.live.size, 0);
  renderer.dispose();
  assert.throws(() => renderer.render(view()), /освобождён/);
});

test('optional ring allocation failure falls back; mandatory temporal or unexpected GL errors propagate', async () => {
  for (const failure of ['oom', 'incomplete', 'temporal', 'unexpected']) {
    const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
    try {
      await renderer.computeReferenceDirect(request);
      if (failure === 'incomplete') fake.incomplete();
      else fake.failAllocation(failure === 'unexpected' ? fake.gl.INVALID_OPERATION : fake.gl.OUT_OF_MEMORY);
      if (failure === 'temporal') assert.throws(() => renderer.render(view(false, true)), /GPU памяти/);
      else if (failure === 'unexpected') assert.throws(() => renderer.render(view(true)), /WebGL allocation/);
      else {
        renderer.render(view(true));
        assert.equal(renderer.stats.ringActive, false);
        assert.equal(fake.draws, 1);
      }
    } finally { renderer.dispose(); }
    assert.equal(fake.live.size, 0, `${failure} resource cleanup`);
  }
});

test('context loss rejects draws and disposal does not delete restored-context resources', async () => {
  const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
  await renderer.computeReferenceDirect(request);
  fake.canvas.dispatchEvent(new Event('webglcontextlost'));
  assert.throws(() => renderer.render(view()), /контекст потерян/);
  fake.live.clear(); // Browser invalidates the old handles on restoration.
  const restoredResource = fake.gl.createTexture(), deletedBefore = fake.deleted.length;
  renderer.dispose();
  assert.equal(fake.deleted.length, deletedBefore);
  assert.ok(fake.live.has(restoredResource));
});

test('cancelled reference calculation leaves the bridge reusable and memory unchanged', async () => {
  const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
  const buffer = renderer.views.buffer;
  try {
    let cancelled = false;
    await assert.rejects(renderer.computeReferenceDirect({ ...request, bits: 4096, iterations: 65536 },
      () => cancelled, async () => { cancelled = true; }), (error: Error) => error.name === 'AbortError');
    assert.equal(fake.uploads.length, 0);
    const reference = await renderer.computeReferenceDirect(request);
    assert.equal(reference.length, 513);
    assert.equal(renderer.views.buffer, buffer);
    assert.equal(fake.uploads.length, 5);
  } finally { renderer.dispose(); }
  assert.equal(fake.live.size, 0);
});

test('disposing during an asynchronous calculation prevents later upload or runtime restart', async () => {
  const fake = canvas(), renderer = new WasmRenderer(fake.canvas);
  await assert.rejects(renderer.computeReferenceDirect({ ...request, bits: 4096, iterations: 65536 },
    () => false, async () => { renderer.dispose(); }), /освобождён/);
  assert.equal(fake.uploads.length, 0);
  assert.equal(fake.live.size, 0);
});
