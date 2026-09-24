import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const binary = await readFile(new URL('../public/wasm/core-simd.wasm', import.meta.url));
const PIXELS = 3840 * 2160;
type Core = {
  memory: WebAssembly.Memory;
  render_input_ptr(): number; render_stats_ptr(): number; render_text_ptr(): number;
  render_camera_snapshot(): number; render_camera_command(op: number): number;
  render_frame(mode: number, now: number): number; render_dispose(): void;
};
type Frame = { ringSamples: number; fullScreen: boolean; assembled: boolean };
type Cadence = (frame: number, work: Frame) => number;

// Exercise the shipped WASM camera/planner. Only external GPU execution is
// replaced, so callback cadence and modeled GPU costs are not hardware results.
async function flight(cadence: Cadence, options: { zoom?: number; width?: number; height?: number; maximum?: number; jump?: boolean; observeFallback?: boolean } = {}) {
  const { zoom = 30, width = 3840, height = 2160, maximum = 16384, jump = false, observeFallback = false } = options;
  let core: Core, preparing = false;
  let work: Frame = { ringSamples: 0, fullScreen: false, assembled: false };
  let referenceUploads = 0, ringAllocations = 0;
  const { instance } = await WebAssembly.instantiate(binary, { gpu: {
    ring_target: (w: number, h: number) => {
      const atlases: Record<number, number[]> = { 1920: [10391, 2891], 3840: [10386, 10353], 5120: [12537, 12487], 7680: [18803, 18708] };
      assert.deepEqual([w, h], atlases[width], 'retain the full-density atlas');
      ringAllocations++; return 1;
    },
    temporal_targets: () => 1,
    reference_texture: () => { referenceUploads++; return 0; },
    draw: (pointer: number) => {
      const header = new Int32Array(core.memory.buffer, pointer, 8);
      const values = new Float32Array(core.memory.buffer, pointer + 32, 40);
      if (preparing) assert.equal(header[0], 1, 'preparation must only fill the cache, never draw another full frame');
      if (header[0] === 1) work.ringSamples += header[5] * header[6];
      if (header[0] === 0 || header[0] === 3) work.fullScreen = true;
      if (header[0] === 2) {
        work.assembled = true;
        assert.equal(values[9], 2, 'preserve requested AA');
        assert.deepEqual(Array.from(header.slice(5, 7)), [width, height]);
      }
      if (header[0] <= 3) assert.equal(values[12], 16384, 'preserve the iteration budget');
      return 0;
    },
  } });
  core = instance.exports as unknown as Core;
  const memory = core.memory.buffer;
  const input = new Float64Array(memory, core.render_input_ptr(), 64);
  const stats = new Float64Array(memory, core.render_stats_ptr(), 16);
  input[0] = width; input[1] = height; input[2] = maximum;
  input[3] = 16384; input[4] = 2; input[5] = 1;
  input[16] = 1; input[17] = 1; input[18] = 1;
  // Stable, previously uploaded reference; pixels are irrelevant to scheduling.
  input[19] = 1; input[20] = 16385; input[21] = 17408; input[23] = 1; input[25] = 16384;
  input[40] = width / height; assert.equal(core.render_camera_command(0), 0);
  input[40] = zoom; assert.equal(core.render_camera_command(3), 0);
  assert.equal(core.render_camera_command(5), 0);
  assert.equal(core.render_frame(1, 1000), 0); // The visible static frame precedes preparation.

  function cameraSnapshot() {
    assert.equal(core.render_camera_snapshot(), 0);
    return { fixed: new TextDecoder().decode(new Uint8Array(memory, core.render_text_ptr(), input[48] + input[49])),
      xLength: input[48], bits: input[50], logScale: input[51] };
  }
  let preparationSamples = 0, preparationPasses = 0;
  function prepare() {
    const camera = cameraSnapshot(), frames = stats[1], fps = stats[15], settling = stats[8], playing = stats[12];
    const inputs = input.slice(0, 48);
    preparing = true;
    try {
      for (let frame = 0; frame < 120; frame++) {
        work = { ringSamples: 0, fullScreen: false, assembled: false };
        assert.equal(core.render_frame(2, 20_000 + frame * 100), 0);
        preparationSamples += work.ringSamples; preparationPasses++;
        assert.ok(work.ringSamples <= width * height / 2, 'preparation work must stay bounded');
        assert.equal(stats[1], frames, 'preparation is not a displayed frame');
        assert.equal(stats[15], fps, 'preparation must not enter the FPS window');
        assert.equal(stats[8], settling, 'preparation must not advance temporal AA');
        assert.equal(stats[12], playing, 'preparation must not change play state');
        if (stats[6]) break;
      }
      assert.equal(stats[6], 1, 'stationary cache preparation must finish in at most 120 passes');
      assert.deepEqual(cameraSnapshot(), camera, 'preparation must not advance or round the actual camera');
      assert.deepEqual(input.slice(0, 48), inputs, 'the virtual preparation scale must not leak into camera input');
    } finally { preparing = false; }
  }

  let elapsed = 0, firstRingMs: number | null = null, ringFrames = 0, doubleWorkFrames = 0, fallbackFrames = 0;
  let firstFallbackMs: number | null = null;
  let jumped = false, zoomBase = zoom, peakFrameMs = 0, finalFrameMs = 0;
  try {
    prepare();
    input[40] = 1; input[41] = 150; assert.equal(core.render_camera_command(4), 0);
    for (let frame = 0; elapsed <= 20_000; frame++) {
      if (jump && !jumped && elapsed >= 2000) {
        assert.notEqual(firstRingMs, null, 'the cache must be active before testing a large jump');
        input[40] = 0; assert.equal(core.render_camera_command(4), 0);
        input[40] = 25; assert.equal(core.render_camera_command(3), 0);
        prepare();
        input[40] = 1; assert.equal(core.render_camera_command(4), 0);
        zoomBase = 25 - elapsed * 0.00055; jumped = true;
      }
      work = { ringSamples: 0, fullScreen: false, assembled: false };
      assert.equal(core.render_frame(1, 1000 + elapsed), 0);
      assert.ok(Math.abs(stats[9] - (zoomBase + elapsed * 0.00055)) < 1e-9, 'flight speed must stay unchanged');
      if (work.assembled) { firstRingMs ??= elapsed; ringFrames++; }
      if (work.fullScreen && work.ringSamples > 0) doubleWorkFrames++;
      if (stats[0] !== 0 && !work.assembled) { fallbackFrames++; firstFallbackMs ??= elapsed; }
      if (!observeFallback && stats[0] !== 0) assert.equal(work.assembled, true, 'the first float/FE flight frame must already use the prepared cache');
      assert.ok(work.fullScreen || work.assembled, 'every flight frame must produce a complete image');
      const frameMs = cadence(frame, work);
      assert.ok(frameMs > 0 && frameMs <= 100, 'the test must not invoke the camera time clamp');
      peakFrameMs = Math.max(peakFrameMs, frameMs); finalFrameMs = frameMs;
      elapsed += frameMs;
    }
    if (!observeFallback) assert.equal(doubleWorkFrames, 0, 'flight must never combine cache preparation and a fullscreen fallback');
    assert.equal(referenceUploads, 0); assert.equal(stats[7], 0);
    assert.equal(ringAllocations, 1, 'reuse the atlas during preparation and flight');
    assert.equal(core.memory.buffer, memory, 'keep fixed WASM memory');
    assert.equal(jumped, jump);
    return { firstRingMs, ringFrames, fallbackFrames, firstFallbackMs, doubleWorkFrames, preparationSamples, preparationPasses, peakFrameMs, finalFrameMs };
  } finally { core.render_dispose(); }
}

for (const fps of [10, 15, 20, 25, 30, 60]) {
  test(`preparing a 4K cache before motion avoids startup fallback at ${fps} FPS`, async () => {
    const result = await flight(() => 1000 / fps);
    assert.equal(result.firstRingMs, 0);
    assert.ok(result.ringFrames > 100);
  });
}
for (const zoom of [0, 3, 115]) {
  test(`a prepared route at zoom ${zoom} uses rings from its first float/FE frame`, async () => {
    const result = await flight(() => 40, { zoom, width: 1920, height: 1080 });
    assert.ok(result.ringFrames > 100);
    if (zoom > 0) assert.equal(result.firstRingMs, 0);
  });
}
test('a prepared cache remains usable when callback cadence varies between 20 and 60 FPS', async () => {
  const frameTimes = [50, 40, 1000 / 30, 1000 / 60, 50, 40, 50, 40];
  const result = await flight(frame => frameTimes[frame % frameTimes.length]);
  assert.equal(result.firstRingMs, 0);
});
test('preparing a new ring window after a large zoom jump reuses the atlas and reference', async () => {
  const result = await flight(() => 1000 / 60, { jump: true });
  assert.ok(result.ringFrames > 100);
});
for (const [fullscreenMs, polarMs] of [[25, 30], [10, 60]]) {
  test(`preparation avoids GPU feedback starvation with fullscreen=${fullscreenMs}ms and polar=${polarMs}ms/screen`, async t => {
    // Deliberately model both the original failure and the divergent increased-
    // budget counterexample. These costs are not measurements of any GPU.
    const result = await flight((_frame, work) => Math.max(1000 / 60,
      (work.fullScreen ? fullscreenMs : 1.5) + polarMs * work.ringSamples / PIXELS));
    t.diagnostic(JSON.stringify({ ...result, modeledPreparationGpuMs: result.preparationSamples / PIXELS * polarMs }));
    assert.equal(result.firstRingMs, 0);
    assert.ok(result.preparationSamples > 0, 'preparation cost must be accounted for explicitly');
    assert.ok(result.ringFrames > 100);
  });
}

for (const stallMs of [80, 100]) for (const fps of [25, 30]) {
  test(`a warmed cache survives a single ${stallMs}ms callback followed by ${fps} FPS`, async t => {
    const result = await flight(frame => frame === 60 ? stallMs : frame > 60 ? 1000 / fps : 1000 / 60, { observeFallback: true });
    t.diagnostic(JSON.stringify(result));
    assert.equal(result.fallbackFrames, 0, 'one dropped frame must not turn a usable cache into a permanent cold fill');
  });
}

test('a warmed cache recovers from a GPU cost spike without cold-fill feedback', async t => {
  const result = await flight((frame, work) => frame === 60 ? 100 : Math.max(1000 / 60,
    (work.fullScreen ? 10 : 1.5) + 60 * work.ringSamples / PIXELS), { observeFallback: true });
  t.diagnostic(JSON.stringify(result));
  assert.equal(result.fallbackFrames, 0);
});

for (const [width, height, maximum] of [[5120, 1440, 16384], [7680, 2160, 32768]]) {
  test(`a ${width}x${height} ultrawide cache survives the maximum normal camera step`, async t => {
    const result = await flight(frame => frame === 60 ? 100 : frame > 60 ? 40 : 1000 / 60,
      { width, height, maximum, observeFallback: true });
    t.diagnostic(JSON.stringify(result));
    assert.equal(result.fallbackFrames, 0, 'the warm limit must account for aspect ratio and layout density');
  });
}

test('unfillable tiny or narrow prewarm budgets report optional unavailability immediately', async () => {
  for (const [width, height] of [[1, 1], [8, 8], [1, 200]]) {
    let core: Core, allocations = 0;
    const draws: number[] = [];
    const { instance } = await WebAssembly.instantiate(binary, { gpu: {
      ring_target: () => { allocations++; return 1; }, temporal_targets: () => 1, reference_texture: () => 0,
      draw: (pointer: number) => { draws.push(new Int32Array(core.memory.buffer, pointer, 8)[0]); return 0; },
    } });
    core = instance.exports as unknown as Core;
    const input = new Float64Array(core.memory.buffer, core.render_input_ptr(), 64);
    input[0] = width; input[1] = height; input[2] = 16384;
    input[3] = 16384; input[4] = 2; input[5] = 1; input[16] = 1;
    input[19] = 1; input[20] = 16385; input[21] = 17408; input[23] = 1; input[25] = 16384;
    input[40] = width / height; assert.equal(core.render_camera_command(0), 0);
    input[40] = 30; assert.equal(core.render_camera_command(3), 0);
    assert.equal(core.render_camera_command(5), 0);
    try {
      assert.equal(core.render_frame(2, 0), 1, `${width}x${height}: a whole polar row cannot fit the budget`);
      assert.equal(allocations, 0, 'do not allocate an atlas that preparation cannot fill');
      assert.deepEqual(draws, []);
      input[40] = 1; input[41] = 100; assert.equal(core.render_camera_command(4), 0);
      assert.equal(core.render_frame(1, 1000), 0);
      assert.deepEqual(draws, [0], 'full-quality rendering remains available');
    } finally { core.render_dispose(); }
  }
});
