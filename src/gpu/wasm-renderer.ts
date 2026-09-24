import type { ReferenceRequest, ReferenceResult, RenderView } from '../types';
import type { GpuTiming, RendererStats } from './renderer';
import { contiguousRingFragment, ringFragment, shipFragment, vertexShader } from './shaders';
import { temporalCompositeFragment, temporalFragment } from './temporal';
import { instantiateRenderWasm, RenderMemoryViews } from './render-wasm';
import type { RenderWasmExports } from './render-wasm';
import { GpuFrameCompletion } from './frame-completion.ts';

interface Program { handle: WebGLProgram; locations: (WebGLUniformLocation | null)[] }
interface Target { texture: WebGLTexture; framebuffer: WebGLFramebuffer; width: number; height: number }
interface TimerExtension { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }
export interface DirectReferenceResult {
  id: number; length: number; capacity: number; iterations: number; bits: number;
  fold: number; celtic: number; computeMs: number; backend: 'wasm'; memoryBytes: number;
}
const UNIFORMS = ['center', 'scale', 'fold', 'celtic', 'hue', 'aspect', 'res', 'aa', 'jitter',
  'iterations', 'referenceLength', 'offsetX', 'offsetY', 'viewScale', 'referenceSize', 'ringBlock',
  'referenceOrbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds', 'ringMap', 'viewHeight', 'bandCount',
  'bandLayout[0]', 'bandPlace[0]', 'currentMap', 'historyMap', 'texelSize', 'historyOffset',
  'historyScale', 'historyWeight', 'moving', 'frameMap'] as const;
const PATHS = ['direct', 'float', 'fe'] as const;
const REFERENCE_ARRAYS = ['orbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
const NEVER_CANCEL = () => false;
const NO_YIELD = async () => {};

/** Browser API bridge only: Rust owns camera arithmetic, temporal state, ring
 * scheduling and the draw loop. Draw descriptors and uniforms stay in WASM. */
export class WasmRenderer {
  readonly exports: RenderWasmExports;
  readonly core: RenderWasmExports;
  readonly views: RenderMemoryViews;
  readonly stats: RendererStats = { path: 'direct', frame: 0, ringsDrawn: 0, ringSamples: 0,
    ringResets: 0, ringFrames: 0, ringActive: false, referenceUploads: 0 };
  private readonly gl: WebGL2RenderingContext;
  private readonly vao: WebGLVertexArrayObject;
  private readonly maxTextureSize: number;
  private readonly programs: (Program | undefined)[] = new Array(16);
  private readonly references: (WebGLTexture | null)[] = [null, null, null, null, null];
  private readonly referenceWidths = new Uint32Array(5);
  private readonly referenceHeights = new Uint32Array(5);
  private readonly boundTextures: (WebGLTexture | null | undefined)[] = new Array(5);
  private activeTexture = -1;
  private currentProgram: WebGLProgram | null = null;
  private currentFramebuffer: WebGLFramebuffer | null | undefined;
  private ringTarget: Target | null = null;
  private readonly temporalTargets: (Target | null)[] = [null, null, null];
  private readonly inputView: Float64Array;
  private readonly statsView: Float64Array;
  private readonly textView: Uint8Array;
  private readonly encoder = new TextEncoder();
  private readonly zeroReference = new Float32Array(4);
  private previousX: bigint | undefined;
  private previousY: bigint | undefined;
  private previousBits = 0;
  private positionXLength = 0;
  private positionYLength = 0;
  private busy = false;
  private disposed = false;
  private contextInvalid = false;
  private bridgeError: Error | null = null;
  private referenceUploadBytes = 0;
  private readonly timer: TimerExtension | null;
  private readonly queries: (WebGLQuery | null)[] = new Array(8).fill(null);
  private readonly queryFrames = new Float64Array(8);
  private readonly queryPaths = new Uint8Array(8);
  private readonly queryRings = new Uint8Array(8);
  private queryHead = 0;
  private queryCount = 0;
  private readonly timingPool: GpuTiming[] = Array.from({ length: 64 }, () => ({ frame: 0, milliseconds: 0, path: 'direct', ring: false }));
  private readonly timings: GpuTiming[] = [];
  private timingNext = 0;
  private lastGpuTime: number | null = null;
  private preparationSync: WebGLSync | null = null;
  private readonly frameCompletion: GpuFrameCompletion | null;
  private readonly onContextLost = () => {
    this.contextInvalid = true;
    this.preparationSync = null;
    this.frameCompletion?.dispose(true);
  };

  constructor(readonly canvas: HTMLCanvasElement | OffscreenCanvas, trackCompletedFrames = false) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('WebGL 2 недоступен.');
    this.gl = gl;
    this.frameCompletion = trackCompletedFrames ? new GpuFrameCompletion(gl) : null;
    const precision = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    if (!precision || precision.precision < 23) throw new Error('GPU не поддерживает требуемую highp float точность.');
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('Не удалось создать WebGL vertex array.');
    this.vao = vao;
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExtension | null;
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.disable(gl.BLEND); gl.disable(gl.DITHER);
    try {
      this.exports = instantiateRenderWasm({
        ring_target: (width, height) => this.ensureRingTarget(width, height),
        temporal_targets: (width, height) => this.ensureTemporalTargets(width, height),
        draw: pointer => this.drawPass(pointer),
        reference_texture: (kind, pointer, length, width, height) => this.uploadReferenceTexture(kind, pointer, length, width, height),
      });
    } catch (error) { gl.deleteVertexArray(vao); throw error; }
    this.core = this.exports;
    this.exports.render_mark_dirty();
    this.views = new RenderMemoryViews(this.exports.memory);
    const buffer = this.views.buffer;
    this.inputView = new Float64Array(buffer, this.exports.render_input_ptr(), 64);
    this.statsView = new Float64Array(buffer, this.exports.render_stats_ptr(), 16);
    this.textView = new Uint8Array(buffer, this.exports.render_text_ptr(), this.exports.render_text_capacity());
    this.inputView[2] = this.maxTextureSize;
    canvas.addEventListener('webglcontextlost', this.onContextLost);
    try {
      if (this.timer) for (let index = 0; index < 8; index++) this.queries[index] = gl.createQuery();
      this.getProgram(0);
      this.checkError('инициализация');
    } catch (error) { this.dispose(); throw error; }
  }

  get renderInput(): Float64Array { return this.inputView; }
  get renderStats(): Float64Array { return this.statsView; }
  get memoryBytes(): number { return this.views.buffer.byteLength; }
  /** Successful reference transfers to WebGL; unused real-orbit texture is 16 bytes. */
  get totalReferenceUploadBytes(): number { return this.referenceUploadBytes; }
  /** Both production and compatibility uploads pass their source arrays directly. */
  get intermediateReferenceCopyBytes(): 0 { return 0; }
  get gpuTimerSupported(): boolean { return this.timer !== null; }
  get gpuTimeMs(): number | null { this.pollGpuTimers(); return this.lastGpuTime; }
  get gpuTimings(): readonly GpuTiming[] { this.pollGpuTimers(); return this.timings; }
  get settling(): boolean { return !this.disposed && !this.contextInvalid && this.statsView[8] !== 0; }

  sampleCompletedFrames(now = performance.now()) {
    return this.frameCompletion?.sample(now) ?? { fps: null, pendingFrames: 0, completionAgeMs: null };
  }
  resetCompletedFrames(now = performance.now()): void { this.frameCompletion?.reset(now); }

  /** Compatibility boundary only: all views remain valid for the renderer's lifetime. */
  refreshViews(): void {}

  resetTemporal(): void { if (!this.disposed) this.exports.render_reset_temporal(); }

  render(view: RenderView): void {
    this.assertAlive();
    const input = this.inputView;
    input[0] = this.canvas.width; input[1] = this.canvas.height;
    input[3] = view.iterations; input[4] = view.aa; input[5] = view.fold; input[6] = view.celtic; input[7] = view.hue;
    input[8] = view.logScale; input[9] = view.scale; input[10] = view.center[0]; input[11] = view.center[1];
    input[12] = view.offsetX[0]; input[13] = view.offsetX[1]; input[14] = view.offsetY[0]; input[15] = view.offsetY[1];
    input[16] = +view.guided; input[17] = +(view.temporal ?? false); input[18] = view.referenceKey;
    const position = view.position;
    input[26] = position ? 1 : 0;
    if (position) {
      if (position.x !== this.previousX || position.y !== this.previousY || position.bits !== this.previousBits) {
        const x = position.x.toString(), y = position.y.toString();
        this.writeText(x, y, this.textView);
        this.positionXLength = x.length; this.positionYLength = y.length;
        this.previousX = position.x; this.previousY = position.y; this.previousBits = position.bits;
      }
      this.checkRuntime(this.exports.render_set_position(this.positionXLength, this.positionYLength, position.bits, view.logScale), 'точные координаты');
    }
    this.frame(0, performance.now());
  }

  /** Production worker uses Rust-owned camera/settings, with the same GPU queries. */
  renderCamera(now: number): void {
    this.assertAlive();
    this.inputView[0] = this.canvas.width; this.inputView[1] = this.canvas.height;
    this.previousX = undefined;
    this.frame(1, now);
  }

  /** Fill the route cache while Rust keeps the visible camera stationary.
   * A positive status means the optional cache is unavailable: use full rendering. */
  prepareCamera(now: number): boolean {
    this.assertAlive();
    const gl = this.gl;
    if (this.preparationSync) {
      const status = gl.clientWaitSync(this.preparationSync, 0, 0);
      if (status === gl.TIMEOUT_EXPIRED) return false;
      this.cancelPreparation();
      if (status === gl.WAIT_FAILED) throw new Error('Не удалось дождаться подготовки GPU-кэша.');
      return true;
    }
    this.inputView[0] = this.canvas.width; this.inputView[1] = this.canvas.height;
    const status = this.frame(2, now);
    if (status === 1) return true;
    if (this.statsView[6] === 0) return false;
    this.preparationSync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!this.preparationSync) throw new Error('Не удалось создать GPU fence подготовки кэша.');
    gl.flush();
    return false;
  }

  cancelPreparation(): void {
    if (this.preparationSync && !this.contextInvalid && !this.disposed) this.gl.deleteSync(this.preparationSync);
    this.preparationSync = null;
  }

  private frame(mode: number, now: number): number {
    this.pollGpuTimers();
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    this.bridgeError = null;
    const previousFrame = this.statsView[1];
    let rendered = false;
    let queryIndex = -1;
    if (mode !== 2 && this.timer && this.queryCount < 8) {
      const slot = (this.queryHead + this.queryCount) % 8;
      if (this.queries[slot]) { queryIndex = slot; gl.beginQuery(this.timer.TIME_ELAPSED_EXT, this.queries[slot]!); }
    }
    try {
      const status = this.exports.render_frame(mode, now);
      this.checkRuntime(status, mode === 2 ? 'подготовка кэша' : 'кадр');
      this.updateStats();
      rendered = mode !== 2 && this.statsView[1] !== previousFrame;
      return status;
    } finally {
      if (queryIndex >= 0 && this.timer) {
        gl.endQuery(this.timer.TIME_ELAPSED_EXT);
        if (this.statsView[1] !== previousFrame) {
          this.queryFrames[queryIndex] = this.statsView[1]; this.queryPaths[queryIndex] = this.statsView[0];
          this.queryRings[queryIndex] = this.statsView[6]; this.queryCount++;
        }
        gl.flush();
      }
      if (mode === 2) gl.flush();
      if (rendered) this.frameCompletion?.recordFrame(performance.now());
    }
  }

  setReference(reference: ReferenceResult): void {
    this.assertAlive();
    this.cancelPreparation();
    const size = reference.capacity * 4;
    if (!Number.isSafeInteger(reference.capacity) || reference.capacity < 1024 || reference.capacity % 1024
        || !Number.isSafeInteger(reference.length) || reference.length < 2 || reference.length > reference.capacity
        || reference.capacity < reference.iterations + 1) {
      throw new Error('Некорректные размеры внешней опорной орбиты.');
    }
    if (this.maxTextureSize < 1024 || reference.capacity / 1024 > this.maxTextureSize) {
      throw new Error('Опорная орбита превышает максимальный размер текстуры GPU.');
    }
    for (let kind = 0; kind < 5; kind++) if ((kind !== 1 || reference.celtic !== 0) && reference[REFERENCE_ARRAYS[kind]].length !== size) {
      throw new Error(`Некорректная длина ${REFERENCE_ARRAYS[kind]}.`);
    }
    this.bridgeError = null;
    for (let kind = 0; kind < 5; kind++) {
      const unused = kind === 1 && reference.celtic === 0;
      this.uploadTextureData(kind, unused ? this.zeroReference : reference[REFERENCE_ARRAYS[kind]],
        unused ? 1 : 1024, unused ? 1 : reference.capacity / 1024, 0);
    }
    const input = this.inputView;
    input[19]++; input[20] = reference.length; input[21] = reference.capacity; input[22] = 1;
    input[23] = reference.fold; input[24] = reference.celtic; input[25] = reference.iterations;
    this.exports.render_external_reference();
    this.updateStats();
  }

  /** Calls Rust's upload dispatch; imports upload directly from WASM result_ptr. */
  refreshReferenceFromCore(): void {
    this.assertAlive(); this.bridgeError = null;
    this.cancelPreparation();
    this.checkRuntime(this.exports.render_reference_ready(), 'загрузка опорной орбиты');
    this.updateStats();
  }

  async computeReferenceDirect(request: ReferenceRequest, cancelled = NEVER_CANCEL, yieldControl = NO_YIELD): Promise<DirectReferenceResult> {
    this.assertAlive();
    if (this.busy) throw new Error('Render WASM уже вычисляет опорную орбиту.');
    this.busy = true;
    const started = performance.now();
    try {
      if (cancelled()) throw this.abortError();
      this.previousX = undefined;
      this.writeText(request.x, request.y, this.textView);
      this.checkRuntime(this.exports.render_set_camera(request.x.length, request.y.length, request.bits, this.inputView[8], 0), 'камера опорной орбиты');
      this.inputView[3] = request.iterations; this.inputView[5] = request.fold; this.inputView[6] = request.celtic;
      this.checkRuntime(this.exports.render_begin_reference(), 'начало опорной орбиты');
      let status = 0;
      while (status === 0) {
        this.assertAlive();
        if (cancelled()) throw this.abortError();
        status = this.exports.step(128);
        if (status < 0) throw new Error(`WASM step: ${this.exports.last_error()}`);
        if (status === 0) await yieldControl();
      }
      this.assertAlive();
      if (cancelled()) throw this.abortError();
      this.refreshReferenceFromCore();
      return { id: request.id, length: this.exports.orbit_length(), capacity: this.exports.capacity(),
        iterations: request.iterations, bits: request.bits, fold: this.inputView[23], celtic: this.inputView[24],
        computeMs: performance.now() - started, backend: 'wasm', memoryBytes: this.memoryBytes };
    } catch (error) {
      if (!this.disposed) this.exports.render_cancel_reference();
      throw error;
    } finally { this.busy = false; }
  }

  readPixels(): Uint8Array {
    this.assertAlive();
    const pixels = new Uint8Array(this.canvas.width * this.canvas.height * 4);
    this.bindFramebuffer(null);
    this.gl.readPixels(0, 0, this.canvas.width, this.canvas.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
    this.checkError('чтение пикселей');
    return pixels;
  }

  pollGpuTimers(): void {
    if (!this.timer || this.disposed || this.contextInvalid) return;
    const gl = this.gl;
    if (gl.getParameter(this.timer.GPU_DISJOINT_EXT)) { this.clearGpuTimings(); return; }
    while (this.queryCount > 0) {
      const slot = this.queryHead, query = this.queries[slot]!;
      if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) break;
      const milliseconds = Number(gl.getQueryParameter(query, gl.QUERY_RESULT)) / 1e6;
      if (Number.isFinite(milliseconds) && milliseconds >= 0) {
        const record = this.timingPool[this.timingNext];
        record.frame = this.queryFrames[slot]; record.milliseconds = milliseconds;
        record.path = PATHS[this.queryPaths[slot]]; record.ring = this.queryRings[slot] !== 0;
        if (this.timings.length < 64) this.timings.push(record);
        this.timingNext = (this.timingNext + 1) % 64;
        this.lastGpuTime = milliseconds;
      }
      this.queryHead = (this.queryHead + 1) % 8; this.queryCount--;
    }
  }

  clearGpuTimings(): void {
    this.queryHead = 0; this.queryCount = 0; this.timings.length = 0; this.timingNext = 0; this.lastGpuTime = null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.cancelPreparation();
    this.frameCompletion?.dispose(this.contextInvalid);
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost);
    this.exports.render_dispose();
    if (!this.contextInvalid && !this.gl.isContextLost()) {
      for (let i = 0; i < this.programs.length; i++) if (this.programs[i]) this.gl.deleteProgram(this.programs[i]!.handle);
      for (let i = 0; i < 5; i++) if (this.references[i]) this.gl.deleteTexture(this.references[i]);
      for (let i = 0; i < 8; i++) if (this.queries[i]) this.gl.deleteQuery(this.queries[i]);
      this.deleteTarget(this.ringTarget);
      for (let i = 0; i < 3; i++) this.deleteTarget(this.temporalTargets[i]);
      this.gl.deleteVertexArray(this.vao);
    }
    this.clearGpuTimings(); this.disposed = true;
  }

  private writeText(x: string, y: string, target: Uint8Array): void {
    if (x.length + y.length > target.length || !/^[+-]?\d+$/.test(x) || !/^[+-]?\d+$/.test(y)) {
      throw new Error('Некорректные или слишком длинные fixed-point координаты.');
    }
    // Coordinates are ASCII integers; no temporary encoded byte arrays.
    this.encoder.encodeInto(x, target);
    for (let i = 0; i < y.length; i++) target[x.length + i] = y.charCodeAt(i);
  }

  private abortError(): Error { const error = new Error('Расчёт отменён'); error.name = 'AbortError'; return error; }
  private updateStats(): void {
    const s = this.statsView, target = this.stats;
    target.path = PATHS[s[0]]; target.frame = s[1]; target.ringsDrawn = s[2]; target.ringSamples = s[3];
    target.ringResets = s[4]; target.ringFrames = s[5]; target.ringActive = s[6] !== 0; target.referenceUploads = s[7];
  }
  private assertAlive(): void {
    if (this.disposed) throw new Error('WasmRenderer уже освобождён.');
    if (this.contextInvalid) throw new Error('WebGL-контекст потерян. Создайте новый WasmRenderer после восстановления.');
  }
  private checkRuntime(status: number, action: string): void {
    if (this.bridgeError) throw this.bridgeError;
    if (status < 0) throw new Error(`Render WASM (${action}): ${this.exports.render_error()} (${status}).`);
  }
  private checkError(action: string): void {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR) throw new Error(`WebGL (${action}): 0x${error.toString(16)}.`);
  }
  private fail(error: unknown): -1 { this.bridgeError = error instanceof Error ? error : new Error(String(error)); return -1; }

  private bindTexture(unit: number, texture: WebGLTexture | null, forUpdate = false): void {
    if (!forUpdate && this.boundTextures[unit] === texture) return;
    const gl = this.gl;
    if (this.activeTexture !== unit) { gl.activeTexture(gl.TEXTURE0 + unit); this.activeTexture = unit; }
    if (this.boundTextures[unit] !== texture) { gl.bindTexture(gl.TEXTURE_2D, texture); this.boundTextures[unit] = texture; }
  }
  private bindFramebuffer(framebuffer: WebGLFramebuffer | null): void {
    if (this.currentFramebuffer !== framebuffer) { this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer); this.currentFramebuffer = framebuffer; }
  }
  private use(program: Program): void {
    if (this.currentProgram !== program.handle) { this.gl.useProgram(program.handle); this.currentProgram = program.handle; }
  }
  private textureParameters(linear: boolean): void {
    const gl = this.gl, filter = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private uploadReferenceTexture(kind: number, pointer: number, length: number, width: number, height: number): number {
    try {
      this.assertAlive();
      if (kind > 4 || pointer % 4 || length !== width * height * 4 || pointer + length * 4 > this.views.buffer.byteLength
          || width > this.maxTextureSize || height > this.maxTextureSize) throw new Error('Некорректный WASM reference_texture pass.');
      this.uploadTextureData(kind, this.views.floats, width, height, pointer >>> 2);
      return 0;
    } catch (error) { return this.fail(error); }
  }

  private uploadTextureData(kind: number, data: Float32Array, width: number, height: number, offset: number): void {
    const gl = this.gl;
    let texture = this.references[kind];
    if (!texture) {
      texture = gl.createTexture();
      if (!texture) throw new Error('Не удалось создать reference texture.');
      this.references[kind] = texture;
      this.bindTexture(kind, texture, true); this.textureParameters(false);
    } else this.bindTexture(kind, texture, true);
    if (this.referenceWidths[kind] === width && this.referenceHeights[kind] === height) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, data, offset);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data, offset);
      this.referenceWidths[kind] = width; this.referenceHeights[kind] = height;
    }
    this.checkError('загрузка reference texture');
    this.referenceUploadBytes += width * height * 16;
  }

  private deleteTarget(target: Target | null): void {
    if (!target) return;
    this.gl.deleteFramebuffer(target.framebuffer); this.gl.deleteTexture(target.texture);
    for (let unit = 0; unit < 5; unit++) if (this.boundTextures[unit] === target.texture) this.boundTextures[unit] = undefined;
    if (this.currentFramebuffer === target.framebuffer) this.currentFramebuffer = undefined;
  }

  private allocateTarget(width: number, height: number, linear: boolean, optional: boolean): Target | null {
    const gl = this.gl;
    if (width < 1 || height < 1 || width > this.maxTextureSize || height > this.maxTextureSize) {
      if (optional) return null;
      throw new Error('Размер framebuffer превышает ограничения GPU.');
    }
    const texture = gl.createTexture(), framebuffer = gl.createFramebuffer();
    let retained = false;
    try {
      if (!texture || !framebuffer) {
        this.checkAllocationErrors();
        if (optional) return null;
        throw new Error('Не удалось создать framebuffer.');
      }
      this.bindTexture(0, texture, true); this.textureParameters(linear);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      if (!this.checkAllocationErrors()) {
        if (optional) return null;
        throw new Error('Недостаточно GPU памяти для temporal AA.');
      }
      this.bindFramebuffer(framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      const noOom = this.checkAllocationErrors();
      if (!noOom || status !== gl.FRAMEBUFFER_COMPLETE) {
        if (optional) return null;
        throw new Error(`Temporal framebuffer неполон: 0x${status.toString(16)}.`);
      }
      retained = true;
      return { texture, framebuffer, width, height };
    } finally {
      this.bindFramebuffer(null);
      if (!retained) {
        if (framebuffer) gl.deleteFramebuffer(framebuffer);
        if (texture) gl.deleteTexture(texture);
        this.boundTextures[0] = undefined;
      }
    }
  }
  private checkAllocationErrors(): boolean {
    const gl = this.gl;
    let success = true;
    for (let error = gl.getError(); error !== gl.NO_ERROR; error = gl.getError()) {
      if (error !== gl.OUT_OF_MEMORY) throw new Error(`WebGL allocation: 0x${error.toString(16)}.`);
      success = false;
    }
    return success;
  }
  private ensureRingTarget(width: number, height: number): number {
    try {
      this.assertAlive();
      if (this.ringTarget?.width === width && this.ringTarget.height === height) return 1;
      this.deleteTarget(this.ringTarget); this.ringTarget = null;
      this.ringTarget = this.allocateTarget(width, height, false, true);
      return this.ringTarget ? 1 : 0;
    } catch (error) { return this.fail(error); }
  }
  private ensureTemporalTargets(width: number, height: number): number {
    try {
      this.assertAlive();
      if (this.temporalTargets[0]?.width === width && this.temporalTargets[0].height === height) return 1;
      for (let index = 0; index < 3; index++) { this.deleteTarget(this.temporalTargets[index]); this.temporalTargets[index] = null; }
      try {
        for (let index = 0; index < 3; index++) this.temporalTargets[index] = this.allocateTarget(width, height, true, false);
      } catch (error) {
        for (let index = 0; index < 3; index++) { this.deleteTarget(this.temporalTargets[index]); this.temporalTargets[index] = null; }
        throw error;
      }
      return 1;
    } catch (error) { return this.fail(error); }
  }

  private compile(source: string, type: number): WebGLShader {
    const gl = this.gl, shader = gl.createShader(type);
    if (!shader) throw new Error('Не удалось создать shader.');
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader); gl.deleteShader(shader);
      throw new Error(`Ошибка компиляции WebGL: ${message}`);
    }
    return shader;
  }
  private getProgram(id: number): Program {
    const cached = this.programs[id];
    if (cached) return cached;
    const gl = this.gl;
    const source = id < 12 ? shipFragment(id % 3 as 0 | 1 | 2, id % 6 >= 3, id >= 6)
      : id === 12 ? ringFragment : id === 13 ? temporalFragment : id === 15 ? contiguousRingFragment : temporalCompositeFragment;
    const vertex = this.compile(vertexShader, gl.VERTEX_SHADER);
    let fragment: WebGLShader | null = null, handle: WebGLProgram | null = null;
    try {
      fragment = this.compile(source, gl.FRAGMENT_SHADER); handle = gl.createProgram();
      if (!handle) throw new Error('Не удалось создать WebGL program.');
      gl.attachShader(handle, vertex); gl.attachShader(handle, fragment); gl.linkProgram(handle);
      if (!gl.getProgramParameter(handle, gl.LINK_STATUS)) throw new Error(`Ошибка линковки WebGL: ${gl.getProgramInfoLog(handle)}`);
      const program: Program = { handle, locations: new Array(UNIFORMS.length) };
      for (let index = 0; index < UNIFORMS.length; index++) program.locations[index] = gl.getUniformLocation(handle, UNIFORMS[index]);
      this.programs[id] = program; this.use(program);
      const l = program.locations;
      gl.uniform1i(l[16], 0); gl.uniform1i(l[17], 1); gl.uniform1i(l[18], 2); gl.uniform1i(l[19], 3); gl.uniform1i(l[20], 4);
      gl.uniform1i(l[21], 0); gl.uniform1i(l[26], 0); gl.uniform1i(l[27], 1); gl.uniform1i(l[33], 0);
      return program;
    } catch (error) { if (handle) gl.deleteProgram(handle); throw error; }
    finally { gl.deleteShader(vertex); if (fragment) gl.deleteShader(fragment); }
  }

  /** Fixed 704-byte pass ABI. This callback performs only WebGL API dispatch. */
  private drawPass(pointer: number): number {
    try {
      this.assertAlive();
      const gl = this.gl, integers = this.views.integers, floats = this.views.floats;
      const h = pointer >>> 2, u = (pointer + 32) >>> 2;
      const kind = integers[h], path = integers[h + 1], index = integers[h + 7];
      const ship = kind === 0 || kind === 1 || kind === 3;
      const programId = ship ? path + (kind === 1 ? 3 : 0) + (floats[u + 34] !== 0 ? 6 : 0)
        : kind === 2 ? (integers[h + 2] & 2 ? 15 : 12) : kind === 4 ? 13 : 14;
      const program = this.getProgram(programId), l = program.locations;
      this.use(program);
      if (kind === 1) this.bindFramebuffer(this.ringTarget!.framebuffer);
      else if (kind === 3) this.bindFramebuffer(this.temporalTargets[0]!.framebuffer);
      else if (kind === 4) this.bindFramebuffer(this.temporalTargets[index + 1]!.framebuffer);
      else this.bindFramebuffer(null);
      gl.viewport(integers[h + 3], integers[h + 4], integers[h + 5], integers[h + 6]);
      if (integers[h + 2] & 1) {
        gl.enable(gl.SCISSOR_TEST); gl.scissor(integers[h + 3], integers[h + 4], integers[h + 5], integers[h + 6]);
      } else gl.disable(gl.SCISSOR_TEST);
      if (ship) {
        this.bindTexture(0, this.references[0]); this.bindTexture(1, this.references[1]); this.bindTexture(2, this.references[2]);
        this.bindTexture(3, this.references[3]); this.bindTexture(4, this.references[4]);
        gl.uniform2fv(l[0], floats, u, 2); gl.uniform1f(l[1], floats[u + 2]); gl.uniform1f(l[2], floats[u + 3]);
        gl.uniform1f(l[3], floats[u + 4]); gl.uniform1f(l[4], floats[u + 5]); gl.uniform1f(l[5], floats[u + 6]);
        gl.uniform2fv(l[6], floats, u + 7, 2); gl.uniform1f(l[7], floats[u + 9]); gl.uniform2fv(l[8], floats, u + 10, 2);
        gl.uniform1i(l[9], floats[u + 12]); gl.uniform1i(l[10], floats[u + 13]);
        gl.uniform2fv(l[11], floats, u + 14, 2); gl.uniform2fv(l[12], floats, u + 16, 2);
        gl.uniform2fv(l[13], floats, u + 18, 2); gl.uniform2fv(l[14], floats, u + 20, 2);
        gl.uniform4fv(l[15], floats, u + 22, 4);
      } else if (kind === 2) {
        this.bindTexture(0, this.ringTarget!.texture);
        gl.uniform1f(l[5], floats[u + 6]); gl.uniform1f(l[22], floats[u + 8]); gl.uniform2fv(l[6], floats, u + 7, 2);
        gl.uniform1f(l[7], floats[u + 9]); gl.uniform1i(l[23], floats[u + 33]);
        gl.uniform4fv(l[24], floats, (pointer + 192) >>> 2, floats[u + 33] * 4);
        gl.uniform4fv(l[25], floats, (pointer + 448) >>> 2, floats[u + 33] * 4);
      } else if (kind === 4) {
        this.bindTexture(0, this.temporalTargets[0]!.texture); this.bindTexture(1, this.temporalTargets[2 - index]!.texture);
        gl.uniform2fv(l[28], floats, u + 31, 2); gl.uniform2fv(l[29], floats, u + 26, 2);
        gl.uniform1f(l[30], floats[u + 28]); gl.uniform1f(l[31], floats[u + 29]); gl.uniform1f(l[32], floats[u + 30]);
      } else if (kind === 5) this.bindTexture(0, this.temporalTargets[index + 1]!.texture);
      else throw new Error(`Неизвестный WASM GPU pass: ${kind}.`);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return 0;
    } catch (error) { return this.fail(error); }
  }
}
