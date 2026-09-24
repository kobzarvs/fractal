import type { ReferenceResult, RenderView } from '../types';
import { ringFragment, shipFragment, vertexShader } from './shaders';
import { createRingLayout, missingRingRanges, positiveModulo, ringWindow } from './rings';
import type { RingBand, RingLayout } from './rings';
import { TemporalAccumulator, temporalFragment, temporalCompositeFragment } from './temporal';

type Path = 0 | 1 | 2;
interface Program { program: WebGLProgram; uniforms: Map<string, WebGLUniformLocation | null> }
interface TimerExtension { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }
export interface GpuTiming {
  frame: number; milliseconds: number;
  path: 'direct' | 'float' | 'fe'; ring: boolean;
}
export interface RendererStats {
  path: 'direct' | 'float' | 'fe'; frame: number;
  ringsDrawn: number; ringSamples: number; ringResets: number;
  ringFrames: number; ringActive: boolean; referenceUploads: number;
}
interface PendingTimer { query: WebGLQuery; sample: Omit<GpuTiming, 'milliseconds'> }
interface RingCache {
  layout: RingLayout; texture: WebGLTexture; framebuffer: WebGLFramebuffer;
  key: string; origin: number;
}
interface TemporalTarget { texture: WebGLTexture; framebuffer: WebGLFramebuffer }
interface TemporalTargets {
  width: number; height: number;
  current: TemporalTarget; history: [TemporalTarget, TemporalTarget]; next: 0 | 1;
}

const TEXTURE_NAMES = ['referenceOrbit', 'realOrbit', 'blaA', 'blaB', 'blaBounds'] as const;
const pathName = (path: Path): 'direct' | 'float' | 'fe' => ['direct', 'float', 'fe'][path] as 'direct' | 'float' | 'fe';

/** Owns a WebGL2 context's resources. Recreate this object after context restoration.
 * Canvas dimensions are controlled by the caller. readPixels returns bottom-up RGBA8.
 * gpuTimeMs is the last completed hardware query, never a requestAnimationFrame delta.
 */
export class FractalRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly vao: WebGLVertexArrayObject;
  private readonly timer: TimerExtension | null;
  private readonly programs = new Map<string, Program>();
  private textures: WebGLTexture[] = [];
  private reference: Pick<ReferenceResult, 'id' | 'length' | 'capacity' | 'iterations' | 'fold' | 'celtic'> | null = null;
  private rings: RingCache | null = null;
  private disabledRingKey = '';
  private previousRingScale = NaN;
  private ringVelocity = 0;
  private readonly pendingTimers: PendingTimer[] = [];
  private readonly completedTimers: GpuTiming[] = [];
  private lastGpuTime: number | null = null;
  private disposed = false;
  // Restoring a canvas keeps its WebGL context object but invalidates all old
  // resource handles. Latch loss permanently for this renderer generation.
  private contextInvalid = false;
  private readonly onContextLost = () => {
    this.contextInvalid = true;
    this.resetTemporal();
  };
  private referenceVersion = 0;
  private readonly temporal = new TemporalAccumulator();
  private temporalTargets: TemporalTargets | null = null;
  readonly stats: RendererStats = {
    path: 'direct', frame: 0,
    ringsDrawn: 0, ringSamples: 0, ringResets: 0,
    ringFrames: 0, ringActive: false, referenceUploads: 0,
  };

  constructor(readonly canvas: HTMLCanvasElement | OffscreenCanvas) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance',
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('WebGL 2 недоступен: нужен браузер с WebGL 2 и включённым GPU.');
    this.gl = gl;
    const precision = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    if (!precision || precision.precision < 23) throw new Error('GPU не поддерживает требуемую highp float точность фрагментного шейдера.');
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('Не удалось создать WebGL vertex array.');
    this.vao = vao;
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExtension | null;
    canvas.addEventListener('webglcontextlost', this.onContextLost);
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    try {
      this.getShipProgram(0, false);
      this.checkError('инициализация');
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  get gpuTimerSupported(): boolean { return this.timer !== null; }
  get gpuTimeMs(): number | null { this.pollGpuTimers(); return this.lastGpuTime; }
  get gpuTimings(): readonly GpuTiming[] { this.pollGpuTimers(); return this.completedTimers; }
  get settling(): boolean { return !this.disposed && !this.contextInvalid && this.temporal.settling; }

  /** Discard accumulated samples without reallocating full-resolution targets. */
  resetTemporal(): void { this.temporal.reset(); }

  /** Non-blocking: consumes only completed EXT_disjoint_timer_query_webgl2 queries. */
  pollGpuTimers(): void {
    if (!this.timer || this.disposed || !this.hasLiveContext()) return;
    const gl = this.gl;
    if (gl.getParameter(this.timer.GPU_DISJOINT_EXT)) {
      for (const pending of this.pendingTimers) gl.deleteQuery(pending.query);
      this.pendingTimers.length = 0;
      this.completedTimers.length = 0;
      this.lastGpuTime = null;
      return;
    }
    while (this.pendingTimers.length && gl.getQueryParameter(this.pendingTimers[0].query, gl.QUERY_RESULT_AVAILABLE)) {
      const pending = this.pendingTimers.shift()!;
      const milliseconds = Number(gl.getQueryParameter(pending.query, gl.QUERY_RESULT)) / 1e6;
      gl.deleteQuery(pending.query);
      if (Number.isFinite(milliseconds) && milliseconds >= 0) {
        this.lastGpuTime = milliseconds;
        this.completedTimers.push({ ...pending.sample, milliseconds });
        if (this.completedTimers.length > 64) this.completedTimers.shift();
      }
    }
  }

  clearGpuTimings(): void {
    if (this.hasLiveContext()) {
      for (const pending of this.pendingTimers) this.gl.deleteQuery(pending.query);
    }
    this.pendingTimers.length = 0;
    this.completedTimers.length = 0;
    this.lastGpuTime = null;
  }

  setReference(result: ReferenceResult): void {
    this.assertAlive();
    const gl = this.gl;
    const size = result.capacity * 4;
    if (!Number.isSafeInteger(result.capacity) || result.capacity < 1024 || result.capacity % 1024 ||
        result.length < 2 || result.length > result.capacity || result.capacity < result.iterations + 1) {
      throw new Error('Некорректные размеры опорной орбиты.');
    }
    const height = result.capacity / 1024;
    if (height > gl.getParameter(gl.MAX_TEXTURE_SIZE) || gl.getParameter(gl.MAX_TEXTURE_SIZE) < 1024) {
      throw new Error('Опорная орбита превышает максимальный размер текстуры GPU.');
    }
    const arrays = [result.orbit, result.realOrbit, result.blaA, result.blaB, result.blaBounds];
    for (let index = 0; index < arrays.length; index++) {
      if ((index !== 1 || result.celtic !== 0) && arrays[index].length !== size) {
        throw new Error(`Некорректная длина ${TEXTURE_NAMES[index]}: ${arrays[index].length}, ожидается ${size}.`);
      }
    }
    const created: WebGLTexture[] = [];
    try {
      for (let index = 0; index < arrays.length; index++) {
        const texture = gl.createTexture();
        if (!texture) throw new Error('Не удалось выделить текстуру опорной орбиты.');
        created.push(texture);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        this.textureParameters();
        const unusedReal = index === 1 && result.celtic === 0;
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, unusedReal ? 1 : 1024, unusedReal ? 1 : height,
          0, gl.RGBA, gl.FLOAT, unusedReal ? new Float32Array(4) : arrays[index]);
      }
      this.checkError('загрузка опорной орбиты');
    } catch (error) {
      for (const texture of created) gl.deleteTexture(texture);
      throw error;
    }
    for (const texture of this.textures) gl.deleteTexture(texture);
    this.textures = created;
    this.reference = { id: result.id, length: result.length, capacity: result.capacity,
      iterations: result.iterations, fold: result.fold, celtic: result.celtic };
    this.referenceVersion++;
    this.stats.referenceUploads++;
    this.clearRings();
    this.resetTemporal();
  }

  render(view: RenderView): void {
    this.assertAlive();
    this.validateView(view);
    const gl = this.gl;
    if (this.canvas.width < 1 || this.canvas.height < 1) { this.resetTemporal(); return; }
    const path: Path = view.logScale >= -8 ? 0 : view.logScale > -80 ? 1 : 2;
    if (path !== 0 && (!this.reference || this.reference.fold !== view.fold || this.reference.celtic !== view.celtic ||
        this.reference.iterations < view.iterations)) {
      throw new Error('Для глубокого масштаба нужна актуальная опорная орбита с теми же fold, Celtic и числом итераций.');
    }
    this.pollGpuTimers();
    this.stats.frame++;
    this.stats.path = pathName(path);
    this.stats.ringActive = false;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.SCISSOR_TEST);
    let query: WebGLQuery | null = null;
    if (this.timer && this.pendingTimers.length < 8) {
      query = gl.createQuery();
      if (query) gl.beginQuery(this.timer.TIME_ELAPSED_EXT, query);
    }
    try {
      const ringUsed = view.guided && path !== 0 && this.renderRings(view);
      this.stats.ringActive = !!ringUsed;
      if (!ringUsed) {
        if (view.temporal && view.aa > 1) this.renderTemporal(view, path);
        else {
          this.resetTemporal();
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.viewport(0, 0, this.canvas.width, this.canvas.height);
          gl.disable(gl.SCISSOR_TEST);
          const program = this.getShipProgram(path, false);
          this.shipUniforms(program, view);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
        }
      } else this.resetTemporal();
      this.checkError('отрисовка кадра');
    } catch (error) {
      this.resetTemporal();
      throw error;
    } finally {
      gl.disable(gl.SCISSOR_TEST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (query && this.timer) {
        gl.endQuery(this.timer.TIME_ELAPSED_EXT);
        this.pendingTimers.push({ query, sample: {
          frame: this.stats.frame, path: pathName(path), ring: this.stats.ringActive,
        } });
        gl.flush();
      }
    }
  }

  readPixels(): Uint8Array {
    this.assertAlive();
    const pixels = new Uint8Array(this.canvas.width * this.canvas.height * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.readPixels(0, 0, this.canvas.width, this.canvas.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
    this.checkError('чтение пикселей');
    return pixels;
  }

  dispose(): void {
    if (this.disposed) return;
    const deleteResources = this.hasLiveContext();
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost);
    this.clearGpuTimings();
    this.clearRings();
    this.clearTemporalTargets();
    if (deleteResources) {
      for (const texture of this.textures) this.gl.deleteTexture(texture);
      for (const program of this.programs.values()) this.gl.deleteProgram(program.program);
      this.gl.deleteVertexArray(this.vao);
    }
    this.textures.length = 0;
    this.programs.clear();
    this.reference = null;
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('FractalRenderer уже освобождён.');
    if (!this.hasLiveContext()) throw new Error('WebGL-контекст потерян. После восстановления создайте новый FractalRenderer.');
  }

  private hasLiveContext(): boolean {
    if (!this.contextInvalid && this.gl.isContextLost()) this.contextInvalid = true;
    return !this.contextInvalid;
  }

  private validateView(view: RenderView): void {
    if (!Number.isInteger(view.iterations) || view.iterations < 1 || view.iterations > 1_000_000) {
      throw new Error('Число итераций должно быть целым от 1 до 1000000.');
    }
    if (!Number.isInteger(view.aa) || view.aa < 1 || view.aa > 5) throw new Error('AA должен быть целым числом от 1 до 5.');
    if (![view.logScale, view.fold, view.celtic, view.hue, ...view.center, ...view.offsetX, ...view.offsetY].every(Number.isFinite)) {
      throw new Error('Координаты, FE-экспоненты и параметры вида должны быть конечными числами.');
    }
    if (view.logScale >= -8 && (!Number.isFinite(view.scale) || view.scale <= 0)) throw new Error('Масштаб прямого рендеринга должен быть положительным.');
    if (view.fold < 0 || view.fold > 1 || view.celtic < 0 || view.celtic > 1) throw new Error('Fold и Celtic должны быть в диапазоне 0…1.');
  }

  private compile(source: string, type: number, label: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error(`Не удалось создать шейдер ${label}.`);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Ошибка компиляции WebGL-шейдера ${label}: ${log || 'неизвестная ошибка'}`);
    }
    return shader;
  }

  private getProgram(key: string, fragment: string): Program {
    const cached = this.programs.get(key);
    if (cached) return cached;
    const gl = this.gl;
    const vertex = this.compile(vertexShader, gl.VERTEX_SHADER, `${key}/vertex`);
    let shader: WebGLShader | null = null;
    let linked: WebGLProgram | null = null;
    try {
      shader = this.compile(fragment, gl.FRAGMENT_SHADER, `${key}/fragment`);
      linked = gl.createProgram();
      if (!linked) throw new Error(`Не удалось создать программу ${key}.`);
      gl.attachShader(linked, vertex);
      gl.attachShader(linked, shader);
      gl.linkProgram(linked);
      if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) {
        throw new Error(`Ошибка линковки WebGL-программы ${key}: ${gl.getProgramInfoLog(linked)}`);
      }
      const program = { program: linked, uniforms: new Map<string, WebGLUniformLocation | null>() };
      this.programs.set(key, program);
      return program;
    } catch (error) {
      if (linked) gl.deleteProgram(linked);
      throw error;
    } finally {
      gl.deleteShader(vertex);
      if (shader) gl.deleteShader(shader);
    }
  }

  private getShipProgram(path: Path, ring: boolean): Program {
    return this.getProgram(`ship/${path}/${+ring}`, shipFragment(path, ring));
  }

  private uniform(program: Program, name: string): WebGLUniformLocation | null {
    if (!program.uniforms.has(name)) program.uniforms.set(name, this.gl.getUniformLocation(program.program, name));
    return program.uniforms.get(name)!;
  }

  private shipUniforms(program: Program, view: RenderView, logScale = view.logScale): void {
    const gl = this.gl;
    gl.useProgram(program.program);
    gl.uniform2f(this.uniform(program, 'center'), view.center[0], view.center[1]);
    gl.uniform1f(this.uniform(program, 'scale'), view.scale);
    gl.uniform1f(this.uniform(program, 'fold'), view.fold);
    gl.uniform1f(this.uniform(program, 'celtic'), view.celtic);
    gl.uniform1f(this.uniform(program, 'hue'), view.hue);
    gl.uniform1f(this.uniform(program, 'aspect'), this.canvas.width / this.canvas.height);
    gl.uniform2f(this.uniform(program, 'res'), this.canvas.width, this.canvas.height);
    gl.uniform1f(this.uniform(program, 'aa'), view.aa);
    gl.uniform2f(this.uniform(program, 'jitter'), 0, 0);
    gl.uniform1i(this.uniform(program, 'iterations'), view.iterations);
    gl.uniform2f(this.uniform(program, 'offsetX'), view.offsetX[0], view.offsetX[1]);
    gl.uniform2f(this.uniform(program, 'offsetY'), view.offsetY[0], view.offsetY[1]);
    const exponent = Math.floor(logScale);
    gl.uniform2f(this.uniform(program, 'viewScale'), 2 ** (logScale - exponent), exponent);
    gl.uniform2f(this.uniform(program, 'referenceSize'), 1024, (this.reference?.capacity ?? 1024) / 1024);
    gl.uniform1i(this.uniform(program, 'referenceLength'), this.reference?.length ?? 0);
    for (let index = 0; index < this.textures.length; index++) {
      gl.activeTexture(gl.TEXTURE0 + index);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[index]);
      gl.uniform1i(this.uniform(program, TEXTURE_NAMES[index]), index);
    }
  }

  private textureParameters(): void {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private checkError(action: string): void {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR) throw new Error(`Ошибка WebGL (${action}): 0x${error.toString(16)}.`);
  }

  private clearTemporalTargets(): void {
    this.resetTemporal();
    if (!this.temporalTargets) return;
    if (this.hasLiveContext()) {
      for (const target of [this.temporalTargets.current, ...this.temporalTargets.history]) {
        this.gl.deleteFramebuffer(target.framebuffer);
        this.gl.deleteTexture(target.texture);
      }
    }
    this.temporalTargets = null;
  }

  private getTemporalTargets(): TemporalTargets {
    const width = this.canvas.width, height = this.canvas.height;
    if (this.temporalTargets?.width === width && this.temporalTargets.height === height) return this.temporalTargets;
    this.clearTemporalTargets();
    const gl = this.gl;
    if (Math.max(width, height) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
      throw new Error('Полное разрешение temporal AA превышает максимальный размер текстуры GPU.');
    }
    const created: TemporalTarget[] = [];
    try {
      for (let index = 0; index < 3; index++) {
        const texture = gl.createTexture(), framebuffer = gl.createFramebuffer();
        if (!texture || !framebuffer) {
          if (texture) gl.deleteTexture(texture);
          if (framebuffer) gl.deleteFramebuffer(framebuffer);
          throw new Error('Не удалось создать framebuffer temporal AA.');
        }
        created.push({ texture, framebuffer });
        gl.bindTexture(gl.TEXTURE_2D, texture);
        this.textureParameters();
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          throw new Error('Framebuffer temporal AA неполон.');
        }
      }
      this.checkError('создание temporal AA');
      return this.temporalTargets = { width, height, current: created[0], history: [created[1], created[2]], next: 0 };
    } catch (error) {
      for (const target of created) { gl.deleteFramebuffer(target.framebuffer); gl.deleteTexture(target.texture); }
      throw error;
    } finally { gl.bindFramebuffer(gl.FRAMEBUFFER, null); }
  }

  private renderTemporal(view: RenderView, path: Path): void {
    const position = view.position;
    if (!position || typeof position.x !== 'bigint' || typeof position.y !== 'bigint' ||
        !Number.isInteger(position.bits) || position.bits < 1 || position.bits > 4096) {
      throw new Error('Temporal AA требует точные fixed-point координаты камеры.');
    }
    const targets = this.getTemporalTargets();
    const gl = this.gl;
    const key = [this.referenceVersion, view.referenceKey, path, view.iterations,
      view.fold, view.celtic, view.hue].join(':');
    const frame = this.temporal.prepare({ ...position, logScale: view.logScale },
      targets.width, targets.height, view.aa, key);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, targets.width, targets.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.current.framebuffer);
    const ship = this.getShipProgram(path, false);
    this.shipUniforms(ship, view);
    gl.uniform1f(this.uniform(ship, 'aa'), 1);
    gl.uniform2f(this.uniform(ship, 'jitter'), frame.jitter.x, frame.jitter.y);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const resolved = targets.history[targets.next];
    const previous = targets.history[1 - targets.next];
    const resolve = this.getProgram('temporal-resolve', temporalFragment);
    gl.bindFramebuffer(gl.FRAMEBUFFER, resolved.framebuffer);
    gl.useProgram(resolve.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, targets.current.texture);
    gl.uniform1i(this.uniform(resolve, 'currentMap'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, previous.texture);
    gl.uniform1i(this.uniform(resolve, 'historyMap'), 1);
    gl.uniform2f(this.uniform(resolve, 'texelSize'), 1 / targets.width, 1 / targets.height);
    gl.uniform2f(this.uniform(resolve, 'historyOffset'), ...frame.historyOffset);
    gl.uniform1f(this.uniform(resolve, 'historyScale'), frame.historyScale);
    gl.uniform1f(this.uniform(resolve, 'historyWeight'), frame.historyWeight);
    gl.uniform1f(this.uniform(resolve, 'moving'), +frame.moving);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const composite = this.getProgram('temporal-composite', temporalCompositeFragment);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(composite.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
    gl.uniform1i(this.uniform(composite, 'frameMap'), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    targets.next = targets.next === 0 ? 1 : 0;
  }

  private clearRings(): void {
    if (this.rings) {
      if (this.hasLiveContext()) {
        this.gl.deleteFramebuffer(this.rings.framebuffer);
        this.gl.deleteTexture(this.rings.texture);
      }
      this.rings = null;
    }
    this.disabledRingKey = '';
    this.previousRingScale = NaN;
    this.ringVelocity = 0;
  }

  private ringKey(view: RenderView): string {
    return [this.canvas.width, this.canvas.height, this.referenceVersion, view.referenceKey,
      view.center[0], view.center[1], ...view.offsetX, ...view.offsetY, view.iterations,
      view.fold, view.celtic, view.hue].join(':');
  }

  private createRings(view: RenderView, key: string): RingCache | null {
    const gl = this.gl;
    const layout = createRingLayout(this.canvas.width, this.canvas.height, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    this.clearRings();
    if (!layout) { this.disabledRingKey = key; return null; }
    const texture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();
    if (!texture || !framebuffer) {
      if (texture) gl.deleteTexture(texture);
      if (framebuffer) gl.deleteFramebuffer(framebuffer);
      throw new Error('Не удалось создать кольцевой кэш GPU.');
    }
    try {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      this.textureParameters();
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, layout.width, layout.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Кольцевой framebuffer неполон: 0x${status.toString(16)}.`);
      this.checkError('создание кольцевого кэша');
    } catch (error) {
      gl.deleteFramebuffer(framebuffer);
      gl.deleteTexture(texture);
      throw error;
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    this.stats.ringResets++;
    return this.rings = {
      layout, texture, framebuffer, key,
      origin: view.logScale + Math.log2(layout.bands[0].radius / this.canvas.height),
    };
  }

  private drawRingRows(cache: RingCache, band: RingBand, first: number, last: number, view: RenderView): void {
    const gl = this.gl;
    for (let index = first; index <= last;) {
      const row = positiveModulo(index, band.rings);
      const rows = Math.min(last - index + 1, band.rings - row);
      const logScale = cache.origin - index * band.step;
      const innermostLogScale = cache.origin - (index + rows - 1) * band.step;
      const path: Path = innermostLogScale > -90 ? 1 : 2;
      const program = this.getShipProgram(path, true);
      this.shipUniforms(program, view, logScale);
      gl.uniform4f(this.uniform(program, 'ringBlock'), band.x, band.row + row, band.angles, band.step);
      gl.bindFramebuffer(gl.FRAMEBUFFER, cache.framebuffer);
      gl.viewport(band.x, band.row + row, band.angles, rows);
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(band.x, band.row + row, band.angles, rows);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.stats.ringsDrawn += rows;
      this.stats.ringSamples += rows * band.angles;
      index += rows;
    }
  }

  private renderRings(view: RenderView): boolean {
    const key = this.ringKey(view);
    if (key === this.disabledRingKey) return false;
    const cache = this.rings?.key === key ? this.rings : this.createRings(view, key);
    if (!cache) return false;
    const movement = Math.abs(view.logScale - this.previousRingScale);
    this.previousRingScale = view.logScale;
    if (Number.isFinite(movement)) this.ringVelocity += (movement - this.ringVelocity) * 0.2;
    // A fast jump can require more polar samples than a full frame. Preserve
    // existing cache rows and use the unchanged full-quality renderer instead.
    const expectedSamples = cache.layout.bands.reduce((sum, band) => sum + this.ringVelocity / band.step * band.angles, 0);
    if (expectedSamples > this.canvas.width * this.canvas.height) return false;
    const work = cache.layout.bands.map(band => {
      const wanted = ringWindow(band, view.logScale, this.canvas.height, cache.origin);
      const old = band.window;
      band.window = old && old[1] >= wanted[0] && old[0] <= wanted[1]
        ? [Math.max(old[0], wanted[0]), Math.min(old[1], wanted[1])] : null;
      return { band, wanted, ranges: missingRingRanges(wanted, band.window) };
    });
    const missing = work.reduce((sum, item) => sum + item.ranges.reduce((n, range) => n + (range[1] - range[0] + 1) * item.band.angles, 0), 0);
    const pixels = this.canvas.width * this.canvas.height;
    let budget = missing <= pixels ? missing : Math.floor(pixels / 2);
    let complete = true;
    for (const item of work) {
      const { band, wanted } = item;
      for (const [first, last] of item.ranges) {
        const rows = Math.min(last - first + 1, Math.floor(budget / band.angles));
        if (rows <= 0) continue;
        // Fill backwards when extending toward the outer rings, keeping one
        // contiguous valid interval even when this frame's work budget ends.
        const from = band.window && last < band.window[0] ? last - rows + 1 : first;
        const to = from + rows - 1;
        this.drawRingRows(cache, band, from, to, view);
        band.window = band.window ? [Math.min(band.window[0], from), Math.max(band.window[1], to)] : [from, to];
        budget -= rows * band.angles;
      }
      if (!band.window || band.window[0] > wanted[0] || band.window[1] < wanted[1]) complete = false;
    }
    if (!complete) return false;
    const gl = this.gl;
    const program = this.getProgram('ring-assembly', ringFragment);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(program.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cache.texture);
    gl.uniform1i(this.uniform(program, 'ringMap'), 0);
    gl.uniform1f(this.uniform(program, 'aspect'), this.canvas.width / this.canvas.height);
    gl.uniform1f(this.uniform(program, 'viewHeight'), this.canvas.height);
    gl.uniform2f(this.uniform(program, 'res'), this.canvas.width, this.canvas.height);
    gl.uniform1f(this.uniform(program, 'aa'), view.aa);
    gl.uniform1i(this.uniform(program, 'bandCount'), cache.layout.bands.length);
    cache.layout.bands.forEach((band, index) => {
      gl.uniform4f(this.uniform(program, `bandLayout[${index}]`), band.row, band.angles, band.step, band.rings);
      gl.uniform3f(this.uniform(program, `bandPlace[${index}]`),
        positiveModulo((cache.origin - view.logScale) / band.step, band.rings), band.radius, band.x);
    });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.stats.ringFrames++;
    return true;
  }
}
