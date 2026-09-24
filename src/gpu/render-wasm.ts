/** Permanent views over the runtime's fixed 256 MiB memory. Initial and maximum
 * WASM memory sizes are identical; neither growth nor resizable buffers are used. */
export class RenderMemoryViews {
  readonly memory: WebAssembly.Memory;
  readonly buffer: ArrayBuffer;
  readonly bytes: Uint8Array;
  readonly floats: Float32Array;
  readonly doubles: Float64Array;
  readonly integers: Int32Array;
  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;
    this.buffer = memory.buffer;
    this.bytes = new Uint8Array(this.buffer);
    this.floats = new Float32Array(this.buffer);
    this.doubles = new Float64Array(this.buffer);
    this.integers = new Int32Array(this.buffer);
  }
  /** Compatibility boundary for callers; the underlying memory never changes. */
  refresh(): false { return false; }
}

export interface RenderGpuImports {
  ring_target(width: number, height: number): number;
  temporal_targets(width: number, height: number): number;
  draw(pointer: number): number;
  reference_texture(kind: number, pointer: number, length: number, width: number, height: number): number;
}

export interface RenderWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  input_ptr(): number;
  input_capacity(): number;
  begin(xLength: number, yLength: number, bits: number, iterations: number, fold: number, celtic: number): number;
  step(steps: number): number;
  result_ptr(kind: number): number;
  result_len(kind: number): number;
  orbit_length(): number;
  capacity(): number;
  last_error(): number;
  render_input_ptr(): number;
  render_text_ptr(): number;
  render_text_capacity(): number;
  render_stats_ptr(): number;
  render_set_position(xLength: number, yLength: number, bits: number, logScale: number): number;
  render_frame(mode: number, now: number): number;
  render_mark_dirty(): void;
  render_external_reference(): void;
  render_camera_command(operation: number): number;
  render_set_camera(xLength: number, yLength: number, bits: number, logScale: number, decimal: number): number;
  render_set_route(xLength: number, yLength: number, endZoom: number, targetZoom: number): number;
  render_camera_snapshot(): number;
  render_begin_reference(): number;
  render_cancel_reference(): void;
  render_reference_ready(): number;
  render_reset_temporal(): void;
  render_reset_gpu(): void;
  render_dispose(): void;
  render_error(): number;
}

let compiledModule: WebAssembly.Module | null = null;
let loading: Promise<void> | null = null;

/** Call once before synchronously creating a WASM renderer (also works in a worker). */
export function preloadRenderWasm(): Promise<void> {
  if (compiledModule) return Promise.resolve();
  if (loading) return loading;
  loading = (async () => {
    const response = await fetch(`${import.meta.env.BASE_URL}wasm/core-simd.wasm`);
    if (!response.ok) throw new Error(`Render WASM: HTTP ${response.status}. Выполните npm run build:wasm.`);
    const fallback = response.clone();
    try { compiledModule = await WebAssembly.compileStreaming(response); }
    catch { compiledModule = await WebAssembly.compile(await fallback.arrayBuffer()); }
  })().catch(error => { loading = null; throw error; });
  return loading;
}

export function instantiateRenderWasm(gpu: RenderGpuImports): RenderWasmExports {
  if (!compiledModule) throw new Error('Перед созданием WASM renderer вызовите await preloadRenderWasm().');
  const instance = new WebAssembly.Instance(compiledModule, { gpu: {
    ring_target: gpu.ring_target, temporal_targets: gpu.temporal_targets,
    draw: gpu.draw, reference_texture: gpu.reference_texture,
  } });
  const core = instance.exports as RenderWasmExports;
  if (!(core.memory instanceof WebAssembly.Memory) || typeof core.render_frame !== 'function'
      || typeof core.render_input_ptr !== 'function' || typeof core.render_stats_ptr !== 'function') {
    throw new Error('WASM не содержит render runtime. Пересоберите ядро: npm run build:wasm.');
  }
  return core;
}
