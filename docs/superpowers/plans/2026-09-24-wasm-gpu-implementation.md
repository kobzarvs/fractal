# Burning Ship WASM + GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Independent modules have disjoint ownership; integration and final review are coordinated in this task.

**Goal:** A working deep-zoom Burning Ship explorer with a tested Rust/WASM orbit/BLA kernel, optimized WebGL2 renderer and reproducible accuracy/performance comparisons.

**Architecture:** Rust exposes a small offset/length ABI for reusable memory and chunked calculation. A worker computes reference arrays; WebGL2 retains pixel-parallel perturbation, extended-range arithmetic, AA and ring reuse. JS BigInt is the independent numerical oracle and selectable benchmark baseline.

**Tech Stack:** Rust wasm32-unknown-unknown, TypeScript, Vite, WebGL2/GLSL ES 3.00, Node test runner.

**Spec:** ../specs/2026-09-24-wasm-gpu-fractal-design.md

## Global Constraints

- Preserve formula, fixed-point rounding, precision, iteration budget and AA when comparing speed.
- No fast-math or relaxed SIMD for the numerical kernel.
- Require stable memory during each hot loop and discard cancelled/stale computations.
- Measure CPU and GPU separately; never report rAF cadence as GPU execution time.
- Chrome capability detection controls resizable-memory use; fixed-buffer fallback remains valid.
- The user explicitly instructed implementation on 2026-09-24. Continue implementation without a further approval round.

## Review Focus

1. Negative arithmetic shifts and decimal parsing must match BigInt even around zero.
2. A late worker response must not overwrite a newer camera view.
3. Memory growth must not leave detached/stale typed-array views in use.
4. Deep exponent ranges and rebase must not collapse a nonzero delta to zero.
5. Resize/context restoration must recreate all GPU resources without using disposed textures.

## Shared contracts

`ReferenceRequest = { id:number; x:string; y:string; bits:number; iterations:number; fold:number; celtic:number }`, where x/y are signed decimal fixed-point integers scaled by 2^bits.

`ReferenceResult = { id:number; length:number; capacity:number; iterations:number; bits:number; fold:number; celtic:number; orbit:Float32Array; realOrbit:Float32Array; blaA:Float32Array; blaB:Float32Array; blaBounds:Float32Array; computeMs:number; backend:'wasm'|'js' }`.

Textures have width 1024 and RGBA float texels. capacity is ceil((iterations+1)/1024)*1024. orbit stores x mantissa/exponent and y mantissa/exponent. BLA levels are 1..10, block index = capacity - (capacity >> (level-1)) + (iteration >> level). realOrbit is optional for Celtic.

### Task 1: Rust numerical kernel

Files: `crates/fractal-core/**`, `Cargo.toml`, `scripts/build-wasm.mjs`.

- [x] Write failing native tests for signed fixed-point multiply/shift and a bounded orbit.
- [x] Implement reusable limb arithmetic, exact orbit export, BLA and chunked C ABI.
- [x] ABI: `input_ptr()->u32`, `input_capacity()->u32`, `begin(x_len,y_len,bits,iterations,fold_steps,celtic_steps)->i32`; x bytes then y bytes in input area. `step(max_steps)->i32`: 0 pending, 1 done, negative error. `result_ptr(kind)->u32` (0 orbit, 1 realOrbit, 2 A, 3 B, 4 bounds), `result_len(kind)->u32` in f32 elements, `orbit_length()->u32`, `capacity()->u32`, `last_error()->i32`.
- [x] Validate parameter limits and stable output lifetime until next begin. Output allocation is reused.
- [x] Build scalar and simd128 release binaries into public/wasm; run cargo tests and exported-ABI smoke checks.

### Task 2: JS oracle and verification fixtures

Files: `src/compute/reference-js.ts`, `tests/reference.test.ts`, `src/tours.ts`.

- [x] Implement a readable independent BigInt oracle with request/result contract above and optional cancellation/yield.
- [x] Export `computeReferenceJs(request, cancelled?, yieldControl?) -> Promise<ReferenceResult>`.
- [x] Port the verified Western armada coordinates and other reference tours as decimal strings.
- [x] Test origin/interior, escaped points, axis/sign cases, fold/celtic quantization and BLA layout. Differential WASM tests compare output bytes and report any deviations.
- [x] Export fixtures at bits 128, 256, 576 and iteration budgets 1024/16384.

### Task 3: GPU renderer

Files: `src/gpu/**`.

- [x] Implement `FractalRenderer(canvas)` with `setReference(result)`, `render(view)`, `readPixels()`, `dispose()`, timer result access and baseline/optimized toggle.
- [x] View fields: `{center:[number,number],scale:number,logScale:number,offsetX:[number,number],offsetY:[number,number],iterations:number,fold:number,celtic:number,aa:number,hue:number,optimized:boolean,guided:boolean,referenceKey:number}`; drawing buffer size is set on canvas externally.
- [x] Implement direct, float perturbation and FE perturbation preserving branch and rounding structure, BLA/rebase and smooth palette.
- [x] Compare specialization and integer BLA alignment to generic shader; preserve baseline for image comparisons.
- [x] Include incremental ring cache for fixed-center guided zoom and AA; validate program/framebuffer failures and context resource lifecycle.

### Task 4: Worker integration and application

Files: `src/compute/wasm.ts`, `src/compute/worker.ts`, `src/compute/client.ts`, `src/camera.ts`, `src/main.ts`, `src/style.css`, package/build configuration.

- [x] Tests first for view refresh after memory growth and cancellation generation.
- [x] Load module with feature detection, expose views on WASM memory, compute in bounded chunks and transfer compact results using buffer reuse.
- [x] Implement exact BigInt camera pan/zoom, Western armada tour, URL parameters, backend comparison, pause/reset and accessible settings.
- [x] Fixed-quality mode is default for measurements; expose actual backend, reference time, GPU time and memory.
- [x] Handle worker errors, restart, resize, hidden pages and WebGL context restore.

### Task 5: Accuracy, performance, review and delivery

Files: `src/benchmark.ts`, `tests/wasm.test.ts`, `README.md`, `docs/performance.md`.

- [x] Run Rust/TS numerical tests and differential orbit/BLA comparisons.
- [x] Browser verification at multiple scales, at least direct/float/FE; image diff generic versus optimized shaders, identical jitter and sizes.
- [x] Benchmark JS versus scalar/SIMD WASM after warm-up and report median/p95 and memory reuse. Benchmark GPU variants with valid timer queries or state their absence.
- [x] Compare output-copy transport with co-located WASM/GPU or shared transport only if useful; report evidence rather than claiming zero-copy GPU.
- [x] Independent review of numerical kernel, lifecycle and final integrated behavior; resolve actionable defects.
- [x] Document build/start/test commands, results, limitations and selected defaults; commit verified implementation.

## Progress ledger

- Ruling: work in the requested project directory on `codex/wasm-gpu` — repository contains only the approved specification and has no competing implementation — moving later remains straightforward.
- Ruling: implement now as explicitly requested, with parallel disjoint modules — another plan approval would contradict the user's instruction to begin — interfaces are reviewed during integration.
- Preflight: Tasks 1/2/4 share only the declared numerical ABI/result contract; Tasks 3/4 share only renderer/view contract; Task 5 consumes all interfaces. Each implementer owns separate files, root owns shared types and integration.

- Complete: Rust/WASM scalar+SIMD; 9 native tests; no-allocation hot loop; SIMD instructions confirmed in binary.
- Complete: 64 Node/TypeScript tests, differential outputs, camera/temporal/memory/input validation.
- Complete: Chromium153/M3Max browser checks all passed, including raw worker cancellation, five GPU scales, rings, temporal reset and context recovery. JSON committed under docs/benchmarks.
- Ruling: reject compile-time fold/celtic specialization after pixel differences; keep integer BLA alignment and equivalent texture reuse. Exact pixels verified after removal.
- Ruling: retain original temporal AA and centre jitter; spatial-only implementation was insufficient for the approved quality requirement.
- Ruling: default auto prefers SIMD with scalar CompileError fallback following user clarification; report the actual loaded variant and retain explicit choices. SIMD advantage over scalar is small on this workload, not assumed universal.
- Ruling: retain pooled transfer transport; measured worker overhead is small. Co-located/shared transport experiment deferred and explicitly recorded in performance.md, no zero-copy claim.
- Limitation: iOS draw batching, cross-GPU equivalence, very-high-precision arithmetic acceleration and energy profiling are not validated; desktop Chromium is the verified target.
- Final review: corrected inverted DOM Y mapping, temporal AA omission, stale-resource disposal after context restoration and final-slice cancellation boundary; all relevant regressions verified.
