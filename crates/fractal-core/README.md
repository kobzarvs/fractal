# Fractal numerical kernel

Build both variants with `node scripts/build-wasm.mjs`. This requires Rust with
`wasm32-unknown-unknown` and Binaryen's `wasm-opt` on PATH. `cargo test --workspace`
runs native arithmetic, orbit, validation and allocation tests.

The kernel preserves the original worker's integer fixed-point arithmetic:
32 guard bits, negative right shifts rounded toward negative infinity, quantized
fold/Celtic factors, 53-bit half-up orbit export and the original BLA operation
order. Integers use signed magnitude with 32-bit limbs; multiplication, squaring,
scaling, addition/subtraction and shifting reuse preallocated buffers.

## Worker ABI

All addresses are WASM byte offsets. Result lengths are counts of `f32` elements.
The module has no imports and exports its linear `memory`.

1. Get `input_ptr()` and `input_capacity()` (16384 bytes), then write decimal
   integer ASCII bytes: first X, then Y, without separators or null terminators.
   Coordinates are integers scaled by `2^bits`; signs and leading zeros are valid.
2. `begin(x_len, y_len, bits, iterations, fold_steps, celtic_steps)` returns 0 or
   a negative error. Limits are bits 1–4096, iterations 1–65536, and fold/Celtic
   steps 0–1024. Floating UI controls must be quantized before this call.
3. `step(max_steps)` returns 0 while pending, 1 when complete, or a negative error.
   Cancellation means stop stepping and begin the next request. The kernel has
   no host calls or asynchronous activity.
4. Read `orbit_length()`, `capacity()`, and `result_ptr(kind)` / `result_len(kind)`.
   Kinds: 0 orbit, 1 realOrbit, 2 BLA A, 3 BLA B, 4 BLA bounds. Invalid kinds or
   unavailable results return pointer/length zero. RealOrbit has four zero floats
   when Celtic is disabled. Capacity is `ceil((iterations + 1) / 1024) * 1024`.

Error codes from `last_error()`: -1 invalid parameters, -2 invalid decimal input,
-3 no current request, -4 zero step size. A new `begin`, including an invalid one,
invalidates the preceding request. Successful calls to `begin` or `step` clear
previous errors. All mutable input access must occur between ABI calls in the
owning worker; simultaneous C/native callers must enforce the same contract.

`begin` may grow memory and invalidate fixed buffer views. Refresh those views
before reading. Pointers and memory size remain stable throughout every `step`
and until the next `begin`. The native allocation probe verifies zero heap
allocations/reallocations inside chunked computation, including the maximum
iteration budget. Output and integer backing allocations are retained for reuse.

Linear memory has a 2 MiB initial size and a declared 256 MiB maximum, permitting
Chrome's resizable buffer API. The largest five logical output arrays occupy
5,324,800 bytes. Worst-case scratch storage for the entire input text budget is
below 165 KiB; allocator capacity and fragmentation are additional overhead.

## Floating-point portability

The SIMD variant vectorizes two independent columns of BLA matrix products using
`f64x2`; the multiply/add order is identical to scalar code. No fast math, relaxed
SIMD or fused multiply-add is enabled. BLA norm/minimum calculations explicitly
propagate NaN like JavaScript rather than adopting Rust's NaN-skipping min/max.

BLA transcendental functions use `libm`. Current differential tests require exact
Float32 bytes, not a tolerance, and pass for both variants. This is not a proof
that every browser engine's Math implementation rounds every possible input
identically; retain cross-engine differential validation when changing engines
or numerical dependencies. Deep orbit mantissa/exponent exports remain nonzero
below the f64 range; BLA uses f64 bounds, matching the original worker.
