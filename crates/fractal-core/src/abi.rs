//! Single-worker C ABI. Result buffers are valid until the next begin call.
//! Coordinates are concatenated UTF-8 decimal integers in the static input area.
//! 0 = success/pending, 1 = complete; errors: -1 parameters, -2 decimal text,
//! -3 no request, -4 zero-sized step. Results are f32 slices, lengths in elements.
use crate::kernel::Kernel;
use std::cell::UnsafeCell;
use std::sync::Mutex;

const INPUT_CAPACITY: usize = 16384;
struct Input(UnsafeCell<[u8; INPUT_CAPACITY]>);
// JavaScript writes the input only between calls in one dedicated worker.
// Native C callers must enforce the same input-area ownership contract.
unsafe impl Sync for Input {}
static INPUT: Input = Input(UnsafeCell::new([0; INPUT_CAPACITY]));
struct State {
    kernel: Option<Kernel>,
    error: i32,
}
static STATE: Mutex<State> = Mutex::new(State {
    kernel: None,
    error: 0,
});

#[unsafe(no_mangle)]
pub extern "C" fn input_ptr() -> u32 {
    INPUT.0.get() as usize as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn input_capacity() -> u32 {
    INPUT_CAPACITY as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn begin(
    x_len: u32,
    y_len: u32,
    bits: u32,
    iterations: u32,
    fold_steps: u32,
    celtic_steps: u32,
) -> i32 {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if x_len as u64 + y_len as u64 > INPUT_CAPACITY as u64 {
        state.error = -1;
        // An invalid begin invalidates the preceding job as well.
        if let Some(kernel) = &mut state.kernel {
            let _ = kernel.begin(b"", b"", 0, 0, 0, 0);
        }
        return -1;
    }
    let input = unsafe { &*INPUT.0.get() };
    let result = state.kernel.get_or_insert_with(Kernel::new).begin(
        &input[..x_len as usize],
        &input[x_len as usize..(x_len + y_len) as usize],
        bits as usize,
        iterations as usize,
        fold_steps as usize,
        celtic_steps as usize,
    );
    state.error = result.err().unwrap_or(0);
    state.error
}

#[unsafe(no_mangle)]
pub extern "C" fn step(max_steps: u32) -> i32 {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let status = state
        .kernel
        .as_mut()
        .map_or(-3, |k| k.step(max_steps as usize));
    state.error = if status < 0 { status } else { 0 };
    status
}

#[unsafe(no_mangle)]
pub extern "C" fn result_ptr(kind: u32) -> u32 {
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.kernel.as_ref().map_or(0, |k| {
        let result = k.result(kind as usize);
        if result.is_empty() {
            0
        } else {
            result.as_ptr() as usize as u32
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn result_len(kind: u32) -> u32 {
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .kernel
        .as_ref()
        .map_or(0, |k| k.result(kind as usize).len() as u32)
}
#[unsafe(no_mangle)]
pub extern "C" fn orbit_length() -> u32 {
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .kernel
        .as_ref()
        .map_or(0, |k| k.length() as u32)
}
#[unsafe(no_mangle)]
pub extern "C" fn capacity() -> u32 {
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .kernel
        .as_ref()
        .map_or(0, |k| k.capacity() as u32)
}
#[unsafe(no_mangle)]
pub extern "C" fn last_error() -> i32 {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).error
}
