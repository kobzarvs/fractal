mod abi;
mod camera;
mod integer;
mod kernel;
mod render;

#[cfg(test)]
mod tests {
    use super::integer::Integer;
    use super::kernel::Kernel;

    #[test]
    fn bounded_ship_orbit_is_independent_of_chunk_boundaries() {
        let mut kernel = Kernel::new();
        kernel.begin(b"-256", b"-256", 8, 5, 1024, 0).unwrap();
        assert_eq!(kernel.step(2), 0);
        assert_eq!(kernel.step(1), 0);
        assert_eq!(kernel.step(20), 1);
        assert_eq!(kernel.length(), 6);
        assert_eq!(
            &kernel.result(0)[..24],
            &[
                0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
                -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
            ]
        );
    }

    fn number(value: i128) -> Integer {
        Integer::parse(value.to_string().as_bytes(), 16).unwrap()
    }

    #[test]
    fn signed_multiply_then_shift_rounds_toward_negative_infinity() {
        // Truncation, or losing a carry between limbs, changes these answers.
        let cases = [
            (-3, 5, 3, -2),
            (-8, 4, 3, -4),
            (-1, 1, 127, -1),
            (3, -5, 3, -2),
            (-3, -5, 3, 1),
            (0, -5, 8, 0),
            (4_294_967_295, 4_294_967_295, 32, 4_294_967_294),
            (-4_294_967_297, 4_294_967_295, 32, -4_294_967_296),
        ];
        let mut product = Integer::zero(16);
        let mut shifted = Integer::zero(16);
        for (a, b, shift, expected) in cases {
            product.multiply(&number(a), &number(b));
            shifted.shift_right(&product, shift);
            assert_eq!(shifted.to_i128(), expected, "{a} * {b} >> {shift}");
        }
    }

    #[test]
    fn decimal_parsing_preserves_large_integer_and_rejects_invalid_text() {
        let value = Integer::parse(b"-00079228162514264337593543950335", 16).unwrap();
        assert_eq!(value.to_i128(), -79_228_162_514_264_337_593_543_950_335i128);
        assert_eq!(Integer::parse(b"-0", 16).unwrap().to_i128(), 0);
        for text in [b"".as_slice(), b"-", b"1.0", b" 1", b"1e3", b"0x10"] {
            assert!(Integer::parse(text, 16).is_err());
        }
    }

    #[test]
    fn float_export_rounds_half_up_and_keeps_deep_exponents() {
        let halfway = number(18_014_398_509_481_987);
        assert_eq!(halfway.to_fe(400), (1.0000000000000002, -346));
        assert_eq!(number(-1).to_fe(4096), (-1.0, -4096));
        assert_eq!(number(0).to_fe(128), (0.0, 0));
    }

    #[test]
    fn limb_arithmetic_matches_independent_i128_across_word_boundaries() {
        let mut state = 0x79b9_d17d_535b_2831u64;
        let mut result = Integer::zero(16);
        let mut shifted = Integer::zero(16);
        for index in 0..3000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let a = (state as i64) as i128;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = (state as i64) as i128;
            let av = number(a);
            let bv = number(b);
            result.multiply(&av, &bv);
            let bits = index % 128;
            shifted.shift_right(&result, bits);
            assert_eq!(shifted.to_i128(), (a * b) >> bits);
            result.combine(&av, &bv, false);
            assert_eq!(result.to_i128(), a + b);
            result.combine(&av, &bv, true);
            assert_eq!(result.to_i128(), a - b);
            result.scale(-2048);
            assert_eq!(result.to_i128(), (a - b) * -2048);
        }
    }

    #[test]
    fn request_validation_invalidates_old_output_and_handles_escaped_orbits() {
        let mut kernel = Kernel::new();
        assert_eq!(kernel.step(1), -3);
        kernel.begin(b"257", b"0", 1, 64, 1024, 1024).unwrap();
        assert_eq!(kernel.step(0), -4);
        assert_eq!(kernel.step(100), 1);
        assert_eq!(kernel.length(), 3);
        assert_eq!(kernel.capacity(), 1024);
        assert_eq!(kernel.result(1).len(), 4096);
        assert_eq!(kernel.begin(b"1.2", b"0", 128, 2, 1024, 0), Err(-2));
        assert!(kernel.result(0).is_empty());
        assert_eq!(kernel.begin(b"0", b"0", 4097, 2, 1024, 0), Err(-1));
        assert_eq!(kernel.begin(b"0", b"0", 128, 65537, 1024, 0), Err(-1));
        assert_eq!(kernel.begin(b"0", b"0", 128, 2, 1025, 0), Err(-1));
    }

    #[test]
    fn chunked_computation_allocates_nothing_and_preserves_result_addresses() {
        let mut kernel = Kernel::new();
        kernel.begin(b"-256", b"-256", 8, 2048, 1024, 0).unwrap();
        let pointers: [usize; 5] =
            std::array::from_fn(|kind| kernel.result(kind).as_ptr() as usize);
        let allocations = super::allocation_probe::measure(|| while kernel.step(37) == 0 {});
        assert_eq!(allocations, 0, "step must not allocate or grow its buffers");
        assert_eq!(kernel.length(), 2049);
        for (kind, pointer) in pointers.iter().enumerate() {
            assert_eq!(kernel.result(kind).as_ptr() as usize, *pointer);
        }
        kernel.begin(b"0", b"0", 4096, 65536, 1024, 1024).unwrap();
        let allocations = super::allocation_probe::measure(|| while kernel.step(128) == 0 {});
        assert_eq!(allocations, 0);
        assert_eq!(kernel.length(), 65537);
    }
}

#[cfg(test)]
mod allocation_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    thread_local! {
        static TRACKING: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }
    struct CountingAllocator;
    fn count() {
        if TRACKING.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|count| count.set(count.get() + 1));
        }
    }
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count();
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count();
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            count();
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;
    pub fn measure(work: impl FnOnce()) -> usize {
        COUNT.with(|count| count.set(0));
        TRACKING.with(|enabled| enabled.set(true));
        work();
        TRACKING.with(|enabled| enabled.set(false));
        COUNT.with(Cell::get)
    }
}
