//! Allocation-free frame planning. JavaScript is only the WebGL host interface.
//! All ABI buffers are owned by this module; imported GPU functions must not
//! re-enter the runtime or retain pointers after memory growth.
use crate::camera::Camera;
use std::cell::UnsafeCell;
use std::sync::Mutex;

const INPUT_LEN: usize = 64;
const TEXT_LEN: usize = 16384;
const MAX_BANDS: usize = 16;
const FLIGHT_SPEED: f64 = 0.55;
const MAX_FRAME_SECONDS: f64 = 0.1;
struct Shared<T>(UnsafeCell<T>);
unsafe impl<T> Sync for Shared<T> {}
static INPUT: Shared<[f64; INPUT_LEN]> = Shared(UnsafeCell::new([0.0; INPUT_LEN]));
static TEXT: Shared<[u8; TEXT_LEN]> = Shared(UnsafeCell::new([0; TEXT_LEN]));
static STATS: Shared<[f64; 16]> = Shared(UnsafeCell::new([0.0; 16]));
static ZERO_PIXEL: [f32; 4] = [0.0; 4];
static RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "gpu")]
unsafe extern "C" {
    fn ring_target(width: u32, height: u32) -> i32;
    fn temporal_targets(width: u32, height: u32) -> i32;
    fn draw(pointer: u32) -> i32;
    fn reference_texture(kind: u32, pointer: u32, length: u32, width: u32, height: u32) -> i32;
}
fn target_ring(width: u32, height: u32) -> i32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ring_target(width, height)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (width, height);
        1
    }
}
fn target_temporal(width: u32, height: u32) -> i32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        temporal_targets(width, height)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (width, height);
        1
    }
}
fn submit(pass: &Pass) -> Result<(), i32> {
    #[cfg(target_arch = "wasm32")]
    let result = unsafe { draw(pass as *const Pass as usize as u32) };
    #[cfg(not(target_arch = "wasm32"))]
    let result = {
        let _ = pass;
        0
    };
    if result < 0 { Err(result) } else { Ok(()) }
}

/// Header: kind,path,flags,x,y,width,height,index. Flags: 1=scissor,
/// 2=contiguous ring assembly. Uniform offsets are documented
/// in src/gpu/render-wasm.ts; the layout is deliberately plain little-endian data.
#[repr(C)]
struct Pass {
    header: [i32; 8],
    values: [f32; 40],
    band_layout: [f32; 64],
    band_place: [f32; 64],
}
impl Pass {
    const fn new() -> Self {
        Self {
            header: [0; 8],
            values: [0.0; 40],
            band_layout: [0.0; 64],
            band_place: [0.0; 64],
        }
    }
}
const _: () = assert!(std::mem::size_of::<Pass>() == 704);

#[derive(Clone, Copy)]
struct Band {
    x: u32,
    row: u32,
    angles: u32,
    columns: u32,
    strips: u32,
    rings: i64,
    step: f64,
    radius: f64,
    log_radius_over_height: f64,
    outwards: f64,
    inwards: f64,
    first: i64,
    last: i64,
    valid: bool,
}
impl Band {
    const EMPTY: Self = Self {
        x: 0,
        row: 0,
        angles: 0,
        columns: 0,
        strips: 0,
        rings: 0,
        step: 0.0,
        radius: 0.0,
        log_radius_over_height: 0.0,
        outwards: 0.0,
        inwards: 0.0,
        first: 0,
        last: 0,
        valid: false,
    };
}
#[derive(Clone, Copy)]
struct Layout {
    bands: [Band; MAX_BANDS],
    count: usize,
    width: u32,
    height: u32,
    contiguous: bool,
    warm_refresh_limit: f64,
}
impl Layout {
    const fn empty() -> Self {
        Self {
            bands: [Band::EMPTY; MAX_BANDS],
            count: 0,
            width: 0,
            height: 0,
            contiguous: false,
            warm_refresh_limit: 0.0,
        }
    }
    fn create(width: u32, height: u32, maximum: u32) -> Option<Self> {
        if width == 0 || height == 0 || maximum == 0 {
            return None;
        }
        let mut radii = [0.0; MAX_BANDS];
        radii[0] = libm::hypot(width as f64, height as f64) / 2.0 + 1.0;
        let mut count = 1;
        while count < MAX_BANDS && radii[count - 1] > 1.0 {
            radii[count] = radii[count - 1] / 2.0;
            count += 1;
        }
        let mut layout = Self::empty();
        layout.count = count;
        layout.contiguous = true;
        let (mut x, mut row, mut row_height, mut texture_width) = (0u64, 0u64, 0u64, 0u64);
        for (index, radius) in radii.iter().copied().enumerate().take(count) {
            let angles_f = libm::ceil(2.0 * std::f64::consts::PI * radius * 1.5).max(16.0);
            if angles_f > u32::MAX as f64 {
                return None;
            }
            let angles = angles_f as u64;
            let strips = angles.div_ceil(maximum as u64);
            layout.contiguous &= strips == 1;
            let columns = angles.div_ceil(strips);
            let step = 2.0 * std::f64::consts::PI / angles as f64 / std::f64::consts::LN_2;
            // One maximum-duration guided camera step, plus two rows per
            // band for ceil/floor boundaries. Unlike a screen-pixel multiple,
            // this limit also covers ultrawide and portrait layouts.
            layout.warm_refresh_limit += angles as f64
                * (FLIGHT_SPEED * MAX_FRAME_SECONDS * std::f64::consts::LOG2_10 / step + 2.0);
            let outwards = if index > 0 { 0.5 } else { 0.0 };
            let inwards = if index + 1 < count { 0.5 } else { 0.0 };
            let rings = libm::ceil((1.0 + outwards + inwards) / step) as u64 + 7;
            let physical_height = rings.checked_mul(strips)?;
            if physical_height > maximum as u64 {
                return None;
            }
            if texture_width == 0 {
                texture_width = columns;
            }
            if x + columns > texture_width {
                row += row_height;
                x = 0;
                row_height = 0;
            }
            layout.bands[index] = Band {
                x: x as u32,
                row: row as u32,
                angles: angles as u32,
                columns: columns as u32,
                strips: strips as u32,
                rings: rings as i64,
                step,
                radius,
                log_radius_over_height: libm::log2(radius / height as f64),
                outwards,
                inwards,
                ..Band::EMPTY
            };
            x += columns;
            row_height = row_height.max(physical_height);
        }
        let height = row + row_height;
        if texture_width > maximum as u64 || height > maximum as u64 {
            return None;
        }
        layout.width = texture_width as u32;
        layout.height = height as u32;
        Some(layout)
    }
}

#[derive(Clone, Copy)]
struct View {
    width: u32,
    height: u32,
    maximum: u32,
    iterations: u32,
    aa: u32,
    fold: f64,
    celtic: f64,
    hue: f64,
    log_scale: f64,
    scale: f64,
    center: [f64; 2],
    offset: [f64; 4],
    guided: bool,
    temporal: bool,
    reference_key: f64,
    reference_version: f64,
    reference_length: u32,
    reference_capacity: u32,
    has_reference: bool,
    reference_fold: f64,
    reference_celtic: f64,
    reference_iterations: u32,
    has_position: bool,
}
fn integer(value: f64, min: f64, max: f64) -> Result<u32, i32> {
    if value.is_finite() && value >= min && value <= max && libm::floor(value) == value {
        Ok(value as u32)
    } else {
        Err(-20)
    }
}
impl View {
    fn read(input: &[f64; INPUT_LEN]) -> Result<Self, i32> {
        for index in [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19] {
            if !input[index].is_finite() {
                return Err(-20);
            }
        }
        if !(0.0..=1.0).contains(&input[5]) || !(0.0..=1.0).contains(&input[6]) {
            return Err(-20);
        }
        if input[8] >= -8.0 && (!input[9].is_finite() || input[9] <= 0.0) {
            return Err(-20);
        }
        Ok(Self {
            width: integer(input[0], 0.0, i32::MAX as f64)?,
            height: integer(input[1], 0.0, i32::MAX as f64)?,
            maximum: integer(input[2], 1.0, i32::MAX as f64)?,
            iterations: integer(input[3], 1.0, 1_000_000.0)?,
            aa: integer(input[4], 1.0, 5.0)?,
            fold: input[5],
            celtic: input[6],
            hue: input[7],
            log_scale: input[8],
            scale: input[9],
            center: [input[10], input[11]],
            offset: [input[12], input[13], input[14], input[15]],
            guided: input[16] != 0.0,
            temporal: input[17] != 0.0,
            reference_key: input[18],
            reference_version: input[19],
            reference_length: integer(input[20], 0.0, 1_000_001.0)?,
            reference_capacity: integer(input[21], 0.0, 1_001_472.0)?,
            has_reference: input[22] != 0.0,
            reference_fold: input[23],
            reference_celtic: input[24],
            reference_iterations: integer(input[25], 0.0, 1_000_000.0)?,
            has_position: input[26] != 0.0,
        })
    }
    fn path(&self) -> i32 {
        if self.log_scale >= -8.0 {
            0
        } else if self.log_scale > -80.0 {
            1
        } else {
            2
        }
    }
    fn ring_key(&self) -> [f64; 15] {
        [
            self.width as f64,
            self.height as f64,
            self.reference_version,
            self.reference_key,
            self.center[0],
            self.center[1],
            self.offset[0],
            self.offset[1],
            self.offset[2],
            self.offset[3],
            self.iterations as f64,
            self.fold,
            self.celtic,
            self.hue,
            self.maximum as f64,
        ]
    }
    fn temporal_key(&self) -> [f64; 10] {
        [
            self.width as f64,
            self.height as f64,
            self.aa as f64,
            self.reference_version,
            self.reference_key,
            self.path() as f64,
            self.iterations as f64,
            self.fold,
            self.celtic,
            self.hue,
        ]
    }
}

struct Runtime {
    camera: Camera,
    reference_camera: Camera,
    candidate_camera: Camera,
    previous_camera: Camera,
    previous_valid: bool,
    position_valid: bool,
    has_reference: bool,
    reference_pending: bool,
    playing: bool,
    end_zoom: f64,
    last_time: f64,
    dirty: bool,
    dragging: bool,
    pointer_x: f64,
    pointer_y: f64,
    frame_times: [f64; 2048],
    frame_head: usize,
    frame_count: usize,
    pass: Pass,
    error: i32,
    stats: [f64; 16],
    ring_layout: Layout,
    ring_key: [f64; 15],
    ring_valid: bool,
    ring_disabled: bool,
    ring_complete: bool,
    ring_origin: f64,
    previous_ring_scale: f64,
    ring_velocity: f64,
    temporal_key: [f64; 10],
    temporal_samples: u32,
    history_samples: u32,
    stationary_samples: u32,
    sample_index: u64,
    temporal_width: u32,
    temporal_height: u32,
    temporal_next: u32,
}
impl Runtime {
    fn new() -> Self {
        Self {
            camera: Camera::new(),
            reference_camera: Camera::new(),
            candidate_camera: Camera::new(),
            previous_camera: Camera::new(),
            previous_valid: false,
            position_valid: false,
            has_reference: false,
            reference_pending: false,
            playing: false,
            end_zoom: 120.0,
            last_time: f64::NAN,
            dirty: true,
            dragging: false,
            pointer_x: 0.0,
            pointer_y: 0.0,
            frame_times: [0.0; 2048],
            frame_head: 0,
            frame_count: 0,
            pass: Pass::new(),
            error: 0,
            stats: [0.0; 16],
            ring_layout: Layout::empty(),
            ring_key: [f64::NAN; 15],
            ring_valid: false,
            ring_disabled: false,
            ring_complete: false,
            ring_origin: 0.0,
            previous_ring_scale: f64::NAN,
            ring_velocity: 0.0,
            temporal_key: [f64::NAN; 10],
            temporal_samples: 1,
            history_samples: 0,
            stationary_samples: 0,
            sample_index: 0,
            temporal_width: 0,
            temporal_height: 0,
            temporal_next: 0,
        }
    }
    fn reset_temporal(&mut self) {
        self.previous_valid = false;
        self.temporal_samples = 1;
        self.history_samples = 0;
        self.stationary_samples = 0;
        self.sample_index = 0;
    }
    fn reset_rings(&mut self) {
        self.ring_valid = false;
        self.ring_disabled = false;
        self.ring_complete = false;
        self.previous_ring_scale = f64::NAN;
        self.ring_velocity = 0.0;
    }
    fn settling(&self) -> bool {
        self.temporal_samples > 1 && self.stationary_samples < self.temporal_samples
    }
    fn publish_stats(&mut self) {
        self.stats[8] = self.settling() as u8 as f64;
        self.stats[9] = self.camera.zoom();
        self.stats[10] = self.camera.log_scale;
        self.stats[11] = self.camera.bits as f64;
        self.stats[12] = self.playing as u8 as f64;
        self.stats[13] = self.reference_pending as u8 as f64;
        unsafe {
            *STATS.0.get() = self.stats;
        }
    }
    fn uniforms(&mut self, view: &View, log_scale: f64) {
        let exponent = libm::floor(log_scale);
        let v = &mut self.pass.values;
        v.fill(0.0);
        v[0] = view.center[0] as f32;
        v[1] = view.center[1] as f32;
        v[2] = view.scale as f32;
        v[3] = view.fold as f32;
        v[4] = view.celtic as f32;
        v[5] = view.hue as f32;
        v[6] = (view.width as f64 / view.height as f64) as f32;
        v[7] = view.width as f32;
        v[8] = view.height as f32;
        v[9] = view.aa as f32;
        v[12] = view.iterations as f32;
        v[13] = view.reference_length as f32;
        for (out, value) in v[14..18].iter_mut().zip(view.offset) {
            *out = value as f32;
        }
        v[18] = crate::camera::pow2(log_scale - exponent) as f32;
        v[19] = exponent as f32;
        v[34] = (view.hue != 0.0) as u8 as f32;
        v[20] = 1024.0;
        v[21] = (if view.has_reference {
            view.reference_capacity
        } else {
            1024
        } as f64
            / 1024.0) as f32;
    }
    // The flat arguments mirror the fixed GPU ABI rather than an allocating command list.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        kind: i32,
        path: i32,
        flags: i32,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        index: u32,
    ) -> Result<(), i32> {
        self.pass.header = [
            kind,
            path,
            flags,
            x,
            y,
            width as i32,
            height as i32,
            index as i32,
        ];
        submit(&self.pass)?;
        self.stats[14] += 1.0;
        Ok(())
    }
    fn draw_rows(&mut self, band: Band, first: i64, last: i64, view: &View) -> Result<(), i32> {
        let mut index = first;
        while index <= last {
            let row = modulo_i(index, band.rings);
            let rows = (last - index + 1).min(band.rings - row);
            let log_scale = self.ring_origin - index as f64 * band.step;
            let inner = self.ring_origin - (index + rows - 1) as f64 * band.step;
            let path = if inner > -90.0 { 1 } else { 2 };
            self.uniforms(view, log_scale);
            for strip in 0..band.strips {
                let first_angle = strip * band.columns;
                let width = band.columns.min(band.angles - first_angle);
                let y = band.row as i64 + strip as i64 * band.rings + row;
                self.pass.values[22] = (band.x as f64 - first_angle as f64) as f32;
                self.pass.values[23] = y as f32;
                self.pass.values[24] = band.angles as f32;
                self.pass.values[25] = band.step as f32;
                self.draw(1, path, 1, band.x as i32, y as i32, width, rows as u32, 0)?;
            }
            self.stats[2] += rows as f64;
            self.stats[3] += rows as f64 * band.angles as f64;
            index += rows;
        }
        Ok(())
    }
    fn render_rings(&mut self, view: &View, preparing: bool) -> Result<bool, i32> {
        let key = view.ring_key();
        let pixels = view.width as f64 * view.height as f64;
        if key != self.ring_key || !self.ring_valid && !self.ring_disabled {
            self.reset_rings();
            self.ring_key = key;
            let Some(layout) = Layout::create(view.width, view.height, view.maximum) else {
                self.ring_disabled = true;
                return Ok(false);
            };
            // Preparation budgets whole rows. The outermost band is widest;
            // if it cannot fit even once, preparation could never make progress.
            if preparing && pixels / 2.0 < layout.bands[0].angles as f64 {
                self.ring_disabled = true;
                return Ok(false);
            }
            let target = target_ring(layout.width, layout.height);
            if target < 0 {
                return Err(target);
            }
            if target == 0 {
                self.ring_disabled = true;
                return Ok(false);
            }
            self.ring_layout = layout;
            self.ring_origin = view.log_scale + layout.bands[0].log_radius_over_height;
            self.ring_valid = true;
            self.stats[4] += 1.0;
        }
        if self.ring_disabled {
            return Ok(false);
        }
        if preparing && pixels / 2.0 < self.ring_layout.bands[0].angles as f64 {
            self.ring_disabled = true;
            return Ok(false);
        }
        let movement = (view.log_scale - self.previous_ring_scale).abs();
        self.previous_ring_scale = view.log_scale;
        if preparing {
            self.ring_velocity = 0.0;
        } else if movement.is_finite() {
            self.ring_velocity += (movement - self.ring_velocity) * 0.2;
        }
        let mut expected = 0.0;
        for band in &self.ring_layout.bands[..self.ring_layout.count] {
            expected += self.ring_velocity / band.step * band.angles as f64;
        }
        if expected > pixels && !self.ring_complete {
            return Ok(false);
        }
        // Fixed work storage replaces per-frame arrays, maps and range objects.
        let mut wanted = [[0i64; 2]; MAX_BANDS];
        let mut ranges = [[[0i64; 2]; 2]; MAX_BANDS];
        let mut counts = [0usize; MAX_BANDS];
        let mut missing = 0.0;
        for i in 0..self.ring_layout.count {
            let band = &self.ring_layout.bands[i];
            let outer = view.log_scale + band.log_radius_over_height + band.outwards;
            let span = 1.0 + band.outwards + band.inwards;
            let first_f = libm::floor((self.ring_origin - outer) / band.step);
            let last_f = libm::ceil((self.ring_origin - outer + span) / band.step);
            if !first_f.is_finite()
                || !last_f.is_finite()
                || first_f.abs() > 9_007_199_254_740_990.0
                || last_f.abs() > 9_007_199_254_740_990.0
            {
                return Ok(false);
            }
            let first = first_f as i64 - 2;
            let last = last_f as i64 + 2;
            wanted[i] = [first, last];
            let overlap_first = band.first.max(first);
            let overlap_last = band.last.min(last);
            if !band.valid || overlap_first > overlap_last {
                ranges[i][0] = [first, last];
                counts[i] = 1;
            } else {
                if first < overlap_first {
                    ranges[i][counts[i]] = [first, overlap_first - 1];
                    counts[i] += 1;
                }
                if last > overlap_last {
                    ranges[i][counts[i]] = [overlap_last + 1, last];
                    counts[i] += 1;
                }
            }
            let mut band_missing = 0.0;
            for range in &ranges[i][..counts[i]] {
                band_missing += (range[1] - range[0] + 1) as f64 * band.angles as f64;
            }
            missing += band_missing;
        }
        // A dropped animation frame can invalidate more than one screen of
        // rows in an otherwise complete cache. Finish this bounded refresh
        // before assembly; reverting to half-screen cold fill can never catch
        // up with a subsequent 25 FPS flight. Large jumps retain the fallback.
        let complete_warm_frame =
            self.ring_complete && missing <= self.ring_layout.warm_refresh_limit;
        if expected > pixels && !complete_warm_frame {
            self.ring_complete = false;
            return Ok(false);
        }
        let mut budget = if preparing {
            missing.min(libm::floor(pixels / 2.0))
        } else if missing <= pixels || complete_warm_frame {
            missing
        } else {
            libm::floor(pixels / 2.0)
        };
        let mut complete = true;
        for i in 0..self.ring_layout.count {
            // Commit overlap changes only after choosing to refresh. Skipped
            // large jumps must not discard rows still useful at the old view.
            let band = &mut self.ring_layout.bands[i];
            if band.valid && band.last >= wanted[i][0] && band.first <= wanted[i][1] {
                band.first = band.first.max(wanted[i][0]);
                band.last = band.last.min(wanted[i][1]);
            } else {
                band.valid = false;
            }
            for range in &ranges[i][..counts[i]] {
                let band = self.ring_layout.bands[i];
                let rows =
                    (range[1] - range[0] + 1).min(libm::floor(budget / band.angles as f64) as i64);
                if rows <= 0 {
                    continue;
                }
                let from = if band.valid && range[1] < band.first {
                    range[1] - rows + 1
                } else {
                    range[0]
                };
                let to = from + rows - 1;
                self.draw_rows(band, from, to, view)?;
                let band = &mut self.ring_layout.bands[i];
                if band.valid {
                    band.first = band.first.min(from);
                    band.last = band.last.max(to);
                } else {
                    band.first = from;
                    band.last = to;
                    band.valid = true;
                }
                budget -= rows as f64 * band.angles as f64;
            }
            let band = self.ring_layout.bands[i];
            if !band.valid || band.first > wanted[i][0] || band.last < wanted[i][1] {
                complete = false;
            }
        }
        self.ring_complete = complete;
        if !complete {
            return Ok(false);
        }
        if preparing {
            return Ok(true);
        }
        self.uniforms(view, view.log_scale);
        self.pass.values[33] = self.ring_layout.count as f32;
        for (index, band) in self.ring_layout.bands[..self.ring_layout.count]
            .iter()
            .enumerate()
        {
            let p = index * 4;
            self.pass.band_layout[p..p + 4].copy_from_slice(&[
                band.row as f32,
                band.angles as f32,
                band.step as f32,
                band.rings as f32,
            ]);
            self.pass.band_place[p..p + 4].copy_from_slice(&[
                modulo_f(
                    (self.ring_origin - view.log_scale) / band.step,
                    band.rings as f64,
                ) as f32,
                band.radius as f32,
                band.x as f32,
                band.columns as f32,
            ]);
        }
        self.draw(
            2,
            view.path(),
            if self.ring_layout.contiguous { 2 } else { 0 },
            0,
            0,
            view.width,
            view.height,
            0,
        )?;
        self.stats[5] += 1.0;
        Ok(true)
    }
    fn render_temporal(&mut self, view: &View) -> Result<(), i32> {
        if !view.has_position || !self.position_valid {
            return Err(-22);
        }
        if view.width > view.maximum || view.height > view.maximum {
            return Err(-23);
        }
        if self.temporal_width != view.width || self.temporal_height != view.height {
            self.reset_temporal();
            let target = target_temporal(view.width, view.height);
            if target < 0 {
                return Err(target);
            }
            if target == 0 {
                return Err(-23);
            }
            self.temporal_width = view.width;
            self.temporal_height = view.height;
            self.temporal_next = 0;
        }
        let key = view.temporal_key();
        if key != self.temporal_key || view.aa != self.temporal_samples {
            self.reset_temporal();
        }
        self.temporal_key = key;
        self.temporal_samples = view.aa;
        let (mut scale, mut x, mut y) = (1.0, 0.0, 0.0);
        if self.previous_valid {
            let delta = self.camera.offset(&self.previous_camera);
            scale = crate::camera::pow2(self.camera.log_scale - self.previous_camera.log_scale);
            x = scaled_fe(delta[0], delta[1] - self.previous_camera.log_scale)
                / (view.width as f64 / view.height as f64);
            y = -scaled_fe(delta[2], delta[3] - self.previous_camera.log_scale);
            if !(x + y).is_finite() || !(0.8..=1.25).contains(&scale) || libm::hypot(x, y) > 0.25 {
                self.reset_temporal();
                self.temporal_samples = view.aa;
                scale = 1.0;
                x = 0.0;
                y = 0.0;
            }
        }
        let moving = self.previous_valid && (scale != 1.0 || x != 0.0 || y != 0.0);
        if moving {
            self.stationary_samples = 0;
        } else if self.stationary_samples == 0 {
            self.history_samples = 0;
            self.sample_index = 0;
        }
        let index = (self.sample_index % view.aa as u64) as f64;
        let mut sum = 0.0;
        for sample in 0..view.aa {
            sum += ((sample as f64 + 0.5) * 0.61803398875) % 1.0;
        }
        let jitter_x = (index + 0.5) / view.aa as f64 - 0.5;
        let jitter_y = ((index + 0.5) * 0.61803398875) % 1.0 - sum / view.aa as f64;
        let weight = if self.previous_valid && self.history_samples > 0 {
            (self.history_samples as f64 / (self.history_samples as f64 + 1.0))
                .min(1.0 - 1.0 / view.aa as f64)
                * if moving {
                    libm::exp(-4.0 * libm::log(scale).abs())
                } else {
                    1.0
                }
        } else {
            0.0
        };
        self.uniforms(view, view.log_scale);
        self.pass.values[9] = 1.0;
        self.pass.values[10] = jitter_x as f32;
        self.pass.values[11] = jitter_y as f32;
        self.draw(
            3,
            view.path(),
            0,
            0,
            0,
            view.width,
            view.height,
            self.temporal_next,
        )?;
        self.pass.values[26] = x as f32;
        self.pass.values[27] = y as f32;
        self.pass.values[28] = scale as f32;
        self.pass.values[29] = weight as f32;
        self.pass.values[30] = moving as u8 as f32;
        self.pass.values[31] = (1.0 / view.width as f64) as f32;
        self.pass.values[32] = (1.0 / view.height as f64) as f32;
        self.draw(
            4,
            view.path(),
            0,
            0,
            0,
            view.width,
            view.height,
            self.temporal_next,
        )?;
        self.draw(
            5,
            view.path(),
            0,
            0,
            0,
            view.width,
            view.height,
            self.temporal_next,
        )?;
        self.previous_camera.copy_from(&self.camera);
        self.previous_valid = true;
        self.history_samples = view.aa.min(self.history_samples + 1);
        if !moving {
            self.stationary_samples = self.stationary_samples.saturating_add(1);
        }
        self.sample_index = self.sample_index.wrapping_add(1);
        self.temporal_next = 1 - self.temporal_next;
        Ok(())
    }
    fn render(&mut self, view: &View) -> Result<(), i32> {
        if view.width < 1 || view.height < 1 {
            self.reset_temporal();
            return Ok(());
        }
        let path = view.path();
        if path != 0
            && (!view.has_reference
                || view.reference_fold != view.fold
                || view.reference_celtic != view.celtic
                || view.reference_iterations < view.iterations)
        {
            return Err(-21);
        }
        self.stats[0] = path as f64;
        self.stats[1] += 1.0;
        self.stats[6] = 0.0;
        let ring_used = view.guided && path != 0 && self.render_rings(view, false)?;
        self.stats[6] = ring_used as u8 as f64;
        if ring_used {
            self.reset_temporal();
        } else if view.temporal && view.aa > 1 {
            self.render_temporal(view)?;
        } else {
            self.reset_temporal();
            self.uniforms(view, view.log_scale);
            self.draw(0, path, 0, 0, 0, view.width, view.height, 0)?;
        }
        Ok(())
    }
    fn prepare_rings(&mut self, input: &[f64; INPUT_LEN]) -> Result<(), i32> {
        if !self.position_valid || !self.has_reference || self.reference_pending {
            return Err(-21);
        }
        // Overview uses the direct shader. Build the cache for the first float
        // view while retaining the actual camera and all visible frame state.
        let mut prepared = *input;
        let log_scale = self.camera.log_scale.min((-8.0_f64).next_down());
        prepared[8] = log_scale;
        prepared[9] = crate::camera::pow2(log_scale);
        let center = self.camera.center();
        prepared[10] = center[0];
        prepared[11] = center[1];
        prepared[12..16].copy_from_slice(&self.camera.offset(&self.reference_camera));
        prepared[22] = 1.0;
        prepared[26] = 1.0;
        let view = View::read(&prepared)?;
        if !view.guided || view.width == 0 || view.height == 0 {
            return Err(-20);
        }
        if view.reference_fold != view.fold
            || view.reference_celtic != view.celtic
            || view.reference_iterations < view.iterations
        {
            return Err(-21);
        }
        self.stats[6] = self.render_rings(&view, true)? as u8 as f64;
        Ok(())
    }
    fn camera_command(&mut self, op: u32, input: &mut [f64; INPUT_LEN]) -> Result<(), i32> {
        match op {
            0 => {
                self.camera.make(input[40])?;
                self.position_valid = true;
                self.has_reference = false;
                self.reference_pending = false;
            }
            1 => {
                if !input[42].is_finite() || input[42] <= 0.0 {
                    return Err(-20);
                }
                self.camera
                    .pan(input[40] / input[42], input[41] / input[42])?;
            }
            2 => {
                if !input[42].is_finite()
                    || !input[43].is_finite()
                    || input[42] <= 0.0
                    || input[43] <= 0.0
                {
                    return Err(-20);
                }
                self.camera.zoom_at(
                    (input[40] - input[42] / 2.0) / input[43],
                    input[41] / input[43] - 0.5,
                    (input[44] * 0.003).clamp(-2.0, 2.0),
                )?;
            }
            3 => self.camera.set_zoom(input[40])?,
            4 => {
                if !input[41].is_finite() {
                    return Err(-20);
                }
                self.playing = input[40] != 0.0;
                self.end_zoom = input[41];
                self.last_time = f64::NAN;
            }
            5 => {
                self.reference_camera.copy_from(&self.camera);
                self.has_reference = true;
            }
            6 => {
                self.last_time = f64::NAN;
                self.frame_count = 0;
                self.frame_head = 0;
                self.stats[15] = 0.0;
            }
            8 => {
                if !input[40].is_finite() || !input[41].is_finite() {
                    return Err(-20);
                }
                self.dragging = true;
                self.pointer_x = input[40];
                self.pointer_y = input[41];
            }
            9 => {
                if !input[40].is_finite()
                    || !input[41].is_finite()
                    || !input[42].is_finite()
                    || input[42] <= 0.0
                {
                    return Err(-20);
                }
                if self.dragging {
                    self.camera.pan(
                        (input[40] - self.pointer_x) / input[42],
                        (input[41] - self.pointer_y) / input[42],
                    )?;
                    self.pointer_x = input[40];
                    self.pointer_y = input[41];
                }
            }
            10 => self.dragging = false,
            _ => return Err(-20),
        }
        self.dirty = true;
        self.reset_temporal();
        Ok(())
    }
    fn record_frame(&mut self, now: f64, rendered: bool) {
        if rendered {
            if self.frame_count == self.frame_times.len() {
                self.frame_head = (self.frame_head + 1) % self.frame_times.len();
                self.frame_count -= 1;
            }
            self.frame_times[(self.frame_head + self.frame_count) % self.frame_times.len()] = now;
            self.frame_count += 1;
        }
        while self.frame_count > 0 && self.frame_times[self.frame_head] <= now - 1000.0 {
            self.frame_head = (self.frame_head + 1) % self.frame_times.len();
            self.frame_count -= 1;
        }
        self.stats[15] = if self.frame_count > 1 {
            let span = now - self.frame_times[self.frame_head];
            if span > 0.0 {
                (self.frame_count - 1) as f64 * 1000.0 / span
            } else {
                0.0
            }
        } else {
            0.0
        };
    }
    fn camera_frame(&mut self, input: &mut [f64; INPUT_LEN], now: f64) -> Result<bool, i32> {
        if !now.is_finite() || !self.position_valid {
            return Err(-20);
        }
        let elapsed = if self.last_time.is_finite() {
            ((now - self.last_time) / 1000.0).clamp(0.0, MAX_FRAME_SECONDS)
        } else {
            0.0
        };
        self.last_time = now;
        if self.playing && !self.reference_pending {
            self.camera.set_zoom(
                self.end_zoom
                    .min(self.camera.zoom() + elapsed * FLIGHT_SPEED),
            )?;
            self.dirty = true;
            if self.camera.zoom() >= self.end_zoom - 1e-10 {
                self.playing = false;
            }
        }
        input[8] = self.camera.log_scale;
        input[9] = crate::camera::pow2(self.camera.log_scale);
        let center = self.camera.center();
        input[10] = center[0];
        input[11] = center[1];
        let offset = if self.has_reference {
            self.camera.offset(&self.reference_camera)
        } else {
            [0.0; 4]
        };
        input[12..16].copy_from_slice(&offset);
        input[22] = self.has_reference as u8 as f64;
        input[26] = 1.0;
        // Input guided expresses route lock. A stationary camera uses temporal AA.
        input[27] = input[16];
        input[16] = if self.playing { input[16] } else { 0.0 };
        Ok((self.dirty || self.settling())
            && (self.camera.log_scale >= -8.0 || self.has_reference && !self.reference_pending))
    }
}
fn modulo_i(value: i64, divisor: i64) -> i64 {
    ((value % divisor) + divisor) % divisor
}
fn modulo_f(value: f64, divisor: f64) -> f64 {
    ((value % divisor) + divisor) % divisor
}
fn scaled_fe(mantissa: f64, exponent: f64) -> f64 {
    if mantissa == 0.0 {
        0.0
    } else {
        mantissa * crate::camera::pow2(exponent)
    }
}
fn with_runtime(work: impl FnOnce(&mut Runtime) -> Result<(), i32>) -> i32 {
    let mut lock = RUNTIME.lock().unwrap_or_else(|error| error.into_inner());
    let runtime = lock.get_or_insert_with(Runtime::new);
    let result = work(runtime);
    runtime.error = result.err().unwrap_or(0);
    if runtime.error != 0 {
        runtime.reset_temporal();
        runtime.reset_rings();
    }
    runtime.publish_stats();
    runtime.error
}

#[unsafe(no_mangle)]
pub extern "C" fn render_input_ptr() -> u32 {
    INPUT.0.get() as usize as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn render_text_ptr() -> u32 {
    TEXT.0.get() as usize as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn render_text_capacity() -> u32 {
    TEXT_LEN as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn render_stats_ptr() -> u32 {
    STATS.0.get() as usize as u32
}
#[unsafe(no_mangle)]
pub extern "C" fn render_error() -> i32 {
    RUNTIME
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |r| r.error)
}
#[unsafe(no_mangle)]
pub extern "C" fn render_reset_temporal() {
    let _ = with_runtime(|r| {
        r.reset_temporal();
        r.dirty = true;
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn render_reset_gpu() {
    let _ = with_runtime(|r| {
        r.reset_temporal();
        r.reset_rings();
        r.temporal_width = 0;
        r.temporal_height = 0;
        r.dirty = true;
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn render_dispose() {
    *RUNTIME.lock().unwrap_or_else(|e| e.into_inner()) = None;
    unsafe {
        *STATS.0.get() = [0.0; 16];
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn render_mark_dirty() {
    let _ = with_runtime(|r| {
        r.dirty = true;
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn render_camera_command(op: u32) -> i32 {
    with_runtime(|r| r.camera_command(op, unsafe { &mut *INPUT.0.get() }))
}
#[unsafe(no_mangle)]
pub extern "C" fn render_set_camera(
    x_len: u32,
    y_len: u32,
    bits: u32,
    log_scale: f64,
    decimal: u32,
) -> i32 {
    with_runtime(|r| {
        if x_len as u64 + y_len as u64 > TEXT_LEN as u64 {
            return Err(-20);
        }
        let text = unsafe { &*TEXT.0.get() };
        let x = &text[..x_len as usize];
        let y = &text[x_len as usize..(x_len + y_len) as usize];
        if decimal != 0 {
            r.camera.set_decimal(x, y, bits as usize, log_scale)?;
        } else {
            r.camera.set_fixed(x, y, bits as usize, log_scale)?;
        }
        r.position_valid = true;
        r.dirty = true;
        r.has_reference = false;
        r.reference_pending = false;
        r.reset_temporal();
        r.reset_rings();
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_set_route(x_len: u32, y_len: u32, end_zoom: f64, target_zoom: f64) -> i32 {
    with_runtime(|r| {
        if x_len as u64 + y_len as u64 > TEXT_LEN as u64
            || !end_zoom.is_finite()
            || !target_zoom.is_finite()
        {
            return Err(-20);
        }
        let text = unsafe { &*TEXT.0.get() };
        let log_scale = libm::log2(3.2);
        let bits = crate::camera::precision_bits(log_scale - end_zoom * libm::log2(10.0));
        r.camera.set_decimal(
            &text[..x_len as usize],
            &text[x_len as usize..(x_len + y_len) as usize],
            bits,
            log_scale,
        )?;
        r.camera.set_zoom(target_zoom)?;
        r.end_zoom = end_zoom;
        r.position_valid = true;
        r.dirty = true;
        r.has_reference = false;
        r.reference_pending = false;
        r.playing = false;
        r.last_time = f64::NAN;
        r.reset_temporal();
        r.reset_rings();
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_external_reference() {
    let _ = with_runtime(|r| {
        r.stats[7] += 1.0;
        r.reset_temporal();
        r.reset_rings();
        r.dirty = true;
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn render_set_position(x_len: u32, y_len: u32, bits: u32, log_scale: f64) -> i32 {
    with_runtime(|r| {
        if x_len as u64 + y_len as u64 > TEXT_LEN as u64 {
            return Err(-20);
        }
        let text = unsafe { &*TEXT.0.get() };
        r.camera.set_fixed(
            &text[..x_len as usize],
            &text[x_len as usize..(x_len + y_len) as usize],
            bits as usize,
            log_scale,
        )?;
        r.position_valid = true;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_camera_snapshot() -> i32 {
    with_runtime(|r| {
        let text = unsafe { &mut *TEXT.0.get() };
        let x_len = r.camera.write_fixed(0, text)?;
        let y_len = r.camera.write_fixed(1, &mut text[x_len..])?;
        let input = unsafe { &mut *INPUT.0.get() };
        input[48] = x_len as f64;
        input[49] = y_len as f64;
        input[50] = r.camera.bits as f64;
        input[51] = r.camera.log_scale;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_frame(mode: u32, now: f64) -> i32 {
    if mode == 2 {
        let mut unavailable = false;
        let status = with_runtime(|r| {
            r.prepare_rings(unsafe { &*INPUT.0.get() })?;
            unavailable = r.ring_disabled;
            Ok(())
        });
        // 0: progress/ready in stats[6]. 1: optional cache unavailable; the
        // caller may use the unchanged fullscreen renderer. Negative: error.
        return if status == 0 && unavailable {
            1
        } else {
            status
        };
    }
    with_runtime(|r| {
        let input = unsafe { &mut *INPUT.0.get() };
        let ready = if mode == 1 {
            r.camera_frame(input, now)?
        } else if mode == 0 {
            true
        } else {
            return Err(-20);
        };
        let result = if ready {
            let view = View::read(input)?;
            r.render(&view)
        } else {
            Ok(())
        };
        if mode == 1 {
            input[16] = input[27];
        }
        if result.is_ok() && ready {
            r.dirty = false;
        }
        r.record_frame(now, result.is_ok() && ready);
        result
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_begin_reference() -> i32 {
    with_runtime(|r| {
        if !r.position_valid {
            return Err(-20);
        }
        let input = unsafe { &mut *INPUT.0.get() };
        let iterations = integer(input[3], 1.0, 65536.0)?;
        if !input[5].is_finite()
            || !input[6].is_finite()
            || !(0.0..=1.0).contains(&input[5])
            || !(0.0..=1.0).contains(&input[6])
        {
            return Err(-20);
        }
        let text = unsafe { &mut *TEXT.0.get() };
        let x_len = r.camera.write_fixed(0, text)?;
        let y_len = r.camera.write_fixed(1, &mut text[x_len..])?;
        let status = crate::abi::begin_from_bytes(
            &text[..x_len],
            &text[x_len..x_len + y_len],
            r.camera.bits,
            iterations as usize,
            libm::floor(input[5] * 1024.0 + 0.5) as usize,
            libm::floor(input[6] * 1024.0 + 0.5) as usize,
        );
        if status < 0 {
            return Err(status);
        }
        r.candidate_camera.copy_from(&r.camera);
        r.reference_pending = true;
        r.dirty = true;
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn render_cancel_reference() {
    let _ = with_runtime(|r| {
        r.reference_pending = false;
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn render_reference_ready() -> i32 {
    with_runtime(|r| {
        let input = unsafe { &mut *INPUT.0.get() };
        let capacity = crate::abi::capacity();
        let length = crate::abi::orbit_length();
        if capacity < 1024 || length < 2 {
            return Err(-21);
        }
        for kind in 0..5u32 {
            let unused = kind == 1 && input[6] == 0.0;
            let pointer = if unused {
                ZERO_PIXEL.as_ptr() as usize as u32
            } else {
                crate::abi::result_ptr(kind)
            };
            let count = if unused {
                4
            } else {
                crate::abi::result_len(kind)
            };
            #[cfg(target_arch = "wasm32")]
            let status = unsafe {
                reference_texture(
                    kind,
                    pointer,
                    count,
                    if unused { 1 } else { 1024 },
                    if unused { 1 } else { capacity / 1024 },
                )
            };
            #[cfg(not(target_arch = "wasm32"))]
            let status = {
                let _ = (pointer, count);
                0
            };
            if status < 0 {
                return Err(status);
            }
        }
        r.reference_camera.copy_from(&r.candidate_camera);
        r.has_reference = true;
        r.reference_pending = false;
        input[19] += 1.0;
        input[20] = length as f64;
        input[21] = capacity as f64;
        input[22] = 1.0;
        input[23] = libm::floor(input[5] * 1024.0 + 0.5) / 1024.0;
        input[24] = libm::floor(input[6] * 1024.0 + 0.5) / 1024.0;
        input[25] = input[3];
        r.stats[7] += 1.0;
        r.reset_rings();
        r.reset_temporal();
        r.dirty = true;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts_preserve_sampling_and_obey_hardware_bounds() {
        let full_hd = Layout::create(1920, 1080, 16384).unwrap();
        assert_eq!((full_hd.width, full_hd.height), (10391, 2891));
        let four_k = Layout::create(3840, 2160, 16384).unwrap();
        assert_eq!((four_k.width, four_k.height), (10386, 10353));
        assert_eq!(four_k.bands[0].angles, 20772);
        assert_eq!(four_k.bands[0].strips, 2);
        assert!(Layout::create(3840, 2160, 4096).is_none());
        assert!(Layout::create(0, 1, 16384).is_none());
        for layout in [full_hd, four_k] {
            for band in &layout.bands[..layout.count] {
                assert!(band.x + band.columns <= layout.width);
                assert!(band.row as i64 + band.rings * band.strips as i64 <= layout.height as i64);
                assert!(band.columns * band.strips >= band.angles);
            }
        }
    }
    fn view() -> View {
        View {
            width: 320,
            height: 200,
            maximum: 16384,
            iterations: 16384,
            aa: 2,
            fold: 1.0,
            celtic: 0.0,
            hue: 0.0,
            log_scale: -100.0,
            scale: libm::pow(2.0, -100.0),
            center: [-1.7, -0.03],
            offset: [0.0; 4],
            guided: true,
            temporal: false,
            reference_key: 1.0,
            reference_version: 1.0,
            reference_length: 16385,
            reference_capacity: 17408,
            has_reference: true,
            reference_fold: 1.0,
            reference_celtic: 0.0,
            reference_iterations: 16384,
            has_position: false,
        }
    }
    #[test]
    fn warm_ring_frames_allocate_nothing_and_reuse_rows() {
        let mut runtime = Runtime::new();
        let mut view = view();
        for _ in 0..100 {
            runtime.render(&view).unwrap();
            if runtime.stats[6] == 1.0 {
                break;
            }
        }
        assert_eq!(runtime.stats[6], 1.0);
        let samples = runtime.stats[3];
        assert_eq!(
            crate::allocation_probe::measure(|| runtime.render(&view).unwrap()),
            0
        );
        assert_eq!(runtime.stats[3], samples);
        view.log_scale -= 0.01;
        assert_eq!(
            crate::allocation_probe::measure(|| runtime.render(&view).unwrap()),
            0
        );
        assert_eq!(runtime.stats[6], 1.0);
        assert!(runtime.stats[3] > samples);
    }
    #[test]
    fn temporal_sequence_settles_without_allocating() {
        let mut runtime = Runtime::new();
        let mut view = view();
        view.guided = false;
        view.temporal = true;
        view.has_position = true;
        runtime
            .camera
            .set_fixed(b"-256", b"-256", 128, view.log_scale)
            .unwrap();
        runtime.position_valid = true;
        assert_eq!(
            crate::allocation_probe::measure(|| runtime.render(&view).unwrap()),
            0
        );
        assert!(runtime.settling());
        assert_eq!(
            crate::allocation_probe::measure(|| runtime.render(&view).unwrap()),
            0
        );
        assert!(!runtime.settling());
        assert_eq!(runtime.stats[14], 6.0);
        assert_eq!(runtime.pass.values[29], 0.5);
    }
    #[test]
    fn modulo_and_path_thresholds_match_javascript() {
        assert_eq!(modulo_i(-1, 17), 16);
        assert_eq!(modulo_f(-0.5, 17.0), 16.5);
        let mut view = view();
        view.log_scale = -8.0;
        assert_eq!(view.path(), 0);
        view.log_scale = -8.00001;
        assert_eq!(view.path(), 1);
        view.log_scale = -80.0;
        assert_eq!(view.path(), 2);
    }
}
