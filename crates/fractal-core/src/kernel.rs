use crate::integer::Integer;

const INVALID_BOUND: f32 = -1_267_650_600_228_229_401_496_703_205_376.0;
const IDENTITY: [f64; 4] = [1.0, 0.0, 0.0, 1.0];

#[derive(Clone, Copy)]
struct Block {
    a: [f64; 4],
    b: [f64; 4],
    radius: f64,
    c_radius: f64,
}
impl Block {
    const fn new() -> Self {
        Self {
            a: IDENTITY,
            b: [0.0; 4],
            radius: f64::INFINITY,
            c_radius: f64::INFINITY,
        }
    }
}

fn norm(matrix: &[f64; 4]) -> f64 {
    let first = matrix[0].abs() + matrix[1].abs();
    let second = matrix[2].abs() + matrix[3].abs();
    if first.is_nan() || second.is_nan() {
        f64::NAN
    } else {
        first.max(second)
    }
}

fn minimum(a: f64, b: f64) -> f64 {
    // Rust f64::min ignores a single NaN; JS Math.min propagates it.
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn matrix_product(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

// Two independent matrix columns, with exactly the scalar multiply/add order.
// Deliberately no relaxed SIMD or multiply-add contraction.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn matrix_product(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    use core::arch::wasm32::*;
    let top = f64x2(b[0], b[1]);
    let bottom = f64x2(b[2], b[3]);
    let row0 = f64x2_add(
        f64x2_mul(f64x2_splat(a[0]), top),
        f64x2_mul(f64x2_splat(a[1]), bottom),
    );
    let row1 = f64x2_add(
        f64x2_mul(f64x2_splat(a[2]), top),
        f64x2_mul(f64x2_splat(a[3]), bottom),
    );
    [
        f64x2_extract_lane::<0>(row0),
        f64x2_extract_lane::<1>(row0),
        f64x2_extract_lane::<0>(row1),
        f64x2_extract_lane::<1>(row1),
    ]
}

fn exponent(matrix: &[f64; 4]) -> i32 {
    let max = matrix.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max > 0.0 {
        libm::floor(libm::log2(max)) as i32
    } else {
        0
    }
}

struct Scratch {
    cx: Integer,
    cy: Integer,
    x: Integer,
    y: Integer,
    xx: Integer,
    yy: Integer,
    real: Integer,
    xy: Integer,
    shifted_x: Integer,
    shifted_y: Integer,
    next_x: Integer,
    next_y: Integer,
}
impl Scratch {
    fn new() -> Self {
        Self {
            cx: Integer::zero(0),
            cy: Integer::zero(0),
            x: Integer::zero(0),
            y: Integer::zero(0),
            xx: Integer::zero(0),
            yy: Integer::zero(0),
            real: Integer::zero(0),
            xy: Integer::zero(0),
            shifted_x: Integer::zero(0),
            shifted_y: Integer::zero(0),
            next_x: Integer::zero(0),
            next_y: Integer::zero(0),
        }
    }
    fn prepare(&mut self, x: &[u8], y: &[u8], bits: usize) -> Result<(), i32> {
        self.cx.reserve(x.len() / 9 + 4);
        self.cy.reserve(y.len() / 9 + 4);
        self.cx.parse_into(x).map_err(|_| -2)?;
        self.cy.parse_into(y).map_err(|_| -2)?;
        self.cx.shift_left_word();
        self.cy.shift_left_word();
        // At the start of a nonescaping iteration |z| <= 256. The next value
        // is bounded by |c| + 2^17; products and signed fold factors fit here.
        let largest = self
            .cx
            .bit_length()
            .max(self.cy.bit_length())
            .max(bits + 18);
        let words = 2 * (largest / 32 + 2) + 4;
        for value in [
            &mut self.cx,
            &mut self.cy,
            &mut self.x,
            &mut self.y,
            &mut self.xx,
            &mut self.yy,
            &mut self.real,
            &mut self.xy,
            &mut self.shifted_x,
            &mut self.shifted_y,
            &mut self.next_x,
            &mut self.next_y,
        ] {
            value.reserve(words);
        }
        self.x.clear();
        self.y.clear();
        Ok(())
    }
}

pub(crate) struct Kernel {
    scratch: Scratch,
    output: [Vec<f32>; 5],
    blocks: [Block; 10],
    bits: usize,
    iterations: usize,
    fold: usize,
    celtic: usize,
    capacity: usize,
    length: usize,
    started: bool,
    done: bool,
}

impl Kernel {
    pub fn new() -> Self {
        Self {
            scratch: Scratch::new(),
            output: std::array::from_fn(|_| Vec::new()),
            blocks: [Block::new(); 10],
            bits: 0,
            iterations: 0,
            fold: 0,
            celtic: 0,
            capacity: 0,
            length: 0,
            started: false,
            done: false,
        }
    }

    pub fn begin(
        &mut self,
        x: &[u8],
        y: &[u8],
        bits: usize,
        iterations: usize,
        fold: usize,
        celtic: usize,
    ) -> Result<(), i32> {
        self.started = false;
        self.done = false;
        self.length = 0;
        self.capacity = 0;
        if !(1..=4096).contains(&bits)
            || !(1..=65536).contains(&iterations)
            || fold > 1024
            || celtic > 1024
            || x.len() + y.len() > 16384
        {
            return Err(-1);
        }
        self.bits = bits + 32;
        self.scratch.prepare(x, y, self.bits)?;
        self.iterations = iterations;
        self.fold = fold;
        self.celtic = celtic;
        self.capacity = (iterations + 1).div_ceil(1024) * 1024;
        for (kind, output) in self.output.iter_mut().enumerate() {
            let len = if kind == 1 && celtic == 0 {
                4
            } else {
                self.capacity * 4
            };
            let fill = if kind == 4 { INVALID_BOUND } else { 0.0 };
            output.resize(len, fill);
            output.fill(fill);
        }
        self.blocks.fill(Block::new());
        self.started = true;
        Ok(())
    }

    pub fn step(&mut self, max_steps: usize) -> i32 {
        if !self.started {
            return -3;
        }
        if max_steps == 0 {
            return -4;
        }
        if self.done {
            return 1;
        }
        for _ in 0..max_steps.min(self.iterations + 1 - self.length) {
            let index = self.length;
            let s = &mut self.scratch;
            let (xm, xe) = s.x.to_fe(self.bits);
            let (ym, ye) = s.y.to_fe(self.bits);
            self.output[0][index * 4..index * 4 + 4]
                .copy_from_slice(&[xm as f32, xe as f32, ym as f32, ye as f32]);
            s.xx.multiply(&s.x, &s.x);
            s.yy.multiply(&s.y, &s.y);
            s.real.combine(&s.xx, &s.yy, true);
            if self.celtic != 0 {
                let (m, e) = s.real.to_fe(self.bits * 2);
                self.output[1][index * 4..index * 4 + 4]
                    .copy_from_slice(&[m as f32, e as f32, 0.0, 0.0]);
            }
            self.length = index + 1;
            let x = xm * libm::scalbn(1.0, xe);
            let y = ym * libm::scalbn(1.0, ye);
            if x * x + y * y > 65536.0 {
                self.done = true;
                return 1;
            }

            let fold = self.fold as f64 / 1024.0;
            let celtic = self.celtic as f64 / 1024.0;
            let celtic_bound = if self.celtic != 0 {
                (x - y).abs().min((x + y).abs()) / 2.0
            } else {
                f64::INFINITY
            };
            let bound = x
                .abs()
                .min(y.abs())
                .min(1e-7 * libm::hypot(x, y))
                .min(celtic_bound);
            // The reference derivative uses rounded f64 signs (including underflow),
            // whereas the Celtic branch uses the exact integer difference's sign.
            let product_sign = if x == 0.0 || y == 0.0 {
                0
            } else if x.is_sign_negative() == y.is_sign_negative() {
                1
            } else {
                -1
            };
            let p = if product_sign < 0 {
                1.0 - 2.0 * fold
            } else if product_sign > 0 {
                1.0
            } else {
                1.0 - fold
            };
            let e = if s.real.sign() < 0 {
                1.0 - 2.0 * celtic
            } else if s.real.sign() > 0 {
                1.0
            } else {
                1.0 - celtic
            };
            let derivative = [2.0 * e * x, -2.0 * e * y, 2.0 * p * y, 2.0 * p * x];
            for level in 0..10 {
                let block = &mut self.blocks[level];
                block.radius = minimum(block.radius, bound / (4.0 * norm(&block.a)));
                let b_norm = norm(&block.b);
                if b_norm > 0.0 {
                    block.c_radius = minimum(block.c_radius, bound / (4.0 * b_norm));
                }
                block.a = matrix_product(&derivative, &block.a);
                block.b = matrix_product(&derivative, &block.b);
                block.b[0] += 1.0;
                block.b[3] += 1.0;
                let stride = 2usize << level;
                if (index + 1) & (stride - 1) == 0 {
                    if norm(&block.a).is_finite()
                        && norm(&block.b).is_finite()
                        && block.radius > 0.0
                        && block.c_radius > 0.0
                    {
                        let offset =
                            (self.capacity - (self.capacity >> level) + index / stride) * 4;
                        let a_exp = exponent(&block.a);
                        let b_exp = exponent(&block.b);
                        // Multiplication by 2^-e matches JS even in extreme exponents.
                        let a_scale = libm::scalbn(1.0, -a_exp);
                        let b_scale = libm::scalbn(1.0, -b_exp);
                        for component in 0..4 {
                            self.output[2][offset + component] =
                                (block.a[component] * a_scale) as f32;
                            self.output[3][offset + component] =
                                (block.b[component] * b_scale) as f32;
                        }
                        self.output[4][offset..offset + 4].copy_from_slice(&[
                            libm::log2(block.radius) as f32,
                            libm::log2(block.c_radius) as f32,
                            a_exp as f32,
                            b_exp as f32,
                        ]);
                    }
                    *block = Block::new();
                }
            }

            let real_shift = if s.real.sign() < 0 && self.celtic != 0 {
                s.real.scale(1024 - 2 * self.celtic as i32);
                self.bits + 10
            } else {
                self.bits
            };
            s.shifted_x.shift_right(&s.real, real_shift);
            s.next_x.combine(&s.shifted_x, &s.cx, false);
            s.xy.multiply(&s.x, &s.y);
            let imaginary_shift = if s.xy.sign() < 0 {
                s.xy.scale(2 * (1024 - 2 * self.fold as i32));
                self.bits + 10
            } else {
                s.xy.scale(2);
                self.bits
            };
            s.shifted_y.shift_right(&s.xy, imaginary_shift);
            s.next_y.combine(&s.shifted_y, &s.cy, false);
            std::mem::swap(&mut s.x, &mut s.next_x);
            std::mem::swap(&mut s.y, &mut s.next_y);
            if index == self.iterations {
                self.done = true;
                return 1;
            }
        }
        0
    }

    pub fn length(&self) -> usize {
        self.length
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn result(&self, kind: usize) -> &[f32] {
        if !self.started {
            return &[];
        }
        self.output.get(kind).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_overflowed_matrix_row_cannot_be_hidden_by_the_other_row() {
        // Math.max propagates NaN; a finite row must not make the BLA admissible.
        assert!(super::norm(&[f64::NAN, 0.0, 1.0, 0.0]).is_nan());
        assert!(super::minimum(1.0, f64::NAN).is_nan());
    }
}
