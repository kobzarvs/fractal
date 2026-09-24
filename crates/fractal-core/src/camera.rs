//! Fixed-point camera matching src/camera.ts. Construction allocates every limb
//! buffer; movement, snapshots, FE offsets and decimal export reuse that storage.
use crate::integer::Integer;

const INPUT_BYTES: usize = 16_384;
// Covers a 16384-digit input, a decimal exponent of 10000, and 4096 binary bits.
// Eight buffers cost 128 KiB per camera; arithmetic visits only occupied limbs.
const LIMBS: usize = 4096;
const LOG2_3_2: f64 = 1.6780719051126378;
const LOG2_10: f64 = std::f64::consts::LOG2_10;
const LOG2_100: f64 = 6.643856189774724;
const TWO_52: f64 = 4503599627370496.0;

pub(crate) struct Camera {
    pub x: Integer,
    pub y: Integer,
    pub bits: usize,
    pub log_scale: f64,
    a: Integer,
    b: Integer,
    c: Integer,
    d: Integer,
    factor: Integer,
    product: Integer,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            x: Integer::zero(LIMBS),
            y: Integer::zero(LIMBS),
            bits: 128,
            log_scale: 0.0,
            a: Integer::zero(LIMBS),
            b: Integer::zero(LIMBS),
            c: Integer::zero(LIMBS),
            d: Integer::zero(LIMBS),
            factor: Integer::zero(LIMBS),
            product: Integer::zero(LIMBS),
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.x.copy_from(&other.x);
        self.y.copy_from(&other.y);
        self.bits = other.bits;
        self.log_scale = other.log_scale;
    }

    pub fn set_fixed(
        &mut self,
        x: &[u8],
        y: &[u8],
        bits: usize,
        log_scale: f64,
    ) -> Result<(), i32> {
        validate_state(bits, log_scale)?;
        if x.len() > INPUT_BYTES || y.len() > INPUT_BYTES {
            return Err(-2);
        }
        self.a.parse_into(x).map_err(|_| -2)?;
        self.b.parse_into(y).map_err(|_| -2)?;
        self.commit(bits, log_scale);
        Ok(())
    }

    pub fn set_decimal(
        &mut self,
        x: &[u8],
        y: &[u8],
        bits: usize,
        log_scale: f64,
    ) -> Result<(), i32> {
        validate_state(bits, log_scale)?;
        decimal_into(&mut self.a, x, bits)?;
        decimal_into(&mut self.b, y, bits)?;
        self.commit(bits, log_scale);
        Ok(())
    }

    fn commit(&mut self, bits: usize, log_scale: f64) {
        self.x.copy_from(&self.a);
        self.y.copy_from(&self.b);
        self.bits = bits;
        self.log_scale = log_scale;
    }

    pub fn make(&mut self, aspect: f64) -> Result<(), i32> {
        if !aspect.is_finite() || aspect <= 0.0 {
            return Err(-1);
        }
        let log_scale = libm::log2(3.2_f64.max(3.5 / aspect));
        self.set_decimal(b"-.45", b"-.45", precision_bits(log_scale), log_scale)
    }

    pub fn pan(&mut self, x: f64, y: f64) -> Result<(), i32> {
        if !x.is_finite() || !y.is_finite() {
            return Err(-1);
        }
        fixed_scale(&mut self.c, &mut self.product, self.log_scale, self.bits)?;
        multiply_factor(&mut self.a, &self.c, x, &mut self.factor, &mut self.product)?;
        multiply_factor(&mut self.b, &self.c, y, &mut self.factor, &mut self.product)?;
        self.d.combine(&self.x, &self.a, true);
        self.x.copy_from(&self.d);
        self.d.combine(&self.y, &self.b, true);
        self.y.copy_from(&self.d);
        Ok(())
    }

    pub fn zoom_at(&mut self, x: f64, y: f64, delta: f64) -> Result<(), i32> {
        if !x.is_finite() || !y.is_finite() || !delta.is_finite() {
            return Err(-1);
        }
        let next = (self.log_scale + delta).clamp(-3900.0, LOG2_100);
        let bits = self.bits.max(precision_bits(next));
        // Guided flight changes the scale about the exact camera centre. The
        // fixed-point scale/multiply work is unnecessary when both anchors are 0.
        if x == 0.0 && y == 0.0 {
            self.x.shift_left(bits - self.bits).map_err(|_| -2)?;
            self.y.shift_left(bits - self.bits).map_err(|_| -2)?;
            self.bits = bits;
            self.log_scale = next;
            return Ok(());
        }
        self.a.copy_from(&self.x);
        self.b.copy_from(&self.y);
        self.a.shift_left(bits - self.bits).map_err(|_| -2)?;
        self.b.shift_left(bits - self.bits).map_err(|_| -2)?;
        fixed_scale(&mut self.c, &mut self.product, self.log_scale, bits)?;
        fixed_scale(&mut self.d, &mut self.product, next, bits)?;
        self.product.combine(&self.c, &self.d, true);
        self.c.copy_from(&self.product);
        multiply_factor(&mut self.d, &self.c, x, &mut self.factor, &mut self.product)?;
        self.product.combine(&self.a, &self.d, false);
        self.a.copy_from(&self.product);
        multiply_factor(&mut self.d, &self.c, y, &mut self.factor, &mut self.product)?;
        self.product.combine(&self.b, &self.d, false);
        self.b.copy_from(&self.product);
        self.commit(bits, next);
        Ok(())
    }

    pub fn set_zoom(&mut self, zoom: f64) -> Result<(), i32> {
        self.zoom_at(0.0, 0.0, LOG2_3_2 - zoom * LOG2_10 - self.log_scale)
    }

    pub fn zoom(&self) -> f64 {
        (LOG2_3_2 - self.log_scale) / LOG2_10
    }

    pub fn center(&self) -> [f64; 2] {
        let (mx, ex) = self.x.to_fe(self.bits);
        let (my, ey) = self.y.to_fe(self.bits);
        [mx * binary_power(ex), my * binary_power(ey)]
    }

    pub fn offset(&mut self, reference: &Self) -> [f64; 4] {
        if self.bits == reference.bits && self.x.equals(&reference.x) && self.y.equals(&reference.y)
        {
            return [0.0; 4];
        }
        let bits = self.bits.max(reference.bits);
        self.a.copy_from(&self.x);
        self.b.copy_from(&reference.x);
        self.a
            .shift_left(bits - self.bits)
            .expect("preallocated camera precision");
        self.b
            .shift_left(bits - reference.bits)
            .expect("preallocated camera precision");
        self.c.combine(&self.a, &self.b, true);
        let (mx, ex) = self.c.to_fe(bits);
        self.a.copy_from(&self.y);
        self.b.copy_from(&reference.y);
        self.a
            .shift_left(bits - self.bits)
            .expect("preallocated camera precision");
        self.b
            .shift_left(bits - reference.bits)
            .expect("preallocated camera precision");
        self.c.combine(&self.a, &self.b, true);
        let (my, ey) = self.c.to_fe(bits);
        [mx, ex as f64, my, ey as f64]
    }

    pub fn write_fixed(&mut self, axis: usize, output: &mut [u8]) -> Result<usize, i32> {
        match axis {
            0 => self.a.copy_from(&self.x),
            1 => self.a.copy_from(&self.y),
            _ => return Err(-1),
        }
        self.a.write_decimal(output).map_err(|_| -2)
    }
}

fn validate_state(bits: usize, log_scale: f64) -> Result<(), i32> {
    if !(1..=4096).contains(&bits) || !log_scale.is_finite() {
        return Err(-1);
    }
    Ok(())
}

pub(crate) fn precision_bits(log_scale: f64) -> usize {
    (libm::ceil((128.0 - log_scale).max(128.0) / 64.0) * 64.0) as usize
}

// Math.round uses ties toward +infinity, unlike Rust's f64::round for negatives.
fn js_round(value: f64) -> f64 {
    let integer = libm::trunc(value);
    let fraction = value - integer;
    if fraction >= 0.5 {
        integer + 1.0
    } else if fraction < -0.5 {
        integer - 1.0
    } else {
        integer
    }
}

// Math.pow is implementation-dependent at its last bit. Dedicated exp2 avoids
// fdlibm pow's extra approximation and keeps camera events inexpensive. Tests
// require exact integer operations and at most 1 ULP against the JS scale input.
pub(crate) fn pow2(exponent: f64) -> f64 {
    libm::exp2(exponent)
}

fn binary_power(exponent: i32) -> f64 {
    if exponent > 1023 {
        f64::INFINITY
    } else if exponent < -1074 {
        0.0
    } else if exponent < -1022 {
        f64::from_bits(1u64 << (exponent + 1074))
    } else {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }
}

fn fixed_scale(
    out: &mut Integer,
    scratch: &mut Integer,
    log_scale: f64,
    bits: usize,
) -> Result<(), i32> {
    let exponent = libm::floor(log_scale);
    let mantissa = js_round(pow2(log_scale - exponent + 52.0));
    out.set_u64(mantissa as u64);
    let shift = exponent + bits as f64 - 52.0;
    if shift >= 0.0 {
        out.shift_left(shift as usize).map_err(|_| -2)?;
    } else {
        scratch.shift_right(out, (-shift) as usize);
        out.copy_from(scratch);
    }
    Ok(())
}

fn multiply_factor(
    out: &mut Integer,
    value: &Integer,
    factor: f64,
    mantissa: &mut Integer,
    product: &mut Integer,
) -> Result<(), i32> {
    if factor == 0.0 || value.sign() == 0 {
        out.clear();
        return Ok(());
    }
    let exponent = libm::floor(libm::log2(factor.abs()));
    let rounded = js_round(factor / binary_power(exponent as i32) * TWO_52);
    mantissa.set_u64(rounded.abs() as u64);
    mantissa.set_negative(rounded < 0.0);
    product.multiply(value, mantissa);
    if exponent >= 52.0 {
        out.copy_from(product);
        out.shift_left((exponent - 52.0) as usize).map_err(|_| -2)?;
    } else {
        out.shift_right(product, (52.0 - exponent) as usize);
    }
    Ok(())
}

fn js_whitespace(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | '\u{20}' | '\u{a0}' | '\u{1680}' |
        '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' |
        '\u{205f}' | '\u{3000}' | '\u{feff}')
}

fn decimal_into(out: &mut Integer, input: &[u8], bits: usize) -> Result<(), i32> {
    if input.len() > INPUT_BYTES {
        return Err(-2);
    }
    let text = std::str::from_utf8(input)
        .map_err(|_| -2)?
        .trim_matches(js_whitespace)
        .as_bytes();
    let (negative, mut index) = match text.first() {
        Some(b'-') => (true, 1),
        Some(b'+') => (false, 1),
        _ => (false, 0),
    };
    out.clear();
    let mut digits = 0;
    let mut fractional = 0i64;
    let mut point = false;
    while let Some(&byte) = text.get(index) {
        if byte.is_ascii_digit() {
            out.append_decimal_digit(byte - b'0').map_err(|_| -2)?;
            digits += 1;
            if point {
                fractional += 1;
            }
        } else if byte == b'.' && !point {
            point = true;
        } else {
            break;
        }
        index += 1;
    }
    if digits == 0 {
        return Err(-2);
    }
    let mut exponent = 0i64;
    if matches!(text.get(index), Some(b'e' | b'E')) {
        index += 1;
        let exponent_negative = text.get(index) == Some(&b'-');
        if matches!(text.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let start = index;
        while let Some(&byte) = text.get(index) {
            if !byte.is_ascii_digit() {
                return Err(-2);
            }
            exponent = exponent
                .checked_mul(10)
                .and_then(|n| n.checked_add((byte - b'0') as i64))
                .ok_or(-2)?;
            index += 1;
        }
        if start == index {
            return Err(-2);
        }
        if exponent_negative {
            exponent = -exponent;
        }
    }
    if index != text.len() {
        return Err(-2);
    }
    let power = exponent.checked_sub(fractional).ok_or(-2)?;
    if !(-10000..=10000).contains(&power) {
        return Err(-2);
    }
    out.shift_left(bits).map_err(|_| -2)?;
    if power >= 0 {
        for _ in 0..power {
            out.append_decimal_digit(0).map_err(|_| -2)?;
        }
    } else {
        let mut remainder = 0;
        for _ in 0..-power {
            remainder = out.divide_small(10);
        }
        if remainder >= 5 {
            out.add_small(1).map_err(|_| -2)?;
        }
    }
    out.set_negative(negative);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Camera;

    fn fixed(camera: &mut Camera, axis: usize) -> String {
        let mut bytes = [0; 32768];
        let length = camera.write_fixed(axis, &mut bytes).unwrap();
        String::from_utf8(bytes[..length].to_vec()).unwrap()
    }

    #[test]
    fn decimal_half_up_rounding_and_negative_pan_keep_integer_precision() {
        let mut camera = Camera::new();
        camera.set_decimal(b" -.125 ", b"1.25e-2", 8, 2.0).unwrap();
        assert_eq!(fixed(&mut camera, 0), "-32");
        assert_eq!(fixed(&mut camera, 1), "3");
        camera.set_decimal(b"-1.25e-2", b".5", 1, 2.0).unwrap();
        assert_eq!(fixed(&mut camera, 0), "0");
        assert_eq!(fixed(&mut camera, 1), "1");
        camera.set_fixed(b"0", b"0", 8, 0.0).unwrap();
        camera.pan(-0.1, 0.1).unwrap();
        // floor(256 * -.1) = -26; floor(256 * .1) = 25.
        assert_eq!(fixed(&mut camera, 0), "26");
        assert_eq!(fixed(&mut camera, 1), "-25");
    }

    #[test]
    fn deep_anchor_precision_growth_and_floor_shift_match_bigint() {
        let mut camera = Camera::new();
        camera.make(1.5).unwrap();
        assert_eq!(camera.center(), [-0.45, -0.45]);
        let mut reference = Camera::new();
        reference.copy_from(&camera);
        camera.set_zoom(120.0).unwrap();
        assert_eq!(camera.bits, 576);
        assert_eq!(camera.offset(&reference), [0.0; 4]);
        camera.pan(0.1, -0.2).unwrap();
        let offset = camera.offset(&reference);
        assert!(offset[0] < 0.0 && offset[1] < -390.0);
        assert!(offset[2] > 0.0 && offset[3] < -390.0);
        camera.zoom_at(0.0, 0.0, -10000.0).unwrap();
        assert_eq!(camera.log_scale, -3900.0);
        assert_eq!(camera.bits, 4032);
    }

    #[test]
    fn invalid_camera_inputs_leave_the_last_valid_camera_unchanged() {
        let mut camera = Camera::new();
        camera
            .set_fixed(b"-123456789", b"987654321", 4096, -3800.0)
            .unwrap();
        for (x, y, bits, log_scale) in [
            (b"1.0".as_slice(), b"2".as_slice(), 128, 0.0),
            (b"1", b"oops", 128, 0.0),
            (b"1", b"2", 0, 0.0),
            (b"1", b"2", 4097, 0.0),
            (b"1", b"2", 128, f64::NAN),
        ] {
            assert!(camera.set_fixed(x, y, bits, log_scale).is_err());
            assert_eq!(fixed(&mut camera, 0), "-123456789");
            assert_eq!(fixed(&mut camera, 1), "987654321");
            assert_eq!(camera.bits, 4096);
        }
        for value in [
            b"".as_slice(),
            b".",
            b"e2",
            b"1e",
            b"1e10001",
            b"1e-10001",
            b"NaN",
        ] {
            assert!(camera.set_decimal(value, b"0", 128, 0.0).is_err());
        }
        assert!(camera.pan(f64::INFINITY, 0.0).is_err());
        assert!(camera.zoom_at(0.0, 0.0, f64::NAN).is_err());
        assert!(camera.write_fixed(2, &mut [0; 32]).is_err());
        assert!(camera.write_fixed(0, &mut [0; 2]).is_err());
    }

    #[test]
    fn camera_frames_copy_offsets_and_decimal_serialization_allocate_nothing() {
        let mut camera = Camera::new();
        let mut reference = Camera::new();
        let mut previous = Camera::new();
        camera.make(1.5).unwrap();
        reference.copy_from(&camera);
        camera.set_zoom(120.0).unwrap();
        let mut output = [0u8; 32768];
        let allocations = crate::allocation_probe::measure(|| {
            for i in 0..120 {
                previous.copy_from(&camera);
                camera.zoom_at(0.1, -0.2, -0.03125).unwrap();
                camera.pan((i as f64 - 60.0) / 1024.0, -0.125).unwrap();
                camera.offset(&reference);
                camera.offset(&previous);
                camera.center();
                camera.write_fixed(0, &mut output).unwrap();
                camera.write_fixed(1, &mut output).unwrap();
            }
        });
        assert_eq!(allocations, 0);
    }

    #[test]
    fn camera_actions_match_the_typescript_bigint_oracle_exactly() {
        // Compare every stored integer and exported f64 bit pattern. This catches
        // changing pow/log operation order as well as signed-shift rounding.
        let script = r#"
import { pathToFileURL } from 'node:url';
const { makeCamera, decimalToFixed, pan, zoomAt, setZoom, cameraOffset, fixedToNumber } =
  await import(pathToFileURL(process.argv[1] + '/src/camera.ts'));
const buffer = new ArrayBuffer(8), bytes = new DataView(buffer);
function hex(n) { bytes.setFloat64(0,n,true); return bytes.getBigUint64(0,true).toString(16).padStart(16,'0'); }
let camera = makeCamera(1.5), reference = { ...camera };
function action(op, ...args) {
  if(op === 'make') camera = makeCamera(Number(args[0]));
  if(op === 'decimal') camera = {x: decimalToFixed(args[0],+args[2]),y: decimalToFixed(args[1],+args[2]),bits:+args[2],logScale:+args[3]};
  if(op === 'fixed') camera = {x: BigInt(args[0]),y: BigInt(args[1]),bits:+args[2],logScale:+args[3]};
  if(op === 'reference') reference = { ...camera };
  if(op === 'zoom') setZoom(camera,Number(args[0]));
  if(op === 'anchor') zoomAt(camera,...args.map(Number));
  if(op === 'pan') pan(camera,...args.map(Number));
  const offset = cameraOffset(camera,reference);
  console.log([op,...args].join('\t'));
  console.log([camera.bits,...[camera.logScale,(Math.log2(3.2)-camera.logScale)/Math.log2(10),fixedToNumber(camera.x,camera.bits),fixedToNumber(camera.y,camera.bits),...offset.x,...offset.y].map(hex),camera.x,camera.y].join('\t'));
}
action('make','1.5'); action('reference'); action('zoom','120');
action('anchor','.125','-.375','-.7'); action('pan','.1','-.2');
for(let i=0;i<40;i++) {
  action('anchor',String((i-20)/113),String((20-i)/127),i%2 ? '1.125' : '-3.25');
  action('pan',String((i-20)*2**-20),String((20-i)*2**-18));
}
for(const bits of [1,8,128,576,2048,4096]) {
  action('decimal','-1.123456789012345678901234567890123456789','1.25e-2',String(bits),'-3.7');
  action('reference'); action('pan','-0.1','0.1'); action('anchor','-.5','.5','-.123');
}
action('decimal','-1.941763662643','-0.00001324567','4096','-3900');
action('reference'); action('pan','0.000000000001','-0.000000000001');
action('anchor','.1','-.2','-10000'); action('anchor','-.2','.3','10000');
action('fixed','-18014398509481987','1','4096','-3900'); action('reference');
action('pan','-5e-324','5e-324'); action('anchor','0','0','-0.125');
for(const aspect of ['0.25','0.7777777777777778','1.3333333333333333','2','19.9']) action('make',aspect);
for(const decimal of ['-.00000000001','1.e2','.5e-3','-0001.2300E+3','1e10000','1e-10000'])
  action('decimal',decimal,'-0','4096','-3900');
let seed = 0x47dce7;
const random = () => { seed = (Math.imul(seed,1664525)+1013904223) >>> 0; return seed/2**32; };
for(let i=0;i<160;i++) {
  // Integer octave scales isolate exact BigInt arithmetic from the separately
  // tested implementation-dependent final bit of Math.pow.
  action('decimal','-.45','.12345678901234567890123456789','4096',String(-8-Math.floor(random()*3800))); action('reference');
  action('anchor',String(random()-.5),String(random()-.5),String(Math.floor(random()*8)-4));
  action('pan',String(random()-.5),String(random()-.5));
}
"#;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let output = std::process::Command::new("node")
            .args(["--input-type=module", "-e", script])
            .arg(root)
            .output()
            .expect("Node is required for the JS camera differential test");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let lines: Vec<_> = text.lines().collect();
        let mut camera = Camera::new();
        let mut reference = Camera::new();
        camera.make(1.5).unwrap();
        reference.copy_from(&camera);
        let (pairs, remainder) = lines.as_chunks::<2>();
        assert!(remainder.is_empty());
        for (index, pair) in pairs.iter().enumerate() {
            let args: Vec<_> = pair[0].split('\t').collect();
            let number = |at: usize| args[at].parse::<f64>().unwrap();
            match args[0] {
                "make" => camera.make(number(1)).unwrap(),
                "decimal" => camera
                    .set_decimal(
                        args[1].as_bytes(),
                        args[2].as_bytes(),
                        args[3].parse().unwrap(),
                        number(4),
                    )
                    .unwrap(),
                "fixed" => camera
                    .set_fixed(
                        args[1].as_bytes(),
                        args[2].as_bytes(),
                        args[3].parse().unwrap(),
                        number(4),
                    )
                    .unwrap(),
                "reference" => reference.copy_from(&camera),
                "zoom" => camera.set_zoom(number(1)).unwrap(),
                "anchor" => camera.zoom_at(number(1), number(2), number(3)).unwrap(),
                "pan" => camera.pan(number(1), number(2)).unwrap(),
                _ => panic!("unknown action"),
            }
            let center = camera.center();
            let offset = camera.offset(&reference);
            let values = [
                camera.log_scale,
                camera.zoom(),
                center[0],
                center[1],
                offset[0],
                offset[1],
                offset[2],
                offset[3],
            ];
            let mut actual = camera.bits.to_string();
            for value in values {
                actual.push_str(&format!("\t{:016x}", value.to_bits()));
            }
            actual.push('\t');
            actual.push_str(&fixed(&mut camera, 0));
            actual.push('\t');
            actual.push_str(&fixed(&mut camera, 1));
            assert_eq!(actual, pair[1], "camera action {index}: {}", pair[0]);
        }
    }

    #[test]
    fn camera_scale_transcendentals_stay_within_one_js_ulp() {
        // JS permits platform-dependent transcendental rounding. In particular,
        // macOS V8 Math.pow rounds this exact value down despite its true tail
        // .5015459745528338956. Do not make Rust less accurate to imitate it.
        assert_eq!(super::pow2(52.71471596875199), 7391127932261240.0);
        assert_eq!(super::pow2(52.97170051862918), 8832238286230673.0);
        let script = r#"
const bytes = new DataView(new ArrayBuffer(8));
const hex = n => { bytes.setFloat64(0,n,true); return bytes.getBigUint64(0,true).toString(16); };
let seed=0x674fd;
for(let i=0;i<3000;i++) {
 seed=(Math.imul(seed,1664525)+1013904223)>>>0;
 const exponent=52+seed/2**32;
 console.log(hex(exponent)+' '+hex(2**exponent));
}
"#;
        let output = std::process::Command::new("node")
            .args(["-e", script])
            .output()
            .unwrap();
        assert!(output.status.success());
        for line in String::from_utf8(output.stdout).unwrap().lines() {
            let (input, expected) = line.split_once(' ').unwrap();
            let input = f64::from_bits(u64::from_str_radix(input, 16).unwrap());
            let expected = u64::from_str_radix(expected, 16).unwrap();
            assert!(
                super::pow2(input).to_bits().abs_diff(expected) <= 1,
                "scale exponent {input}"
            );
        }
        assert_eq!(super::js_round(-2.5), -2.0);
        assert_eq!(super::js_round(2.5), 3.0);
        assert_eq!(super::js_round(4503599627370497.0), 4503599627370497.0);
    }

    #[test]
    #[ignore = "manual release-mode camera performance measurement"]
    fn measure_camera_hot_path() {
        use std::hint::black_box;
        use std::time::Instant;
        const COUNT: usize = 200_000;
        let start = Instant::now();
        for index in 0..COUNT {
            black_box(super::pow2(black_box(
                52.0 + (index % 1000) as f64 / 1000.0,
            )));
        }
        let helper = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for index in 0..COUNT {
            black_box(2.0_f64.powf(black_box(52.0 + (index % 1000) as f64 / 1000.0)));
        }
        let native = start.elapsed().as_secs_f64();
        let mut camera = Camera::new();
        let mut reference = Camera::new();
        camera.make(1.5).unwrap();
        camera.set_zoom(120.0).unwrap();
        reference.copy_from(&camera);
        let start = Instant::now();
        for index in 0..COUNT {
            camera
                .set_zoom(black_box(120.0 + (index % 1000) as f64 * 0.0001))
                .unwrap();
            black_box(camera.center());
            black_box(camera.offset(&reference));
        }
        let flight = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for index in 0..COUNT {
            camera
                .zoom_at(
                    black_box(0.125),
                    black_box(-0.25),
                    black_box(if index % 2 == 0 { -0.03125 } else { 0.03125 }),
                )
                .unwrap();
        }
        let anchored = start.elapsed().as_secs_f64();
        eprintln!(
            "camera release microbenchmark: exp2 {:.1} ns; native pow {:.1} ns; flight set_zoom+center+offset {:.1} ns; anchored zoom {:.1} ns",
            helper * 1e9 / COUNT as f64,
            native * 1e9 / COUNT as f64,
            flight * 1e9 / COUNT as f64,
            anchored * 1e9 / COUNT as f64
        );
    }
}
