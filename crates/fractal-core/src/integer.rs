//! Signed magnitude integers with preallocated base-2^32 limbs.
//! All arithmetic mutates an existing buffer; only setup resizes storage.
use std::cmp::Ordering;

pub(crate) struct Integer {
    words: Vec<u32>,
    len: usize,
    negative: bool,
}

impl Integer {
    pub fn zero(capacity: usize) -> Self {
        Self {
            words: vec![0; capacity],
            len: 0,
            negative: false,
        }
    }

    pub fn reserve(&mut self, capacity: usize) {
        if self.words.len() < capacity {
            self.words.resize(capacity, 0);
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.negative = false;
    }

    /// Copy into storage allocated during camera/kernel construction.
    pub fn copy_from(&mut self, other: &Self) {
        self.words[..other.len].copy_from_slice(&other.words[..other.len]);
        self.len = other.len;
        self.negative = other.negative;
    }

    pub fn set_u64(&mut self, value: u64) {
        self.clear();
        if value != 0 {
            self.words[0] = value as u32;
            self.len = 1;
            if value >> 32 != 0 {
                self.words[1] = (value >> 32) as u32;
                self.len = 2;
            }
        }
    }

    pub fn set_negative(&mut self, negative: bool) {
        self.negative = negative && self.len != 0;
    }

    pub fn append_decimal_digit(&mut self, digit: u8) -> Result<(), ()> {
        if digit > 9 {
            return Err(());
        }
        let mut carry = digit as u64;
        for word in &mut self.words[..self.len] {
            let value = *word as u64 * 10 + carry;
            *word = value as u32;
            carry = value >> 32;
        }
        if carry != 0 {
            if self.len == self.words.len() {
                return Err(());
            }
            self.words[self.len] = carry as u32;
            self.len += 1;
        }
        Ok(())
    }

    /// Add to the magnitude, used for half-up rounding of positive decimal input.
    pub fn add_small(&mut self, value: u32) -> Result<(), ()> {
        let mut carry = value as u64;
        let mut index = 0;
        while carry != 0 {
            if index == self.words.len() {
                return Err(());
            }
            if index == self.len {
                self.words[index] = 0;
                self.len += 1;
            }
            let sum = self.words[index] as u64 + carry;
            self.words[index] = sum as u32;
            carry = sum >> 32;
            index += 1;
        }
        Ok(())
    }

    /// Shift exactly in-place. Unlike resize/reserve this can never allocate.
    pub fn shift_left(&mut self, bits: usize) -> Result<(), ()> {
        if self.len == 0 || bits == 0 {
            return Ok(());
        }
        let length = self.bit_length().checked_add(bits).ok_or(())?.div_ceil(32);
        if length > self.words.len() {
            return Err(());
        }
        let whole = bits / 32;
        let part = bits % 32;
        self.words.copy_within(..self.len, whole);
        self.words[..whole].fill(0);
        let old_end = self.len + whole;
        let mut carry = 0;
        if part != 0 {
            for word in &mut self.words[whole..old_end] {
                let next = *word >> (32 - part);
                *word = (*word << part) | carry;
                carry = next;
            }
        }
        if carry != 0 {
            self.words[old_end] = carry;
        }
        self.len = length;
        Ok(())
    }

    /// Magnitude division, retaining sign and returning a nonnegative remainder.
    pub fn divide_small(&mut self, divisor: u32) -> u32 {
        assert_ne!(divisor, 0);
        let mut remainder = 0u64;
        for word in self.words[..self.len].iter_mut().rev() {
            let value = (remainder << 32) | *word as u64;
            *word = (value / divisor as u64) as u32;
            remainder = value % divisor as u64;
        }
        self.normalize();
        remainder as u32
    }

    /// Destructive decimal formatting for a caller-owned scratch integer.
    pub fn write_decimal(&mut self, output: &mut [u8]) -> Result<usize, ()> {
        let negative = self.negative;
        if self.len == 0 {
            *output.first_mut().ok_or(())? = b'0';
            return Ok(1);
        }
        let mut length = 0;
        while self.len != 0 {
            let mut chunk = self.divide_small(1_000_000_000);
            let mut digits = 0;
            while digits < 9 && (self.len != 0 || chunk != 0) {
                *output.get_mut(length).ok_or(())? = b'0' + (chunk % 10) as u8;
                length += 1;
                digits += 1;
                chunk /= 10;
            }
        }
        if negative {
            *output.get_mut(length).ok_or(())? = b'-';
            length += 1;
        }
        output[..length].reverse();
        Ok(length)
    }

    #[cfg(test)]
    pub fn parse(bytes: &[u8], capacity: usize) -> Result<Self, ()> {
        let mut value = Self::zero(capacity);
        value.parse_into(bytes)?;
        Ok(value)
    }

    pub fn parse_into(&mut self, bytes: &[u8]) -> Result<(), ()> {
        self.clear();
        let (negative, digits) = match bytes.first() {
            Some(b'-') => (true, &bytes[1..]),
            Some(b'+') => (false, &bytes[1..]),
            _ => (false, bytes),
        };
        if digits.is_empty() {
            return Err(());
        }
        for &digit in digits {
            if !digit.is_ascii_digit() {
                return Err(());
            }
            let mut carry = (digit - b'0') as u64;
            for word in &mut self.words[..self.len] {
                let value = *word as u64 * 10 + carry;
                *word = value as u32;
                carry = value >> 32;
            }
            if carry != 0 {
                if self.len == self.words.len() {
                    return Err(());
                }
                self.words[self.len] = carry as u32;
                self.len += 1;
            }
        }
        self.negative = negative && self.len != 0;
        Ok(())
    }

    fn normalize(&mut self) {
        while self.len != 0 && self.words[self.len - 1] == 0 {
            self.len -= 1;
        }
        if self.len == 0 {
            self.negative = false;
        }
    }

    pub fn sign(&self) -> i32 {
        if self.len == 0 {
            0
        } else if self.negative {
            -1
        } else {
            1
        }
    }

    pub fn equals(&self, other: &Self) -> bool {
        self.negative == other.negative
            && self.len == other.len
            && self.words[..self.len] == other.words[..other.len]
    }

    pub fn bit_length(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            32 * (self.len - 1) + (32 - self.words[self.len - 1].leading_zeros()) as usize
        }
    }

    fn compare_magnitude(&self, other: &Self) -> Ordering {
        match self.len.cmp(&other.len) {
            Ordering::Equal => self.words[..self.len]
                .iter()
                .rev()
                .cmp(other.words[..other.len].iter().rev()),
            order => order,
        }
    }

    pub fn combine(&mut self, a: &Self, b: &Self, subtract: bool) {
        let b_negative = b.negative ^ subtract;
        if a.negative == b_negative {
            self.negative = a.negative;
            self.len = a.len.max(b.len);
            let mut carry = 0u64;
            for index in 0..self.len {
                let av = if index < a.len {
                    a.words[index] as u64
                } else {
                    0
                };
                let bv = if index < b.len {
                    b.words[index] as u64
                } else {
                    0
                };
                let sum = av + bv + carry;
                self.words[index] = sum as u32;
                carry = sum >> 32;
            }
            if carry != 0 {
                self.words[self.len] = carry as u32;
                self.len += 1;
            }
        } else {
            let (large, small, negative) = match a.compare_magnitude(b) {
                Ordering::Less => (b, a, b_negative),
                Ordering::Equal => {
                    self.clear();
                    return;
                }
                Ordering::Greater => (a, b, a.negative),
            };
            self.negative = negative;
            self.len = large.len;
            let mut borrow = 0u64;
            for index in 0..large.len {
                let av = large.words[index] as u64;
                let bv = if index < small.len {
                    small.words[index] as u64 + borrow
                } else {
                    borrow
                };
                self.words[index] = av.wrapping_sub(bv) as u32;
                borrow = u64::from(av < bv);
            }
        }
        self.normalize();
    }

    pub fn multiply(&mut self, a: &Self, b: &Self) {
        if a.len == 0 || b.len == 0 {
            self.clear();
            return;
        }
        self.len = a.len + b.len;
        self.negative = a.negative ^ b.negative;
        self.words[..self.len].fill(0);
        for i in 0..a.len {
            let mut carry = 0u64;
            for j in 0..b.len {
                // The maximum is (2^32-1)^2 + 2*(2^32-1) = 2^64-1.
                let product =
                    a.words[i] as u64 * b.words[j] as u64 + self.words[i + j] as u64 + carry;
                self.words[i + j] = product as u32;
                carry = product >> 32;
            }
            self.words[i + b.len] = carry as u32;
        }
        self.normalize();
    }

    pub fn scale(&mut self, factor: i32) {
        if factor == 0 || self.len == 0 {
            self.clear();
            return;
        }
        self.negative ^= factor < 0;
        let magnitude = factor.unsigned_abs() as u64;
        let mut carry = 0u64;
        for word in &mut self.words[..self.len] {
            let product = *word as u64 * magnitude + carry;
            *word = product as u32;
            carry = product >> 32;
        }
        if carry != 0 {
            self.words[self.len] = carry as u32;
            self.len += 1;
        }
    }

    pub fn shift_left_word(&mut self) {
        if self.len != 0 {
            self.words.copy_within(0..self.len, 1);
            self.words[0] = 0;
            self.len += 1;
        }
    }

    pub fn shift_right(&mut self, a: &Self, bits: usize) {
        let whole = bits / 32;
        let part = bits % 32;
        let discarded = a.negative
            && (a.words[..whole.min(a.len)].iter().any(|&w| w != 0)
                || (part != 0 && whole < a.len && a.words[whole] & ((1u32 << part) - 1) != 0));
        self.len = a.len.saturating_sub(whole);
        self.negative = a.negative;
        for index in 0..self.len {
            let low = a.words[index + whole];
            self.words[index] = if part == 0 {
                low
            } else {
                let high = if index + whole + 1 < a.len {
                    a.words[index + whole + 1]
                } else {
                    0
                };
                (low >> part) | (high << (32 - part))
            };
        }
        // Signed BigInt >> floors: -15 >> 3 is -2, even when all retained limbs vanish.
        if discarded {
            let mut index = 0;
            while index < self.len && self.words[index] == u32::MAX {
                self.words[index] = 0;
                index += 1;
            }
            if index == self.len {
                self.words[index] = 1;
                self.len += 1;
            } else {
                self.words[index] += 1;
            }
        }
        self.normalize();
    }

    pub fn to_fe(&self, bits: usize) -> (f64, i32) {
        let length = self.bit_length();
        if length == 0 {
            return (0.0, 0);
        }
        let shift = length.saturating_sub(53);
        let start = shift / 32;
        let offset = shift % 32;
        let mut top = (self.words[start] as u64) >> offset;
        if start + 1 < self.len {
            top |= (self.words[start + 1] as u64) << (32 - offset);
        }
        if offset != 0 && start + 2 < self.len {
            top |= (self.words[start + 2] as u64) << (64 - offset);
        }
        if shift != 0 && (self.words[(shift - 1) / 32] >> ((shift - 1) % 32)) & 1 != 0 {
            top += 1;
        }
        let mut mantissa = top as f64 / (1u64 << (length - shift - 1)) as f64;
        let mut exponent = length as i32 - 1 - bits as i32;
        if mantissa >= 2.0 {
            mantissa *= 0.5;
            exponent += 1;
        }
        if self.negative {
            mantissa = -mantissa;
        }
        (mantissa, exponent)
    }

    #[cfg(test)]
    pub fn to_i128(&self) -> i128 {
        assert!(self.len <= 4);
        let magnitude = self.words[..self.len]
            .iter()
            .rev()
            .fold(0u128, |n, &w| (n << 32) | w as u128);
        if self.negative {
            -(magnitude as i128)
        } else {
            magnitude as i128
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Integer;

    #[test]
    fn a_4096_bit_square_propagates_every_carry_and_floors_a_negative_shift() {
        // (2^4096 - 1)^2 = 2^8192 - 2^4097 + 1.
        let value = Integer {
            words: vec![u32::MAX; 128],
            len: 128,
            negative: false,
        };
        let mut square = Integer::zero(260);
        square.multiply(&value, &value);
        assert_eq!(square.len, 256);
        assert_eq!(square.words[0], 1);
        assert!(square.words[1..128].iter().all(|&word| word == 0));
        assert_eq!(square.words[128], u32::MAX - 1);
        assert!(square.words[129..256].iter().all(|&word| word == u32::MAX));
        square.scale(-1);
        let mut shifted = Integer::zero(260);
        shifted.shift_right(&square, 4096);
        // floor(-(2^4096 - 2 + 2^-4096)) = -(2^4096 - 1).
        assert!(shifted.negative);
        assert_eq!(shifted.len, 128);
        assert!(shifted.words[..128].iter().all(|&word| word == u32::MAX));
        assert_eq!(shifted.to_fe(4096), (-1.0, 0));
    }
}
