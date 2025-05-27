use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::buf::SymbolicByte;

pub const FALSE: SymbolicBit = SymbolicBit::Literal(false);
pub const TRUE: SymbolicBit = SymbolicBit::Literal(true);

/// A value that can be used to represent a variable bit, possibly with constraints on its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicBit {
    /// A literal `true` or `false` value.
    Literal(bool),

    /// A variable value. The parameter is the identifier for this variable. Two variables with the
    /// same identifier are equivalent.
    Variable(usize),

    /// The negation of a symbolic bit. The `!` operator should be preferred to this, as it has the
    /// opportunity to perform simplications where a direct construction does not.
    Not(Rc<Self>),

    /// The conjunction of two symbolic bits. The `&` operator should be preferred to this, as it
    /// has the opportunity to perform simpliciations where a direct construction does not.
    And(Rc<Self>, Rc<Self>),
}

#[derive(thiserror::Error, Debug)]
pub enum ConcretizationError {
    #[error("non-literal bit at index {bit_index}")]
    NonLiteralBit { bit_index: usize },

    #[error("value exceeded maximum number of bytes ({max_bytes})")]
    Overflow { max_bytes: usize },
}

impl SymbolicBit {
    pub fn equals(self, rhs: Self) -> Self {
        (self.clone() & rhs.clone()) | (!self & !rhs)
    }

    pub fn select(self, lhs: Self, rhs: Self) -> Self {
        (self.clone() & lhs) | (!self & rhs)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Evaluator {
    variables: std::collections::HashMap<usize, bool>,
    and_gates: std::collections::HashMap<(usize, usize), bool>,
}

impl Evaluator {
    pub fn with_concrete_values(variables: &SymbolicBitVec, literals: &SymbolicBitVec) -> Self {
        let variables = std::iter::zip(variables.iter(), literals.iter())
            .filter_map(|(variable, literal)| {
                if let SymbolicBit::Variable(variable) = variable {
                    if let SymbolicBit::Literal(literal) = literal {
                        return Some((*variable, *literal));
                    }
                }

                None
            })
            .collect();

        Self {
            variables,
            and_gates: Default::default(),
        }
    }

    pub fn evaluate(&mut self, bit: &SymbolicBit) -> bool {
        match bit {
            SymbolicBit::Literal(x) => *x,
            SymbolicBit::Variable(id) => *self
                .variables
                .get(id)
                .expect("variable should be defined in cache"),
            SymbolicBit::Not(bit) => !self.evaluate(bit),
            SymbolicBit::And(lhs, rhs) => {
                let lhs_addr = Rc::as_ptr(lhs).addr();
                let rhs_addr = Rc::as_ptr(rhs).addr();
                let cache_key = (
                    usize::min(lhs_addr, rhs_addr),
                    usize::max(lhs_addr, rhs_addr),
                );

                if let Some(value) = self.and_gates.get(&cache_key).copied() {
                    value
                } else {
                    let value = self.evaluate(lhs) && self.evaluate(rhs);
                    self.and_gates.insert(cache_key, value);
                    value
                }
            }
        }
    }
}

impl FromIterator<(usize, bool)> for Evaluator {
    fn from_iter<T: IntoIterator<Item = (usize, bool)>>(iter: T) -> Self {
        Self {
            variables: iter.into_iter().collect(),
            and_gates: Default::default(),
        }
    }
}

impl Default for SymbolicBit {
    fn default() -> Self {
        FALSE
    }
}

impl std::ops::Not for SymbolicBit {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            SymbolicBit::Literal(false) => SymbolicBit::Literal(true),
            SymbolicBit::Literal(true) => SymbolicBit::Literal(false),

            // TODO: Use Rc::unwrap_or_clone(y) once feature is stable
            // See https://github.com/rust-lang/rust/issues/93610
            SymbolicBit::Not(y) => (*y).clone(),
            _ => SymbolicBit::Not(Rc::new(self)),
        }
    }
}

impl std::ops::BitAnd for SymbolicBit {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match self {
            SymbolicBit::Literal(false) => return SymbolicBit::Literal(false),
            SymbolicBit::Literal(true) => return rhs,
            SymbolicBit::Variable(x) => match rhs {
                SymbolicBit::Not(y) if *y == SymbolicBit::Variable(x) => {
                    return SymbolicBit::Literal(false)
                }
                _ => (),
            },
            _ => (),
        }

        match rhs {
            SymbolicBit::Literal(false) => return SymbolicBit::Literal(false),
            SymbolicBit::Literal(true) => return self,
            SymbolicBit::Variable(x) => match self {
                SymbolicBit::Not(y) if *y == SymbolicBit::Variable(x) => {
                    return SymbolicBit::Literal(false)
                }
                _ => (),
            },
            _ => (),
        }

        SymbolicBit::And(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::BitOr for SymbolicBit {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        !(!self & !rhs)
    }
}

impl std::ops::BitXor for SymbolicBit {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        (self.clone() & !rhs.clone()) | (!self & rhs)
    }
}

#[derive(Debug, Clone)]
pub struct SymbolicBitVec {
    bits: VecDeque<SymbolicBit>,
}

impl TryInto<Vec<SymbolicByte>> for SymbolicBitVec {
    type Error = String;

    fn try_into(self) -> Result<Vec<SymbolicByte>, Self::Error> {
        if self.bits.len() % 8 == 0 {
            Ok(self.into_bytes())
        } else {
            Err(format!(
                "invalid number of bits: {len}",
                len = self.bits.len()
            ))
        }
    }
}

impl IntoIterator for SymbolicBitVec {
    type Item = SymbolicBit;
    type IntoIter = std::collections::vec_deque::IntoIter<SymbolicBit>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.into_iter()
    }
}

impl std::ops::Index<usize> for SymbolicBitVec {
    type Output = SymbolicBit;

    fn index(&self, index: usize) -> &Self::Output {
        &self.bits[index]
    }
}

impl FromIterator<SymbolicBit> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicBit>>(iter: T) -> Self {
        Self {
            bits: iter.into_iter().collect(),
        }
    }
}

static START_SYMBOL: AtomicUsize = AtomicUsize::new(0);
impl SymbolicBitVec {
    pub fn msb(&self) -> Option<&SymbolicBit> {
        self.bits.back()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn num_bytes(&self) -> usize {
        self.bits.len().next_multiple_of(8) / 8
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SymbolicBit> {
        self.bits.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SymbolicBit> {
        self.bits.iter_mut()
    }

    pub fn with_size(num_bits: usize) -> Self {
        let start_symbol = START_SYMBOL.fetch_add(num_bits, Ordering::SeqCst);
        let mut bits = VecDeque::with_capacity(num_bits);
        for i in 0..num_bits {
            bits.push_back(SymbolicBit::Variable(start_symbol + i));
        }

        Self { bits }
    }

    pub fn into_bytes(self) -> Vec<SymbolicByte> {
        assert_eq!(self.bits.len() % 8, 0);
        let num_bytes = self.bits.len() / 8;

        const FALSE: SymbolicBit = SymbolicBit::Literal(false);
        let mut bits = [FALSE; 8];
        let mut bytes = Vec::with_capacity(num_bytes);

        for (i, bit) in self.bits.into_iter().enumerate() {
            bits[i % 8] = bit;
            if (i + 1) % 8 == 0 {
                bytes.push(bits.into());
                bits = [FALSE; 8];
            }
        }

        bytes
    }

    pub fn into_parts(self, num_bits: usize) -> Vec<Self> {
        assert_eq!(self.bits.len() % num_bits, 0);
        let mut parts = Vec::new();
        let mut remainder = self;
        while remainder.len() > num_bits {
            let mut next_split = remainder.bits;
            remainder.bits = next_split.split_off(num_bits);
            parts.push(SymbolicBitVec { bits: next_split });
        }
        parts.push(remainder);
        parts
    }

    pub fn empty() -> Self {
        Self {
            bits: VecDeque::with_capacity(0),
        }
    }

    pub fn constant(mut value: usize, num_bits: usize) -> Self {
        let mut bits = VecDeque::with_capacity(num_bits);
        for _ in 0..num_bits {
            bits.push_back(SymbolicBit::Literal(value & 0x1 > 0));
            value >>= 1;
        }

        if value > 0 {
            // TODO Throw error, this should be a result
        }

        Self { bits }
    }

    pub fn signed_minimum_value(num_bits: usize) -> Self {
        Self {
            bits: std::iter::repeat(SymbolicBit::Literal(false))
                .take(num_bits - 1)
                .chain(std::iter::once(SymbolicBit::Literal(true)))
                .collect(),
        }
    }

    pub fn signed_maximum_value(num_bits: usize) -> Self {
        Self {
            bits: std::iter::repeat(SymbolicBit::Literal(true))
                .take(num_bits - 1)
                .chain(std::iter::once(SymbolicBit::Literal(false)))
                .collect(),
        }
    }

    pub fn contains_variable(&self) -> bool {
        self.bits
            .iter()
            .any(|bit| !matches!(*bit, SymbolicBit::Literal(_)))
    }

    pub fn equals(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.bits.len(), rhs.bits.len());
        self.equals_unchecked(rhs)
    }

    fn equals_unchecked(mut self, mut rhs: Self) -> SymbolicBit {
        if self.bits.len() == 1 {
            // SAFETY: Know both self and rhs bits length is 1
            unsafe {
                self.msb()
                    .cloned()
                    .unwrap_unchecked()
                    .equals(rhs.msb().cloned().unwrap_unchecked())
            }
        } else {
            let partition = self.bits.len() / 2;
            let self_part: Self = self.bits.split_off(partition).into_iter().collect();
            let rhs_part: Self = rhs.bits.split_off(partition).into_iter().collect();
            self_part.equals_unchecked(rhs_part) & self.equals_unchecked(rhs)
        }
    }

    /// Concatenates the left-hand side with the right-hand side, creating a new `SymbolicBitVec`
    /// with a combined length of both inputs.
    pub fn concat(mut self, mut rhs: Self) -> Self {
        let mut bits = VecDeque::with_capacity(self.bits.len() + rhs.bits.len());
        bits.append(&mut self.bits);
        bits.append(&mut rhs.bits);
        Self { bits }
    }

    /// Creates a new `SymbolicBitVec` with the specified number of least-significant bits removed.
    pub fn truncate_lsb(self, num_bits_truncated: usize) -> Self {
        Self {
            bits: self.bits.into_iter().skip(num_bits_truncated).collect(),
        }
    }

    /// Creates a new `SymbolicBitVec` with the specified number of most-significant bits removed.
    pub fn truncate_msb(self, num_bits_truncated: usize) -> Self {
        let num_bits = self.bits.len();
        Self {
            bits: self
                .bits
                .into_iter()
                .take(num_bits - num_bits_truncated)
                .collect(),
        }
    }

    /// Create a new `SymbolicBitVec` with the number of additional zero bits specified as the
    /// most-significant bits.
    pub fn zero_extend(self, num_bits: usize) -> Self {
        self.concat(SymbolicBitVec::constant(0, num_bits))
    }

    /// Create a new `SymbolicBitVec` with the number of additional bits specified as the
    /// most-significant bits. The additional bits are clones of the original most significant-bit.
    pub fn sign_extend(self, num_bits: usize) -> Self {
        let mut extension = VecDeque::with_capacity(num_bits);
        let msb = self.msb().unwrap();
        for _ in 0..num_bits {
            extension.push_back(msb.clone());
        }

        let extension = Self { bits: extension };
        self.concat(extension)
    }

    pub fn addition_with_carry(self, rhs: Self) -> (Self, SymbolicBit) {
        let mut carry = self.clone().addition_carry_bits(rhs.clone());
        let overflow = carry.bits.pop_back().unwrap();
        let sum = self ^ rhs ^ carry;
        (sum, overflow)
    }

    pub fn subtraction_with_borrow(self, rhs: Self) -> (Self, SymbolicBit) {
        let diff = self.clone() - rhs.clone();

        // Positive overflow occurs if sign bits are (0, 1) but 1 in sum
        let positive_overflow = !self.msb().cloned().unwrap()
            & rhs.msb().cloned().unwrap()
            & diff.msb().cloned().unwrap();

        // Negative overflow occurs if sign bits are (1, 0) but 0 in sum
        let negative_overflow = self.msb().cloned().unwrap()
            & !rhs.msb().cloned().unwrap()
            & !diff.msb().cloned().unwrap();

        // Overflow occurs if either positive or negative overflow occurs
        (diff, positive_overflow | negative_overflow)
    }

    /// Multiply two integers together. When multiplying unsigned integers together, the number of
    /// output bits should be the sum of input sizes unless a lesser size is known in advance.
    ///
    /// # Unsigned Example
    ///
    /// ```
    /// # use sym::SymbolicBitVec;
    /// let x = SymbolicBitVec::constant(2, 4);
    /// let y = SymbolicBitVec::constant(15, 4);
    ///
    /// let output_bits = x.len() + y.len();
    /// let product: u8 = x.multiply(y, output_bits).try_into().unwrap();
    /// assert_eq!(product, 30);
    /// ```
    ///
    /// When multiplying signed integers, both inputs should be sign-extended to the requisite
    /// output size. The `output_bits` argument should be set to this size as well.
    ///
    /// # Signed Example
    ///
    /// ```
    /// # use sym::SymbolicBitVec;
    /// let x = SymbolicBitVec::constant(0x2, 4); // Positive 2
    /// let y = SymbolicBitVec::constant(0xF, 4); // Negative 1
    ///
    /// // Need 8 bits to represent the product of 4-bit numbers
    /// let output_bits = x.len() + y.len();
    ///
    /// // Sign extend both x and y to match output length
    /// let x = x.sign_extend(output_bits - 4);
    /// let y = y.sign_extend(output_bits - 4);
    ///
    /// // Multiply the values, but only keep 8 bits of output
    /// let product: u8 = x.multiply(y, output_bits).try_into().unwrap();
    /// let product = product as i8;
    /// assert_eq!(product, -2);
    /// ```
    pub fn multiply(self, rhs: Self, output_bits: usize) -> Self {
        let rhs_size = rhs.len();

        // The initial zero sum may need to be fewer bits than normal if the number of output bits
        // is smaller than self. While additional zeros would not negatively impact the
        // computation, this ensures that the final product has the expected number of output bits.
        let zero = SymbolicBitVec::constant(0, usize::min(self.len(), output_bits));
        let mut product = rhs.concat(zero.clone());

        for n in 0..rhs_size {
            let selector = product.bits[0].clone();

            // The number of bits to sum depends on the number of output bits requested.
            // We can avoid unnecessary additions in this manner. Consider the following
            // where only 4 output bits are requested:
            //
            // a0 a1 a2 a3
            // b0 b1 b2 b3
            // -----------
            //  0  0  0  0 |           [initial sum]
            // y0 y1 y2 y3 |           [b0]
            //    y0 y1 y2 | y3        [b1]
            //       y0 y1 | y2 y3     [b2]
            //          y0 | y1 y2 y3  [b3]
            // ----------------------
            //
            // Everything to the right of the | separator does not need to be computed.
            let max_sum_bits = output_bits.saturating_sub(n);
            let num_sum_bits = usize::min(self.len(), max_sum_bits);
            let carry = if num_sum_bits > 0 {
                // Select the bits to sum, truncating the most significant bits as necessary

                // Extract num_sum_bits from product starting from rhs_size and store in x.
                // The value that's replaced into the product is unimportant since these
                // bits will be overwritten by the sum x + y.
                let mut x = SymbolicBitVec::constant(0, num_sum_bits);
                product
                    .iter_mut()
                    .skip(rhs_size)
                    .take(num_sum_bits)
                    .zip(x.iter_mut())
                    .for_each(|(p, x)| std::mem::swap(p, x));

                let mut y = self.clone().truncate_msb(self.len() - num_sum_bits);
                y.mux_mut(SymbolicBitVec::constant(0, num_sum_bits), selector);

                let (sum, carry) = x.addition_with_carry(y);
                for (i, bit) in sum.bits.into_iter().enumerate() {
                    product.bits[rhs_size + i] = bit;
                }

                Some(carry)
            } else {
                None
            };

            // Remove the current least significant bit corresponding to the RHS and shift
            // everything down.
            product.bits.pop_front();

            // The carry bit should only be added if the next iteration would not result in a
            // truncated addition. This is to ensure the number of product bits at the conclusion
            // of this loop contains the expected number of output bits.
            if max_sum_bits > self.len() {
                product.bits.push_back(carry.unwrap());
            }
        }

        product
    }

    /// Computes the unsigned integer division of `dividend` / `self` and returns
    /// `(quotient, remainder)`.
    ///
    /// # Division by zero
    ///
    /// Division by zero is undefined. It is the responsibility of the caller to ensure the divisor
    /// is non-zero.
    pub fn unsigned_divide(self, dividend: Self) -> (Self, Self) {
        // Zero extend so self and rhs are the same length
        let divisor = match dividend.len().saturating_sub(self.len()) {
            0 => self,
            n => self.zero_extend(n),
        };

        let dividend = match divisor.len().saturating_sub(dividend.len()) {
            0 => dividend,
            n => dividend.zero_extend(n),
        };

        let mut quotient = SymbolicBitVec::constant(0, dividend.len());
        let mut remainder = SymbolicBitVec::constant(0, dividend.len());

        for next_bit in dividend.bits.into_iter().rev() {
            remainder.bits.rotate_right(1);
            remainder.bits[0] = next_bit;

            // Check if RHS is less than working set
            let selector = remainder.clone().less_than(divisor.clone());
            let diff = remainder.clone() - divisor.clone();
            remainder.mux_mut(diff, selector.clone());

            quotient.bits.rotate_right(1);
            quotient.bits[0] = !selector;
        }

        (quotient, remainder)
    }

    pub fn signed_divide(self, dividend: Self) -> (Self, Self) {
        let divisor_msb = self.msb().cloned().unwrap();
        let dividend_msb = dividend.msb().cloned().unwrap();

        let mut unsigned_divisor = self.clone();
        unsigned_divisor.mux_mut(-self, !divisor_msb.clone());

        let mut unsigned_dividend = dividend.clone();
        unsigned_dividend.mux_mut(-dividend, !dividend_msb.clone());

        let (mut quotient, mut remainder) = unsigned_divisor.unsigned_divide(unsigned_dividend);

        let negated_quotient = -quotient.clone();
        quotient.mux_mut(
            negated_quotient,
            divisor_msb.clone().equals(dividend_msb.clone()),
        );

        let negated_remainder = -remainder.clone();
        remainder.mux_mut(negated_remainder, !dividend_msb);

        (quotient, remainder)
    }

    pub fn addition_carry_bits(self, rhs: Self) -> Self {
        assert_eq!(self.bits.len(), rhs.bits.len());
        let mut carry = VecDeque::with_capacity(self.bits.len() + 1);
        carry.push_back(SymbolicBit::Literal(false));
        carry.push_back(self[0].clone() & rhs[0].clone());
        for i in 1..self.bits.len() {
            carry.push_back(
                (self[i].clone() & rhs[i].clone())
                    | (self[i].clone() & carry[i].clone())
                    | (rhs[i].clone() & carry[i].clone()),
            );
        }

        Self { bits: carry }
    }

    pub fn unsigned_addition_overflow(self, rhs: Self) -> SymbolicBit {
        let len = self.len();
        let carry_bits = self.addition_carry_bits(rhs);
        carry_bits[len].clone()
    }

    pub fn signed_addition_overflow(self, rhs: Self) -> SymbolicBit {
        let sum = self.clone() + rhs.clone();

        // Positive overflow occurs if sign bit is 0 for LHS and RHS but 1 in sum
        let positive_overflow = !self[self.len() - 1].clone()
            & !rhs[rhs.len() - 1].clone()
            & sum[sum.len() - 1].clone();

        // Negative overflow occurs if sign bit is 1 for LHS and RHS but 0 in sum
        let negative_overflow =
            self[self.len() - 1].clone() & rhs[rhs.len() - 1].clone() & !sum[sum.len() - 1].clone();

        // Overflow occurs if either positive or negative overflow occurs
        positive_overflow | negative_overflow
    }

    pub fn less_than(mut self, mut rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        if self.is_empty() {
            return FALSE;
        }

        let lhs_msb = self.bits.pop_back().unwrap();
        let rhs_msb = rhs.bits.pop_back().unwrap();
        let less_than = lhs_msb.clone().equals(SymbolicBit::Literal(false))
            & rhs_msb.clone().equals(SymbolicBit::Literal(true));

        if self.bits.is_empty() {
            less_than
        } else {
            less_than | (lhs_msb.equals(rhs_msb) & self.less_than(rhs))
        }
    }

    pub fn less_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().less_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn greater_than(mut self, mut rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        if self.is_empty() {
            return FALSE;
        }

        let lhs_msb = self.bits.pop_back().unwrap();
        let rhs_msb = rhs.bits.pop_back().unwrap();
        let greater_than = lhs_msb.clone().equals(SymbolicBit::Literal(true))
            & rhs_msb.clone().equals(SymbolicBit::Literal(false));

        if self.is_empty() {
            greater_than
        } else {
            greater_than | (lhs_msb.equals(rhs_msb) & self.greater_than(rhs))
        }
    }

    pub fn greater_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().greater_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn signed_less_than(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        if self.is_empty() {
            return FALSE;
        }

        let lhs_sign_bit = self.msb().cloned().unwrap();
        let rhs_sign_bit = rhs.msb().cloned().unwrap();
        let mixed_sign_case = lhs_sign_bit.clone().equals(SymbolicBit::Literal(true))
            & rhs_sign_bit.clone().equals(SymbolicBit::Literal(false));
        let same_sign_case = lhs_sign_bit.equals(rhs_sign_bit) & self.less_than(rhs);

        mixed_sign_case | same_sign_case
    }

    pub fn signed_less_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().signed_less_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn signed_greater_than(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        let lhs_sign_bit = self[self.len() - 1].clone();
        let rhs_sign_bit = rhs[rhs.len() - 1].clone();
        let mixed_sign_case = lhs_sign_bit.clone().equals(SymbolicBit::Literal(false))
            & rhs_sign_bit.clone().equals(SymbolicBit::Literal(true));
        let same_sign_case = lhs_sign_bit.equals(rhs_sign_bit) & self.greater_than(rhs);

        mixed_sign_case | same_sign_case
    }

    pub fn signed_greater_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().signed_greater_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn popcount(self) -> Self {
        let len = if self.is_empty() {
            // This special case is needed to avoid taking log2 of 0
            1
        } else {
            // ilog2 rounds down. If it rounds down, need to add 1. If it is a power of 2 then also
            // need to add 1 to account for if all bits are set
            self.len().ilog2() as usize + 1
        };

        let mut result = SymbolicBitVec::constant(0, len);
        for bit in self.bits.into_iter() {
            let bit: SymbolicBitVec = bit.into();
            let bit = bit.zero_extend(len - 1);
            result = result + bit;
        }
        result
    }

    pub fn signed_shift_right(mut self, rhs: Self) -> Self {
        let sign_bit = self.msb().cloned().unwrap();
        for (i, shift_bit) in rhs.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift_mut(1 << i, sign_bit.clone(), ShiftDirection::Right);
            self.mux_mut(shifted_value, !shift_bit);
        }
        self
    }

    fn shift_mut(&mut self, amount: usize, shift_in: SymbolicBit, direction: ShiftDirection) {
        let len = self.len();

        match direction {
            ShiftDirection::Left => {
                // [ 0 1 2 3 4 5 6 7 ] << 3
                // [ x x x 0 1 2 3 4 ]
                for i in (amount..len).rev() {
                    self.bits.swap(i, i - amount);
                }
                for i in 0..usize::min(len, amount) {
                    self.bits[i] = shift_in.clone();
                }
            }
            ShiftDirection::Right => {
                // [ 0 1 2 3 4 5 6 7 ] >> 3
                // [ 3 4 5 6 7 x x x ]
                for i in amount..len {
                    self.bits.swap(i, i - amount);
                }
                for i in 0..usize::min(len, amount) {
                    self.bits[len - 1 - i] = shift_in.clone();
                }
            }
        }
    }

    fn mux_mut(&mut self, rhs: Self, selector: SymbolicBit) {
        rhs.bits.into_iter().enumerate().for_each(|(i, rhs)| {
            let lhs = std::mem::take(&mut self.bits[i]);
            self.bits[i] = selector.clone().select(lhs, rhs);
        });
    }
}

enum ShiftDirection {
    Left,
    Right,
}

impl std::ops::Not for SymbolicBitVec {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            bits: self.bits.into_iter().map(|bit| !bit).collect(),
        }
    }
}

impl std::ops::BitAnd for SymbolicBitVec {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs & rhs)
                .collect(),
        }
    }
}

impl std::ops::BitOr for SymbolicBitVec {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs | rhs)
                .collect(),
        }
    }
}

impl std::ops::BitXor for SymbolicBitVec {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs ^ rhs)
                .collect(),
        }
    }
}

impl std::ops::Shl<usize> for SymbolicBitVec {
    type Output = Self;

    fn shl(mut self, rhs: usize) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl std::ops::Shl for SymbolicBitVec {
    type Output = Self;

    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl std::ops::ShlAssign<usize> for SymbolicBitVec {
    fn shl_assign(&mut self, rhs: usize) {
        self.shift_mut(rhs, SymbolicBit::Literal(false), ShiftDirection::Left);
    }
}

impl std::ops::ShlAssign for SymbolicBitVec {
    fn shl_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift_mut(1 << i, SymbolicBit::Literal(false), ShiftDirection::Left);
            self.mux_mut(shifted_value, !shift_bit);
        }
    }
}

/// Performs an _unsigned_ right shift.
impl std::ops::ShrAssign for SymbolicBitVec {
    fn shr_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift_mut(1 << i, SymbolicBit::Literal(false), ShiftDirection::Right);
            self.mux_mut(shifted_value, !shift_bit);
        }
    }
}

impl std::ops::ShrAssign<usize> for SymbolicBitVec {
    fn shr_assign(&mut self, rhs: usize) {
        self.shift_mut(rhs, SymbolicBit::Literal(false), ShiftDirection::Right);
    }
}

impl std::ops::Shr for SymbolicBitVec {
    type Output = Self;

    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl std::ops::Shr<usize> for SymbolicBitVec {
    type Output = Self;

    fn shr(mut self, rhs: usize) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl std::ops::Add for SymbolicBitVec {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        // The carry bit size is actually N+1 in order to track whether an overflow has occurred.
        // The addition does not care about the overflow so remove this bit
        let (sum, _) = self.addition_with_carry(rhs);
        sum
    }
}

impl std::ops::Neg for SymbolicBitVec {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let num_bits = self.bits.len();
        !self + SymbolicBitVec::constant(1, num_bits)
    }
}

impl std::ops::Sub for SymbolicBitVec {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        self + (-rhs)
    }
}

impl std::ops::Mul for SymbolicBitVec {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // The output size is the sum of the number of bits. Clippy is overly zealous here flagging
        // the addition as erroneous.
        #[allow(clippy::suspicious_arithmetic_impl)]
        let output_size = self.len() + rhs.len();

        self.multiply(rhs, output_size)
    }
}
