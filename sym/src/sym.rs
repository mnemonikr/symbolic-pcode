use std::rc::Rc;

use thiserror;

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
pub enum ConcretizationError<T>
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::error::Error + 'static,
{
    #[error("Non-literal bit at index {0}")]
    NonLiteralBit(usize),

    #[error("Failed to convert {value} to type: {source}")]
    ConversionError {
        value: usize,
        source: <T as TryFrom<usize>>::Error,
    },
}

pub fn concretize_bit_iter<'a, T, I>(iter: I) -> Result<T, ConcretizationError<T>>
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::error::Error,
    I: Iterator<Item = &'a SymbolicBit>,
{
    let mut result: usize = 0;
    let mut bit_value = 1;
    let mut bit_index = 0;
    for bit in iter {
        match *bit {
            SymbolicBit::Literal(is_set) => {
                if is_set {
                    result += bit_value;
                }
            }
            _ => return Err(ConcretizationError::NonLiteralBit(bit_index)),
        }
        bit_value <<= 1;
        bit_index += 1;
    }

    result
        .try_into()
        .map_err(|err| ConcretizationError::ConversionError {
            value: result,
            source: err,
        })
}

impl SymbolicBit {
    fn equals(self, rhs: Self) -> Self {
        (self.clone() & rhs.clone()) | (!self & !rhs)
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
            _ => return SymbolicBit::Not(Rc::new(self)),
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

pub struct SymbolicBitBuf<const N: usize> {
    pub(crate) bits: [SymbolicBit; N],
}

impl<const N: usize> SymbolicBitBuf<N> {
    pub fn new(start_symbol: usize) -> Self {
        // SAFETY: The `assume_init` call is on a MaybeUninit array holding MaybeUninits.
        // It is safe to assume the outer array is initialized since the inner values
        // are still appropriately marked as MaybeUninit.
        let mut bits: [std::mem::MaybeUninit<SymbolicBit>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..N {
            bits[i].write(SymbolicBit::Variable(start_symbol + i));
        }

        Self {
            // SAFETY: Array values are initialized, now safe to convert from array of MaybeUninits
            // to array of initialized values.
            bits: unsafe { (&bits as *const _ as *const [SymbolicBit; N]).read() },
        }
    }
}

impl<const N: usize> std::ops::BitAnd for SymbolicBitBuf<N> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut bits: [std::mem::MaybeUninit<SymbolicBit>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        self.bits
            .into_iter()
            .zip(rhs.bits.into_iter())
            .enumerate()
            .for_each(|(i, (lhs, rhs))| {
                bits[i].write(lhs & rhs);
            });

        Self {
            // SAFETY: Array values are initialized, now safe to convert from array of MaybeUninits
            // to array of initialized values.
            bits: unsafe { (&bits as *const _ as *const [SymbolicBit; N]).read() },
        }
    }
}

impl<const N: usize> From<[SymbolicBit; N]> for SymbolicBitBuf<N> {
    fn from(bits: [SymbolicBit; N]) -> Self {
        Self { bits }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolicBitVec {
    pub(crate) bits: Vec<SymbolicBit>,
}

impl SymbolicBitVec {
    pub fn new(start_symbol: usize) -> Self {
        let mut bits = Vec::with_capacity(8);
        for i in 0..8 {
            bits.push(SymbolicBit::Variable(start_symbol + i));
        }
        Self { bits }
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
            bits: Vec::with_capacity(0),
        }
    }

    pub fn constant(mut value: usize, num_bits: usize) -> Self {
        let mut bits = Vec::with_capacity(num_bits);
        for _ in 0..num_bits {
            bits.push(SymbolicBit::Literal(value & 0x1 > 0));
            value >>= 1;
        }

        if value != 0 {
            // TODO Throw error, this should be a result
        }

        Self { bits }
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
                self.bits
                    .pop()
                    .unwrap_unchecked()
                    .equals(rhs.bits.pop().unwrap_unchecked())
            }
        } else {
            let partition = self.bits.len() / 2;
            let self_part: Self = self.bits.split_off(partition).into();
            let rhs_part: Self = rhs.bits.split_off(partition).into();
            self_part.equals_unchecked(rhs_part) & self.equals_unchecked(rhs)
        }
    }

    /// Concatenates the left-hand side with the right-hand side, creating a new `SymbolicBitVec`
    /// with a combined length of both inputs.
    pub fn concat(mut self, mut rhs: Self) -> Self {
        let mut bits = Vec::with_capacity(self.bits.len() + rhs.bits.len());
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
        let mut extension = Vec::with_capacity(num_bits);
        let msb = &self.bits[self.bits.len() - 1];
        for _ in 0..num_bits {
            extension.push(msb.clone());
        }

        let extension = Self { bits: extension };
        self.concat(extension)
    }

    pub fn addition_carry_bits(self, rhs: Self) -> Self {
        assert_eq!(self.bits.len(), rhs.bits.len());
        let mut carry = Vec::with_capacity(self.bits.len() + 1);
        carry.push(SymbolicBit::Literal(false));
        carry.push(self[0].clone() & rhs[0].clone());
        for i in 1..self.bits.len() {
            carry.push(
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

    pub fn less_than(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        let mut result = SymbolicBit::Literal(false);
        for i in 0..self.len() {
            result = result
                | (self[i].clone().equals(SymbolicBit::Literal(false))
                    & rhs[i].clone().equals(SymbolicBit::Literal(true)));
        }
        result
    }

    pub fn less_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().less_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn greater_than(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        let mut result = SymbolicBit::Literal(false);
        for i in 0..self.len() {
            result = result
                | (self[i].clone().equals(SymbolicBit::Literal(true))
                    & rhs[i].clone().equals(SymbolicBit::Literal(false)));
        }
        result
    }

    pub fn greater_than_eq(self, rhs: Self) -> SymbolicBit {
        self.clone().greater_than(rhs.clone()) | self.equals(rhs)
    }

    pub fn signed_less_than(self, rhs: Self) -> SymbolicBit {
        assert_eq!(self.len(), rhs.len());
        let lhs_sign_bit = self[self.len() - 1].clone();
        let rhs_sign_bit = rhs[rhs.len() - 1].clone();
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
        let len = if self.len() == 0 {
            // This special case is needed to avoid taking log2 of 0
            1
        } else if self.len().is_power_of_two() {
            self.len().ilog2() as usize
        } else {
            // Add 1 if not power of 2 since ilog2 rounds down
            self.len().ilog2() as usize + 1
        };

        let mut result = SymbolicBitVec::constant(0, len);
        for bit in self.bits.into_iter() {
            let bit: SymbolicBitVec = vec![bit].into();
            let bit = bit.zero_extend(len - 1);
            result = result + bit;
        }
        result
    }

    pub fn signed_shift_right(self, rhs: Self) -> Self {
        let sign_bit = self.bits.last().unwrap().clone();
        let mut bit_shift_value = 1;
        let mut result = self;
        for bit in rhs.bits {
            let shifted_value = result.shift(bit_shift_value, sign_bit.clone(), false);
            result = shifted_value.mux(result, bit);
            bit_shift_value <<= 1;
        }

        result
    }

    fn shift(&self, amount: usize, shift_in: SymbolicBit, is_left_shift: bool) -> Self {
        let size = self.len();
        let shift_in = std::iter::repeat(shift_in).take(usize::min(size, amount));
        let bits = if is_left_shift {
            shift_in
                .chain(self.bits.iter().take(size.saturating_sub(amount)).cloned())
                .collect()
        } else {
            self.bits
                .iter()
                .skip(amount)
                .cloned()
                .chain(shift_in)
                .collect()
        };

        Self { bits }
    }

    fn mux(self, rhs: Self, selector: SymbolicBit) -> Self {
        let positive = vec![selector.clone(); self.len()].into();
        let negative = vec![!selector; self.len()].into();
        (self & positive) | (rhs & negative)
    }
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
                .zip(rhs.bits.into_iter())
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
                .zip(rhs.bits.into_iter())
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
                .zip(rhs.bits.into_iter())
                .map(|(lhs, rhs)| lhs ^ rhs)
                .collect(),
        }
    }
}

impl std::ops::Shl<usize> for SymbolicBitVec {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        let num_bits = self.bits.len();
        let mut bits = Vec::with_capacity(num_bits);
        let rhs = usize::min(rhs, num_bits);
        for _ in 0..rhs {
            bits.push(SymbolicBit::Literal(false));
        }
        bits.append(&mut self.bits[..num_bits - rhs].to_vec());

        Self { bits }
    }
}

impl std::ops::Shl for SymbolicBitVec {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        let mut bit_shift_value = 1;
        let mut result = self;
        for bit in rhs.bits {
            let shifted_value = result.shift(bit_shift_value, SymbolicBit::Literal(false), true);
            result = shifted_value.mux(result, bit);
            bit_shift_value <<= 1;
        }

        result
    }
}

/// Performs an _unsigned_ right shift.
impl std::ops::Shr for SymbolicBitVec {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        let mut bit_shift_value = 1;
        let mut result = self;
        for bit in rhs.bits {
            let shifted_value = result.shift(bit_shift_value, SymbolicBit::Literal(false), false);
            result = shifted_value.mux(result, bit);
            bit_shift_value <<= 1;
        }

        result
    }
}

impl std::ops::Add for SymbolicBitVec {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        // The carry bit size is actually N+1 in order to track whether an overflow has occurred.
        // The addition does not care about the overflow so remove this bit
        let carry = self
            .clone()
            .addition_carry_bits(rhs.clone())
            .truncate_msb(1);
        self ^ rhs ^ carry
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_equality() {
        let x = SymbolicBit::Literal(true);
        assert_eq!(x.clone().equals(x), SymbolicBit::Literal(true));

        let x = SymbolicBit::Literal(false);
        assert_eq!(x.clone().equals(x), SymbolicBit::Literal(true));
    }

    #[test]
    fn double_negation() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(!!x.clone(), x);
    }

    #[test]
    fn conjunction_with_false() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(
            x.clone() & SymbolicBit::Literal(false),
            SymbolicBit::Literal(false),
        );
        assert_eq!(
            SymbolicBit::Literal(false) & x.clone(),
            SymbolicBit::Literal(false),
        );
    }

    #[test]
    fn conjunction_with_true() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(x.clone() & SymbolicBit::Literal(true), x);
        assert_eq!(SymbolicBit::Literal(true) & x.clone(), x);
    }

    #[test]
    fn conjunction_with_negated_self() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(x.clone() & !x.clone(), SymbolicBit::Literal(false));
        assert_eq!(!x.clone() & x.clone(), SymbolicBit::Literal(false));
    }

    #[test]
    fn disjunction_with_false() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(x.clone() | SymbolicBit::Literal(false), x);
        assert_eq!(SymbolicBit::Literal(false) | x.clone(), x);
    }

    #[test]
    fn disjunction_with_true() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(
            x.clone() | SymbolicBit::Literal(true),
            SymbolicBit::Literal(true)
        );
        assert_eq!(
            SymbolicBit::Literal(true) | x.clone(),
            SymbolicBit::Literal(true),
        );
    }

    #[test]
    fn exclusive_or_with_self() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(x.clone() ^ x.clone(), SymbolicBit::Literal(false));
    }

    #[test]
    fn exclusive_or_with_zero() {
        let x = SymbolicBit::Variable(0);
        assert_eq!(x.clone() ^ SymbolicBit::Literal(false), x);
        assert_eq!(SymbolicBit::Literal(false) ^ x.clone(), x);
    }

    #[test]
    fn add_bytes_no_carry() {
        let x: SymbolicBitVec = 0xAAu8.into();
        let y: SymbolicBitVec = 0x55u8.into();
        let sum: u8 = (x + y).try_into().expect("failed byte converison");
        assert_eq!(sum, 0xFF);
    }

    #[test]
    fn one_plus_one() {
        let x: SymbolicBitVec = 0x01u8.into();
        let y: SymbolicBitVec = 0x01u8.into();
        let sum: u8 = (x + y).try_into().expect("failed byte converison");
        assert_eq!(sum, 0x02);
    }

    #[test]
    fn maximal_carry_addition() {
        let x: SymbolicBitVec = 0x7Fu8.into();
        let y: SymbolicBitVec = 0x7Fu8.into();
        let sum: u8 = (x + y).try_into().expect("failed byte converison");
        assert_eq!(sum, 0xFE);
    }

    #[test]
    fn negative_one() {
        let one: SymbolicBitVec = 0x01u8.into();
        let negative_one: u8 = (-one).try_into().expect("failed byte converison");
        assert_eq!(negative_one, u8::MAX);
    }

    #[test]
    fn addition_overflow() {
        let x: SymbolicBitVec = u8::MAX.into();
        let sum: u8 = (x + 1u8.into()).try_into().expect("failed byte converison");
        assert_eq!(sum, 0x00);
    }

    #[test]
    fn addition_overflow_carry() {
        let x: SymbolicBitVec = u8::MAX.into();
        let y: SymbolicBitVec = 1u8.into();
        let carry = x.addition_carry_bits(y);
        assert_eq!(carry[8], SymbolicBit::Literal(true));

        let x: SymbolicBitVec = (u8::MAX - 1).into();
        let y: SymbolicBitVec = 1u8.into();
        let carry = x.addition_carry_bits(y);
        assert_eq!(carry[8], SymbolicBit::Literal(false));
    }

    #[test]
    fn subtraction() {
        for n in u8::MIN..=u8::MAX {
            let x: SymbolicBitVec = n.into();
            let y: SymbolicBitVec = n.into();
            let diff: u8 = (x - y).try_into().expect("failed byte converison");
            assert_eq!(diff, 0);
        }
    }

    #[test]
    fn left_shift() {
        for n in u8::MIN..u8::MAX {
            let x: SymbolicBitVec = 1u8.into();
            let result: u8 = (x << n as usize)
                .try_into()
                .expect("failed byte conversion");

            // Rust panics if shifting more than byte size
            let expected_result = if n < 8 { 1u8 << n } else { 0 };
            assert_eq!(result, expected_result);
        }
    }

    #[test]
    fn symbolic_concrete_conversions() {
        for n in 0..u8::MAX {
            let symbolic_byte: SymbolicBitVec = n.into();
            let concrete_byte: u8 = symbolic_byte
                .try_into()
                .expect("failed to convert back into byte");
            assert_eq!(concrete_byte, n);
        }
    }

    #[test]
    fn sign_extension() {
        let x: SymbolicBitVec = u8::MAX.into();
        let sext: u16 = x
            .sign_extend(8)
            .try_into()
            .expect("failed concrete conversion");
        assert_eq!(sext, 0xFFFFu16);
    }

    #[test]
    fn zero_extension() {
        let x: SymbolicBitVec = u8::MAX.into();
        let zext: u16 = x
            .zero_extend(8)
            .try_into()
            .expect("failed concrete conversion");
        assert_eq!(zext, 0x00FFu16);
    }

    #[test]
    fn concatenation() {
        let x: SymbolicBitVec = 0xADu8.into();
        let y: SymbolicBitVec = 0xDEu8.into();
        let concat: u16 = x.concat(y).try_into().expect("failed type conversion");
        assert_eq!(concat, 0xDEADu16);
    }

    #[test]
    fn truncation_lsb() {
        let x: SymbolicBitVec = 0xDEADu16.into();
        let truncated: u8 = x
            .truncate_lsb(8)
            .try_into()
            .expect("failed type converison");
        assert_eq!(truncated, 0xDEu8);
    }

    #[test]
    fn truncation_msb() {
        let x: SymbolicBitVec = 0xDEADu16.into();
        let truncated: u8 = x
            .truncate_msb(8)
            .try_into()
            .expect("failed type converison");
        assert_eq!(truncated, 0xADu8);
    }

    #[test]
    fn split_into_bytes() {
        let x: SymbolicBitVec = 0xDEADu16.into();
        let split = x.into_parts(8);
        assert_eq!(split.len(), 2);

        let byte: u8 = (&split[0]).try_into().expect("failed type conversion");
        assert_eq!(byte, 0xADu8);

        let byte: u8 = (&split[1]).try_into().expect("failed type conversion");
        assert_eq!(byte, 0xDEu8);
    }

    #[test]
    fn one_bit_equality() {
        let x = SymbolicBitVec::constant(0, 1);
        let y = SymbolicBitVec::constant(0, 1);
        let eq = x.equals(y);
        assert_eq!(eq, SymbolicBit::Literal(true));

        let x = SymbolicBitVec::constant(0, 1);
        let y = SymbolicBitVec::constant(1, 1);
        let eq = x.equals(y);
        assert_eq!(eq, SymbolicBit::Literal(false));
    }

    #[test]
    fn equals() {
        let x: SymbolicBitVec = 0xFEEDF00Du32.into();
        let y: SymbolicBitVec = 0xDEEDF00Du32.into();
        let eq = x.clone().equals(x.clone());
        let neq = x.equals(y);
        assert_eq!(eq, SymbolicBit::Literal(true));
        assert_eq!(neq, SymbolicBit::Literal(false));
    }

    #[test]
    fn less_than() {
        let x: SymbolicBitVec = 1u32.into();
        let y: SymbolicBitVec = 0u32.into();
        let less_than = y.clone().less_than(x);
        let not_less_than = y.clone().less_than(y);
        assert_eq!(less_than, SymbolicBit::Literal(true));
        assert_eq!(not_less_than, SymbolicBit::Literal(false));
    }

    #[test]
    fn greater_than() {
        let x: SymbolicBitVec = 0u32.into();
        let y: SymbolicBitVec = 1u32.into();
        let greater_than = y.clone().greater_than(x);
        let not_greater_than = y.clone().greater_than(y);
        assert_eq!(greater_than, SymbolicBit::Literal(true));
        assert_eq!(not_greater_than, SymbolicBit::Literal(false));
    }

    #[test]
    fn signed_less_than() {
        let neg_one: SymbolicBitVec = 0xFFu8.into();
        let neg_two: SymbolicBitVec = 0xFEu8.into();
        let pos_one: SymbolicBitVec = 0x01u8.into();
        let pos_two: SymbolicBitVec = 0x02u8.into();

        assert_eq!(
            neg_two.clone().signed_less_than(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -2 < -1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_less_than(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 < 1 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_less_than(pos_two.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 < 2 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_less_than(pos_one.clone()),
            SymbolicBit::Literal(false),
            "Expect 1 < 1 to be false (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_less_than(neg_one.clone()),
            SymbolicBit::Literal(false),
            "Expect -1 < -1 to be false (signed comparison)",
        );
    }

    #[test]
    fn signed_greater_than() {
        let neg_one: SymbolicBitVec = 0xFFu8.into();
        let neg_two: SymbolicBitVec = 0xFEu8.into();
        let pos_one: SymbolicBitVec = 0x01u8.into();
        let pos_two: SymbolicBitVec = 0x02u8.into();

        assert_eq!(
            pos_two.clone().signed_greater_than(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 2 > 1 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_greater_than(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 > -1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_greater_than(neg_two.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 > -2 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_greater_than(pos_one.clone()),
            SymbolicBit::Literal(false),
            "Expect 1 > 1 to be false (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_greater_than(neg_one.clone()),
            SymbolicBit::Literal(false),
            "Expect -1 > -1 to be false (signed comparison)",
        );
    }

    #[test]
    fn signed_less_than_eq() {
        let neg_one: SymbolicBitVec = 0xFFu8.into();
        let neg_two: SymbolicBitVec = 0xFEu8.into();
        let pos_one: SymbolicBitVec = 0x01u8.into();
        let pos_two: SymbolicBitVec = 0x02u8.into();

        assert_eq!(
            neg_two.clone().signed_less_than_eq(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -2 < -1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_less_than_eq(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 < 1 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_less_than_eq(pos_two.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 < 2 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_less_than_eq(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 < 1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_less_than_eq(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 < -1 (signed comparison)",
        );
    }

    #[test]
    fn signed_greater_than_eq() {
        let neg_one: SymbolicBitVec = 0xFFu8.into();
        let neg_two: SymbolicBitVec = 0xFEu8.into();
        let pos_one: SymbolicBitVec = 0x01u8.into();
        let pos_two: SymbolicBitVec = 0x02u8.into();

        assert_eq!(
            pos_two.clone().signed_greater_than_eq(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 2 >= 1 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_greater_than_eq(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 >= -1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_greater_than_eq(neg_two.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 >= -2 (signed comparison)",
        );
        assert_eq!(
            pos_one.clone().signed_greater_than_eq(pos_one.clone()),
            SymbolicBit::Literal(true),
            "Expect 1 >= 1 (signed comparison)",
        );
        assert_eq!(
            neg_one.clone().signed_greater_than_eq(neg_one.clone()),
            SymbolicBit::Literal(true),
            "Expect -1 >= -1 (signed comparison)",
        );
    }

    #[test]
    fn popcount() {
        for n in 0..8 {
            let value = SymbolicBitVec::constant((1 << n) - 1, 8);
            let popcount = value.popcount();
            let popcount: u8 = popcount.try_into().expect("failed converison");
            assert_eq!(popcount, n);
        }
    }

    #[test]
    fn shift_left() {
        for n in 0..=8 {
            let value = SymbolicBitVec::constant(0b0000_0001, 8);
            let shift_amount = SymbolicBitVec::constant(n, 8);
            let expected = if n < 8 { 1 << n } else { 0 };
            let result: u8 = (value << shift_amount)
                .try_into()
                .expect("failed conversion");
            assert_eq!(result, expected, "failed 1 << {n}");
        }
    }

    #[test]
    fn shift_right() {
        for n in 0..=8 {
            let value = SymbolicBitVec::constant(0b1000_0000, 8);
            let shift_amount = SymbolicBitVec::constant(n, 8);
            let expected = if n < 8 { 0x80 >> n } else { 0 };
            let result: u8 = (value >> shift_amount)
                .try_into()
                .expect("failed conversion");
            assert_eq!(result, expected, "failed 0x80 >> {n}");
        }
    }

    #[test]
    fn signed_shift_right_negative() {
        for n in 0..=8 {
            let value = SymbolicBitVec::constant(0b1000_0000, 8);
            let shift_amount = SymbolicBitVec::constant(n, 8);
            let expected = if n < 8 { (-128 as i8 >> n) as u8 } else { 0xFF };
            let result: u8 = value
                .signed_shift_right(shift_amount)
                .try_into()
                .expect("failed conversion");
            assert_eq!(result, expected, "failed signed shift 0x80 >> {n}");
        }
    }

    #[test]
    fn signed_shift_right_positive() {
        for n in 0..=8 {
            let value = SymbolicBitVec::constant(0b0111_1111, 8);
            let shift_amount = SymbolicBitVec::constant(n, 8);
            let expected = if n < 8 { 0x7F >> n } else { 0 };
            let result: u8 = value
                .signed_shift_right(shift_amount)
                .try_into()
                .expect("failed conversion");
            assert_eq!(result, expected, "failed signed shift 0x7F >> {n}");
        }
    }
}
