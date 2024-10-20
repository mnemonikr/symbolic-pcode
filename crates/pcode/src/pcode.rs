// Note: Will move the pcode traits from sym into here once this crate no longer depends on sym
// Copied into sym for testing purposes only at the moment
pub trait PcodeOps: BitwisePcodeOps {
    type Bit: BitwisePcodeOps;

    fn add(&self, other: &Self) -> Self;
    fn carry(&self, other: &Self, sign: SignType) -> Self::Bit;
    fn negate(&self, other: &Self) -> Self;
    fn subtract(&self, other: &Self) -> Self;
    fn borrow(&self, other: &Self) -> Self::Bit;
    fn multiply(&self, other: &Self) -> Self;
    fn divide(&self, other: &Self, sign: SignType) -> Self;
    fn remainder(&self, other: &Self, sign: SignType) -> Self;
    fn extend(&self, other: &Self, sign: SignType) -> Self;
    fn piece(&self, other: &Self) -> Self;

    // Amount here is a constant. Constants are stored in the offset field of an address which is
    // u64.
    //
    // TODO Abstract that field (and maybe others?) into their own types??
    fn subpiece(&self, amount: u64) -> Self;

    fn shift_left(&self, other: &Self) -> Self;
    fn shift_right(&self, other: &Self, sign: SignType) -> Self;

    fn equals(&self, other: &Self) -> Self::Bit;
    fn less_than(&self, other: &Self, sign: SignType) -> Self::Bit;

    fn not_equals(&self, other: &Self) -> Self::Bit {
        self.equals(other).negate()
    }

    fn less_than_or_equals(&self, other: &Self, sign: SignType) -> Self::Bit {
        self.less_than(other, sign).or(&self.equals(other))
    }

    fn greater_than(&self, other: &Self, sign: SignType) -> Self::Bit {
        self.less_than_or_equals(other, sign).negate()
    }

    fn greater_than_or_equals(&self, other: &Self, sign: SignType) -> Self::Bit {
        self.less_than(other, sign).negate()
    }
}

pub trait BitwisePcodeOps {
    fn and(&self, other: &Self) -> Self;
    fn negate(&self) -> Self;
    fn or(&self, other: &Self) -> Self
    where
        Self: Sized,
    {
        (self.negate()).and(&other.negate()).negate()
    }

    fn xor(&self, other: &Self) -> Self
    where
        Self: Sized,
    {
        self.and(&other.negate())
            .negate()
            .and(&self.negate().and(other).negate())
            .negate()
    }
}

pub enum SignType {
    Signed,
    Unsigned,
}

impl PcodeOps for u64 {
    type Bit = bool;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn carry(&self, other: &Self, sign: SignType) -> Self::Bit {
        match sign {
            SignType::Signed => {
                let x = *self as i64;
                let y = *other as i64;
                x.checked_add(y).is_none()
            }
            SignType::Unsigned => self.checked_add(*other).is_none(),
        }
    }

    fn negate(&self, other: &Self) -> Self {
        todo!()
    }

    fn subtract(&self, other: &Self) -> Self {
        todo!()
    }

    fn borrow(&self, other: &Self) -> Self::Bit {
        todo!()
    }

    fn multiply(&self, other: &Self) -> Self {
        todo!()
    }

    fn divide(&self, other: &Self, sign: SignType) -> Self {
        todo!()
    }

    fn remainder(&self, other: &Self, sign: SignType) -> Self {
        todo!()
    }

    fn extend(&self, other: &Self, sign: SignType) -> Self {
        todo!()
    }

    fn piece(&self, other: &Self) -> Self {
        todo!()
    }

    fn subpiece(&self, amount: u64) -> Self {
        todo!()
    }

    fn shift_left(&self, other: &Self) -> Self {
        todo!()
    }

    fn shift_right(&self, other: &Self, sign: SignType) -> Self {
        todo!()
    }

    fn equals(&self, other: &Self) -> Self::Bit {
        todo!()
    }

    fn less_than(&self, other: &Self, sign: SignType) -> Self::Bit {
        todo!()
    }
}

impl BitwisePcodeOps for u64 {
    fn and(&self, other: &Self) -> Self {
        self & other
    }

    fn negate(&self) -> Self {
        !self
    }

    fn or(&self, other: &Self) -> Self {
        self | other
    }

    fn xor(&self, other: &Self) -> Self {
        self ^ other
    }
}

impl BitwisePcodeOps for bool {
    fn and(&self, other: &Self) -> Self {
        self & other
    }

    fn negate(&self) -> Self {
        !self
    }

    fn or(&self, other: &Self) -> Self {
        self | other
    }

    fn xor(&self, other: &Self) -> Self {
        self ^ other
    }
}
