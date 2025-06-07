use crate::*;

pub fn tests() -> Result {
    if or(0x5A, 0xA5) != 0xFF {
        return Err(TestError::BitwiseOr);
    }

    if and(0x5A, 0xA5) != 0x00 {
        return Err(TestError::BitwiseAnd);
    }

    if xor(0xFF, 0xF0) != 0x0F {
        return Err(TestError::BitwiseXor);
    }

    if shl(0x5A, 4) != 0xA0 {
        return Err(TestError::BitwiseShl);
    }

    if shr(0x5A, 4) != 0x05 {
        return Err(TestError::BitwiseShr);
    }

    if shr_signed(i8::MIN, 3) != 0xF0u8 as i8 {
        return Err(TestError::BitwiseShrSigned);
    }

    if negate(0x0) != u128::MAX {
        return Err(TestError::BitwiseNegate);
    }

    Ok(())
}

#[inline(never)]
fn or(x: u8, y: u8) -> u8 {
    x | y
}

#[inline(never)]
fn and(x: u8, y: u8) -> u8 {
    x & y
}

#[inline(never)]
fn xor(x: u8, y: u8) -> u8 {
    x ^ y
}

#[inline(never)]
fn shl(x: u8, y: u8) -> u8 {
    x << y
}

#[inline(never)]
fn shr(x: u8, y: u8) -> u8 {
    x >> y
}

#[inline(never)]
fn shr_signed(x: i8, y: i8) -> i8 {
    x >> y
}

#[inline(never)]
fn negate(x: u128) -> u128 {
    !x
}
