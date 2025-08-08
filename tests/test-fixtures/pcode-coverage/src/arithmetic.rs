use crate::*;

pub fn tests() -> Result {
    if add(u64::MAX, 1) != (0, true) {
        return Err(TestError::IntAdd);
    }

    if subtract(0, 1) != (u64::MAX, true) {
        return Err(TestError::IntSubtract);
    }

    if multiply(u64::MAX, 16) != (0xFFFF_FFFF_FFFF_FFF0, true) {
        return Err(TestError::IntMultiply);
    }

    if large_multiply(u64::MAX.into(), u64::MAX.into()) != 0xfffffffffffffffe0000000000000001 {
        return Err(TestError::IntLargeMultiply);
    }

    if divide(0x12345678, 16) != 0x01234567 {
        return Err(TestError::IntDivide);
    }

    if divide_signed(0x12345678, -16) != -0x01234567 {
        return Err(TestError::IntDivide);
    }

    if remainder(0x12345678, 16) != 8 {
        return Err(TestError::IntRemainder);
    }

    if remainder_signed(-32, 5) != -2 {
        return Err(TestError::IntRemainder);
    }

    if negate(-32) != 32 {
        return Err(TestError::IntNegate);
    }

    Ok(())
}

#[inline(never)]
fn add(x: u64, y: u64) -> (u64, bool) {
    x.overflowing_add(y)
}

#[inline(never)]
fn subtract(x: u64, y: u64) -> (u64, bool) {
    x.overflowing_sub(y)
}

#[inline(never)]
fn multiply(x: u64, y: u64) -> (u64, bool) {
    x.overflowing_mul(y)
}

#[inline(never)]
fn divide(x: u64, y: u64) -> u64 {
    x / y
}

#[inline(never)]
fn divide_signed(x: i64, y: i64) -> i64 {
    x / y
}

#[inline(never)]
fn remainder(x: u64, y: u64) -> u64 {
    x % y
}

#[inline(never)]
fn remainder_signed(x: i64, y: i64) -> i64 {
    x % y
}

#[inline(never)]
fn large_multiply(x: u128, y: u128) -> u128 {
    x * y
}

#[inline(never)]
#[cfg(target_arch = "x86_64")]
fn negate(mut x: i64) -> i64 {
    // Required to force the neg instruction since the default assembly produced does not use it
    unsafe {
        core::arch::asm!("neg {x}", x = inout(reg) x);
    }
    x
}

#[inline(never)]
#[cfg(not(target_arch = "x86_64"))]
fn negate(x: i64) -> i64 {
    -x
}
