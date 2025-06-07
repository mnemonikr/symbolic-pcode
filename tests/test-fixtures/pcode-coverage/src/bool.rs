use crate::*;

pub fn tests() -> Result {
    let x = true;
    let y = false;

    #[allow(clippy::nonminimal_bool)]
    #[allow(clippy::eq_op)]
    if boolean_or(y, y) {
        return Err(TestError::BooleanOr);
    }

    if boolean_and(x, y) {
        return Err(TestError::BooleanAnd);
    }

    if boolean_neq(x, x) {
        return Err(TestError::BooleanNotEq);
    }

    if boolean_not(x) {
        return Err(TestError::BooleanNot);
    }

    if boolean_eq(x, y) {
        return Err(TestError::BooleanEq);
    }

    if boolean_xor(x, x) {
        return Err(TestError::BooleanXor);
    }

    Ok(())
}

#[inline(never)]
fn boolean_or(x: bool, y: bool) -> bool {
    x || y
}

#[inline(never)]
fn boolean_and(x: bool, y: bool) -> bool {
    x && y
}

#[inline(never)]
fn boolean_xor(x: bool, y: bool) -> bool {
    x ^ y
}

#[inline(never)]
fn boolean_neq(x: bool, y: bool) -> bool {
    x != y
}

#[inline(never)]
fn boolean_eq(x: bool, y: bool) -> bool {
    x == y
}

#[inline(never)]
fn boolean_not(x: bool) -> bool {
    !x
}
