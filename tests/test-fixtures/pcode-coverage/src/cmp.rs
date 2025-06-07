use crate::*;

pub fn tests() -> Result {
    if !(less_than(0, 1) && less_than(-1, 0)) {
        return Err(TestError::CmpLessThan);
    }

    if !(less_than_eq(1, 1) && less_than_eq(-1, -1)) {
        return Err(TestError::CmpLessThanEq);
    }

    if !(greater_than(10, 2) && greater_than(-2, -10)) {
        return Err(TestError::CmpGreaterThan);
    }

    if !(greater_than_eq(10, 2) && greater_than_eq(-2, -10)) {
        return Err(TestError::CmpGreaterThanEq);
    }

    if !(eq(10, 10) && eq(-10, -10)) {
        return Err(TestError::CmpEq);
    }

    if !(neq(55, 10) && neq(-55, -10)) {
        return Err(TestError::CmpNeq);
    }

    // Trying to force use of greater than comparisons
    if cmp(10, 5) != core::cmp::Ordering::Greater {
        return Err(TestError::CmpGreaterThan);
    }

    Ok(())
}

#[inline(never)]
fn less_than<T>(x: T, y: T) -> bool
where
    T: PartialOrd,
{
    x < y
}

#[inline(never)]
fn less_than_eq<T>(x: T, y: T) -> bool
where
    T: PartialOrd,
{
    x <= y
}

#[inline(never)]
fn greater_than<T>(x: T, y: T) -> bool
where
    T: PartialOrd,
{
    x > y
}

#[inline(never)]
fn greater_than_eq<T>(x: T, y: T) -> bool
where
    T: PartialOrd,
{
    x >= y
}

#[inline(never)]
fn neq<T>(x: T, y: T) -> bool
where
    T: PartialEq,
{
    x != y
}

#[inline(never)]
fn eq<T>(x: T, y: T) -> bool
where
    T: PartialEq,
{
    x == y
}

#[inline(never)]
fn cmp(x: u64, y: u64) -> core::cmp::Ordering {
    // CLIPPY: Allow this to try and trigger greater than comparisons
    #[allow(clippy::collapsible_if)]
    if x >= y {
        if x > y {
            return core::cmp::Ordering::Greater;
        }
    } else if x <= y {
        if x < y {
            return core::cmp::Ordering::Less;
        }
    }

    core::cmp::Ordering::Equal
}
