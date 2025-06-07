use crate::*;

pub fn tests() -> Result {
    indirect_call(callback)?;
    Ok(())
}

#[inline(never)]
fn callback(value: u64) -> bool {
    value > 0
}

#[inline(never)]
fn indirect_call(callback: impl Fn(u64) -> bool) -> Result {
    if !(callback)(100) {
        return Err(TestError::ControlFlowIndirectCall);
    }

    Ok(())
}
