#![no_std]
#![no_main]

mod arithmetic;
mod bitwise;
mod bool;
mod cmp;
mod controlflow;

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> u32 {
    match test() {
        Ok(()) => TestError::Success as u32,
        Err(err) => err as u32,
    }
}

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[repr(u32)]
enum TestError {
    Success = 0x0,
    BooleanOr = 0x11,
    BooleanAnd = 0x12,
    BooleanNotEq = 0x13,
    BooleanNot = 0x14,
    BooleanEq = 0x15,
    BooleanXor = 0x16,
    BitwiseOr = 0x21,
    BitwiseAnd = 0x22,
    BitwiseXor = 0x23,
    BitwiseShl = 0x24,
    BitwiseShr = 0x25,
    BitwiseShrSigned = 0x26,
    BitwiseNegate = 0x27,
    IntAdd = 0x31,
    IntSubtract = 0x32,
    IntMultiply = 0x33,
    IntLargeMultiply = 0x34,
    IntDivide = 0x35,
    IntRemainder = 0x36,
    IntNegate = 0x37,
    ControlFlowIndirectCall = 0x41,
    CmpLessThan = 0x51,
    CmpLessThanEq = 0x52,
    CmpGreaterThan = 0x53,
    CmpGreaterThanEq = 0x54,
    CmpEq = 0x55,
    CmpNeq = 0x56,
}

type Result = core::result::Result<(), TestError>;

fn test() -> Result {
    arithmetic::tests()?;
    bitwise::tests()?;
    bool::tests()?;
    cmp::tests()?;
    controlflow::tests()?;

    Ok(())
}
