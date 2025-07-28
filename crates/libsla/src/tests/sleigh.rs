use crate::*;
use libsla_sys::sys;

#[test]
pub fn addr_space_type() -> Result<()> {
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_IOP),
        AddressSpaceType::PcodeOp
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_CONSTANT),
        AddressSpaceType::Constant
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_PROCESSOR),
        AddressSpaceType::Processor
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_JOIN),
        AddressSpaceType::Join
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_FSPEC),
        AddressSpaceType::FuncCallSpecs
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_INTERNAL),
        AddressSpaceType::Internal
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_SPACEBASE),
        AddressSpaceType::BaseRegister
    );

    Ok(())
}
