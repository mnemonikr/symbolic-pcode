use crate::*;
use pcode_ops::validate;

#[test]
fn validate_pcode() -> validate::Result {
    validate::Validator::<SymbolicBitVec>::validate()
}
