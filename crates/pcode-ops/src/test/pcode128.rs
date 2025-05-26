use crate::validate::*;
use crate::*;

#[test]
fn validate_pcode128_impl() -> Result {
    Validator::<Pcode128>::validate()
}
