use crate::convert::{PcodeValue, TryFromPcodeValueError};
use crate::pcode128::Pcode128;

#[test]
fn into_byte_buffer() -> Result<(), TryFromPcodeValueError> {
    let value = PcodeValue::from(Pcode128::new(u64::MAX.into(), u64::BITS));
    let bytes: [u8; 8] = value.try_into()?;
    for byte in bytes {
        assert_eq!(byte, 0xff);
    }
    Ok(())
}

#[test]
fn into_byte_buffer_too_small() -> Result<(), TryFromPcodeValueError> {
    let value = PcodeValue::from(Pcode128::new(u64::MAX.into(), u64::BITS));
    let result: Result<[u8; 7], _> = value.try_into();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TryFromPcodeValueError::InvalidSize
    ));
    Ok(())
}

#[test]
fn into_byte_buffer_too_big() -> Result<(), TryFromPcodeValueError> {
    let value = PcodeValue::from(Pcode128::new(u64::MAX.into(), u64::BITS));
    let result: Result<[u8; 9], _> = value.try_into();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TryFromPcodeValueError::InvalidSize
    ));
    Ok(())
}

#[test]
fn into_vec() -> Result<(), TryFromPcodeValueError> {
    let value = PcodeValue::from(Pcode128::new(u64::MAX.into(), u64::BITS));
    let bytes: Vec<u8> = value.try_into()?;
    for byte in bytes {
        assert_eq!(byte, 0xff);
    }
    Ok(())
}
