use std::borrow::Cow;

use crate::emulator::{Error, Result, *};
use crate::mem::{GenericMemory, VarnodeDataStore};
use libsla::*;
use pcode_ops::*;

fn unique_address(offset: u64) -> Address {
    Address {
        offset,
        address_space: unique_address_space(),
    }
}

fn processor_address(offset: u64) -> Address {
    Address {
        offset,
        address_space: processor_address_space(),
    }
}

fn constant_address(offset: u64) -> Address {
    Address {
        offset,
        address_space: constant_address_space(),
    }
}

fn instruction_address() -> Address {
    processor_address(0xFF00000000)
}

fn processor_address_space() -> AddressSpace {
    AddressSpace {
        id: AddressSpaceId::new(0),
        name: Cow::Borrowed("ram"),
        word_size: 1,
        address_size: 4,
        space_type: AddressSpaceType::Processor,
        big_endian: false,
    }
}

fn unique_address_space() -> AddressSpace {
    AddressSpace {
        id: AddressSpaceId::new(1),
        name: Cow::Borrowed("unique"),
        word_size: 1,
        address_size: 8,
        space_type: AddressSpaceType::Internal,
        big_endian: false,
    }
}

fn constant_address_space() -> AddressSpace {
    AddressSpace {
        id: AddressSpaceId::new(2),
        name: Cow::Borrowed("constant"),
        word_size: 1,
        address_size: 8,
        space_type: AddressSpaceType::Constant,
        big_endian: false,
    }
}

fn write_value(
    memory: &mut GenericMemory<Pcode128>,
    offset: u64,
    value: impl Into<Pcode128>,
) -> Result<VarnodeData> {
    let value = value.into();
    let varnode = VarnodeData {
        address: processor_address(offset),
        size: value.num_bytes(),
    };

    memory.write(&varnode, value)?;
    Ok(varnode)
}

#[test]
fn copy() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);
    let data = 0xDEADBEEFu32;
    let input = VarnodeData {
        address: processor_address(0),
        size: 4,
    };
    let output = VarnodeData {
        address: processor_address(4),
        size: 4,
    };
    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Copy,
        inputs: vec![input.clone()],
        output: Some(output.clone()),
    };

    memory.write(&input, data.into())?;
    emulator.emulate(&mut memory, &instruction)?;
    memory.read(&output)?;
    let result: u32 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, data);

    Ok(())
}

#[test]
fn load() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    // Write 0xDEADBEEF to 0x04030201
    let data = 0xDEADBEEFu32;
    let offset = 0x04030201u64;
    write_value(&mut memory, offset, data)?;

    // Write 0x04030201 to 0x0. This is the load indirection
    let offset_data = offset as u32;
    let offset_input = VarnodeData {
        address: processor_address(0),
        size: 4,
    };
    memory.write(&offset_input, offset_data.into())?;

    // Set the address space input offset to the space id of the processor addr space
    // It is important that the address space of this varnode is the constant space.
    let addr_space_input = VarnodeData {
        address: constant_address(processor_address_space().id.raw_id() as u64),
        size: 8, // This value doesn't really matter
    };

    // The output varnode will the location the data is stored at.
    let output = VarnodeData {
        address: unique_address(0),
        size: 4,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Load,
        inputs: vec![addr_space_input.clone(), offset_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u32 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xDEADBEEF);
    Ok(())
}

#[test]
fn store() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    // Write 0xDEADBEEF somewhere. This value will be retrieved and stored to the specified
    // address determined through the space id and offset indirection.
    let data = 0xDEADBEEFu32;
    let data_input = VarnodeData {
        address: unique_address(0xD0D0DADA),
        size: 4,
    };
    memory.write(&data_input, data.into())?;

    // Write 0x04030201 to 0x0. This is the store indirection
    let offset_data = 0x04030201u32;
    let offset_input = VarnodeData {
        address: processor_address(0),
        size: 4,
    };
    memory.write(&offset_input, offset_data.into())?;

    // Set the address space input offset to the space id of the processor addr space
    // It is important that the address space of this varnode is the constant space.
    let addr_space_input = VarnodeData {
        address: constant_address(processor_address_space().id.raw_id() as u64),
        size: 8, // This value doesn't really matter
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Store,
        inputs: vec![
            addr_space_input.clone(),
            offset_input.clone(),
            data_input.clone(),
        ],
        output: None,
    };

    emulator.emulate(&mut memory, &instruction)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0x04030201, // The data stored at offset_input determines this offset
        },
        size: 4,
    };
    let result: u32 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xDEADBEEF);
    Ok(())
}

#[test]
fn int_sub() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let lhs_data = 0xDEADBEEFu32;
    let lhs_input = VarnodeData {
        address: unique_address(0),
        size: 4,
    };
    memory.write(&lhs_input, lhs_data.into())?;

    let rhs_data = 0xBEEFu32;
    let rhs_input = VarnodeData {
        address: unique_address(4),
        size: 4,
    };
    memory.write(&rhs_input, rhs_data.into())?;

    let output = VarnodeData {
        address: processor_address(0),
        size: 4,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::Subtract),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u32 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xDEAD0000);
    Ok(())
}

#[test]
fn int_sborrow() -> Result<()> {
    let test_data = (u8::MIN..=u8::MAX)
        .map(|value| (value, value, false))
        .chain(vec![
            (0x00, 0x80, true),  // 0 - (-128) != -128
            (0x01, 0x81, true),  // 1 - (-127) != -128
            (0x80, 0x00, false), // -128 - 0 = -128
            (0x80, 0x01, true),  // -128 - 1 != 127
        ])
        .collect::<Vec<_>>();

    for (lhs, rhs, expected_result) in test_data {
        let expected_result = if expected_result { 1 } else { 0 };
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, lhs)?;
        let rhs_input = write_value(&mut memory, 1, rhs)?;

        let output = VarnodeData {
            address: processor_address(2),
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: instruction_address(),
            op_code: OpCode::Int(IntOp::Borrow),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;

        let result: u8 = memory.read(&output)?.try_into().unwrap();
        assert_eq!(
            expected_result, result,
            "failed borrow of {lhs} - {rhs}, expected {expected_result} but got {result}"
        );
    }

    Ok(())
}

#[test]
fn int_add() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let lhs_data = 0xDEAD0000u32;
    let lhs_input = VarnodeData {
        address: unique_address(0),
        size: 4,
    };
    memory.write(&lhs_input, lhs_data.into())?;

    let rhs_data = 0xBEEFu32;
    let rhs_input = VarnodeData {
        address: unique_address(4),
        size: 4,
    };
    memory.write(&rhs_input, rhs_data.into())?;

    let output = VarnodeData {
        address: processor_address(0),
        size: 4,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::Add),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u32 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xDEADBEEF);
    Ok(())
}

#[test]
fn int_multiply() -> Result<()> {
    for lhs in 0..16u8 {
        for rhs in 0..16u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: processor_address(2),
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: instruction_address(),
                op_code: OpCode::Int(IntOp::Multiply),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();
            assert_eq!(result, lhs * rhs, "failed {lhs} * {rhs}");
        }
    }

    Ok(())
}

#[test]
fn int_multiply_multibyte() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs: u16 = 0xFF;
    let lhs_input = write_value(&mut memory, 0, lhs)?;

    let rhs: u16 = 0x80;
    let rhs_input = write_value(&mut memory, 2, rhs)?;

    let output = VarnodeData {
        address: processor_address(2),
        size: 2,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::Multiply),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, lhs * rhs, "failed {lhs} * {rhs}");

    Ok(())
}

#[test]
fn int_divide() -> Result<()> {
    for lhs in 0..16u8 {
        for rhs in 1..16u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 2,
                },
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: instruction_address(),
                // This will compute LHS / RHS
                op_code: OpCode::Int(IntOp::Divide(IntSign::Unsigned)),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();
            assert_eq!(result, lhs / rhs, "failed {lhs} / {rhs}");
        }
    }

    Ok(())
}

#[test]
fn int_remainder() -> Result<()> {
    for lhs in 0..16u8 {
        for rhs in 1..16u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: processor_address(2),
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: instruction_address(),
                // This will compute LHS / RHS
                op_code: OpCode::Int(IntOp::Remainder(IntSign::Unsigned)),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();
            assert_eq!(result, lhs % rhs, "failed {lhs} % {rhs}");
        }
    }

    Ok(())
}

#[test]
fn int_signed_divide() -> Result<()> {
    for lhs in 0..16u8 {
        for rhs in 1..16u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_value = Pcode128::new(lhs.into(), 4).sign_extend(1);
            let lhs: i8 = lhs_value.signed_value() as i8;
            let lhs_input = write_value(&mut memory, 0, lhs_value)?;

            let rhs_value = Pcode128::new(rhs.into(), 4).sign_extend(1);
            let rhs: i8 = rhs_value.signed_value() as i8;
            let rhs_input = write_value(&mut memory, 1, rhs_value)?;

            let output = VarnodeData {
                address: processor_address(2),
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: instruction_address(),
                // This will compute LHS / RHS
                op_code: OpCode::Int(IntOp::Divide(IntSign::Signed)),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();
            let expected = lhs / rhs;
            assert_eq!(
                result as i8,
                lhs / rhs,
                "failed signed {lhs} / {rhs}, got {result} but expected {expected}"
            );
        }
    }

    Ok(())
}

#[test]
fn int_signed_remainder() -> Result<()> {
    for lhs in 0..16u8 {
        for rhs in 1..16u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_value = Pcode128::new(lhs.into(), 4).sign_extend(1);
            let lhs: i8 = lhs_value.signed_value() as i8;
            let lhs_input = write_value(&mut memory, 0, lhs_value)?;

            let rhs_value = Pcode128::new(rhs.into(), 4).sign_extend(1);
            let rhs: i8 = rhs_value.signed_value() as i8;
            let rhs_input = write_value(&mut memory, 1, rhs_value)?;

            let output = VarnodeData {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 2,
                },
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 0xFF00000000,
                },
                // This will compute LHS % RHS
                op_code: OpCode::Int(IntOp::Remainder(IntSign::Signed)),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();
            assert_eq!(result as i8, lhs % rhs, "failed signed {lhs} % {rhs}");
        }
    }

    Ok(())
}

#[test]
fn int_zext() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let data = 0xFFu8;
    let input = VarnodeData {
        address: unique_address(0),
        size: 1,
    };
    memory.write(&input, data.into())?;

    let output = VarnodeData {
        address: processor_address(0),
        size: 2,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::Extension(IntSign::Unsigned)),
        inputs: vec![input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x00FF);
    Ok(())
}

#[test]
fn int_sext() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let data = 0x807Fu16;
    let data_varnode = VarnodeData {
        address: unique_address(0),
        size: 2,
    };
    memory.write(&data_varnode, data.into())?;

    let input_positive = VarnodeData {
        address: unique_address(0),
        size: 1,
    };

    let input_negative = VarnodeData {
        address: unique_address(1),
        size: 1,
    };

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0,
        },
        size: 2,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::Extension(IntSign::Signed)),
        inputs: vec![input_positive.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x007F);

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Extension(IntSign::Signed)),
        inputs: vec![input_negative.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xFF80);
    Ok(())
}

#[test]
fn int_equal() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let data = 0xDEADBEEFu32;
    let lhs_input = VarnodeData {
        address: unique_address(0),
        size: 4,
    };
    memory.write(&lhs_input, data.into())?;

    let rhs_input = VarnodeData {
        address: Address {
            address_space: unique_address_space(),
            offset: 4,
        },
        size: 4,
    };
    memory.write(&rhs_input, data.into())?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0,
        },
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Equal),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x1, "Expected 0xDEADBEEF == 0xDEADBEEF to be 1");

    memory.write(&rhs_input, 0u32.into())?;

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x0, "Expected 0xDEADBEEF == 0x0 to be 0");
    Ok(())
}

#[test]
fn int_not_equal() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let data = 0xDEADBEEFu32;
    let lhs_input = VarnodeData {
        address: unique_address(0),
        size: 4,
    };
    memory.write(&lhs_input, data.into())?;

    let rhs_input = VarnodeData {
        address: unique_address(4),
        size: 4,
    };
    memory.write(&rhs_input, data.into())?;

    let output = VarnodeData {
        address: processor_address(0),
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Int(IntOp::NotEqual),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x0, "Expected 0xDEADBEEF != 0xDEADBEEF to be 0");

    memory.write(&rhs_input, 0u32.into())?;
    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0x1, "Expected 0xDEADBEEF != 0x0 to be 1");
    Ok(())
}

#[test]
fn piece() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let msb_input = write_value(&mut memory, 0, 0xDEADu16)?;
    let lsb_input = write_value(&mut memory, 2, 0xBEEFu16)?;

    let output = VarnodeData {
        address: processor_address(4),
        size: 4,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Piece,
        inputs: vec![msb_input.clone(), lsb_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u32 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, 0xDEADBEEF);
    Ok(())
}

#[test]
fn subpiece() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator =
        StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

    let data = 0xDEADBEEFu32;
    let data_input = VarnodeData {
        address: unique_address(0),
        size: 4,
    };
    memory.write(&data_input, data.into())?;

    let truncation_input = VarnodeData {
        address: constant_address(2),
        size: 1,
    };

    let output = VarnodeData {
        address: processor_address(0),
        size: 2,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Subpiece,
        inputs: vec![data_input.clone(), truncation_input.clone()],
        output: Some(output.clone()),
    };

    // Expect to truncate 2 least-significant bytes
    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xDEAD);

    let output = VarnodeData {
        address: processor_address(4),
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::Subpiece,
        inputs: vec![data_input.clone(), truncation_input.clone()],
        output: Some(output.clone()),
    };

    // Expect to truncate 2 least-significant bytes and 1 most significant byte
    // since the output size is less than the input size
    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, 0xAD);
    Ok(())
}

#[test]
fn branch_ind() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let data = 0xDEADBEEFu32;
    let data_input = write_value(&mut memory, 0, data)?;
    let instruction = PcodeInstruction {
        address: instruction_address(),
        op_code: OpCode::BranchIndirect,
        inputs: vec![data_input.clone()],
        output: None,
    };
    let branch_addr = emulator.emulate(&mut memory, &instruction)?;
    let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
        address_space: processor_address_space(),
        offset: 0xDEADBEEF,
    }));
    assert_eq!(branch_addr, expected_addr);
    Ok(())
}

#[test]
fn call_ind() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let data = 0xDEADBEEFu32;
    let data_input = write_value(&mut memory, 0, data)?;

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::CallIndirect,
        inputs: vec![data_input.clone()],
        output: None,
    };
    let branch_addr = emulator.emulate(&mut memory, &instruction)?;
    let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
        address_space: processor_address_space(),
        offset: 0xDEADBEEF,
    }));
    assert_eq!(branch_addr, expected_addr);
    Ok(())
}

#[test]
fn bool_negate() -> Result<()> {
    for value in 0..=1u8 {
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let input = write_value(&mut memory, 0, value)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Bool(BoolOp::Negate),
            inputs: vec![input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;

        let result: u8 = memory.read(&output)?.try_into().unwrap();
        assert_eq!(result, (!value) & 0x1, "failed !{value}");
    }

    Ok(())
}

#[test]
fn bool_and() -> Result<()> {
    for lhs in 0..=1u8 {
        for rhs in 0..=1u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 2,
                },
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 0xFF00000000,
                },
                op_code: OpCode::Bool(BoolOp::And),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();

            assert_eq!(result, lhs & rhs, "failed {lhs} & {rhs}");
        }
    }

    Ok(())
}

#[test]
fn bool_or() -> Result<()> {
    for lhs in 0..=1u8 {
        for rhs in 0..=1u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 2,
                },
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 0xFF00000000,
                },
                op_code: OpCode::Bool(BoolOp::Or),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;

            let result: u8 = memory.read(&output)?.try_into().unwrap();

            assert_eq!(result, lhs | rhs, "failed {lhs} | {rhs}");
        }
    }

    Ok(())
}

#[test]
fn bool_xor() -> Result<()> {
    for lhs in 0..=1u8 {
        for rhs in 0..=1u8 {
            let mut memory = GenericMemory::<Pcode128>::default();
            let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs)?;
            let rhs_input = write_value(&mut memory, 1, rhs)?;

            let output = VarnodeData {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 2,
                },
                size: 1,
            };

            let instruction = PcodeInstruction {
                address: Address {
                    address_space: processor_address_space(),
                    offset: 0xFF00000000,
                },
                op_code: OpCode::Bool(BoolOp::Xor),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&mut memory, &instruction)?;
            let result: u8 = memory.read(&output)?.try_into().unwrap();

            assert_eq!(result, lhs ^ rhs, "failed {lhs} ^ {rhs}");
        }
    }

    Ok(())
}

#[test]
fn int_negate() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs = 0b1010_0101;
    let lhs_input = write_value(&mut memory, 0, lhs)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 2,
        },
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Bitwise(BoolOp::Negate)),
        inputs: vec![lhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;

    let result: u8 = memory.read(&output)?.try_into().unwrap();
    assert_eq!(result, !lhs, "failed !{lhs}");

    Ok(())
}

#[test]
fn int_2comp() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs = 1u8;
    let lhs_input = write_value(&mut memory, 0, lhs)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 2,
        },
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Negate),
        inputs: vec![lhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, -1i8 as u8, "failed -{lhs}");

    Ok(())
}

#[test]
fn int_and() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs = 0b0011_1100;
    let rhs = 0b1010_0101;
    let lhs_input = write_value(&mut memory, 0, lhs)?;
    let rhs_input = write_value(&mut memory, 1, rhs)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 2,
        },
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Bitwise(BoolOp::And)),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, lhs & rhs, "failed {lhs} & {rhs}");

    Ok(())
}

#[test]
fn int_or() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs = 0b0011_1100;
    let rhs = 0b1010_0101;
    let lhs_input = write_value(&mut memory, 0, lhs)?;
    let rhs_input = write_value(&mut memory, 1, rhs)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 2,
        },
        size: 1,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Bitwise(BoolOp::Or)),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u8 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, lhs | rhs, "failed {lhs} | {rhs}");

    Ok(())
}

#[test]
fn int_xor() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let lhs = 0b1111_0000_0011_1100;
    let rhs = 0b0000_1111_1010_0101;
    let lhs_input = write_value(&mut memory, 0, lhs)?;
    let rhs_input = write_value(&mut memory, 2, rhs)?;

    let output = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 4,
        },
        size: 2,
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Int(IntOp::Bitwise(BoolOp::Xor)),
        inputs: vec![lhs_input.clone(), rhs_input.clone()],
        output: Some(output.clone()),
    };

    emulator.emulate(&mut memory, &instruction)?;
    let result: u16 = memory.read(&output)?.try_into().unwrap();

    assert_eq!(result, lhs ^ rhs, "failed {lhs} ^ {rhs}");

    Ok(())
}

#[test]
fn int_less_than() -> Result<()> {
    let test_data: Vec<(u8, u8, bool)> = vec![
        (0x00, 0x00, false),
        (0x00, 0x01, true),
        (0x01, 0x80, true),
        (0x80, 0xFF, true),
    ];
    for (lhs, rhs, expected_result) in test_data {
        let expected_result = if expected_result { 1 } else { 0 };
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, lhs)?;
        let rhs_input = write_value(&mut memory, 1, rhs)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::LessThan(IntSign::Unsigned)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;

        let result: u8 = memory.read(&output)?.try_into().unwrap();
        assert_eq!(result, expected_result, "failed {lhs} < {rhs}");
    }

    Ok(())
}

#[test]
fn int_less_than_eq() -> Result<()> {
    let test_data: Vec<(u8, u8, bool)> = vec![
        (0x00, 0x00, true),
        (0x00, 0x01, true),
        (0x01, 0x80, true),
        (0x80, 0xFF, true),
    ];
    for (lhs, rhs, expected_result) in test_data {
        let expected_result = if expected_result { 1 } else { 0 };
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, lhs)?;
        let rhs_input = write_value(&mut memory, 1, rhs)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();

        assert_eq!(result, expected_result, "failed {lhs} <= {rhs}");
    }

    Ok(())
}

#[test]
fn int_signed_less_than() -> Result<()> {
    let test_data: Vec<(u8, u8, bool)> = vec![
        (0x00, 0x00, false),
        (0x00, 0x01, true),
        (0x01, 0x80, false),
        (0x80, 0xFF, true),
    ];
    for (lhs, rhs, expected_result) in test_data {
        let expected_result = if expected_result { 1 } else { 0 };
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, lhs)?;
        let rhs_input = write_value(&mut memory, 1, rhs)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::LessThan(IntSign::Signed)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();

        assert_eq!(
            result, expected_result,
            "failed signed comparison {lhs} < {rhs}"
        );
    }

    Ok(())
}

#[test]
fn int_signed_less_than_eq() -> Result<()> {
    let test_data: Vec<(u8, u8, bool)> = vec![
        (0x00, 0x00, true),
        (0x00, 0x01, true),
        (0x01, 0x80, false),
        (0x80, 0xFF, true),
    ];
    for (lhs, rhs, expected_result) in test_data {
        let expected_result = if expected_result { 1 } else { 0 };
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, lhs)?;
        let rhs_input = write_value(&mut memory, 1, rhs)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();

        assert_eq!(
            result, expected_result,
            "failed signed comparison {lhs} <= {rhs}"
        );
    }

    Ok(())
}

#[test]
fn shift_left() -> Result<()> {
    for n in 0..=8u8 {
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, 0x01u8)?;
        let rhs_input = write_value(&mut memory, 1, n)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::ShiftLeft),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();
        let expected_result = if n < 8 { 1 << n } else { 0 };

        assert_eq!(result, expected_result, "failed 1 << {n}");
    }

    Ok(())
}

#[test]
fn shift_right() -> Result<()> {
    for n in 0..=8u8 {
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, 0x80u8)?;
        let rhs_input = write_value(&mut memory, 1, n)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();
        let expected_result = if n < 8 { 0x80 >> n } else { 0 };

        assert_eq!(result, expected_result, "failed 0x80 >> {n}");
    }

    Ok(())
}

#[test]
fn signed_shift_right() -> Result<()> {
    for n in 0..=8u8 {
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, 0x80u8)?;
        let rhs_input = write_value(&mut memory, 1, n)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::ShiftRight(IntSign::Signed)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let result: u8 = memory.read(&output)?.try_into().unwrap();
        let expected_result = if n < 8 { (-128i8 >> n) as u8 } else { 0xFF };

        assert_eq!(result, expected_result, "failed signed shift 0x80 >> {n}");
    }

    Ok(())
}

#[test]
fn call() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let data_input = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        },
        size: 0, // This value is irrelevant
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Call,
        inputs: vec![data_input.clone()],
        output: None,
    };

    let branch_addr = emulator.emulate(&mut memory, &instruction)?;
    let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
        address_space: processor_address_space(),
        offset: 0xDEADBEEF,
    }));

    assert_eq!(branch_addr, expected_addr);
    Ok(())
}

#[test]
fn branch_absolute() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let data_input = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        },
        size: 0, // This value is irrelevant
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Branch,
        inputs: vec![data_input.clone()],
        output: None,
    };

    let branch_addr = emulator.emulate(&mut memory, &instruction)?;
    let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
        address_space: processor_address_space(),
        offset: 0xDEADBEEF,
    }));

    assert_eq!(branch_addr, expected_addr);
    Ok(())
}

#[test]
fn branch_pcode_relative() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let data_input = VarnodeData {
        address: Address {
            address_space: constant_address_space(),
            offset: u64::MAX,
        },
        size: 0, // This value is irrelevant
    };

    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Branch,
        inputs: vec![data_input.clone()],
        output: None,
    };

    let branch_addr = emulator.emulate(&mut memory, &instruction)?;
    let expected_addr = ControlFlow::Jump(Destination::PcodeAddress(-1));

    assert_eq!(branch_addr, expected_addr);
    Ok(())
}

#[test]
fn conditional_branch_absolute() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let destination_input = VarnodeData {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        },
        size: 0, // This value is irrelevant
    };

    let condition_input = write_value(&mut memory, 1, 0x1u8)?;
    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::BranchConditional,
        inputs: vec![destination_input.clone(), condition_input.clone()],
        output: None,
    };

    let control_flow = emulator.emulate(&mut memory, &instruction)?;
    let expected_destination = Destination::MachineAddress(Address {
        address_space: processor_address_space(),
        offset: 0xDEADBEEF,
    });
    match control_flow {
        ControlFlow::ConditionalBranch {
            condition_origin,
            condition,
            destination,
        } => {
            assert_eq!(condition_origin, condition_input);
            assert_eq!(condition, Some(true));
            assert_eq!(
                destination, expected_destination,
                "invalid branch destination"
            );
        }
        _ => panic!("unexpected control flow instruction: {control_flow:?}"),
    }

    Ok(())
}

#[test]
fn conditional_branch_pcode_relative() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let destination_input = VarnodeData {
        address: Address {
            address_space: constant_address_space(),
            offset: u64::MAX,
        },
        size: 0, // This value is irrelevant
    };

    let condition_input = write_value(&mut memory, 1, 0x1u8)?;
    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::BranchConditional,
        inputs: vec![destination_input.clone(), condition_input.clone()],
        output: None,
    };

    let control_flow = emulator.emulate(&mut memory, &instruction)?;
    let expected_destination = Destination::PcodeAddress(-1);
    match control_flow {
        ControlFlow::ConditionalBranch {
            condition_origin,
            condition,
            destination,
        } => {
            assert_eq!(condition_input, condition_origin);
            assert_eq!(condition, Some(true));
            assert_eq!(
                destination, expected_destination,
                "invalid branch destination"
            );
        }
        _ => panic!("unexpected control flow instruction: {control_flow:?}"),
    }

    Ok(())
}

#[test]
fn popcount() -> Result<()> {
    for n in 0..=8u8 {
        let value: u8 = ((1u16 << n) - 1) as u8;
        let mut memory = GenericMemory::<Pcode128>::default();
        let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs_input = write_value(&mut memory, 0, value)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Popcount,
            inputs: vec![lhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        let expected_result = n;
        let result: u8 = memory.read(&output)?.try_into().unwrap();

        assert_eq!(result, expected_result, "failed popcount of {value:#02x}");
    }

    Ok(())
}

#[test]
fn unsupported_opcode() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
    let instruction = PcodeInstruction {
        address: Address {
            address_space: processor_address_space(),
            offset: 0xFF00000000,
        },
        op_code: OpCode::Unknown(0),
        inputs: Vec::new(),
        output: None,
    };

    let result = emulator.emulate(&mut memory, &instruction);
    assert!(matches!(result, Err(Error::UnsupportedInstruction { .. })));

    Ok(())
}
