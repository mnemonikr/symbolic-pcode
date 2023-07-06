use thiserror;

use crate::mem;
use crate::sym;
use sla::{
    Address, AddressSpace, AddressSpaceType, LoadImage, OpCode, PcodeInstruction, VarnodeData,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MemoryAccess(#[from] mem::Error),

    #[error("illegal instruction {0:?}: {1}")]
    IllegalInstruction(PcodeInstruction, String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct PcodeEmulator {
    memory: mem::Memory,
}

impl LoadImage for PcodeEmulator {
    fn instruction_bytes(&self, input: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        let bytes = self.memory.read_bytes(&input);

        // The number of bytes requested may exceed valid data in memory.
        // In that case only read and return the defined bytes.
        let bytes = match bytes {
            Err(mem::Error::UndefinedData(addr)) => {
                let input = VarnodeData {
                    size: addr.offset - input.address.offset,
                    address: input.address.clone(),
                };
                self.memory.read_bytes(&input)
            }
            _ => bytes,
        };

        bytes
            .map_err(|err| err.to_string())?
            .into_iter()
            .map(|x| x.try_into())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_err| "symbolic byte".to_string())
    }
}

fn check_num_inputs(instruction: &PcodeInstruction, num_inputs: usize) -> Result<()> {
    if instruction.inputs.len() == num_inputs {
        Ok(())
    } else {
        Err(Error::IllegalInstruction(
            instruction.clone(),
            format!(
                "expected {num_inputs} inputs, found {}",
                instruction.inputs.len()
            ),
        ))
    }
}

fn check_has_output(instruction: &PcodeInstruction, has_output: bool) -> Result<()> {
    if instruction.output.is_some() == has_output {
        Ok(())
    } else {
        Err(Error::IllegalInstruction(
            instruction.clone(),
            format!(
                "instruction has unexpected output: {:?}",
                instruction.output
            ),
        ))
    }
}

impl PcodeEmulator {
    pub fn new(address_spaces: Vec<AddressSpace>) -> Self {
        Self {
            memory: mem::Memory::new(address_spaces),
        }
    }

    pub fn memory(&self) -> &mem::Memory {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut mem::Memory {
        &mut self.memory
    }

    pub fn emulate(&mut self, instruction: &PcodeInstruction) -> Result<Option<Address>> {
        match instruction.op_code {
            OpCode::CPUI_COPY => self.copy(&instruction)?,
            OpCode::CPUI_LOAD => self.load(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_STORE => self.store(
                &instruction.inputs[0],
                &instruction.inputs[1],
                &instruction.inputs[2],
            )?,
            OpCode::CPUI_INT_AND => self.int_and(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_ADD => self.int_add(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_CARRY => self.int_carry(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_SCARRY => self.int_scarry(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_SUB => self.int_sub(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_SEXT => {
                self.int_sext(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::CPUI_INT_ZEXT => {
                self.int_zext(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::CPUI_POPCOUNT => {
                self.popcount(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::CPUI_SUBPIECE => self.subpiece(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_EQUAL => self.int_equal(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_NOTEQUAL => self.int_not_equal(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_INT_SLESS => self.int_signed_less_than(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::CPUI_RETURN => return self.return_instruction(&instruction).map(Option::Some),
            _ => unimplemented!("Operation not yet implemented: {:?}", instruction.op_code),
        }

        Ok(None)
    }

    /// Copy a sequence of contiguous bytes from anywhere to anywhere. Size of input0 and output
    /// must be the same.
    pub fn copy(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, true)?;

        let input = &instruction.inputs[0];
        let output = unsafe { instruction.output.as_ref().unwrap_unchecked() };

        if input.size != output.size {
            return Err(Error::IllegalInstruction(
                instruction.clone(),
                format!(
                    "input varnode {} and output varnode {} sizes differ",
                    input, output,
                ),
            ));
        }

        self.memory
            .write_bytes(self.memory.read_bytes_owned(&input)?, &output)?;

        Ok(())
    }

    /// This instruction loads data from a dynamic location into the output variable by
    /// dereferencing a pointer. The "pointer" comes in two pieces. One piece, input1, is a normal
    /// variable containing the offset of the object being pointed at. The other piece, input0, is a
    /// constant indicating the space into which the offset applies. The data in input1 is
    /// interpreted as an unsigned offset and should have the same size as the space referred to by
    /// the ID, i.e. a 4-byte address space requires a 4-byte offset. The space ID is not manually
    /// entered by a user but is automatically generated by the p-code compiler. The amount of data
    /// loaded by this instruction is determined by the size of the output variable. It is easy to
    /// confuse the address space of the output and input1 variables and the Address Space
    /// represented by the ID, which could all be different. Unlike many programming models, there
    /// are multiple spaces that a "pointer" can refer to, and so an extra ID is required.
    pub fn load(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(
            input_0.address.address_space.space_type,
            AddressSpaceType::Constant
        );

        let input = VarnodeData {
            address: Address {
                address_space: self.memory.address_space(&input_0)?.clone(),
                offset: self.memory.read_concrete_value(&input_1)?,
            },
            size: output.size,
        };

        self.memory
            .write_bytes(self.memory.read_bytes_owned(&input)?, &output)?;

        Ok(())
    }

    ///  This instruction is the complement of LOAD. The data in the variable input2 is stored at a
    ///  dynamic location by dereferencing a pointer. As with LOAD, the “pointer” comes in two
    ///  pieces: a space ID part, and an offset variable. The size of input1 must match the address
    ///  space specified by the ID, and the amount of data stored is determined by the size of
    ///  input2.
    ///
    /// Its possible for the addressable unit of an address space to be bigger than a single byte.
    /// If the wordsize attribute of the space given by the ID is bigger than one, the offset into
    /// the space obtained from input1 must be multiplied by this value in order to obtain the
    /// correct byte offset into the space.
    pub fn store(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        input_2: &VarnodeData,
    ) -> Result<()> {
        let output = VarnodeData {
            address: Address {
                address_space: self.memory.address_space(&input_0)?.clone(),
                offset: self.memory.read_concrete_value(&input_1)?,
            },
            size: input_2.size,
        };

        self.memory
            .write_bytes(self.memory.read_bytes_owned(&input_2)?, &output)?;

        Ok(())
    }

    pub fn int_and(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(input_1.size, output.size);

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let value = lhs & rhs;
        self.memory.write_bytes(value.into_parts(8), &output)?;

        Ok(())
    }

    /// This is standard integer addition. It works for either unsigned or signed interpretations
    /// of the integer encoding (twos complement). Size of both inputs and output must be the same.
    /// The addition is of course performed modulo this size. Overflow and carry conditions are
    /// calculated by other operations. See INT_CARRY and INT_SCARRY.
    pub fn int_add(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(input_1.size, output.size);
        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let sum = lhs + rhs;
        self.memory.write_bytes(sum.into_parts(8), &output)?;

        Ok(())
    }

    /// This operation checks for unsigned addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as unsigned integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    pub fn int_carry(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(output.size, 1);
        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();

        let overflow: sym::SymbolicBitVec = vec![lhs.unsigned_addition_overflow(rhs)].into();
        self.memory
            .write_bytes(vec![overflow.zero_extend(7)], &output)?;

        Ok(())
    }

    /// This operation checks for signed addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as signed integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    pub fn int_scarry(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(output.size, 1);
        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let overflow: sym::SymbolicBitVec = vec![lhs.signed_addition_overflow(rhs)].into();
        self.memory
            .write_bytes(vec![overflow.zero_extend(7)], &output)?;

        Ok(())
    }

    ///  This is standard integer subtraction. It works for either unsigned or signed
    ///  interpretations of the integer encoding (twos complement). Size of both inputs and output
    ///  must be the same. The subtraction is of course performed modulo this size. Overflow and
    ///  borrow conditions are calculated by other operations. See INT_SBORROW and INT_LESS.
    pub fn int_sub(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(input_1.size, output.size);
        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let diff = lhs - rhs;
        self.memory.write_bytes(diff.into_parts(8), &output)?;

        Ok(())
    }

    pub fn int_zext(&mut self, input_0: &VarnodeData, output: &VarnodeData) -> Result<()> {
        assert!(output.size > input_0.size);
        let data: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let data = data.zero_extend(8 * (output.size - input_0.size) as usize);
        self.memory.write_bytes(data.into_parts(8), &output)?;

        Ok(())
    }

    pub fn int_sext(&mut self, input_0: &VarnodeData, output: &VarnodeData) -> Result<()> {
        assert!(output.size > input_0.size);
        let data: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let data = data.sign_extend(8 * (output.size - input_0.size) as usize);
        self.memory.write_bytes(data.into_parts(8), &output)?;

        Ok(())
    }

    pub fn int_equal(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(output.size, 1);

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let bit: sym::SymbolicBitVec = vec![lhs.equals(rhs)].into();
        self.memory.write_bytes(vec![bit.zero_extend(7)], &output)?;
        Ok(())
    }

    pub fn int_not_equal(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(output.size, 1);

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let bit: sym::SymbolicBitVec = vec![!lhs.equals(rhs)].into();
        self.memory.write_bytes(vec![bit.zero_extend(7)], &output)?;
        Ok(())
    }

    /// This is a signed integer comparison operator. If the signed integer input0 is strictly less
    /// than the signed integer input1, output is set to true. Both inputs must be the same size,
    /// and the output must have a size of 1.
    pub fn int_signed_less_than(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(input_0.size, input_1.size);
        assert_eq!(output.size, 1);

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_1)?.into();
        let bit: sym::SymbolicBitVec = vec![lhs.signed_less_than(rhs)].into();
        self.memory.write_bytes(vec![bit.zero_extend(7)], &output)?;

        Ok(())
    }

    pub fn popcount(&mut self, input_0: &VarnodeData, output: &VarnodeData) -> Result<()> {
        let value: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let result = value.popcount();
        let num_bits = result.len();
        let requested_bits = 8 * output.size;
        if num_bits > requested_bits {
            // TODO Return error here if popcount bits exceeds number requested
        }

        self.memory.write_bytes(
            result.zero_extend(requested_bits - num_bits).into_parts(8),
            &output,
        )?;

        Ok(())
    }

    pub fn subpiece(
        &mut self,
        input_0: &VarnodeData,
        input_1: &VarnodeData,
        output: &VarnodeData,
    ) -> Result<()> {
        assert_eq!(
            input_1.address.address_space.space_type,
            AddressSpaceType::Constant
        );
        let mut data = self.memory.read_bytes_owned(input_0)?;

        // Remove this number of least significant bytes
        data.drain(..input_1.address.offset);

        // Remove any excess from most significant bytes
        data.drain(output.size..);

        self.memory.write_bytes(data, &output)?;
        Ok(())
    }

    /// This is an indirect branching instruction. The address to branch to is determined
    /// dynamically (at runtime) by examining the contents of the variable input0. As this
    /// instruction is currently defined, the variable input0 only contains the offset of the
    /// destination, and the address space is taken from the address associated with the branching
    /// instruction itself. So execution can only branch within the same address space via this
    /// instruction. The size of the variable input0 must match the size of offsets for the current
    /// address space. P-code relative branching is not possible with BRANCHIND.
    pub fn branch_ind(&mut self, instruction: &PcodeInstruction) -> Result<Address> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, false)?;

        let input_0 = &instruction.inputs[0];

        if input_0.size != instruction.address.address_space.address_size {
            return Err(Error::IllegalInstruction(
                instruction.clone(),
                format!(
                    "Branch target offset size {} does not match address space size {}",
                    input_0.size, instruction.address.address_space.address_size
                ),
            ));
        }

        if instruction.address.address_space.space_type == AddressSpaceType::PcodeOp {
            return Err(Error::IllegalInstruction(
                instruction.clone(),
                format!(
                    "P-code relative branching is not possible with {:?}",
                    instruction.op_code
                ),
            ));
        }

        // TODO Enforce restriction that this is not P-code relative branch

        Ok(Address {
            address_space: instruction.address.address_space.clone(),
            offset: self.memory.read_concrete_value::<usize>(input_0)?,
        })
    }

    pub fn return_instruction(&mut self, instruction: &PcodeInstruction) -> Result<Address> {
        self.branch_ind(instruction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor_address_space() -> AddressSpace {
        AddressSpace {
            id: 0,
            name: "ram".to_string(),
            word_size: 1,
            address_size: 4,
            space_type: AddressSpaceType::Processor,
            big_endian: false,
        }
    }

    fn unique_address_space() -> AddressSpace {
        AddressSpace {
            id: 1,
            name: "unique".to_string(),
            word_size: 1,
            address_size: 8,
            space_type: AddressSpaceType::Internal,
            big_endian: false,
        }
    }

    fn constant_address_space() -> AddressSpace {
        AddressSpace {
            id: 2,
            name: "constant".to_string(),
            word_size: 1,
            address_size: 8,
            space_type: AddressSpaceType::Constant,
            big_endian: false,
        }
    }

    #[test]
    fn copy() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };
        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 4,
            },
            size: 4,
        };
        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::CPUI_COPY,
            inputs: vec![input.clone()],
            output: Some(output.clone()),
        };

        emulator.memory.write_bytes(data, &input)?;
        emulator.copy(&instruction)?;
        emulator.memory.read_bytes(&output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&input)?,
            0xDEADBEEF
        );

        Ok(())
    }

    #[test]
    fn load() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        // Write 0xDEADBEEF to 0x04030201
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0x04030201,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data, &data_input)?;

        // Write 0x04030201 to 0x0. This is the load indirection
        let offset_data = vec![0x01u8.into(), 0x02u8.into(), 0x03u8.into(), 0x04u8.into()];
        let offset_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(offset_data, &offset_input)?;

        // Set the address space input offset to the space id of the processor addr space
        // It is important that the address space of this varnode is the constant space.
        let addr_space_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: processor_address_space().id,
            },
            size: 8, // This value doesn't really matter
        };

        // The output varnode will the location the data is stored at.
        let output = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.load(&addr_space_input, &offset_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&output)?,
            0xDEADBEEF
        );
        Ok(())
    }

    #[test]
    fn store() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        // Write 0xDEADBEEF somewhere. This value will be retrieved and stored to the specified
        // address determined through the space id and offset indirection.
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0xD0D0DADA,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data, &data_input)?;

        // Write 0x04030201 to 0x0. This is the store indirection
        let offset_data = vec![0x01u8.into(), 0x02u8.into(), 0x03u8.into(), 0x04u8.into()];
        let offset_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(offset_data, &offset_input)?;

        // Set the address space input offset to the space id of the processor addr space
        // It is important that the address space of this varnode is the constant space.
        let addr_space_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: processor_address_space().id,
            },
            size: 8, // This value doesn't really matter
        };

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0x04030201, // The data stored at offset_input determines this offset
            },
            size: 4,
        };

        emulator.store(&addr_space_input, &offset_input, &data_input)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&output)?,
            0xDEADBEEF
        );
        Ok(())
    }

    #[test]
    fn int_sub() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let lhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(lhs_data, &lhs_input)?;

        let rhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0x00u8.into(), 0x00u8.into()];
        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        emulator.memory.write_bytes(rhs_data, &rhs_input)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };

        emulator.int_sub(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&output)?,
            0xDEAD0000
        );
        Ok(())
    }

    #[test]
    fn int_add() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let lhs_data = vec![0x00u8.into(), 0x00u8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(lhs_data, &lhs_input)?;

        let rhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0x00u8.into(), 0x00u8.into()];
        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        emulator.memory.write_bytes(rhs_data, &rhs_input)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };

        emulator.int_add(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&output)?,
            0xDEADBEEF
        );
        Ok(())
    }

    #[test]
    fn int_zext() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xFFu8.into()];
        let input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 1,
        };
        emulator.memory.write_bytes(data, &input)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 2,
        };

        emulator.int_zext(&input, &output)?;
        assert_eq!(emulator.memory.read_concrete_value::<u16>(&output)?, 0x00FF);
        Ok(())
    }

    #[test]
    fn int_sext() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0x7Fu8.into(), 0x80u8.into()];
        let data_varnode = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 2,
        };
        emulator.memory.write_bytes(data, &data_varnode)?;

        let input_positive = VarnodeData {
            address: Address {
                address_space: data_varnode.address.address_space.clone(),
                offset: 0,
            },
            size: 1,
        };

        let input_negative = VarnodeData {
            address: Address {
                address_space: data_varnode.address.address_space.clone(),
                offset: 1,
            },
            size: 1,
        };

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 2,
        };

        emulator.int_sext(&input_positive, &output)?;
        assert_eq!(emulator.memory.read_concrete_value::<u16>(&output)?, 0x007F);

        emulator.int_sext(&input_negative, &output)?;
        assert_eq!(emulator.memory.read_concrete_value::<u16>(&output)?, 0xFF80);
        Ok(())
    }

    #[test]
    fn int_equal() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data.clone(), &lhs_input)?;

        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data.clone(), &rhs_input)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 1,
        };

        emulator.int_equal(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            0x1,
            "Expected 0xDEADBEEF == 0xDEADBEEF to be 1"
        );

        emulator.memory.write_bytes(
            vec![0x0u8.into(), 0x0u8.into(), 0x0u8.into(), 0x0u8.into()],
            &rhs_input,
        )?;
        emulator.int_equal(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            0x0,
            "Expected 0xDEADBEEF == 0x0 to be 0"
        );
        Ok(())
    }

    #[test]
    fn int_not_equal() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data.clone(), &lhs_input)?;

        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data.clone(), &rhs_input)?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 1,
        };

        emulator.int_not_equal(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            0x0,
            "Expected 0xDEADBEEF != 0xDEADBEEF to be 0"
        );

        emulator.memory.write_bytes(
            vec![0x0u8.into(), 0x0u8.into(), 0x0u8.into(), 0x0u8.into()],
            &rhs_input,
        )?;
        emulator.int_not_equal(&lhs_input, &rhs_input, &output)?;
        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            0x1,
            "Expected 0xDEADBEEF != 0x0 to be 1"
        );
        Ok(())
    }

    #[test]
    fn subpiece() -> Result<()> {
        let mut emulator =
            PcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        emulator.memory.write_bytes(data, &data_input)?;

        let truncation_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: 2,
            },
            size: 1,
        };

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 2,
        };

        // Expect to truncate 2 least-significant bytes
        emulator.subpiece(&data_input, &truncation_input, &output)?;
        assert_eq!(emulator.memory.read_concrete_value::<u16>(&output)?, 0xDEAD);

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 4,
            },
            size: 1,
        };

        // Expect to truncate 2 least-significant bytes and 1 most significant byte
        // since the output size is less than the input size
        emulator.subpiece(&data_input, &truncation_input, &output)?;
        assert_eq!(emulator.memory.read_concrete_value::<u8>(&output)?, 0xAD);
        Ok(())
    }

    #[test]
    fn branch_ind() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };

        emulator.memory.write_bytes(data, &data_input)?;

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::CPUI_BRANCHIND,
            inputs: vec![data_input.clone()],
            output: None,
        };
        let branch_addr = emulator.branch_ind(&instruction)?;
        let expected_addr = Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        };
        assert_eq!(branch_addr, expected_addr);
        Ok(())
    }

    // CPUI_COPY ✅
    // CPUI_INT_SUB ✅
    // CPUI_STORE ✅
    // CPUI_INT_ADD ✅
    // CPUI_LOAD ✅
    // CPUI_INT_ZEXT ✅
    // CPUI_INT_SEXT ✅
    // CPUI_SUBPIECE ✅
    // CPUI_INT_NOTEQUAL ✅
    // CPUI_RETURN ✅
}
