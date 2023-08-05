use thiserror;

use crate::mem;
use sla::{
    Address, AddressSpace, AddressSpaceType, BoolOp, IntOp, IntSign, LoadImage, OpCode,
    PcodeInstruction, VarnodeData,
};
use sym;

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
                    // SAFETY: This new size MUST be less than the existing input size
                    size: unsafe {
                        (addr.offset - input.address.offset)
                            .try_into()
                            .unwrap_unchecked()
                    },
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
        if has_output
            && instruction
                .output
                .as_ref()
                .unwrap()
                .address
                .address_space
                .is_constant()
        {
            Err(Error::IllegalInstruction(
                instruction.clone(),
                format!(
                    "instruction output address space is constant: {:?}",
                    instruction.output
                ),
            ))
        } else {
            Ok(())
        }
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

fn check_input_sizes_match(instruction: &PcodeInstruction) -> Result<()> {
    check_input_sizes_equal(instruction, (&instruction.inputs[0]).size)
}

fn check_input_sizes_match_output(instruction: &PcodeInstruction) -> Result<()> {
    check_input_sizes_equal(instruction, instruction.output.as_ref().unwrap().size)
}

fn check_input_sizes_equal(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    (0..instruction.inputs.len())
        .map(|i| check_input_size_equals(instruction, i, expected_size))
        .collect()
}

fn check_input_size_equals(
    instruction: &PcodeInstruction,
    input_index: usize,
    expected_size: usize,
) -> Result<()> {
    let input = &instruction.inputs[input_index];
    if input.size != expected_size {
        Err(Error::IllegalInstruction(
            instruction.clone(),
            format!(
                "input[{input_index}] size {} != {expected_size}",
                input.size
            ),
        ))
    } else {
        Ok(())
    }
}

fn check_output_size_equals(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    let output_size = instruction.output.as_ref().unwrap().size;
    if output_size != expected_size {
        return Err(Error::IllegalInstruction(
            instruction.clone(),
            format!("output size {output_size} != {expected_size}"),
        ));
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Destination {
    MachineAddress(Address),
    PcodeAddress(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlow {
    NextInstruction,
    Jump(Destination),
    ConditionalBranch(sym::SymbolicBit, Destination),
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

    pub fn emulate(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        match instruction.op_code {
            OpCode::Copy => self.copy(&instruction)?,
            OpCode::Load => self.load(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Store => self.store(
                &instruction.inputs[0],
                &instruction.inputs[1],
                &instruction.inputs[2],
            )?,
            OpCode::Int(IntOp::Bitwise(BoolOp::And)) => self.int_and(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Or)) => self.int_or(&instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Xor)) => self.int_xor(&instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Negate)) => self.int_negate(&instruction)?,
            OpCode::Int(IntOp::Add) => self.int_add(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Carry(IntSign::Unsigned)) => self.int_carry(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Carry(IntSign::Signed)) => self.int_scarry(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Subtract) => self.int_sub(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Negate) => self.int_2comp(&instruction)?,
            OpCode::Int(IntOp::Borrow) => self.int_sub_borrow(&instruction)?,
            OpCode::Int(IntOp::Multiply) => self.int_multiply(&instruction)?,
            OpCode::Int(IntOp::Divide(IntSign::Unsigned)) => self.int_divide(&instruction)?,
            OpCode::Int(IntOp::Divide(IntSign::Signed)) => self.int_signed_divide(&instruction)?,
            OpCode::Int(IntOp::Remainder(IntSign::Unsigned)) => self.int_remainder(&instruction)?,
            OpCode::Int(IntOp::Remainder(IntSign::Signed)) => {
                self.int_signed_remainder(&instruction)?
            }
            OpCode::Int(IntOp::Extension(IntSign::Unsigned)) => {
                self.int_zext(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::Int(IntOp::Extension(IntSign::Signed)) => {
                self.int_sext(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::Popcount => {
                self.popcount(&instruction.inputs[0], instruction.output.as_ref().unwrap())?
            }
            OpCode::Piece => self.piece(&instruction)?,
            OpCode::Subpiece => self.subpiece(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::Equal) => self.int_equal(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::NotEqual) => self.int_not_equal(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::LessThan(IntSign::Signed)) => self.int_signed_less_than(
                &instruction.inputs[0],
                &instruction.inputs[1],
                instruction.output.as_ref().unwrap(),
            )?,
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)) => {
                self.int_signed_less_than_eq(&instruction)?
            }
            OpCode::Int(IntOp::LessThan(IntSign::Unsigned)) => self.int_less_than(&instruction)?,
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)) => {
                self.int_less_than_eq(&instruction)?
            }
            OpCode::Int(IntOp::ShiftLeft) => self.shift_left(&instruction)?,
            OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)) => self.shift_right(&instruction)?,
            OpCode::Int(IntOp::ShiftRight(IntSign::Signed)) => {
                self.signed_shift_right(&instruction)?
            }
            OpCode::Bool(BoolOp::Negate) => self.bool_negate(&instruction)?,
            OpCode::Bool(BoolOp::And) => self.bool_and(&instruction)?,
            OpCode::Bool(BoolOp::Or) => self.bool_or(&instruction)?,
            OpCode::Bool(BoolOp::Xor) => self.bool_xor(&instruction)?,
            OpCode::Return => return self.return_instruction(&instruction),
            OpCode::BranchIndirect => return self.branch_ind(&instruction),
            OpCode::Branch => return self.branch(&instruction),
            OpCode::BranchConditional => return self.conditional_branch(&instruction),
            OpCode::Call => return self.call(&instruction),
            OpCode::CallIndirect => return self.call_ind(&instruction),
            _ => unimplemented!("Operation not yet implemented: {:?}", instruction.op_code),
        }

        Ok(ControlFlow::NextInstruction)
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

    /// This is an absolute jump instruction. The varnode parameter input0 encodes the destination
    /// address (address space and offset) of the jump. The varnode is not treated as a variable for
    /// this instruction and does not store the destination. Its address space and offset are the
    /// destination. The size of input0 is irrelevant.
    ///
    /// Confusion about the meaning of this instruction can result because of the translation from
    /// machine instructions to p-code. The destination of the jump is a machine address and refers
    /// to the machine instruction at that address. When attempting to determine which p-code
    /// instruction is executed next, the rule is: execute the first p-code instruction resulting
    /// from the translation of the machine instruction(s) at that address. The resulting p-code
    /// instruction may not be attached directly to the indicated address due to NOP instruction
    /// s and delay slots.
    ///
    /// If input0 is constant, i.e. its address space is the constant address space, then it encodes
    /// a p-code relative branch. In this case, the offset of input0 is considered a relative offset
    /// into the indexed list of p-code operations corresponding to the translation of the current
    /// machine instruction. This allows branching within the operations forming a single
    /// instruction. For example, if the BRANCH occurs as the pcode operation with index 5 for the
    /// instruction, it can branch to operation with index 8 by specifying a constant destination
    /// "address" of 3. Negative constants can be used for backward branches.
    fn branch(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, false)?;
        Ok(ControlFlow::Jump(Self::branch_destination(
            &instruction.inputs[0],
        )))
    }

    /// Determine the destination of a branch instruction based on its input address.
    fn branch_destination(destination: &VarnodeData) -> Destination {
        if destination.address.address_space.space_type == AddressSpaceType::Constant {
            // The p-code relative branch is permitted to be negative. The commentary for this
            // function is that the size of the input is irrelevant. I have double checked the
            // Ghidra p-code generator and confirmed that the relative offset is assigned to an
            // unsigned 64-bit value through the subtraction of two unsigned 64-bit values.
            // Therefore any negative offset interpretation would be determined by treating this
            // value as a *signed* 64-bit offset.
            //
            // See: PcodeCacher::resolveRelatives(void) in sleigh.cc
            //
            // For this reason it is absolutely paramount that the offset received from Ghidra be
            // preserved as a 64-bit value and *NOT* converted at any point. Since this value may
            // be either signed or unsigned depending on context, any conversion would necessarily
            // fail to preserve the correct value in all contexts.
            Destination::PcodeAddress(destination.address.offset as i64)
        } else {
            Destination::MachineAddress(destination.address.clone())
        }
    }

    /// This instruction is semantically equivalent to the BRANCH instruction. Beware: This
    /// instruction does not behave like a typical function call. In particular, there is no
    /// internal stack in p-code for saving the return address. Use of this instruction instead of
    /// BRANCH is intended to provide a hint to algorithms that try to follow code flow. It
    /// indicates that the original machine instruction, of which this p-code instruction is only a
    /// part, is intended to be a function call. The p-code instruction does not implement the full
    /// semantics of the call itself; it only implements the final branch.
    ///
    /// In the raw p-code translation process, this operation can only take input0, but in follow-on
    /// analysis, it can take arbitrary additional inputs. These represent (possibly partial)
    /// recovery of the parameters being passed to the logical call represented by this operation.
    /// These additional parameters have no effect on the original semantics of the raw p-code but
    /// naturally hold the varnode values flowing into the call.
    fn call(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        self.branch(instruction)
    }

    /// This instruction is semantically equivalent to the BRANCHIND instruction. It does not
    /// perform a function call in the usual sense of the term. It merely indicates that the
    /// original machine instruction is intended to be an indirect call. See the discussion for the
    /// CALL instruction.
    ///
    /// As with the CALL instruction, this operation may take additional inputs when not in raw
    /// form, representing the parameters being passed to the logical call.
    fn call_ind(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        self.branch_ind(instruction)
    }

    /// This is a conditional branch instruction where the dynamic condition for taking the branch
    /// is determined by the 1 byte variable input1. If this variable is non-zero, the condition is
    /// considered true and the branch is taken. As in the BRANCH instruction the parameter input0
    /// is not treated as a variable but as an address and is interpreted in the same way.
    /// Furthermore, a constant space address is also interpreted as a relative address so that a
    /// CBRANCH can do p-code relative branching. See the discussion for the BRANCH operation.
    fn conditional_branch(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, false)?;
        check_input_size_equals(&instruction, 1, 1)?;

        let selector: sym::SymbolicBit = self
            .memory
            .read_bytes_owned(&instruction.inputs[1])?
            .pop()
            .unwrap()
            .truncate_msb(7)
            .try_into()
            .unwrap();

        Ok(ControlFlow::ConditionalBranch(
            selector,
            Self::branch_destination(&instruction.inputs[0]),
        ))
    }

    /// This operation performs a Logical-And on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
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

    /// This operation performs a Logical-Or on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
    fn int_or(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs = self.memory.read_bytes_owned(&instruction.inputs[0])?;
        let rhs = self.memory.read_bytes_owned(&instruction.inputs[1])?;

        let or = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs | rhs)
            .collect();

        self.memory
            .write_bytes(or, &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This operation performs a logical Exclusive-Or on the bits of input0 and input1. Both
    /// inputs and output must be the same size.
    fn int_xor(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs = self.memory.read_bytes_owned(&instruction.inputs[0])?;
        let rhs = self.memory.read_bytes_owned(&instruction.inputs[1])?;

        let xor = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs ^ rhs)
            .collect();

        self.memory
            .write_bytes(xor, &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This is the twos complement or arithmetic negation operation. Treating input0 as a signed
    /// integer, the result is the same integer value but with the opposite sign. This is equivalent
    /// to doing a bitwise negation of input0 and then adding one. Both input0 and output must be
    /// the same size.
    fn int_2comp(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let negative = -lhs;

        self.memory.write_bytes(
            negative.into_parts(8),
            &instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    /// This is the bitwise negation operation. Output is the result of taking every bit of input0
    /// and flipping it. Both input0 and output must be the same size.
    fn int_negate(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs = self.memory.read_bytes_owned(&instruction.inputs[0])?;

        let negation = lhs.into_iter().map(|value| !value).collect();

        self.memory
            .write_bytes(negation, &instruction.output.as_ref().unwrap())?;

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

    /// This operation checks for signed subtraction overflow or borrow conditions. If the result of
    /// subtracting input1 from input0 as signed integers overflows the size of the varnodes, output
    /// is assigned true. Both inputs must be the same size, and output must be size 1. Note that
    /// the equivalent unsigned subtraction overflow condition is INT_LESS.
    fn int_sub_borrow(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match(&instruction)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let overflow: sym::SymbolicBitVec = vec![lhs.subtraction_with_borrow(rhs).1].into();

        self.memory.write_bytes(
            vec![overflow.zero_extend(7)],
            instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    fn int_multiply(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let output = instruction.output.as_ref().unwrap();

        let product = lhs.multiply(rhs, 8 * output.size);
        self.memory.write_bytes(product.into_parts(8), output)?;

        Ok(())
    }

    /// This is an unsigned integer division operation. Divide input0 by input1, truncating the
    /// result to the nearest integer, and store the result in output. Both inputs and output must
    /// be the same size. There is no handling of division by zero. To simulate a processor's
    /// handling of a division-by-zero trap, other operations must be used before the INT_DIV.
    fn int_divide(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let output = instruction.output.as_ref().unwrap();

        let (quotient, _) = rhs.unsigned_divide(lhs);
        self.memory.write_bytes(quotient.into_parts(8), output)?;

        Ok(())
    }

    /// This is an unsigned integer remainder operation. The remainder of performing the unsigned
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0/input1, using the INT_DIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn int_remainder(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let output = instruction.output.as_ref().unwrap();

        let (_, remainder) = rhs.unsigned_divide(lhs);
        self.memory.write_bytes(remainder.into_parts(8), output)?;

        Ok(())
    }

    /// This is a signed integer division operation. The resulting integer is the one closest to
    /// the rational value input0/input1 but which is still smaller in absolute value. Both inputs
    /// and output must be the same size. There is no handling of division by zero. To simulate a
    /// processor's handling of a division-by-zero trap, other operations must be used before the
    /// INT_SDIV.
    fn int_signed_divide(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let output = instruction.output.as_ref().unwrap();

        let (quotient, _) = rhs.signed_divide(lhs);
        self.memory.write_bytes(quotient.into_parts(8), output)?;

        Ok(())
    }

    /// This is a signed integer remainder operation. The remainder of performing the signed
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0 s/ input1, using the INT_SDIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn int_signed_remainder(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let output = instruction.output.as_ref().unwrap();

        let (_, remainder) = rhs.signed_divide(lhs);
        self.memory.write_bytes(remainder.into_parts(8), output)?;

        Ok(())
    }

    /// Zero-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with zero. The size of output must be strictly bigger than the
    /// size of input.
    pub fn int_zext(&mut self, input_0: &VarnodeData, output: &VarnodeData) -> Result<()> {
        assert!(output.size > input_0.size);
        let data: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let data = data.zero_extend(8 * (output.size - input_0.size) as usize);
        self.memory.write_bytes(data.into_parts(8), &output)?;

        Ok(())
    }

    /// Sign-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with either zero or all ones (0xff) depending on the most
    /// significant bit of input0. The size of output must be strictly bigger than the size of
    /// input0.
    pub fn int_sext(&mut self, input_0: &VarnodeData, output: &VarnodeData) -> Result<()> {
        assert!(output.size > input_0.size);
        let data: sym::SymbolicBitVec = self.memory.read_bytes_owned(input_0)?.into();
        let data = data.sign_extend(8 * (output.size - input_0.size) as usize);
        self.memory.write_bytes(data.into_parts(8), &output)?;

        Ok(())
    }

    /// This is the integer equality operator. Output is assigned true, if input0 equals input1. It
    /// works for signed, unsigned, or any contiguous data where the match must be down to the bit.
    /// Both inputs must be the same size, and the output must have a size of 1.
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

    /// This is the integer inequality operator. Output is assigned true, if input0 does not equal
    /// input1. It works for signed, unsigned, or any contiguous data where the match must be down
    /// to the bit. Both inputs must be the same size, and the output must have a size of 1.
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

    /// This is a signed integer comparison operator. If the signed integer input0 is less than or
    /// equal to the signed integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_signed_less_than_eq(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match(&instruction)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let bit: sym::SymbolicBitVec = vec![lhs.signed_less_than_eq(rhs)].into();
        self.memory.write_bytes(
            vec![bit.zero_extend(7)],
            instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is strictly
    /// less than the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_less_than(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match(&instruction)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let bit: sym::SymbolicBitVec = vec![lhs.less_than(rhs)].into();
        self.memory.write_bytes(
            vec![bit.zero_extend(7)],
            instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is less than
    /// or equal to the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_less_than_eq(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_match(&instruction)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let bit: sym::SymbolicBitVec = vec![lhs.less_than_eq(rhs)].into();
        self.memory.write_bytes(
            vec![bit.zero_extend(7)],
            instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    /// This is a bit count (population count) operator. Within the binary representation of the
    /// value contained in the input varnode, the number of 1 bits are counted and then returned
    /// in the output varnode. A value of 0 returns 0, a 4-byte varnode containing the value
    /// 2<sup>32</sup>-1 (all bits set) returns 32, for instance. The input and output varnodes can
    /// have any size. The resulting count is zero extended into the output varnode.
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

    fn piece(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_output_size_equals(
            &instruction,
            instruction.inputs[0].size + instruction.inputs[1].size,
        )?;

        let msb: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let lsb: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();

        self.memory.write_bytes(
            lsb.concat(msb).into_parts(8),
            instruction.output.as_ref().unwrap(),
        )?;

        Ok(())
    }

    /// This is a truncation operator that understands the endianess of the data. Input1 indicates
    /// the number of least significant bytes of input0 to be thrown away. Output is then filled
    /// with any remaining bytes of input0 up to the size of output. If the size of output is
    /// smaller than the size of input0 minus the constant input1, then the additional most
    /// significant bytes of input0 will also be truncated.
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

        // Remove this number of least significant bytes. If for some reason the offset exceeds
        // the maximum usize value, then by definition all of the data would be drained anyway.
        data.drain(..input_1.address.offset.try_into().unwrap_or(usize::MAX));

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
    pub fn branch_ind(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, false)?;

        let input_0 = &instruction.inputs[0];

        if input_0.size != instruction.address.address_space.address_size {
            return Err(Error::IllegalInstruction(
                instruction.clone(),
                format!(
                    "Branch target offset size {target_size} does not match address space size {space_size}",
                    target_size=input_0.size, space_size=instruction.address.address_space.address_size
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

        Ok(ControlFlow::Jump(Destination::MachineAddress(Address {
            address_space: instruction.address.address_space.clone(),
            offset: self.memory.read_concrete_value::<u64>(input_0)?,
        })))
    }

    /// This instruction is semantically equivalent to the BRANCHIND instruction. It does not
    /// perform a return from subroutine in the usual sense of the term. It merely indicates that
    /// the original machine instruction is intended to be a return from subroutine. See the
    /// discussion for the CALL instruction.
    ///
    /// Similarly to CALL and CALLIND, this operation may take an additional input when not in raw
    /// form. If input1 is present it represents the value being returned by this operation. This is
    /// used by analysis algorithms to hold the value logically flowing back to the parent
    /// subroutine.
    pub fn return_instruction(&mut self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        self.branch_ind(instruction)
    }

    /// This is a logical negate operator, where we assume input0 and output are boolean values. It
    /// puts the logical complement of input0, treated as a single bit, into output. Both input0 and
    /// output are size 1. Boolean values are implemented with a full byte, but are still considered
    /// to only support a value of true or false.
    fn bool_negate(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 1)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_equal(&instruction, 1)?;
        check_output_size_equals(&instruction, 1)?;

        let input = self
            .memory
            .read_bytes_owned(&instruction.inputs[0])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let negation = !input;
        let negation = negation.zero_extend(7);

        self.memory
            .write_bytes(vec![negation], &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This is an Exclusive-Or operator, where we assume the inputs and output are boolean values.
    /// It puts the exclusive-or of input0 and input1, treated as single bits, into output. Both
    /// inputs and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_xor(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_equal(&instruction, 1)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[0])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let rhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[1])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let xor = lhs ^ rhs;
        let xor = xor.zero_extend(7);

        self.memory
            .write_bytes(vec![xor], &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This is a Logical-And operator, where we assume the inputs and output are boolean values. It
    /// puts the logical-and of input0 and input1, treated as single bits, into output. Both inputs
    /// and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_and(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_equal(&instruction, 1)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[0])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let rhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[1])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let and = lhs & rhs;
        let and = and.zero_extend(7);

        self.memory
            .write_bytes(vec![and], &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This is a Logical-Or operator, where we assume the inputs and output are boolean values. It
    /// puts the logical-or of input0 and input1, treated as single bits, into output. Both inputs
    /// and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_or(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_input_sizes_equal(&instruction, 1)?;
        check_output_size_equals(&instruction, 1)?;

        let lhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[0])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let rhs = self
            .memory
            .read_bytes_owned(&instruction.inputs[1])?
            .pop()
            .unwrap()
            .truncate_msb(7);

        let or = lhs | rhs;
        let or = or.zero_extend(7);

        self.memory
            .write_bytes(vec![or], &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This operation performs a left shift on input0. The value given by input1, interpreted as an
    /// unsigned integer, indicates the number of bits to shift. The vacated (least significant)
    /// bits are filled with zero. If input1 is zero, no shift is performed and input0 is copied
    /// into output. If input1 is larger than the number of bits in output, the result is zero. Both
    /// input0 and output must be the same size. Input1 can be any size.
    fn shift_left(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let result = (lhs << rhs).into_parts(8);

        self.memory
            .write_bytes(result, &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This operation performs an unsigned (logical) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated (most significant) bits are filled with zero. If input1 is zero, no shift is
    /// performed and input0 is copied into output. If input1 is larger than the number of bits in
    /// output, the result is zero. Both input0 and output must be the same size. Input1 can be any
    /// size.
    fn shift_right(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let result = (lhs >> rhs).into_parts(8);

        self.memory
            .write_bytes(result, &instruction.output.as_ref().unwrap())?;

        Ok(())
    }

    /// This operation performs a signed (arithmetic) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated bits are filled with the original value of the most significant (sign) bit of
    /// input0. If input1 is zero, no shift is performed and input0 is copied into output. If input1
    /// is larger than the number of bits in output, the result is zero or all 1-bits (-1),
    /// depending on the original sign of input0. Both input0 and output must be the same size.
    /// Input1 can be any size.
    fn signed_shift_right(&mut self, instruction: &PcodeInstruction) -> Result<()> {
        check_num_inputs(&instruction, 2)?;
        check_has_output(&instruction, true)?;
        check_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[0])?.into();
        let rhs: sym::SymbolicBitVec = self.memory.read_bytes_owned(&instruction.inputs[1])?.into();
        let result = lhs.signed_shift_right(rhs).into_parts(8);

        self.memory
            .write_bytes(result, &instruction.output.as_ref().unwrap())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sym::SymbolicBitVec;

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

    fn write_bytes(
        emulator: &mut PcodeEmulator,
        offset: u64,
        bytes: Vec<SymbolicBitVec>,
    ) -> Result<VarnodeData> {
        let varnode = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset,
            },
            size: bytes.len(),
        };

        emulator.memory_mut().write_bytes(bytes, &varnode)?;
        Ok(varnode)
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
            op_code: OpCode::Copy,
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
                offset: processor_address_space().id as u64,
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
                offset: processor_address_space().id as u64,
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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
                op_code: OpCode::Int(IntOp::Borrow),
                inputs: vec![lhs_input.clone(), rhs_input.clone()],
                output: Some(output.clone()),
            };

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed borrow of {lhs} - {rhs}"
            );
        }

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
    fn int_multiply() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 0..16u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value: SymbolicBitVec = lhs.into();
                let lhs_value = lhs_value.zero_extend(4);
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs_value])?;

                let rhs_value: SymbolicBitVec = rhs.into();
                let rhs_value = rhs_value.zero_extend(4);
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs_value])?;

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
                    op_code: OpCode::Int(IntOp::Multiply),
                    inputs: vec![lhs_input.clone(), rhs_input.clone()],
                    output: Some(output.clone()),
                };

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs * rhs,
                    "failed {lhs} * {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_multiply_multibyte() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs: u8 = 0xFF;
        let lhs_value: SymbolicBitVec = lhs.into();
        let lhs_value = lhs_value.zero_extend(8);
        let lhs_input = write_bytes(&mut emulator, 0, lhs_value.into_parts(8))?;

        let rhs: u8 = 0x80;
        let rhs_value: SymbolicBitVec = rhs.into();
        let rhs_value = rhs_value.zero_extend(8);
        let rhs_input = write_bytes(&mut emulator, 1, rhs_value.into_parts(8))?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 2,
            },
            size: 2,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::Multiply),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u16>(&output)?,
            lhs as u16 * rhs as u16,
            "failed {lhs} * {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_divide() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 1..16u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value: SymbolicBitVec = lhs.into();
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs_value])?;

                let rhs_value: SymbolicBitVec = rhs.into();
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs_value])?;

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
                    // This will compute LHS / RHS
                    op_code: OpCode::Int(IntOp::Divide(IntSign::Unsigned)),
                    inputs: vec![lhs_input.clone(), rhs_input.clone()],
                    output: Some(output.clone()),
                };

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs / rhs,
                    "failed {lhs} / {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_remainder() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 1..16u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value: SymbolicBitVec = lhs.into();
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs_value])?;

                let rhs_value: SymbolicBitVec = rhs.into();
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs_value])?;

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
                    // This will compute LHS / RHS
                    op_code: OpCode::Int(IntOp::Remainder(IntSign::Unsigned)),
                    inputs: vec![lhs_input.clone(), rhs_input.clone()],
                    output: Some(output.clone()),
                };

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs % rhs,
                    "failed {lhs} % {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_signed_divide() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 1..16u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = SymbolicBitVec::constant(lhs.into(), 4).sign_extend(4);
                let lhs: u8 = lhs_value.clone().try_into().unwrap();
                let lhs = lhs as i8;
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs_value])?;

                let rhs_value = SymbolicBitVec::constant(rhs.into(), 4).sign_extend(4);
                let rhs: u8 = rhs_value.clone().try_into().unwrap();
                let rhs = rhs as i8;
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs_value])?;

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
                    // This will compute LHS / RHS
                    op_code: OpCode::Int(IntOp::Divide(IntSign::Signed)),
                    inputs: vec![lhs_input.clone(), rhs_input.clone()],
                    output: Some(output.clone()),
                };

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)? as i8,
                    lhs / rhs,
                    "failed signed {lhs} / {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_signed_remainder() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 1..16u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = SymbolicBitVec::constant(lhs.into(), 4).sign_extend(4);
                let lhs: u8 = lhs_value.clone().try_into().unwrap();
                let lhs = lhs as i8;
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs_value])?;

                let rhs_value = SymbolicBitVec::constant(rhs.into(), 4).sign_extend(4);
                let rhs: u8 = rhs_value.clone().try_into().unwrap();
                let rhs = rhs as i8;
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs_value])?;

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

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)? as i8,
                    lhs % rhs,
                    "failed signed {lhs} % {rhs}"
                );
            }
        }

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
    fn piece() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let msb_input = write_bytes(&mut emulator, 0, vec![0xADu8.into(), 0xDEu8.into()])?;
        let lsb_input = write_bytes(&mut emulator, 2, vec![0xEFu8.into(), 0xBEu8.into()])?;

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
            op_code: OpCode::Piece,
            inputs: vec![msb_input.clone(), lsb_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u32>(&output)?,
            0xDEADBEEF
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
            op_code: OpCode::BranchIndirect,
            inputs: vec![data_input.clone()],
            output: None,
        };
        let branch_addr = emulator.emulate(&instruction)?;
        let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        }));
        assert_eq!(branch_addr, expected_addr);
        Ok(())
    }

    #[test]
    fn call_ind() -> Result<()> {
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
            op_code: OpCode::CallIndirect,
            inputs: vec![data_input.clone()],
            output: None,
        };
        let branch_addr = emulator.emulate(&instruction)?;
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let input = write_bytes(&mut emulator, 0, vec![value.into()])?;

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

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                (!value) & 0x1,
                "failed !{value}"
            );
        }

        Ok(())
    }

    #[test]
    fn bool_and() -> Result<()> {
        for lhs in 0..=1u8 {
            for rhs in 0..=1u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs & rhs,
                    "failed {lhs} & {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn bool_or() -> Result<()> {
        for lhs in 0..=1u8 {
            for rhs in 0..=1u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs | rhs,
                    "failed {lhs} | {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn bool_xor() -> Result<()> {
        for lhs in 0..=1u8 {
            for rhs in 0..=1u8 {
                let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

                emulator.emulate(&instruction)?;

                assert_eq!(
                    emulator.memory.read_concrete_value::<u8>(&output)?,
                    lhs ^ rhs,
                    "failed {lhs} ^ {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_negate() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;

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

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            !lhs,
            "failed !{lhs}"
        );

        Ok(())
    }

    #[test]
    fn int_2comp() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 1u8;
        let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;

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

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            0xFF, // Negative 1
            "failed -{lhs}"
        );

        Ok(())
    }

    #[test]
    fn int_and() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
        let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            lhs & rhs,
            "failed {lhs} & {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_or() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
        let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            lhs | rhs,
            "failed {lhs} | {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_xor() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
        let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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
            op_code: OpCode::Int(IntOp::Bitwise(BoolOp::Xor)),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&instruction)?;

        assert_eq!(
            emulator.memory.read_concrete_value::<u8>(&output)?,
            lhs ^ rhs,
            "failed {lhs} ^ {rhs}"
        );

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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed {lhs} < {rhs}"
            );
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed {lhs} <= {rhs}"
            );
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![rhs.into()])?;

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

            emulator.emulate(&instruction)?;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed signed comparison {lhs} <= {rhs}"
            );
        }

        Ok(())
    }

    #[test]
    fn shift_left() -> Result<()> {
        for n in 0..=8u8 {
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![0x01u8.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![n.into()])?;

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

            emulator.emulate(&instruction)?;
            let expected_result = if n < 8 { 1 << n } else { 0 };

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed 1 << {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn shift_right() -> Result<()> {
        for n in 0..=8u8 {
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![0x80u8.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![n.into()])?;

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

            emulator.emulate(&instruction)?;
            let expected_result = if n < 8 { 0x80 >> n } else { 0 };

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed 0x80 >> {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn signed_shift_right() -> Result<()> {
        for n in 0..=8u8 {
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![0x80u8.into()])?;
            let rhs_input = write_bytes(&mut emulator, 1, vec![n.into()])?;

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

            emulator.emulate(&instruction)?;
            let expected_result = if n < 8 { (-128i8 >> n) as u8 } else { 0xFF };

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed signed shift 0x80 >> {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn call() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
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

        let branch_addr = emulator.emulate(&instruction)?;
        let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        }));

        assert_eq!(branch_addr, expected_addr);
        Ok(())
    }

    #[test]
    fn branch_absolute() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
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

        let branch_addr = emulator.emulate(&instruction)?;
        let expected_addr = ControlFlow::Jump(Destination::MachineAddress(Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        }));

        assert_eq!(branch_addr, expected_addr);
        Ok(())
    }

    #[test]
    fn branch_pcode_relative() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
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

        let branch_addr = emulator.emulate(&instruction)?;
        let expected_addr = ControlFlow::Jump(Destination::PcodeAddress(-1));

        assert_eq!(branch_addr, expected_addr);
        Ok(())
    }

    #[test]
    fn conditional_branch_absolute() -> Result<()> {
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xDEADBEEF,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_bytes(&mut emulator, 1, vec![0x1u8.into()])?;
        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::BranchConditional,
            inputs: vec![destination_input.clone(), condition_input.clone()],
            output: None,
        };

        let control_flow = emulator.emulate(&instruction)?;
        let expected_destination = Destination::MachineAddress(Address {
            address_space: processor_address_space(),
            offset: 0xDEADBEEF,
        });
        match control_flow {
            ControlFlow::ConditionalBranch(condition, destination) => {
                assert_eq!(condition, sym::SymbolicBit::Literal(true));
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
        let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: u64::MAX,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_bytes(&mut emulator, 1, vec![0x1u8.into()])?;
        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::BranchConditional,
            inputs: vec![destination_input.clone(), condition_input.clone()],
            output: None,
        };

        let control_flow = emulator.emulate(&instruction)?;
        let expected_destination = Destination::PcodeAddress(-1);
        match control_flow {
            ControlFlow::ConditionalBranch(condition, destination) => {
                assert_eq!(condition, sym::SymbolicBit::Literal(true));
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
            let mut emulator = PcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut emulator, 0, vec![value.into()])?;

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

            emulator.emulate(&instruction)?;
            let expected_result = n;

            assert_eq!(
                emulator.memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed popcount of {value:#02x}"
            );
        }

        Ok(())
    }
}
