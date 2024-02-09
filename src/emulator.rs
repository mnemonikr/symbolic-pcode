use sla::{
    Address, AddressSpace, AddressSpaceType, BoolOp, IntOp, IntSign, OpCode, PcodeInstruction,
    VarnodeData,
};
use sym::SymbolicByte;
use thiserror;

use crate::mem::{self, Memory, SymbolicMemory};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Error occurred while accessing a memory location
    #[error(transparent)]
    MemoryAccess(#[from] mem::Error),

    /// The provided instruction violates some invariant. An example of this could be missing an
    /// output varnode for an instruction that requires an output.
    #[error("illegal instruction {instruction:?}: {reason}")]
    IllegalInstruction {
        instruction: PcodeInstruction,
        reason: String,
    },

    /// Emulation of this instruction is not implemented
    #[error("unsupported instruction {instruction:?}")]
    UnsupportedInstruction { instruction: PcodeInstruction },

    #[error("symbolic address loaded from {varnode} in instruction {instruction:?}")]
    SymbolicAddress {
        instruction: PcodeInstruction,
        varnode: VarnodeData,
        address: Vec<SymbolicByte>,
    },

    #[error("unknown address space id {space_id} referenced by {varnode} in instruction: {instruction:?}")]
    UnknownAddressSpace {
        instruction: PcodeInstruction,
        varnode: VarnodeData,
        space_id: usize,
    },

    #[error("internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The pcode emulator structure that holds the necessary data for emulation.
pub struct StandardPcodeEmulator {
    address_spaces_by_id: std::collections::BTreeMap<usize, AddressSpace>,
}

/// Destination of a control flow instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Destination {
    /// An address in machine memory
    MachineAddress(Address),

    /// A pcode instruction offset relative to the currently executing pcode instruction. The
    /// offset may be negative.
    PcodeAddress(i64),
}

/// Describes which instruction should be executed next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ControlFlow {
    /// The next pcode instruction. This could be another pcode instruction translated from the
    /// same machine instruction or, if this is the last pcode instruction, then the first pcode
    /// instruction of the next machine instruction.
    #[default]
    NextInstruction,

    /// Execution should continue at the provided destination
    Jump(Destination),

    /// Execution should continue at the provided destination if the condition evaluates to true.
    /// Otherwise execution should continue with the next instruction.
    ConditionalBranch(sym::SymbolicBit, Destination),
}

pub trait PcodeEmulator {
    fn emulate(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow>;
}

impl PcodeEmulator for StandardPcodeEmulator {
    fn emulate(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        match instruction.op_code {
            OpCode::Copy => self.copy(memory, &instruction)?,
            OpCode::Load => self.load(memory, &instruction)?,
            OpCode::Store => self.store(memory, &instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::And)) => self.int_and(memory, &instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Or)) => self.int_or(memory, &instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Xor)) => self.int_xor(memory, &instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::Negate)) => self.int_negate(memory, &instruction)?,
            OpCode::Int(IntOp::Add) => self.int_add(memory, &instruction)?,
            OpCode::Int(IntOp::Carry(IntSign::Unsigned)) => self.int_carry(memory, &instruction)?,
            OpCode::Int(IntOp::Carry(IntSign::Signed)) => self.int_scarry(memory, &instruction)?,
            OpCode::Int(IntOp::Subtract) => self.int_sub(memory, &instruction)?,
            OpCode::Int(IntOp::Negate) => self.int_2comp(memory, &instruction)?,
            OpCode::Int(IntOp::Borrow) => self.int_sub_borrow(memory, &instruction)?,
            OpCode::Int(IntOp::Multiply) => self.int_multiply(memory, &instruction)?,
            OpCode::Int(IntOp::Divide(IntSign::Unsigned)) => {
                self.int_divide(memory, &instruction)?
            }
            OpCode::Int(IntOp::Divide(IntSign::Signed)) => {
                self.int_signed_divide(memory, &instruction)?
            }
            OpCode::Int(IntOp::Remainder(IntSign::Unsigned)) => {
                self.int_remainder(memory, &instruction)?
            }
            OpCode::Int(IntOp::Remainder(IntSign::Signed)) => {
                self.int_signed_remainder(memory, &instruction)?
            }
            OpCode::Int(IntOp::Extension(IntSign::Unsigned)) => {
                self.int_zext(memory, &instruction)?
            }
            OpCode::Int(IntOp::Extension(IntSign::Signed)) => {
                self.int_sext(memory, &instruction)?
            }
            OpCode::Popcount => self.popcount(memory, &instruction)?,
            OpCode::Piece => self.piece(memory, &instruction)?,
            OpCode::Subpiece => self.subpiece(memory, &instruction)?,
            OpCode::Int(IntOp::Equal) => self.int_equal(memory, &instruction)?,
            OpCode::Int(IntOp::NotEqual) => self.int_not_equal(memory, &instruction)?,
            OpCode::Int(IntOp::LessThan(IntSign::Signed)) => {
                self.int_signed_less_than(memory, &instruction)?
            }
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)) => {
                self.int_signed_less_than_eq(memory, &instruction)?
            }
            OpCode::Int(IntOp::LessThan(IntSign::Unsigned)) => {
                self.int_less_than(memory, &instruction)?
            }
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)) => {
                self.int_less_than_eq(memory, &instruction)?
            }
            OpCode::Int(IntOp::ShiftLeft) => self.shift_left(memory, &instruction)?,
            OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)) => {
                self.shift_right(memory, &instruction)?
            }
            OpCode::Int(IntOp::ShiftRight(IntSign::Signed)) => {
                self.signed_shift_right(memory, &instruction)?
            }
            OpCode::Bool(BoolOp::Negate) => self.bool_negate(memory, &instruction)?,
            OpCode::Bool(BoolOp::And) => self.bool_and(memory, &instruction)?,
            OpCode::Bool(BoolOp::Or) => self.bool_or(memory, &instruction)?,
            OpCode::Bool(BoolOp::Xor) => self.bool_xor(memory, &instruction)?,
            OpCode::Return => return self.return_instruction(memory, &instruction),
            OpCode::BranchIndirect => return self.branch_ind(memory, &instruction),
            OpCode::Branch => return self.branch(memory, &instruction),
            OpCode::BranchConditional => return self.conditional_branch(memory, &instruction),
            OpCode::Call => return self.call(memory, &instruction),
            OpCode::CallIndirect => return self.call_ind(memory, &instruction),
            _ => {
                return Err(Error::UnsupportedInstruction {
                    instruction: instruction.clone(),
                })
            }
        }

        Ok(ControlFlow::NextInstruction)
    }
}

impl StandardPcodeEmulator {
    pub fn new(address_spaces: impl IntoIterator<Item = AddressSpace>) -> Self {
        Self {
            address_spaces_by_id: address_spaces
                .into_iter()
                .map(|space| (space.id, space))
                .collect(),
        }
    }

    fn write_bitvec(
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
        result: sym::SymbolicBitVec,
    ) -> Result<()> {
        let result: Vec<SymbolicByte> = result.try_into().map_err(|err| {
            Error::InternalError(format!("cannot convert result to bytes: {err}"))
        })?;
        Self::write_bytes(memory, instruction, result)
    }

    fn write_bytes(
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
        result: Vec<SymbolicByte>,
    ) -> Result<()> {
        memory.write(&instruction.output.as_ref().unwrap(), result.into_iter())?;
        Ok(())
    }

    /// Copy a sequence of contiguous bytes from anywhere to anywhere. Size of input0 and output
    /// must be the same.
    fn copy(&self, memory: &mut impl SymbolicMemory, instruction: &PcodeInstruction) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let input = &instruction.inputs[0];
        let data = memory.read(input)?.into_iter().collect();

        Self::write_bytes(memory, instruction, data)?;

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
    ///
    /// It is possible for the addressable unit of an address space to be bigger than a single byte.
    /// If the wordsize attribute of the space given by the ID is bigger than one, the offset into
    /// the space obtained from input1 must be multiplied by this value in order to obtain the
    /// correct byte offset into the space.
    fn load(&self, memory: &mut impl SymbolicMemory, instruction: &PcodeInstruction) -> Result<()> {
        require_num_inputs(instruction, 2)?;
        require_has_output(instruction, true)?;

        let output = instruction.output.as_ref().unwrap();
        let input = VarnodeData {
            address: self.indirect_address(memory, instruction)?,
            size: output.size,
        };

        let data = memory.read(&input)?.into_iter().collect();
        Self::write_bytes(memory, instruction, data)?;

        Ok(())
    }

    /// This instruction is the complement of LOAD. The data in the variable input2 is stored at a
    /// dynamic location by dereferencing a pointer. As with LOAD, the “pointer” comes in two
    /// pieces: a space ID part, and an offset variable. The size of input1 must match the address
    /// space specified by the ID, and the amount of data stored is determined by the size of
    /// input2.
    ///
    /// Its possible for the addressable unit of an address space to be bigger than a single byte.
    /// If the wordsize attribute of the space given by the ID is bigger than one, the offset into
    /// the space obtained from input1 must be multiplied by this value in order to obtain the
    /// correct byte offset into the space.
    fn store(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 3)?;
        require_has_output(instruction, false)?;

        let input = &instruction.inputs[2];
        let output = VarnodeData {
            address: self.indirect_address(memory, instruction)?,
            size: input.size,
        };

        let data = memory.read(&input)?;
        memory.write(&output, data.into_iter())?;

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
    /// instruction may not be attached directly to the indicated address due to NOP instructions
    /// and delay slots.
    ///
    /// If input0 is constant, i.e. its address space is the constant address space, then it encodes
    /// a p-code relative branch. In this case, the offset of input0 is considered a relative offset
    /// into the indexed list of p-code operations corresponding to the translation of the current
    /// machine instruction. This allows branching within the operations forming a single
    /// instruction. For example, if the BRANCH occurs as the pcode operation with index 5 for the
    /// instruction, it can branch to operation with index 8 by specifying a constant destination
    /// "address" of 3. Negative constants can be used for backward branches.
    fn branch(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, false)?;
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
    fn call(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        self.branch(memory, instruction)
    }

    /// This instruction is semantically equivalent to the BRANCHIND instruction. It does not
    /// perform a function call in the usual sense of the term. It merely indicates that the
    /// original machine instruction is intended to be an indirect call. See the discussion for the
    /// CALL instruction.
    ///
    /// As with the CALL instruction, this operation may take additional inputs when not in raw
    /// form, representing the parameters being passed to the logical call.
    fn call_ind(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        self.branch_ind(memory, instruction)
    }

    /// This is a conditional branch instruction where the dynamic condition for taking the branch
    /// is determined by the 1 byte variable input1. If this variable is non-zero, the condition is
    /// considered true and the branch is taken. As in the BRANCH instruction the parameter input0
    /// is not treated as a variable but as an address and is interpreted in the same way.
    /// Furthermore, a constant space address is also interpreted as a relative address so that a
    /// CBRANCH can do p-code relative branching. See the discussion for the BRANCH operation.
    fn conditional_branch(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, false)?;
        require_input_size_equals(&instruction, 1, 1)?;

        let selector: sym::SymbolicBit = memory.read_bit(&instruction.inputs[1])?;

        Ok(ControlFlow::ConditionalBranch(
            selector,
            Self::branch_destination(&instruction.inputs[0]),
        ))
    }

    /// This operation performs a Logical-And on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
    fn int_and(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let and = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs & rhs)
            .collect();

        Self::write_bytes(memory, instruction, and)?;

        Ok(())
    }

    /// This operation performs a Logical-Or on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
    fn int_or(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let or = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs | rhs)
            .collect();

        Self::write_bytes(memory, instruction, or)?;

        Ok(())
    }

    /// This operation performs a logical Exclusive-Or on the bits of input0 and input1. Both
    /// inputs and output must be the same size.
    fn int_xor(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: Vec<SymbolicByte> = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let xor = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs ^ rhs)
            .collect();

        Self::write_bytes(memory, instruction, xor)?;

        Ok(())
    }

    /// This is the twos complement or arithmetic negation operation. Treating input0 as a signed
    /// integer, the result is the same integer value but with the opposite sign. This is equivalent
    /// to doing a bitwise negation of input0 and then adding one. Both input0 and output must be
    /// the same size.
    fn int_2comp(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let negative = -lhs;

        Self::write_bitvec(memory, instruction, negative)?;

        Ok(())
    }

    /// This is the bitwise negation operation. Output is the result of taking every bit of input0
    /// and flipping it. Both input0 and output must be the same size.
    fn int_negate(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs = memory.read(&instruction.inputs[0])?;
        let negation = lhs.into_iter().map(|value| !value).collect();

        Self::write_bytes(memory, instruction, negation)?;

        Ok(())
    }

    /// This is standard integer addition. It works for either unsigned or signed interpretations
    /// of the integer encoding (twos complement). Size of both inputs and output must be the same.
    /// The addition is of course performed modulo this size. Overflow and carry conditions are
    /// calculated by other operations. See INT_CARRY and INT_SCARRY.
    fn int_add(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let sum = lhs + rhs;
        Self::write_bitvec(memory, instruction, sum)?;

        Ok(())
    }

    /// This operation checks for unsigned addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as unsigned integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    fn int_carry(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;
        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let overflow = lhs.unsigned_addition_overflow(rhs);
        Self::write_bytes(memory, instruction, vec![overflow.into()])?;

        Ok(())
    }

    /// This operation checks for signed addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as signed integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    fn int_scarry(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;
        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let overflow = lhs.signed_addition_overflow(rhs);

        Self::write_bytes(memory, instruction, vec![overflow.into()])?;

        Ok(())
    }

    ///  This is standard integer subtraction. It works for either unsigned or signed
    ///  interpretations of the integer encoding (twos complement). Size of both inputs and output
    ///  must be the same. The subtraction is of course performed modulo this size. Overflow and
    ///  borrow conditions are calculated by other operations. See INT_SBORROW and INT_LESS.
    fn int_sub(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let diff = lhs - rhs;
        Self::write_bitvec(memory, instruction, diff)?;

        Ok(())
    }

    /// This operation checks for signed subtraction overflow or borrow conditions. If the result of
    /// subtracting input1 from input0 as signed integers overflows the size of the varnodes, output
    /// is assigned true. Both inputs must be the same size, and output must be size 1. Note that
    /// the equivalent unsigned subtraction overflow condition is INT_LESS.
    fn int_sub_borrow(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let overflow = lhs.subtraction_with_borrow(rhs).1;

        Self::write_bytes(memory, instruction, vec![overflow.into()])?;

        Ok(())
    }

    fn int_multiply(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let output = instruction.output.as_ref().unwrap();

        let product = lhs.multiply(rhs, 8 * output.size);
        Self::write_bitvec(memory, instruction, product)?;

        Ok(())
    }

    /// This is an unsigned integer division operation. Divide input0 by input1, truncating the
    /// result to the nearest integer, and store the result in output. Both inputs and output must
    /// be the same size. There is no handling of division by zero. To simulate a processor's
    /// handling of a division-by-zero trap, other operations must be used before the INT_DIV.
    fn int_divide(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let (quotient, _) = rhs.unsigned_divide(lhs);
        Self::write_bitvec(memory, instruction, quotient)?;

        Ok(())
    }

    /// This is an unsigned integer remainder operation. The remainder of performing the unsigned
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0/input1, using the INT_DIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn int_remainder(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let (_, remainder) = rhs.unsigned_divide(lhs);
        Self::write_bitvec(memory, instruction, remainder)?;

        Ok(())
    }

    /// This is a signed integer division operation. The resulting integer is the one closest to
    /// the rational value input0/input1 but which is still smaller in absolute value. Both inputs
    /// and output must be the same size. There is no handling of division by zero. To simulate a
    /// processor's handling of a division-by-zero trap, other operations must be used before the
    /// INT_SDIV.
    fn int_signed_divide(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let (quotient, _) = rhs.signed_divide(lhs);
        Self::write_bitvec(memory, instruction, quotient)?;

        Ok(())
    }

    /// This is a signed integer remainder operation. The remainder of performing the signed
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0 s/ input1, using the INT_SDIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn int_signed_remainder(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match_output(&instruction)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        let (_, remainder) = rhs.signed_divide(lhs);
        Self::write_bitvec(memory, instruction, remainder)?;

        Ok(())
    }

    /// Zero-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with zero. The size of output must be strictly bigger than the
    /// size of input.
    fn int_zext(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        let input = &instruction.inputs[0];
        require_output_size_exceeds(&instruction, input.size)?;
        let output = instruction.output.as_ref().unwrap();

        let data: sym::SymbolicBitVec = memory.read(&input)?.into_iter().collect();
        let data = data.zero_extend(8 * (output.size - input.size) as usize);
        Self::write_bitvec(memory, instruction, data)?;

        Ok(())
    }

    /// Sign-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with either zero or all ones (0xff) depending on the most
    /// significant bit of input0. The size of output must be strictly bigger than the size of
    /// input0.
    fn int_sext(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        let input = &instruction.inputs[0];
        require_output_size_exceeds(&instruction, input.size)?;
        let output = instruction.output.as_ref().unwrap();

        let data: sym::SymbolicBitVec = memory.read(&input)?.into_iter().collect();
        let data = data.sign_extend(8 * (output.size - input.size) as usize);
        Self::write_bitvec(memory, instruction, data)?;

        Ok(())
    }

    /// This is the integer equality operator. Output is assigned true, if input0 equals input1. It
    /// works for signed, unsigned, or any contiguous data where the match must be down to the bit.
    /// Both inputs must be the same size, and the output must have a size of 1.
    fn int_equal(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = lhs.equals(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is the integer inequality operator. Output is assigned true, if input0 does not equal
    /// input1. It works for signed, unsigned, or any contiguous data where the match must be down
    /// to the bit. Both inputs must be the same size, and the output must have a size of 1.
    fn int_not_equal(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = !lhs.equals(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is a signed integer comparison operator. If the signed integer input0 is strictly less
    /// than the signed integer input1, output is set to true. Both inputs must be the same size,
    /// and the output must have a size of 1.
    fn int_signed_less_than(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = lhs.signed_less_than(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is a signed integer comparison operator. If the signed integer input0 is less than or
    /// equal to the signed integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_signed_less_than_eq(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = lhs.signed_less_than_eq(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is strictly
    /// less than the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_less_than(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = lhs.less_than(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is less than
    /// or equal to the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn int_less_than_eq(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_match(&instruction)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let bit = lhs.less_than_eq(rhs);
        Self::write_bytes(memory, instruction, vec![bit.into()])?;

        Ok(())
    }

    /// This is a bit count (population count) operator. Within the binary representation of the
    /// value contained in the input varnode, the number of 1 bits are counted and then returned
    /// in the output varnode. A value of 0 returns 0, a 4-byte varnode containing the value
    /// 2<sup>32</sup>-1 (all bits set) returns 32, for instance. The input and output varnodes can
    /// have any size. The resulting count is zero extended into the output varnode.
    fn popcount(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        let value: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let result = value.popcount();
        let num_bits = result.len();

        // The input and output varnodes can have any size. It is not documented what should occur
        // if the resulting popcount value exceeds the varnode output size. Since the behavior is
        // undocumented assuming this should never occur and returning an error.
        require_output_size_at_least(instruction, num_bits / 8)?;

        // The resulting count is zero extended into the output varnode.
        let output = instruction.output.as_ref().unwrap();
        Self::write_bitvec(
            memory,
            instruction,
            result.zero_extend(8 * output.size - num_bits),
        )?;

        Ok(())
    }

    /// This is a concatenation operator that understands the endianess of the data. The size of
    /// input0 and input1 must add up to the size of output. The data from the inputs is
    /// concatenated in such a way that, if the inputs and output are considered integers, the first
    /// input makes up the most significant part of the output.
    fn piece(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_output_size_equals(
            &instruction,
            instruction.inputs[0].size + instruction.inputs[1].size,
        )?;

        let msb: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let lsb: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();

        Self::write_bitvec(memory, instruction, lsb.concat(msb))?;

        Ok(())
    }

    /// This is a truncation operator that understands the endianess of the data. Input1 indicates
    /// the number of least significant bytes of input0 to be thrown away. Output is then filled
    /// with any remaining bytes of input0 up to the size of output. If the size of output is
    /// smaller than the size of input0 minus the constant input1, then the additional most
    /// significant bytes of input0 will also be truncated.
    fn subpiece(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_address_space_type(&instruction, 1, AddressSpaceType::Constant)?;
        let mut data = memory.read(&instruction.inputs[0])?;

        // Remove this number of least significant bytes. If for some reason the offset exceeds
        // the maximum usize value, then by definition all of the data would be drained anyway.
        let truncate_count = instruction.inputs[1]
            .address
            .offset
            .try_into()
            .unwrap_or(usize::MAX);

        // Remove this number of least significant bytes. If for some reason the offset exceeds
        // the maximum usize value, then by definition all of the data would be drained anyway.
        data.drain(..truncate_count);

        // Remove any excess from most significant bytes
        let output = instruction.output.as_ref().unwrap();
        data.drain(output.size..);

        // Clone remaining bytes
        let data = data.into_iter().collect();
        Self::write_bytes(memory, instruction, data)?;

        Ok(())
    }

    /// This is an indirect branching instruction. The address to branch to is determined
    /// dynamically (at runtime) by examining the contents of the variable input0. As this
    /// instruction is currently defined, the variable input0 only contains the offset of the
    /// destination, and the address space is taken from the address associated with the branching
    /// instruction itself. So execution can only branch within the same address space via this
    /// instruction. The size of the variable input0 must match the size of offsets for the current
    /// address space. P-code relative branching is not possible with BRANCHIND.
    fn branch_ind(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, false)?;

        // Constant address space indicates a p-code relative branch.
        let address_space = &instruction.address.address_space;
        if address_space.space_type == AddressSpaceType::Constant {
            return Err(Error::IllegalInstruction {
                instruction: instruction.clone(),
                reason: format!(
                    "P-code relative branching is not possible with {:?}",
                    instruction.op_code
                ),
            });
        }

        let offset = Self::indirect_offset(memory, instruction, 0, address_space)?;

        Ok(ControlFlow::Jump(Destination::MachineAddress(Address {
            address_space: address_space.clone(),
            offset,
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
    fn return_instruction(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        self.branch_ind(memory, instruction)
    }

    /// This is a logical negate operator, where we assume input0 and output are boolean values. It
    /// puts the logical complement of input0, treated as a single bit, into output. Both input0 and
    /// output are size 1. Boolean values are implemented with a full byte, but are still considered
    /// to only support a value of true or false.
    fn bool_negate(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 1)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_equal(&instruction, 1)?;
        require_output_size_equals(&instruction, 1)?;

        let input = memory.read_bit(&instruction.inputs[0])?;
        let negation = !input;

        Self::write_bytes(memory, instruction, vec![negation.into()])?;

        Ok(())
    }

    /// This is an Exclusive-Or operator, where we assume the inputs and output are boolean values.
    /// It puts the exclusive-or of input0 and input1, treated as single bits, into output. Both
    /// inputs and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_xor(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_equal(&instruction, 1)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs = memory.read_bit(&instruction.inputs[0])?;
        let rhs = memory.read_bit(&instruction.inputs[1])?;
        let xor = lhs ^ rhs;

        Self::write_bytes(memory, instruction, vec![xor.into()])?;

        Ok(())
    }

    /// This is a Logical-And operator, where we assume the inputs and output are boolean values. It
    /// puts the logical-and of input0 and input1, treated as single bits, into output. Both inputs
    /// and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_and(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_equal(&instruction, 1)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs = memory.read_bit(&instruction.inputs[0])?;
        let rhs = memory.read_bit(&instruction.inputs[1])?;
        let and = lhs & rhs;

        Self::write_bytes(memory, instruction, vec![and.into()])?;

        Ok(())
    }

    /// This is a Logical-Or operator, where we assume the inputs and output are boolean values. It
    /// puts the logical-or of input0 and input1, treated as single bits, into output. Both inputs
    /// and output are size 1. Boolean values are implemented with a full byte, but are still
    /// considered to only support a value of true or false.
    fn bool_or(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_input_sizes_equal(&instruction, 1)?;
        require_output_size_equals(&instruction, 1)?;

        let lhs = memory.read_bit(&instruction.inputs[0])?;
        let rhs = memory.read_bit(&instruction.inputs[1])?;
        let or = lhs | rhs;

        Self::write_bytes(memory, instruction, vec![or.into()])?;

        Ok(())
    }

    /// This operation performs a left shift on input0. The value given by input1, interpreted as an
    /// unsigned integer, indicates the number of bits to shift. The vacated (least significant)
    /// bits are filled with zero. If input1 is zero, no shift is performed and input0 is copied
    /// into output. If input1 is larger than the number of bits in output, the result is zero. Both
    /// input0 and output must be the same size. Input1 can be any size.
    fn shift_left(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let result = lhs << rhs;

        Self::write_bitvec(memory, instruction, result)?;

        Ok(())
    }

    /// This operation performs an unsigned (logical) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated (most significant) bits are filled with zero. If input1 is zero, no shift is
    /// performed and input0 is copied into output. If input1 is larger than the number of bits in
    /// output, the result is zero. Both input0 and output must be the same size. Input1 can be any
    /// size.
    fn shift_right(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let result = lhs >> rhs;

        Self::write_bitvec(memory, instruction, result)?;

        Ok(())
    }

    /// This operation performs a signed (arithmetic) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated bits are filled with the original value of the most significant (sign) bit of
    /// input0. If input1 is zero, no shift is performed and input0 is copied into output. If input1
    /// is larger than the number of bits in output, the result is zero or all 1-bits (-1),
    /// depending on the original sign of input0. Both input0 and output must be the same size.
    /// Input1 can be any size.
    fn signed_shift_right(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(&instruction, 2)?;
        require_has_output(&instruction, true)?;
        require_output_size_equals(&instruction, instruction.inputs[0].size)?;

        let lhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[0])?.into_iter().collect();
        let rhs: sym::SymbolicBitVec = memory.read(&instruction.inputs[1])?.into_iter().collect();
        let result = lhs.signed_shift_right(rhs);

        Self::write_bitvec(memory, instruction, result)?;

        Ok(())
    }

    /// Construct an address from runtime values. The address space of the address is encoded in
    /// the first input as the address space identifier. The address offset is stored in memory at
    /// the address specified by the second input.
    ///
    /// The address size of the address space referred to by the first input should match the size
    /// of the varnode referred to by the second input. In other words, an address space which
    /// requires 4-bytes to address should load a 4-byte offset from memory.
    fn indirect_address(
        &self,
        memory: &impl SymbolicMemory,
        instruction: &PcodeInstruction,
    ) -> Result<Address> {
        // Space identifier must be a constant value
        require_input_address_space_type(&instruction, 0, AddressSpaceType::Constant)?;
        let address_space = self.address_space(&instruction, 0)?.clone();
        let offset = Self::indirect_offset(memory, instruction, 1, &address_space)?;

        Ok(Address {
            address_space,
            offset,
        })
    }

    fn indirect_offset(
        memory: &impl SymbolicMemory,
        instruction: &PcodeInstruction,
        input_index: usize,
        target_space: &AddressSpace,
    ) -> Result<u64> {
        // This assumes the number of inputs has already been validated
        require_input_size_equals(instruction, input_index, target_space.address_size)?;

        // Get concrete bytes. Can return an error if byte is symbolic
        let offset_bytes = memory.read(&instruction.inputs[input_index])?;
        let offset_bytes: Vec<u8> = memory
            .read(&instruction.inputs[input_index])?
            .into_iter()
            .map(u8::try_from)
            .collect::<std::result::Result<_, sym::ConcretizationError<_>>>()
            .map_err(|_| Error::SymbolicAddress {
                instruction: instruction.clone(),
                varnode: instruction.inputs[input_index].clone(),
                address: offset_bytes,
            })?;

        // Convert vector into array of bytes
        let offset = match target_space.address_size {
            1 => u8::from_le_bytes(offset_bytes.try_into().unwrap()).into(),
            2 => u16::from_le_bytes(offset_bytes.try_into().unwrap()).into(),
            4 => u32::from_le_bytes(offset_bytes.try_into().unwrap()).into(),
            8 => u64::from_le_bytes(offset_bytes.try_into().unwrap()).into(),
            _ => {
                // Other values less than 8 could be supported but requires zext first.
                // Values greater than 8 are impossible since the offset type is u64
                return Err(Error::InternalError(format!(
                    "address space size {size} not supported for indirect lookups",
                    size = target_space.address_size
                )));
            }
        };

        Ok(offset)
    }

    /// Get the address space encoded by the address offset of the specified input.
    fn address_space(
        &self,
        instruction: &PcodeInstruction,
        input_index: usize,
    ) -> Result<&AddressSpace> {
        let input = &instruction.inputs[input_index];
        let space_id = input.address.offset.try_into().map_err(|err| {
            Error::InternalError(format!(
                "unable to convert offset {offset} into address space id: {err}",
                offset = input.address.offset,
            ))
        })?;

        self.address_spaces_by_id
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace {
                instruction: instruction.clone(),
                varnode: input.clone(),
                space_id,
            })
    }
}

/// Require that the number of inputs matches the number expected by the instruction
fn require_num_inputs(instruction: &PcodeInstruction, num_inputs: usize) -> Result<()> {
    if instruction.inputs.len() == num_inputs {
        Ok(())
    } else {
        Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!(
                "expected {num_inputs} inputs, found {actual}",
                actual = instruction.inputs.len()
            ),
        })
    }
}

/// Require the instruction output existence to match the expected value
fn require_has_output(instruction: &PcodeInstruction, has_output: bool) -> Result<()> {
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
            Err(Error::IllegalInstruction {
                instruction: instruction.clone(),
                reason: format!(
                    "instruction output address space is constant: {:?}",
                    instruction.output
                ),
            })
        } else {
            Ok(())
        }
    } else {
        Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!(
                "instruction has unexpected output: {:?}",
                instruction.output
            ),
        })
    }
}

/// Require that the input sizes match the value expected by the instruction
fn require_input_sizes_match(instruction: &PcodeInstruction) -> Result<()> {
    require_input_sizes_equal(instruction, (&instruction.inputs[0]).size)
}

/// Require that the input sizes match the size of the instruction output
fn require_input_sizes_match_output(instruction: &PcodeInstruction) -> Result<()> {
    require_input_sizes_equal(instruction, instruction.output.as_ref().unwrap().size)
}

/// Require that the input sizes are all equal
fn require_input_sizes_equal(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    (0..instruction.inputs.len())
        .map(|i| require_input_size_equals(instruction, i, expected_size))
        .collect()
}

/// Require the input address space to be of the expected type
fn require_input_address_space_type(
    instruction: &PcodeInstruction,
    input_index: usize,
    expected_space_type: AddressSpaceType,
) -> Result<()> {
    let space_type = instruction.inputs[input_index]
        .address
        .address_space
        .space_type;
    if space_type != expected_space_type {
        return Err(Error::IllegalInstruction {
                instruction: instruction.clone(),
                reason: format!(
                    "input[{input_index}] address space type is {space_type:?}, expected {expected_space_type:?}"
                ),
        });
    }

    Ok(())
}

/// Require that the instruction input identified by its index has the expected size
fn require_input_size_equals(
    instruction: &PcodeInstruction,
    input_index: usize,
    expected_size: usize,
) -> Result<()> {
    let input = &instruction.inputs[input_index];
    if input.size != expected_size {
        Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!(
                "input[{input_index}] size {} != {expected_size}",
                input.size
            ),
        })
    } else {
        Ok(())
    }
}

/// Require that the instruction output size equals the expected value
fn require_output_size_equals(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    let output_size = instruction.output.as_ref().unwrap().size;
    if output_size != expected_size {
        return Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!("output size {output_size} != {expected_size}"),
        });
    }

    Ok(())
}

/// Require that the instruction output size is strictly greater than the expected size
fn require_output_size_exceeds(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    let output_size = instruction.output.as_ref().unwrap().size;
    if output_size <= expected_size {
        return Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!("output size {output_size} must exceed {expected_size}"),
        });
    }

    Ok(())
}

/// Require that the instruction output size is at least the expected size
fn require_output_size_at_least(
    instruction: &PcodeInstruction,
    expected_size: usize,
) -> Result<()> {
    let output_size = instruction.output.as_ref().unwrap().size;
    if output_size < expected_size {
        return Err(Error::IllegalInstruction {
            instruction: instruction.clone(),
            reason: format!("output size {output_size} must be at least {expected_size}"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem::Memory;
    use sym::{SymbolicBitVec, SymbolicByte};

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
        memory: &mut Memory,
        offset: u64,
        bytes: Vec<SymbolicByte>,
    ) -> Result<VarnodeData> {
        let varnode = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset,
            },
            size: bytes.len(),
        };

        memory.write(&varnode, bytes.into_iter())?;
        Ok(varnode)
    }

    #[test]
    fn copy() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);
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

        memory.write(&input, data.into_iter())?;
        emulator.emulate(&mut memory, &instruction)?;
        memory.read(&output)?;
        assert_eq!(memory.read_concrete_value::<u32>(&input)?, 0xDEADBEEF);

        Ok(())
    }

    #[test]
    fn load() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        // Write 0xDEADBEEF to 0x04030201
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0x04030201,
            },
            size: 4,
        };
        memory.write(&data_input, data.into_iter())?;

        // Write 0x04030201 to 0x0. This is the load indirection
        let offset_data = vec![0x01u8.into(), 0x02u8.into(), 0x03u8.into(), 0x04u8.into()];
        let offset_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&offset_input, offset_data.into_iter())?;

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

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Load,
            inputs: vec![addr_space_input.clone(), offset_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u32>(&output)?, 0xDEADBEEF);
        Ok(())
    }

    #[test]
    fn store() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

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
        memory.write(&data_input, data.into_iter())?;

        // Write 0x04030201 to 0x0. This is the store indirection
        let offset_data = vec![0x01u8.into(), 0x02u8.into(), 0x03u8.into(), 0x04u8.into()];
        let offset_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&offset_input, offset_data.into_iter())?;

        // Set the address space input offset to the space id of the processor addr space
        // It is important that the address space of this varnode is the constant space.
        let addr_space_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: processor_address_space().id as u64,
            },
            size: 8, // This value doesn't really matter
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
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
        assert_eq!(memory.read_concrete_value::<u32>(&output)?, 0xDEADBEEF);
        Ok(())
    }

    #[test]
    fn int_sub() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let lhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&lhs_input, lhs_data.into_iter())?;

        let rhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0x00u8.into(), 0x00u8.into()];
        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        memory.write(&rhs_input, rhs_data.into_iter())?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::Subtract),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u32>(&output)?, 0xDEAD0000);
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

            emulator.emulate(&mut memory, &instruction)?;

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed borrow of {lhs} - {rhs}"
            );
        }

        Ok(())
    }

    #[test]
    fn int_add() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let lhs_data = vec![0x00u8.into(), 0x00u8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&lhs_input, lhs_data.into_iter())?;

        let rhs_data = vec![0xEFu8.into(), 0xBEu8.into(), 0x00u8.into(), 0x00u8.into()];
        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        memory.write(&rhs_input, rhs_data.into_iter())?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 4,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::Add),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u32>(&output)?, 0xDEADBEEF);
        Ok(())
    }

    #[test]
    fn int_multiply() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 0..16u8 {
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = SymbolicBitVec::constant(lhs.into(), 4);
                let lhs_value = lhs_value.zero_extend(4).into_bytes().pop().unwrap();
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs_value])?;

                let rhs_value = SymbolicBitVec::constant(rhs.into(), 4);
                let rhs_value = rhs_value.zero_extend(4).into_bytes().pop().unwrap();
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs_value])?;

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

                emulator.emulate(&mut memory, &instruction)?;

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
                    lhs * rhs,
                    "failed {lhs} * {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_multiply_multibyte() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs: u8 = 0xFF;
        let lhs_value: SymbolicBitVec = lhs.into();
        let lhs_value = lhs_value.zero_extend(8);
        let lhs_input = write_bytes(&mut memory, 0, lhs_value.into_bytes())?;

        let rhs: u8 = 0x80;
        let rhs_value: SymbolicBitVec = rhs.into();
        let rhs_value = rhs_value.zero_extend(8);
        let rhs_input = write_bytes(&mut memory, 1, rhs_value.into_bytes())?;

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

        emulator.emulate(&mut memory, &instruction)?;

        assert_eq!(
            memory.read_concrete_value::<u16>(&output)?,
            lhs as u16 * rhs as u16,
            "failed {lhs} * {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_divide() -> Result<()> {
        for lhs in 0..16u8 {
            for rhs in 1..16u8 {
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value: SymbolicByte = lhs.into();
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs_value])?;

                let rhs_value: SymbolicByte = rhs.into();
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs_value])?;

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

                emulator.emulate(&mut memory, &instruction)?;

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value: SymbolicByte = lhs.into();
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs_value])?;

                let rhs_value: SymbolicByte = rhs.into();
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs_value])?;

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

                emulator.emulate(&mut memory, &instruction)?;

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = SymbolicBitVec::constant(lhs.into(), 4)
                    .sign_extend(4)
                    .into_bytes()
                    .pop()
                    .unwrap();
                let lhs: u8 = lhs_value.clone().try_into().unwrap();
                let lhs = lhs as i8;
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs_value])?;

                let rhs_value = SymbolicBitVec::constant(rhs.into(), 4)
                    .sign_extend(4)
                    .into_bytes()
                    .pop()
                    .unwrap();
                let rhs: u8 = rhs_value.clone().try_into().unwrap();
                let rhs = rhs as i8;
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs_value])?;

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

                emulator.emulate(&mut memory, &instruction)?;

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)? as i8,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = SymbolicBitVec::constant(lhs.into(), 4)
                    .sign_extend(4)
                    .into_bytes()
                    .pop()
                    .unwrap();
                let lhs: u8 = lhs_value.clone().try_into().unwrap();
                let lhs = lhs as i8;
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs_value])?;

                let rhs_value = SymbolicBitVec::constant(rhs.into(), 4)
                    .sign_extend(4)
                    .into_bytes()
                    .pop()
                    .unwrap();
                let rhs: u8 = rhs_value.clone().try_into().unwrap();
                let rhs = rhs as i8;
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs_value])?;

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

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)? as i8,
                    lhs % rhs,
                    "failed signed {lhs} % {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_zext() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xFFu8.into()];
        let input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 1,
        };
        memory.write(&input, data.into_iter())?;

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0,
            },
            size: 2,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::Extension(IntSign::Unsigned)),
            inputs: vec![input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u16>(&output)?, 0x00FF);
        Ok(())
    }

    #[test]
    fn int_sext() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0x7Fu8.into(), 0x80u8.into()];
        let data_varnode = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 2,
        };
        memory.write(&data_varnode, data.into_iter())?;

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

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Int(IntOp::Extension(IntSign::Signed)),
            inputs: vec![input_positive.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u16>(&output)?, 0x007F);

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
        assert_eq!(memory.read_concrete_value::<u16>(&output)?, 0xFF80);
        Ok(())
    }

    #[test]
    fn int_equal() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&lhs_input, data.iter().cloned())?;

        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        memory.write(&rhs_input, data.iter().cloned())?;

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
        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            0x1,
            "Expected 0xDEADBEEF == 0xDEADBEEF to be 1"
        );

        memory.write(
            &rhs_input,
            vec![0x0u8.into(), 0x0u8.into(), 0x0u8.into(), 0x0u8.into()].into_iter(),
        )?;

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            0x0,
            "Expected 0xDEADBEEF == 0x0 to be 0"
        );
        Ok(())
    }

    #[test]
    fn int_not_equal() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let lhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&lhs_input, data.iter().cloned())?;

        let rhs_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 4,
            },
            size: 4,
        };
        memory.write(&rhs_input, data.iter().cloned())?;

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
            op_code: OpCode::Int(IntOp::NotEqual),
            inputs: vec![lhs_input.clone(), rhs_input.clone()],
            output: Some(output.clone()),
        };

        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            0x0,
            "Expected 0xDEADBEEF != 0xDEADBEEF to be 0"
        );

        memory.write(
            &rhs_input,
            vec![0x0u8.into(), 0x0u8.into(), 0x0u8.into(), 0x0u8.into()].into_iter(),
        )?;
        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            0x1,
            "Expected 0xDEADBEEF != 0x0 to be 1"
        );
        Ok(())
    }

    #[test]
    fn piece() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let msb_input = write_bytes(&mut memory, 0, vec![0xADu8.into(), 0xDEu8.into()])?;
        let lsb_input = write_bytes(&mut memory, 2, vec![0xEFu8.into(), 0xBEu8.into()])?;

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

        emulator.emulate(&mut memory, &instruction)?;

        assert_eq!(memory.read_concrete_value::<u32>(&output)?, 0xDEADBEEF);
        Ok(())
    }

    #[test]
    fn subpiece() -> Result<()> {
        let mut memory = Memory::new();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = VarnodeData {
            address: Address {
                address_space: unique_address_space(),
                offset: 0,
            },
            size: 4,
        };
        memory.write(&data_input, data.into_iter())?;

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

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Subpiece,
            inputs: vec![data_input.clone(), truncation_input.clone()],
            output: Some(output.clone()),
        };

        // Expect to truncate 2 least-significant bytes
        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u16>(&output)?, 0xDEAD);

        let output = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 4,
            },
            size: 1,
        };

        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
            op_code: OpCode::Subpiece,
            inputs: vec![data_input.clone(), truncation_input.clone()],
            output: Some(output.clone()),
        };

        // Expect to truncate 2 least-significant bytes and 1 most significant byte
        // since the output size is less than the input size
        emulator.emulate(&mut memory, &instruction)?;
        assert_eq!(memory.read_concrete_value::<u8>(&output)?, 0xAD);
        Ok(())
    }

    #[test]
    fn branch_ind() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = write_bytes(&mut memory, 0, data)?;
        let instruction = PcodeInstruction {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xFF00000000,
            },
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
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let data = vec![0xEFu8.into(), 0xBEu8.into(), 0xADu8.into(), 0xDEu8.into()];
        let data_input = write_bytes(&mut memory, 0, data)?;

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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let input = write_bytes(&mut memory, 0, vec![value.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
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
                let mut memory = Memory::new();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
                let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

                assert_eq!(
                    memory.read_concrete_value::<u8>(&output)?,
                    lhs ^ rhs,
                    "failed {lhs} ^ {rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn int_negate() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;

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

        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            !lhs,
            "failed !{lhs}"
        );

        Ok(())
    }

    #[test]
    fn int_2comp() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 1u8;
        let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;

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

        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            0xFF, // Negative 1
            "failed -{lhs}"
        );

        Ok(())
    }

    #[test]
    fn int_and() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
        let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            lhs & rhs,
            "failed {lhs} & {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_or() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
        let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

        assert_eq!(
            memory.read_concrete_value::<u8>(&output)?,
            lhs | rhs,
            "failed {lhs} | {rhs}"
        );

        Ok(())
    }

    #[test]
    fn int_xor() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b1111_0000_0011_1100;
        let rhs = 0b0000_1111_1010_0101;
        let lhs_input = write_bytes(&mut memory, 0, sym::SymbolicBitVec::from(lhs).into_bytes())?;
        let rhs_input = write_bytes(&mut memory, 2, sym::SymbolicBitVec::from(rhs).into_bytes())?;

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

        assert_eq!(
            memory.read_concrete_value::<u16>(&output)?,
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![lhs.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![rhs.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed signed comparison {lhs} <= {rhs}"
            );
        }

        Ok(())
    }

    #[test]
    fn shift_left() -> Result<()> {
        for n in 0..=8u8 {
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![0x01u8.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![n.into()])?;

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
            let expected_result = if n < 8 { 1 << n } else { 0 };

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed 1 << {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn shift_right() -> Result<()> {
        for n in 0..=8u8 {
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![0x80u8.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![n.into()])?;

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
            let expected_result = if n < 8 { 0x80 >> n } else { 0 };

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed 0x80 >> {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn signed_shift_right() -> Result<()> {
        for n in 0..=8u8 {
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![0x80u8.into()])?;
            let rhs_input = write_bytes(&mut memory, 1, vec![n.into()])?;

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
            let expected_result = if n < 8 { (-128i8 >> n) as u8 } else { 0xFF };

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed signed shift 0x80 >> {n}"
            );
        }

        Ok(())
    }

    #[test]
    fn call() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
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
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
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
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
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
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xDEADBEEF,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_bytes(&mut memory, 1, vec![0x1u8.into()])?;
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
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: u64::MAX,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_bytes(&mut memory, 1, vec![0x1u8.into()])?;
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
            let mut memory = Memory::new();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_bytes(&mut memory, 0, vec![value.into()])?;

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

            assert_eq!(
                memory.read_concrete_value::<u8>(&output)?,
                expected_result,
                "failed popcount of {value:#02x}"
            );
        }

        Ok(())
    }

    #[test]
    fn unsupported_opcode() -> Result<()> {
        let mut memory = Memory::new();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
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
}
