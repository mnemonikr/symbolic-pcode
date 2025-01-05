use pcode_ops::{BitwisePcodeOps, PcodeOps};
use sla::{
    Address, AddressSpace, AddressSpaceId, AddressSpaceType, BoolOp, IntOp, IntSign, OpCode,
    PcodeInstruction, VarnodeData,
};
use thiserror;

use crate::mem::{self, VarnodeDataStore};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Error occurred while accessing a memory location.
    #[error(transparent)]
    MemoryAccess(#[from] mem::Error),

    /// The provided instruction violates an invariant described by the error kind.
    #[error("illegal instruction {instruction}: {kind}")]
    IllegalInstruction {
        instruction: Box<PcodeInstruction>,
        kind: IllegalInstructionKind,
    },

    /// The offset stored in memory was retrieved successfully but failed to be converted to the
    /// type required to construct an address offset. This can occur if the offset stored in memory
    /// is symbolic.
    #[error("failed to construct indirect address for {instruction:?}")]
    IndirectAddressOffset {
        instruction: Box<PcodeInstruction>,
        offset_varnode: VarnodeData,
        target_address_space: Box<AddressSpace>,
    },

    /// Emulation of this instruction is not implemented.
    #[error("unsupported instruction {instruction:?}")]
    UnsupportedInstruction { instruction: Box<PcodeInstruction> },

    /// No address space with the given identifier is associated with this emulator. This can occur
    /// if the instruction was decoded with a different set of address spaces than the ones
    /// registered with this emulator.
    #[error("unknown address space id {space_id} referenced by {varnode} in instruction: {instruction:?}")]
    UnknownAddressSpace {
        instruction: Box<PcodeInstruction>,
        varnode: VarnodeData,
        space_id: AddressSpaceId,
    },

    /// An internal error occurred. This is a fatal error that cannot be safely handled.
    #[error("internal error: {0}")]
    InternalError(String),
}

#[derive(Debug)]
pub enum IllegalInstructionKind {
    VarnodeNotPermitted(usize),
    VarnodeMissing(usize),
    InvalidVarnodeSize(usize),
    InvalidVarnodeAddressSpace(usize),
    InvalidInstructionAddressSpace,
}

pub type Result<T> = std::result::Result<T, Error>;

/// The pcode emulator structure that holds the necessary data for emulation.
#[derive(Debug, Clone)]
pub struct StandardPcodeEmulator {
    address_spaces_by_id: std::collections::BTreeMap<AddressSpaceId, AddressSpace>,
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
    ConditionalBranch {
        condition_origin: VarnodeData,
        condition: Option<bool>,
        destination: Destination,
    },
}

pub trait PcodeEmulator {
    fn emulate<M: VarnodeDataStore>(
        &self,
        memory: &mut M,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow>;
}

macro_rules! binary_shift_op {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 2)?;
        require_has_output($instr, true)?;
        require_output_size_equals(&$instr, $instr.inputs[0].size)?;

        let lhs = $mem.read(&$instr.inputs[0])?;
        let rhs = $mem.read(&$instr.inputs[1])?;
        $mem.write($instr.output.as_ref().unwrap(), lhs.$op(rhs))?;
    }};
}

macro_rules! binary_op {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 2)?;
        require_has_output($instr, true)?;
        require_input_sizes_match_output($instr)?;

        let lhs = $mem.read(&$instr.inputs[0])?;
        let rhs = $mem.read(&$instr.inputs[1])?;
        $mem.write($instr.output.as_ref().unwrap(), lhs.$op(rhs))?;
    }};
}

macro_rules! unary_op {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 1)?;
        require_has_output($instr, true)?;
        require_input_sizes_match_output($instr)?;

        let lhs = $mem.read(&$instr.inputs[0])?;
        $mem.write($instr.output.as_ref().unwrap(), lhs.$op())?;
    }};
}

macro_rules! binary_op_bit {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 2)?;
        require_has_output($instr, true)?;
        require_output_size_equals($instr, 1)?;

        let lhs = $mem.read(&$instr.inputs[0])?;
        let rhs = $mem.read(&$instr.inputs[1])?;
        $mem.write_bit($instr.output.as_ref().unwrap(), lhs.$op(rhs))?;
    }};
}

macro_rules! bool_unary_op {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 1)?;
        require_has_output($instr, true)?;
        require_input_sizes_equal($instr, 1)?;
        require_output_size_equals($instr, 1)?;

        let lhs = $mem.read_bit(&$instr.inputs[0])?;
        $mem.write_bit($instr.output.as_ref().unwrap(), lhs.$op())?;
    }};
}

macro_rules! bool_binary_op {
    ($mem:ident, $instr:ident, $op:ident) => {{
        require_num_inputs($instr, 2)?;
        require_has_output($instr, true)?;
        require_input_sizes_equal($instr, 1)?;
        require_output_size_equals($instr, 1)?;

        let lhs = $mem.read_bit(&$instr.inputs[0])?;
        let rhs = $mem.read_bit(&$instr.inputs[1])?;
        $mem.write_bit($instr.output.as_ref().unwrap(), lhs.$op(rhs))?;
    }};
}

impl PcodeEmulator for StandardPcodeEmulator {
    fn emulate<M: VarnodeDataStore>(
        &self,
        memory: &mut M,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        match instruction.op_code {
            OpCode::Copy => self.copy(memory, instruction)?,
            OpCode::Load => self.load(memory, instruction)?,
            OpCode::Store => self.store(memory, instruction)?,
            OpCode::Int(IntOp::Bitwise(BoolOp::And)) => binary_op!(memory, instruction, and),
            OpCode::Int(IntOp::Bitwise(BoolOp::Or)) => binary_op!(memory, instruction, or),
            OpCode::Int(IntOp::Bitwise(BoolOp::Xor)) => binary_op!(memory, instruction, xor),
            OpCode::Int(IntOp::Bitwise(BoolOp::Negate)) => unary_op!(memory, instruction, not),
            OpCode::Int(IntOp::Add) => binary_op!(memory, instruction, add),
            OpCode::Int(IntOp::Carry(IntSign::Unsigned)) => {
                binary_op_bit!(memory, instruction, unsigned_carry)
            }
            OpCode::Int(IntOp::Carry(IntSign::Signed)) => {
                binary_op_bit!(memory, instruction, signed_carry)
            }
            OpCode::Int(IntOp::Subtract) => binary_op!(memory, instruction, subtract),
            OpCode::Int(IntOp::Negate) => unary_op!(memory, instruction, negate),
            OpCode::Int(IntOp::Borrow) => binary_op_bit!(memory, instruction, borrow),
            OpCode::Int(IntOp::Multiply) => binary_op!(memory, instruction, multiply),
            OpCode::Int(IntOp::Divide(IntSign::Unsigned)) => {
                binary_op!(memory, instruction, unsigned_divide)
            }
            OpCode::Int(IntOp::Divide(IntSign::Signed)) => {
                binary_op!(memory, instruction, signed_divide)
            }
            OpCode::Int(IntOp::Remainder(IntSign::Unsigned)) => {
                binary_op!(memory, instruction, unsigned_remainder)
            }
            OpCode::Int(IntOp::Remainder(IntSign::Signed)) => {
                binary_op!(memory, instruction, signed_remainder)
            }
            OpCode::Int(IntOp::Extension(IntSign::Unsigned)) => {
                self.int_zext(memory, instruction)?
            }
            OpCode::Int(IntOp::Extension(IntSign::Signed)) => self.int_sext(memory, instruction)?,
            OpCode::Popcount => self.popcount(memory, instruction)?,
            OpCode::Piece => self.piece(memory, instruction)?,
            OpCode::Subpiece => self.subpiece(memory, instruction)?,
            OpCode::Int(IntOp::Equal) => binary_op_bit!(memory, instruction, equals),
            OpCode::Int(IntOp::NotEqual) => binary_op_bit!(memory, instruction, not_equals),
            OpCode::Int(IntOp::LessThan(IntSign::Signed)) => {
                binary_op_bit!(memory, instruction, signed_less_than)
            }
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)) => {
                binary_op_bit!(memory, instruction, signed_less_than_or_equals)
            }
            OpCode::Int(IntOp::LessThan(IntSign::Unsigned)) => {
                binary_op_bit!(memory, instruction, unsigned_less_than)
            }
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)) => {
                binary_op_bit!(memory, instruction, unsigned_less_than_or_equals)
            }
            OpCode::Int(IntOp::ShiftLeft) => binary_shift_op!(memory, instruction, shift_left),
            OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)) => {
                binary_shift_op!(memory, instruction, unsigned_shift_right)
            }
            OpCode::Int(IntOp::ShiftRight(IntSign::Signed)) => {
                binary_shift_op!(memory, instruction, signed_shift_right)
            }
            OpCode::Bool(BoolOp::Negate) => bool_unary_op!(memory, instruction, not),
            OpCode::Bool(BoolOp::And) => bool_binary_op!(memory, instruction, and),
            OpCode::Bool(BoolOp::Or) => bool_binary_op!(memory, instruction, or),
            OpCode::Bool(BoolOp::Xor) => bool_binary_op!(memory, instruction, xor),
            OpCode::Return => return self.return_instruction(memory, instruction),
            OpCode::BranchIndirect => return self.branch_ind(memory, instruction),
            OpCode::Branch => return self.branch(instruction),
            OpCode::BranchConditional => return self.conditional_branch(memory, instruction),
            OpCode::Call => return self.call(instruction),
            OpCode::CallIndirect => return self.call_ind(memory, instruction),
            _ => {
                return Err(Error::UnsupportedInstruction {
                    instruction: Box::new(instruction.clone()),
                })
            }
        }

        Ok(ControlFlow::NextInstruction)
    }
}

impl StandardPcodeEmulator {
    /// Create a new emulator over the given set of address spaces. These address spaces are
    /// necessary to support indirect memory lookups, since such lookups are encoded into the
    /// pcode operands as address space ids.
    pub fn new(address_spaces: impl IntoIterator<Item = AddressSpace>) -> Self {
        Self {
            address_spaces_by_id: address_spaces
                .into_iter()
                .map(|space| (space.id, space))
                .collect(),
        }
    }

    /// Copy a sequence of contiguous bytes from anywhere to anywhere. Size of input0 and output
    /// must be the same.
    fn copy(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, true)?;
        require_input_sizes_match_output(instruction)?;

        let input = &instruction.inputs[0];
        memory.write(instruction.output.as_ref().unwrap(), memory.read(input)?)?;

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
    fn load(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 2)?;
        require_has_output(instruction, true)?;

        let output = instruction.output.as_ref().unwrap();
        let input = VarnodeData {
            address: self.indirect_address(memory, instruction)?,
            size: output.size,
        };

        memory.write(instruction.output.as_ref().unwrap(), memory.read(&input)?)?;

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
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 3)?;
        require_has_output(instruction, false)?;

        let input = &instruction.inputs[2];
        let output = VarnodeData {
            address: self.indirect_address(memory, instruction)?,
            size: input.size,
        };

        memory.write(&output, memory.read(input)?)?;

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
    fn branch(&self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, false)?;
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
    fn call(&self, instruction: &PcodeInstruction) -> Result<ControlFlow> {
        self.branch(instruction)
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
        memory: &mut impl VarnodeDataStore,
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
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        require_num_inputs(instruction, 2)?;
        require_has_output(instruction, false)?;
        require_input_size_equals(instruction, 1, 1)?;

        Ok(ControlFlow::ConditionalBranch {
            condition: memory.read_bit(&instruction.inputs[1])?.try_into().ok(),
            condition_origin: instruction.inputs[1].clone(),
            destination: Self::branch_destination(&instruction.inputs[0]),
        })
    }

    /// Zero-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with zero. The size of output must be strictly bigger than the
    /// size of input.
    fn int_zext(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, true)?;
        let input = &instruction.inputs[0];
        require_output_size_exceeds(instruction, input.size)?;
        let output = instruction.output.as_ref().unwrap();

        let lhs = memory.read(&instruction.inputs[0])?;
        memory.write(output, lhs.zero_extend(output.size))?;

        Ok(())
    }

    /// Sign-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with either zero or all ones (0xff) depending on the most
    /// significant bit of input0. The size of output must be strictly bigger than the size of
    /// input0.
    fn int_sext(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, true)?;
        let input = &instruction.inputs[0];
        require_output_size_exceeds(instruction, input.size)?;
        let output = instruction.output.as_ref().unwrap();

        let lhs = memory.read(&instruction.inputs[0])?;
        memory.write(output, lhs.sign_extend(output.size))?;

        Ok(())
    }

    /// This is a bit count (population count) operator. Within the binary representation of the
    /// value contained in the input varnode, the number of 1 bits are counted and then returned
    /// in the output varnode. A value of 0 returns 0, a 4-byte varnode containing the value
    /// 2<sup>32</sup>-1 (all bits set) returns 32, for instance. The input and output varnodes can
    /// have any size. The resulting count is zero extended into the output varnode.
    fn popcount(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, true)?;

        let output = instruction.output.as_ref().unwrap();
        let popcount = memory.read(&instruction.inputs[0])?.popcount();
        memory.write(output, popcount.zero_extend(output.size))?;

        Ok(())
    }

    /// This is a concatenation operator that understands the endianess of the data. The size of
    /// input0 and input1 must add up to the size of output. The data from the inputs is
    /// concatenated in such a way that, if the inputs and output are considered integers, the first
    /// input makes up the most significant part of the output.
    fn piece(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 2)?;
        require_has_output(instruction, true)?;
        require_output_size_equals(
            instruction,
            instruction.inputs[0].size + instruction.inputs[1].size,
        )?;

        let msb = memory.read(&instruction.inputs[0])?;
        let lsb = memory.read(&instruction.inputs[1])?;
        memory.write(instruction.output.as_ref().unwrap(), msb.piece(lsb))?;

        Ok(())
    }

    /// This is a truncation operator that understands the endianess of the data. Input1 indicates
    /// the number of least significant bytes of input0 to be thrown away. Output is then filled
    /// with any remaining bytes of input0 up to the size of output. If the size of output is
    /// smaller than the size of input0 minus the constant input1, then the additional most
    /// significant bytes of input0 will also be truncated.
    fn subpiece(
        &self,
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<()> {
        require_num_inputs(instruction, 2)?;
        require_has_output(instruction, true)?;
        require_input_address_space_type(instruction, 1, AddressSpaceType::Constant)?;
        let value = memory.read(&instruction.inputs[0])?;

        // Remove this number of least significant bytes. If for some reason the offset exceeds
        // the maximum usize value, then by definition all of the data would be drained anyway.
        let truncate_count = instruction.inputs[1].address.offset;
        let value = value.truncate_trailing_bytes(truncate_count);

        // Remove any excess from most significant bytes
        let output = instruction.output.as_ref().unwrap();
        let value = value.truncate_to_size(output.size);

        // Clone remaining bytes
        memory.write(output, value)?;

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
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        require_num_inputs(instruction, 1)?;
        require_has_output(instruction, false)?;

        // Constant address space indicates a p-code relative branch.
        let address_space = &instruction.address.address_space;
        if address_space.space_type == AddressSpaceType::Constant {
            return Err(Error::IllegalInstruction {
                instruction: Box::new(instruction.clone()),
                kind: IllegalInstructionKind::InvalidInstructionAddressSpace,
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
        memory: &mut impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        self.branch_ind(memory, instruction)
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
        memory: &impl VarnodeDataStore,
        instruction: &PcodeInstruction,
    ) -> Result<Address> {
        // Space identifier must be a constant value
        require_input_address_space_type(instruction, 0, AddressSpaceType::Constant)?;
        let address_space = self.address_space(instruction, 0)?.clone();
        let offset = Self::indirect_offset(memory, instruction, 1, &address_space)?;

        Ok(Address {
            address_space,
            offset,
        })
    }

    fn indirect_offset(
        memory: &impl VarnodeDataStore,
        instruction: &PcodeInstruction,
        input_index: usize,
        target_space: &AddressSpace,
    ) -> Result<u64> {
        // This assumes the number of inputs has already been validated
        require_input_size_equals(instruction, input_index, target_space.address_size)?;

        // Get concrete bytes. Can return an error if byte is symbolic
        memory
            .read(&instruction.inputs[input_index])?
            .try_into()
            .map_err(|_err| Error::IndirectAddressOffset {
                instruction: Box::new(instruction.clone()),
                offset_varnode: instruction.inputs[input_index].clone(),
                target_address_space: Box::new(target_space.clone()),
            })
    }

    /// Get the address space encoded by the address offset of the specified input.
    fn address_space(
        &self,
        instruction: &PcodeInstruction,
        input_index: usize,
    ) -> Result<&AddressSpace> {
        let input = &instruction.inputs[input_index];
        let space_id = usize::try_from(input.address.offset).map_err(|err| {
            Error::InternalError(format!(
                "unable to convert offset {offset} into address space id: {err}",
                offset = input.address.offset,
            ))
        })?;
        let space_id = AddressSpaceId::new(space_id);

        self.address_spaces_by_id
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace {
                instruction: Box::new(instruction.clone()),
                varnode: input.clone(),
                space_id,
            })
    }
}

impl std::fmt::Display for IllegalInstructionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let varnode_name_lookup = |index: &usize| {
            if *index == 0 {
                "output".to_string()
            } else {
                format!("input {input_index}", input_index = index - 1)
            }
        };

        match self {
            IllegalInstructionKind::VarnodeNotPermitted(varnode_index) => {
                write!(
                    f,
                    "{name} varnode not permitted",
                    name = varnode_name_lookup(varnode_index)
                )
            }
            IllegalInstructionKind::VarnodeMissing(varnode_index) => {
                write!(
                    f,
                    "{name} varnode missing",
                    name = varnode_name_lookup(varnode_index)
                )
            }
            IllegalInstructionKind::InvalidVarnodeSize(varnode_index) => {
                write!(
                    f,
                    "{name} varnode size is invalid",
                    name = varnode_name_lookup(varnode_index)
                )
            }
            IllegalInstructionKind::InvalidVarnodeAddressSpace(varnode_index) => {
                write!(
                    f,
                    "{name} varnode address space is invalid",
                    name = varnode_name_lookup(varnode_index)
                )
            }
            IllegalInstructionKind::InvalidInstructionAddressSpace => {
                write!(f, "instruction address space is invalid")
            }
        }
    }
}

/// Require that the number of inputs matches the number expected by the instruction
fn require_num_inputs(instruction: &PcodeInstruction, num_inputs: usize) -> Result<()> {
    match instruction.inputs.len().cmp(&num_inputs) {
        std::cmp::Ordering::Less => Err(Error::IllegalInstruction {
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::VarnodeMissing(num_inputs - instruction.inputs.len()),
        }),
        std::cmp::Ordering::Equal => Ok(()),
        std::cmp::Ordering::Greater => Err(Error::IllegalInstruction {
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::VarnodeNotPermitted(num_inputs + 1),
        }),
    }
}

/// Require the instruction output existence to match the expected value
fn require_has_output(instruction: &PcodeInstruction, has_output: bool) -> Result<()> {
    if has_output {
        if instruction.output.is_none() {
            return Err(Error::IllegalInstruction {
                instruction: Box::new(instruction.clone()),
                kind: IllegalInstructionKind::VarnodeMissing(0),
            });
        } else if instruction
            .output
            .as_ref()
            .unwrap()
            .address
            .address_space
            .is_constant()
        {
            return Err(Error::IllegalInstruction {
                instruction: Box::new(instruction.clone()),
                kind: IllegalInstructionKind::InvalidVarnodeAddressSpace(0),
            });
        }
    } else if instruction.output.is_some() {
        return Err(Error::IllegalInstruction {
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::VarnodeNotPermitted(0),
        });
    }

    Ok(())
}

/// Require that the input sizes match the size of the instruction output
fn require_input_sizes_match_output(instruction: &PcodeInstruction) -> Result<()> {
    require_input_sizes_equal(instruction, instruction.output.as_ref().unwrap().size)
}

/// Require that the input sizes are all equal
fn require_input_sizes_equal(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    (0..instruction.inputs.len())
        .try_for_each(|i| require_input_size_equals(instruction, i, expected_size))
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
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::InvalidVarnodeAddressSpace(input_index + 1),
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
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::InvalidVarnodeSize(input_index + 1),
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
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::InvalidVarnodeSize(0),
        });
    }

    Ok(())
}

/// Require that the instruction output size is strictly greater than the expected size
fn require_output_size_exceeds(instruction: &PcodeInstruction, expected_size: usize) -> Result<()> {
    let output_size = instruction.output.as_ref().unwrap().size;
    if output_size <= expected_size {
        return Err(Error::IllegalInstruction {
            instruction: Box::new(instruction.clone()),
            kind: IllegalInstructionKind::InvalidVarnodeSize(0),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::test_fixture::ConcreteValue;
    use mem::{GenericMemory, VarnodeDataStore};

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
        memory: &mut GenericMemory<ConcreteValue>,
        offset: u64,
        value: ConcreteValue,
    ) -> Result<VarnodeData> {
        let varnode = VarnodeData {
            address: processor_address(offset),
            size: value.num_bytes(),
        };

        memory.write(&varnode, value)?;
        Ok(varnode)
    }

    #[test]
    fn copy() -> Result<()> {
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
            StandardPcodeEmulator::new(vec![processor_address_space(), unique_address_space()]);

        // Write 0xDEADBEEF to 0x04030201
        let data = 0xDEADBEEFu32;
        let offset = 0x04030201u64;
        write_value(&mut memory, offset, data.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs.into())?;
            let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs: u16 = 0xFF;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;

        let rhs: u16 = 0x80;
        let rhs_input = write_value(&mut memory, 2, rhs.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = ConcreteValue::new(lhs.into(), 4).sign_extend(1);
                let lhs: i8 = lhs_value.signed_value() as i8;
                let lhs_input = write_value(&mut memory, 0, lhs_value)?;

                let rhs_value = ConcreteValue::new(rhs.into(), 4).sign_extend(1);
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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_value = ConcreteValue::new(lhs.into(), 4).sign_extend(1);
                let lhs: i8 = lhs_value.signed_value() as i8;
                let lhs_input = write_value(&mut memory, 0, lhs_value)?;

                let rhs_value = ConcreteValue::new(rhs.into(), 4).sign_extend(1);
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let msb_input = write_value(&mut memory, 0, 0xDEADu16.into())?;
        let lsb_input = write_value(&mut memory, 2, 0xBEEFu16.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator =
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let data = 0xDEADBEEFu32;
        let data_input = write_value(&mut memory, 0, data.into())?;
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let data = 0xDEADBEEFu32;
        let data_input = write_value(&mut memory, 0, data.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let input = write_value(&mut memory, 0, value.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
                let mut memory = GenericMemory::<ConcreteValue>::default();
                let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
                let lhs_input = write_value(&mut memory, 0, lhs.into())?;
                let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b1010_0101;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 1u8;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;
        let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b0011_1100;
        let rhs = 0b1010_0101;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;
        let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let lhs = 0b1111_0000_0011_1100;
        let rhs = 0b0000_1111_1010_0101;
        let lhs_input = write_value(&mut memory, 0, lhs.into())?;
        let rhs_input = write_value(&mut memory, 2, rhs.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs.into())?;
            let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs.into())?;
            let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs.into())?;
            let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, lhs.into())?;
            let rhs_input = write_value(&mut memory, 1, rhs.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, 0x01u8.into())?;
            let rhs_input = write_value(&mut memory, 1, n.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, 0x80u8.into())?;
            let rhs_input = write_value(&mut memory, 1, n.into())?;

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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, 0x80u8.into())?;
            let rhs_input = write_value(&mut memory, 1, n.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: processor_address_space(),
                offset: 0xDEADBEEF,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_value(&mut memory, 1, 0x1u8.into())?;
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
        let mut memory = GenericMemory::<ConcreteValue>::default();
        let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
        let destination_input = VarnodeData {
            address: Address {
                address_space: constant_address_space(),
                offset: u64::MAX,
            },
            size: 0, // This value is irrelevant
        };

        let condition_input = write_value(&mut memory, 1, 0x1u8.into())?;
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
            let mut memory = GenericMemory::<ConcreteValue>::default();
            let emulator = StandardPcodeEmulator::new(vec![processor_address_space()]);
            let lhs_input = write_value(&mut memory, 0, value.into())?;

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
        let mut memory = GenericMemory::<ConcreteValue>::default();
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
