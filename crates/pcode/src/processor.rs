use pcode_ops::BitwisePcodeOps;
use thiserror;

use crate::emulator::{ControlFlow, Destination, PcodeEmulator};
use crate::mem::{ExecutableMemory, MemoryBranch, VarnodeDataStore};
use libsla::{
    Address, AddressSpace, NativeDisassembly, PcodeDisassembly, PcodeInstruction, Sleigh,
    VarnodeData,
};

// TODO Emulator can also have memory access errors. Probably better to write a custom
// derivation that converts emulator errors into processor errors.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Error occurred during emulation
    #[error(transparent)]
    Emulation(#[from] crate::emulator::Error),

    /// Error occurred while accessing a memory location
    #[error(transparent)]
    MemoryAccess(#[from] crate::mem::Error),

    #[error("failed to decode instruction: {0}")]
    InstructionDecoding(#[from] libsla::Error),

    #[error("address {address} not in expected space {expected}")]
    InvalidAddressSpace {
        address: Address,
        expected: AddressSpace,
    },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("symbolic branch condition at {condition_origin}")]
    SymbolicBranch { condition_origin: VarnodeData },

    #[error("internal error: {0}")]
    InternalError(String),

    #[error("operation not permitted for state {0:?}")]
    InvalidState(ProcessorState),
}

pub type Result<T> = std::result::Result<T, Error>;

enum NextExecution {
    NextInstruction,
    Jump(Address),
    PcodeOffset(i64),
    Halt,
}

enum BranchingNextExecution {
    Branch(VarnodeData, NextExecution, NextExecution),
    Flow(NextExecution),
}

pub trait ProcessorResponseHandler: Clone {
    fn fetched<M: VarnodeDataStore>(&mut self, memory: &mut M) -> Result<Address>;
    fn decoded<M: VarnodeDataStore>(
        &mut self,
        memory: &mut M,
        execution: &PcodeExecution,
    ) -> Result<()>;
    fn jumped<M: VarnodeDataStore>(&mut self, memory: &mut M, address: &Address) -> Result<()>;
}

pub struct Processor<
    E: PcodeEmulator + Clone,
    M: VarnodeDataStore + Default,
    H: ProcessorResponseHandler + Clone,
> {
    memory: M,
    state: ProcessorState,
    handler: H,
    emulator: E,
}

impl<E: PcodeEmulator + Clone, M: VarnodeDataStore + Default, H: ProcessorResponseHandler + Clone>
    Processor<E, M, H>
{
    pub fn new(memory: M, emulator: E, handler: H) -> Self {
        Self {
            memory,
            emulator,
            handler,
            state: ProcessorState::Fetch,
        }
    }

    pub fn memory(&self) -> &M {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut M {
        &mut self.memory
    }

    pub fn emulator(&self) -> &E {
        &self.emulator
    }

    pub fn state(&self) -> &ProcessorState {
        &self.state
    }

    pub fn disassemble(&self, sleigh: &impl Sleigh) -> Result<NativeDisassembly> {
        if let ProcessorState::Decode(d) = &self.state {
            d.disassemble(sleigh, ExecutableMemory(&self.memory))
        } else {
            Err(Error::InvalidArgument(format!(
                "cannot disassemble outside of decoding state: {state:?}",
                state = self.state
            )))
        }
    }

    pub fn step(&mut self, sleigh: &impl Sleigh) -> Result<()> {
        self.step_internal(sleigh, None)
    }

    pub fn step_branch(
        &mut self,
        sleigh: &impl Sleigh,
        branch_condition_evaluation: bool,
    ) -> Result<()> {
        self.step_internal(sleigh, Some(branch_condition_evaluation))
    }

    fn step_internal(
        &mut self,
        sleigh: &impl Sleigh,
        branch_condition_evaluation: Option<bool>,
    ) -> Result<()> {
        match &mut self.state {
            ProcessorState::Fetch => {
                let fetched_instruction = self.handler.fetched(&mut self.memory)?;
                self.state = ProcessorState::Decode(DecodeInstruction::new(fetched_instruction));
            }
            ProcessorState::Decode(d) => {
                let execution = d.decode(sleigh, ExecutableMemory(&self.memory))?;
                self.handler.decoded(&mut self.memory, &execution)?;
                self.state = ProcessorState::Execute(execution);
            }
            ProcessorState::Execute(x) if x.is_empty() => {
                // Some instructions are decoded as noops.
                self.state = ProcessorState::Fetch;
            }
            ProcessorState::Execute(x) => {
                let control_flow = self
                    .emulator
                    .emulate(&mut self.memory, x.current_instruction())?;
                match x.next_execution(control_flow) {
                    BranchingNextExecution::Flow(e1) => self.update_execution(e1)?,
                    BranchingNextExecution::Branch(condition, e1, e2) => {
                        match branch_condition_evaluation {
                            Some(true) => self.update_execution(e1)?,
                            Some(false) => self.update_execution(e2)?,
                            _ => {
                                return Err(Error::SymbolicBranch {
                                    condition_origin: condition,
                                });
                            }
                        }
                    }
                }
            }
            state => return Err(Error::InvalidState(state.clone())),
        }

        Ok(())
    }

    fn update_execution(&mut self, next_execution: NextExecution) -> Result<()> {
        if let ProcessorState::Execute(execution) = &mut self.state {
            match next_execution {
                NextExecution::Jump(address) => {
                    self.handler.jumped(&mut self.memory, &address)?;
                    self.state = ProcessorState::Fetch;
                }
                NextExecution::NextInstruction => self.state = ProcessorState::Fetch,
                NextExecution::PcodeOffset(offset) => execution.update_index(offset)?,
                NextExecution::Halt => self.state = ProcessorState::Halt,
            }

            Ok(())
        } else {
            Err(Error::InternalError(format!(
                "cannot update execution, current state is {state:?}",
                state = self.state
            )))
        }
    }
}

pub struct BranchingProcessor<
    E: PcodeEmulator + Clone,
    M: VarnodeDataStore + Default,
    H: ProcessorResponseHandler + Clone,
> {
    processor: Processor<E, MemoryBranch<M>, H>,
}

impl<E: PcodeEmulator + Clone, M: VarnodeDataStore + Default, H: ProcessorResponseHandler + Clone>
    BranchingProcessor<E, M, H>
{
    pub fn new(memory: M, emulator: E, handler: H) -> Self {
        Self {
            processor: Processor::new(MemoryBranch::new(memory), emulator, handler),
        }
    }

    pub fn processor(&self) -> &Processor<E, MemoryBranch<M>, H> {
        &self.processor
    }

    pub fn processor_mut(&mut self) -> &mut Processor<E, MemoryBranch<M>, H> {
        &mut self.processor
    }

    /// Step the processor. In the event of a symbolic branch a new processor will be returned. The
    /// new processor will have taken the branch (i.e. evaluated the branch condition to `true`).
    /// This processor will **not** have taken the branch (i.e. evaluated the branch condition to
    /// `false`).
    pub fn step(&mut self, sleigh: &impl Sleigh) -> Result<Option<Self>> {
        match self.processor.step(sleigh) {
            Err(e) => {
                if let Error::SymbolicBranch { condition_origin } = &e {
                    let mut branched_processor = self.branch(condition_origin);
                    self.processor.step_branch(sleigh, false)?;
                    branched_processor.processor.step_branch(sleigh, true)?;
                    Ok(Some(branched_processor))
                } else {
                    Err(e)
                }
            }
            result => result.map(|_| None),
        }
    }

    /// Create a new processor with a branch of memory that takes this branch
    fn branch(&mut self, condition_origin: &VarnodeData) -> Self {
        Self {
            processor: Processor {
                memory: self.processor.memory.new_branch(
                    self.processor
                        .memory
                        .read_bit(condition_origin)
                        .unwrap()
                        // Need to negate this condition because the new memory is the one that
                        // does NOT take the branch
                        .not(),
                ),
                state: self.processor.state.clone(),
                handler: self.processor.handler.clone(),
                emulator: self.processor.emulator.clone(),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum ProcessorState {
    Fetch,
    Decode(DecodeInstruction),
    Execute(PcodeExecution),
    Halt,
}

impl std::fmt::Display for ProcessorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fetch => {
                write!(f, "Fetch")?;
            }
            Self::Halt => {
                write!(f, "Halt")?;
            }
            Self::Decode(decode_instr) => {
                write!(f, "Decode {address}", address = decode_instr.address())?;
            }
            Self::Execute(execution) => {
                write!(
                    f,
                    "Execute {pcode_instr}",
                    pcode_instr = execution.current_instruction()
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeInstruction {
    address: Address,
}

impl DecodeInstruction {
    pub fn new(address: Address) -> Self {
        Self { address }
    }

    pub fn address(&self) -> &Address {
        &self.address
    }

    pub fn decode<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: ExecutableMemory<M>,
    ) -> Result<PcodeExecution> {
        let pcode = sleigh
            .disassemble_pcode(&memory, self.address.clone())
            .map_err(Error::InstructionDecoding)?;

        Ok(PcodeExecution::new(pcode))
    }

    pub fn disassemble<M: VarnodeDataStore>(
        &self,
        sleigh: &impl Sleigh,
        memory: ExecutableMemory<M>,
    ) -> Result<NativeDisassembly> {
        let assembly = sleigh
            .disassemble_native(&memory, self.address.clone())
            .map_err(Error::InstructionDecoding)?;

        Ok(assembly)
    }
}

#[derive(Clone, Debug)]
pub struct PcodeExecution {
    pcode: PcodeDisassembly,
    index: usize,
}

impl PcodeExecution {
    pub fn new(pcode: PcodeDisassembly) -> Self {
        Self { pcode, index: 0 }
    }

    pub fn pcode(&self) -> &PcodeDisassembly {
        &self.pcode
    }

    pub fn origin(&self) -> &VarnodeData {
        &self.pcode.origin
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn current_instruction(&self) -> &PcodeInstruction {
        &self.pcode.instructions[self.index]
    }

    pub fn is_empty(&self) -> bool {
        self.pcode.instructions.is_empty()
    }

    pub fn is_final_instruction(&self) -> bool {
        self.index + 1 == self.pcode.instructions.len()
    }

    pub fn update_index(&mut self, offset: i64) -> Result<()> {
        // Index is always safe to convert to i64
        let index = i64::try_from(self.index).unwrap() + offset;
        let index = index.try_into().map_err(|err| {
            Error::InvalidArgument(format!(
                "Offset {offset} from index {index} is invalid: {err}"
            ))
        })?;

        if index >= self.pcode.instructions.len() {
            return Err(Error::InvalidArgument(format!(
                "Offset {offset} from index {index} exceeds pcode instruction length {len}",
                len = self.pcode.instructions.len(),
                index = self.index
            )));
        }

        self.index = index;
        Ok(())
    }

    pub fn index_from_offset(&self, offset: i64) -> Result<usize> {
        // Index is always safe to convert to i64
        let index = i64::try_from(self.index).unwrap() + offset;
        let index = index.try_into().map_err(|err| {
            Error::InvalidArgument(format!(
                "Offset {offset} from index {index} is invalid: {err}"
            ))
        })?;

        if index < self.pcode.instructions.len() {
            Ok(index)
        } else {
            Err(Error::InvalidArgument(format!(
                "Offset {offset} from index {index} exceeds pcode instruction length {len}",
                len = self.pcode.instructions.len(),
                index = self.index
            )))
        }
    }

    pub fn offset(self, offset: i64) -> Result<Self> {
        // Index is always safe to convert to i64
        let index = i64::try_from(self.index).unwrap() + offset;
        let index = index.try_into().map_err(|err| {
            Error::InvalidArgument(format!(
                "Offset {offset} from index {index} is invalid: {err}"
            ))
        })?;

        if index < self.pcode.instructions.len() {
            Ok(Self {
                pcode: self.pcode,
                index,
            })
        } else {
            Err(Error::InvalidArgument(format!(
                "Offset {offset} from index {index} exceeds pcode instruction length {len}",
                len = self.pcode.instructions.len(),
                index = self.index
            )))
        }
    }

    fn next_instruction(&self) -> NextExecution {
        if self.is_final_instruction() {
            NextExecution::NextInstruction
        } else {
            NextExecution::PcodeOffset(1)
        }
    }

    fn jump(&self, destination: &Destination) -> NextExecution {
        match destination {
            Destination::MachineAddress(address) => NextExecution::Jump(address.clone()),
            Destination::PcodeAddress(offset) => NextExecution::PcodeOffset(*offset),
        }
    }

    fn next_execution(&self, flow: ControlFlow) -> BranchingNextExecution {
        match flow {
            ControlFlow::NextInstruction
            | ControlFlow::ConditionalBranch {
                condition: Some(false),
                ..
            } => BranchingNextExecution::Flow(self.next_instruction()),
            ControlFlow::Jump(destination)
            | ControlFlow::ConditionalBranch {
                condition: Some(true),
                destination,
                ..
            } => BranchingNextExecution::Flow(self.jump(&destination)),
            ControlFlow::ConditionalBranch {
                condition_origin,
                destination,
                ..
            } => BranchingNextExecution::Branch(
                condition_origin,
                self.jump(&destination),
                self.next_instruction(),
            ),
            ControlFlow::Halt => BranchingNextExecution::Flow(NextExecution::Halt),
        }
    }
}
