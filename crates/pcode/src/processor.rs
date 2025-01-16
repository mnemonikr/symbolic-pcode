use thiserror;

use crate::emulator::{ControlFlow, Destination, PcodeEmulator};
use crate::mem::{ExecutableMemory, MemoryBranch, VarnodeDataStore};
use libsla::{
    Address, AddressSpace, AssemblyInstruction, Disassembly, GhidraSleigh, PcodeInstruction,
    Sleigh, VarnodeData,
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

    #[error("internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

enum NextExecution {
    NextInstruction,
    Jump(Address),
    PcodeOffset(i64),
}

enum BranchingNextExecution {
    Branch(VarnodeData, NextExecution, NextExecution),
    Flow(NextExecution),
}

pub struct ProcessorManager<
    E: PcodeEmulator + Clone,
    M: VarnodeDataStore + Default,
    H: ProcessorResponseHandler<M>,
> {
    processors: Vec<Processor<E, M, H>>,
    sleigh: GhidraSleigh,
}

impl<E: PcodeEmulator + Clone, M: VarnodeDataStore + Default, H: ProcessorResponseHandler<M>>
    ProcessorManager<E, M, H>
{
    pub fn new(sleigh: GhidraSleigh, processor: Processor<E, M, H>) -> Self {
        Self {
            sleigh,
            processors: vec![processor],
        }
    }

    pub fn sleigh(&self) -> &GhidraSleigh {
        &self.sleigh
    }

    pub fn step_all(&mut self) -> Result<()> {
        for i in 0..self.processors.len() {
            if let Some(new_processor) = self.processors[i].step(&self.sleigh)? {
                self.processors.push(new_processor);
            }
        }

        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    pub fn remove_all(
        &mut self,
        filter: impl Fn(&Processor<E, M, H>) -> bool,
    ) -> Vec<Processor<E, M, H>> {
        let mut removed = Vec::new();
        for i in (0..self.processors.len()).rev() {
            if filter(&self.processors[i]) {
                removed.push(self.processors.swap_remove(i));
            }
        }

        removed
    }

    pub fn processors(&self) -> impl Iterator<Item = &Processor<E, M, H>> {
        self.processors.iter()
    }

    pub fn processors_mut(&mut self) -> impl Iterator<Item = &mut Processor<E, M, H>> {
        self.processors.iter_mut()
    }
}

pub trait ProcessorResponseHandler<M: VarnodeDataStore + Default>: Clone {
    fn fetched(&mut self, memory: &mut MemoryBranch<M>) -> Result<Address>;
    fn decoded(&mut self, memory: &mut MemoryBranch<M>, execution: &PcodeExecution) -> Result<()>;
    fn jumped(&mut self, memory: &mut MemoryBranch<M>, address: &Address) -> Result<()>;
}

pub struct Processor<
    E: PcodeEmulator + Clone,
    M: VarnodeDataStore + Default,
    H: ProcessorResponseHandler<M> + Clone,
> {
    memory: MemoryBranch<M>,
    state: ProcessorState,
    handler: H,
    emulator: E,
}

impl<
        E: PcodeEmulator + Clone,
        M: VarnodeDataStore + Default,
        H: ProcessorResponseHandler<M> + Clone,
    > Processor<E, M, H>
{
    pub fn new(memory: M, emulator: E, handler: H) -> Self {
        Self {
            memory: MemoryBranch::new(memory),
            emulator,
            handler,
            state: ProcessorState::Fetch,
        }
    }

    pub fn memory(&self) -> &MemoryBranch<M> {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut MemoryBranch<M> {
        &mut self.memory
    }

    pub fn emulator(&self) -> &E {
        &self.emulator
    }

    pub fn state(&self) -> &ProcessorState {
        &self.state
    }

    pub fn disassemble(&self, sleigh: &impl Sleigh) -> Result<Disassembly<AssemblyInstruction>> {
        if let ProcessorState::Decode(d) = &self.state {
            d.disassemble(sleigh, ExecutableMemory(&self.memory))
        } else {
            Err(Error::InvalidArgument(format!(
                "cannot disassemble outside of decoding state: {state:?}",
                state = self.state
            )))
        }
    }

    pub fn step(&mut self, sleigh: &impl Sleigh) -> Result<Option<Self>> {
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
                match x.next_execution(control_flow)? {
                    BranchingNextExecution::Flow(e1) => self.update_execution(e1)?,
                    BranchingNextExecution::Branch(condition, e1, e2) => {
                        let mut branched_processor = self.branch(condition);
                        self.update_execution(e1)?;
                        branched_processor.update_execution(e2)?;
                        return Ok(Some(branched_processor));
                    }
                }
            }
        }

        Ok(None)
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
            }

            Ok(())
        } else {
            Err(Error::InternalError(format!(
                "cannot update execution, current state is {state:?}",
                state = self.state
            )))
        }
    }

    fn branch(&mut self, condition_origin: VarnodeData) -> Self {
        Processor {
            memory: self
                .memory
                .new_branch(self.memory.read_bit(&condition_origin).unwrap()),
            state: self.state.clone(),
            handler: self.handler.clone(),
            emulator: self.emulator.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ProcessorState {
    Fetch,
    Decode(DecodeInstruction),
    Execute(PcodeExecution),
}

#[derive(Clone, Debug)]
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
    ) -> Result<Disassembly<AssemblyInstruction>> {
        let assembly = sleigh
            .disassemble_native(&memory, self.address.clone())
            .map_err(Error::InstructionDecoding)?;

        Ok(assembly)
    }
}

#[derive(Clone, Debug)]
pub struct PcodeExecution {
    pcode: Disassembly<PcodeInstruction>,
    index: usize,
}

impl PcodeExecution {
    pub fn new(pcode: Disassembly<PcodeInstruction>) -> Self {
        Self { pcode, index: 0 }
    }

    pub fn origin(&self) -> &VarnodeData {
        &self.pcode.origin
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

    fn next_instruction(&self) -> Result<NextExecution> {
        if self.is_final_instruction() {
            Ok(NextExecution::NextInstruction)
        } else {
            Ok(NextExecution::PcodeOffset(1))
        }
    }

    fn jump(&self, destination: &Destination) -> Result<NextExecution> {
        match destination {
            Destination::MachineAddress(address) => Ok(NextExecution::Jump(address.clone())),
            Destination::PcodeAddress(offset) => Ok(NextExecution::PcodeOffset(*offset)),
        }
    }

    fn branch(
        &self,
        condition_origin: VarnodeData,
        destination: &Destination,
    ) -> Result<BranchingNextExecution> {
        Ok(BranchingNextExecution::Branch(
            condition_origin,
            self.jump(destination)?,
            self.next_instruction()?,
        ))
    }

    fn next_execution(&self, flow: ControlFlow) -> Result<BranchingNextExecution> {
        let result = match flow {
            ControlFlow::NextInstruction
            | ControlFlow::ConditionalBranch {
                condition: Some(false),
                ..
            } => BranchingNextExecution::Flow(self.next_instruction()?),
            ControlFlow::Jump(destination)
            | ControlFlow::ConditionalBranch {
                condition: Some(true),
                destination,
                ..
            } => BranchingNextExecution::Flow(self.jump(&destination)?),
            ControlFlow::ConditionalBranch {
                condition_origin,
                destination,
                ..
            } => self.branch(condition_origin, &destination)?,
        };

        Ok(result)
    }
}
