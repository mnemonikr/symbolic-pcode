use thiserror;

use crate::emulator::{ControlFlow, Destination, PcodeEmulator};
use crate::mem::{ExecutableMemory, SymbolicMemory, SymbolicMemoryReader, SymbolicMemoryWriter};
use sla::{Address, AddressSpace, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};

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
    InstructionDecoding(String),

    #[error("symbolic condition")]
    SymbolicCondition(SymbolicBit),

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

pub struct Processor<E: PcodeEmulator, M: SymbolicMemory> {
    sleigh: Sleigh,
    emulator: E,
    memory: ExecutableMemory<M>,
}

impl<E: PcodeEmulator, M: SymbolicMemory> Processor<E, M> {
    pub fn new(sleigh: Sleigh, emulator: E, memory: M) -> Self {
        Processor {
            memory: ExecutableMemory(memory),
            emulator,
            sleigh,
        }
    }

    pub fn sleigh(&self) -> &Sleigh {
        &self.sleigh
    }

    pub fn emulator(&self) -> &E {
        &self.emulator
    }

    pub fn write_register_concrete(
        &mut self,
        register_name: impl AsRef<str>,
        data: impl AsRef<[u8]>,
    ) -> crate::mem::Result<()> {
        let output = self.sleigh.register_from_name(register_name);
        self.write_concrete(output, data)
    }

    pub fn register(&self, name: impl AsRef<str>) -> VarnodeData {
        self.sleigh.register_from_name(name)
    }

    pub fn address_space(&self, name: impl AsRef<str>) -> Result<AddressSpace> {
        let name = name.as_ref();
        self.sleigh
            .address_spaces()
            .into_iter()
            .find(|addr_space| addr_space.name == name)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown address space: {name}")))
    }

    pub fn memory_mut(&mut self) -> &mut impl SymbolicMemory {
        &mut self.memory
    }

    pub fn read_register(&self, name: impl AsRef<str>) -> Result<Vec<SymbolicByte>> {
        let register = self.register(name);
        let result = self.memory.read(&register)?;
        Ok(result)
    }

    pub fn write_register(
        &mut self,
        name: impl AsRef<str>,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        let register = self.register(name);
        let result = self.memory.write(&register, data)?;
        Ok(result)
    }

    pub fn read_address_space(
        &self,
        name: impl AsRef<str>,
        offset: u64,
        size: usize,
    ) -> Result<Vec<SymbolicByte>> {
        let varnode = VarnodeData {
            address: Address {
                address_space: self.address_space(name)?,
                offset,
            },
            size,
        };

        let result = self.memory.read(&varnode)?;
        Ok(result)
    }

    pub fn write_address(
        &mut self,
        address: Address,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        let varnode = VarnodeData {
            address,
            size: data.len(),
        };
        self.memory.write(&varnode, data)?;
        Ok(())
    }

    pub fn write_concrete(
        &mut self,
        varnode: VarnodeData,
        data: impl AsRef<[u8]>,
    ) -> crate::mem::Result<()> {
        let bytes = data
            .as_ref()
            .into_iter()
            .copied()
            .map(Into::<SymbolicByte>::into);

        self.memory.write(&varnode, bytes)
    }

    pub fn default_code_space(&self) -> AddressSpace {
        self.sleigh.default_code_space()
    }

    pub fn write_instructions(
        &mut self,
        base_address: u64,
        instructions: impl AsRef<[u8]>,
    ) -> Result<()> {
        let varnode = VarnodeData {
            address: Address {
                offset: base_address,
                address_space: self.sleigh.default_code_space(),
            },
            size: instructions.as_ref().len(),
        };
        self.write_concrete(varnode, instructions)?;
        Ok(())
    }

    /// Emulates the current instruction in the instruction register. This assumes the instruction
    /// in the register is an address offset into the [Sleigh::default_code_space].
    ///
    /// If the next instruction is an address into a different address space, the error variant
    /// [Error::InvalidAddressSpace] is returned.
    pub fn single_step(&mut self, instruction_register_name: impl AsRef<str>) -> Result<()> {
        // TODO Better error here
        let rip = self
            .read_register(&instruction_register_name)?
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .map_err(|_| Error::InternalError("symbolic instruction address".into()))?;

        let next_instr = self.emulate(Address {
            offset: rip,
            address_space: self.sleigh.default_code_space(),
        })?;

        if self.sleigh.default_code_space() != next_instr.address_space {
            return Err(Error::InvalidAddressSpace {
                address: next_instr,
                expected: self.sleigh.default_code_space(),
            });
        }

        self.write_register(
            instruction_register_name,
            next_instr
                .offset
                .to_le_bytes()
                .into_iter()
                .map(SymbolicByte::from),
        )?;

        Ok(())
    }

    pub fn emulate(&mut self, address: Address) -> Result<Address> {
        let pcode = self
            .sleigh
            .pcode(&self.memory, &address)
            .map_err(|err| Error::InstructionDecoding(err))?;
        let next_addr = address.offset + pcode.num_bytes_consumed as u64;
        let mut instruction_index = 0;
        let max_index =
            i64::try_from(pcode.pcode_instructions.len()).expect("too many instructions");
        while (0..max_index).contains(&instruction_index) {
            // SAFETY: Index is already bound by size of array
            let instruction = unsafe {
                &pcode.pcode_instructions[usize::try_from(instruction_index).unwrap_unchecked()]
            };

            match self.emulator.emulate(&mut self.memory, &instruction)? {
                ControlFlow::NextInstruction => {
                    instruction_index += 1;
                }
                ControlFlow::Jump(destination) => match destination {
                    Destination::MachineAddress(addr) => {
                        return Ok(addr);
                    }
                    Destination::PcodeAddress(offset) => {
                        instruction_index += offset;
                    }
                },
                ControlFlow::ConditionalBranch(sym::SymbolicBit::Literal(true), destination) => {
                    match destination {
                        Destination::MachineAddress(addr) => {
                            assert_eq!(addr.address_space, self.sleigh.default_code_space());
                            return Ok(addr);
                        }
                        Destination::PcodeAddress(offset) => {
                            instruction_index += offset;
                        }
                    }
                }
                ControlFlow::ConditionalBranch(sym::SymbolicBit::Literal(false), _) => {
                    instruction_index += 1;
                }
                ControlFlow::ConditionalBranch(condition, _) => {
                    return Err(Error::SymbolicCondition(condition));
                }
            }
        }

        // TODO: This assumes that any instruction index outside the valid bounds means proceed to
        // the next instruction. For branchless pcode instructions this is correct. However if
        // there is a relative pcode branch that reaches out-of-bounds it is unclear what the
        // correct behavior is.
        Ok(Address {
            offset: next_addr,
            address_space: address.address_space,
        })
    }
}
