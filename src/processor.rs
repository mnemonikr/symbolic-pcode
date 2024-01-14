use thiserror;

use crate::emulator::{ControlFlow, Destination, PcodeEmulator};
use sla::{Address, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicByte};

// TODO Emulator can also have memory access errors. Probably better to write a custom
// derivation that converts emulator errors into processor errors.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Error occurred while accessing a memory location
    #[error(transparent)]
    Emulation(#[from] crate::emulator::Error),

    /// Error occurred while accessing a memory location
    #[error(transparent)]
    MemoryAccess(#[from] crate::mem::Error),

    #[error("failed to decode instruction: {0}")]
    InstructionDecoding(String),

    #[error("symbolic condition")]
    SymbolicCondition(SymbolicBit),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Processor {
    sleigh: Sleigh,
    emulator: PcodeEmulator,
}

impl Processor {
    pub fn new(sleigh: Sleigh) -> Self {
        Processor {
            emulator: PcodeEmulator::new(sleigh.address_spaces()),
            sleigh,
        }
    }

    pub fn write_register_concrete(
        &mut self,
        register_name: impl AsRef<str>,
        data: impl AsRef<[u8]>,
    ) -> crate::mem::Result<()> {
        let output = self.sleigh.register_from_name(register_name);
        self.write_concrete(output, data)
    }

    pub fn read_register<T>(&mut self, register_name: impl AsRef<str>) -> Result<T>
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error + 'static,
    {
        let input = self.sleigh.register_from_name(register_name);
        let value = self.emulator.memory().read_concrete_value::<T>(&input)?;
        Ok(value)
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
            .map(Into::<SymbolicByte>::into)
            .collect::<Vec<_>>();

        self.emulator.memory_mut().write_bytes(bytes, &varnode)
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

    pub fn single_step(&mut self, instruction_register_name: impl AsRef<str>) -> Result<()> {
        let rip: u64 = self.read_register(&instruction_register_name)?;
        let next_instr = self.emulate(rip)?;
        self.write_register_concrete(instruction_register_name, next_instr.to_le_bytes())?;
        Ok(())
    }

    pub fn emulate(&mut self, offset: u64) -> Result<u64> {
        let pcode = self
            .sleigh
            .pcode(&self.emulator, offset as u64)
            .map_err(|err| Error::InstructionDecoding(err))?;
        let next_addr = offset + pcode.num_bytes_consumed as u64;
        let mut instruction_index = 0;
        let max_index =
            i64::try_from(pcode.pcode_instructions.len()).expect("too many instructions");
        while (0..max_index).contains(&instruction_index) {
            // SAFETY: Index is already bound by size of array
            let instruction = unsafe {
                &pcode.pcode_instructions[usize::try_from(instruction_index).unwrap_unchecked()]
            };

            match self.emulator.emulate(&instruction)? {
                ControlFlow::Jump(destination) => match destination {
                    Destination::MachineAddress(addr) => {
                        assert_eq!(addr.address_space, self.sleigh.default_code_space());
                        return Ok(addr.offset);
                    }
                    Destination::PcodeAddress(offset) => {
                        instruction_index += offset;
                    }
                },
                ControlFlow::ConditionalBranch(condition, destination) => {
                    if let sym::SymbolicBit::Literal(condition) = condition {
                        if condition {
                            match destination {
                                Destination::MachineAddress(addr) => {
                                    assert_eq!(
                                        addr.address_space,
                                        self.sleigh.default_code_space()
                                    );
                                    return Ok(addr.offset);
                                }
                                Destination::PcodeAddress(offset) => {
                                    instruction_index += offset;
                                }
                            }
                        }
                    } else {
                        return Err(Error::SymbolicCondition(condition));
                    }
                }
                ControlFlow::NextInstruction => {
                    instruction_index += 1;
                }
            }
        }

        // TODO: This assumes that any instruction index outside the valid bounds means proceed to
        // the next instruction. For branchless pcode instructions this is correct. However if
        // there is a relative pcode branch that reaches out-of-bounds it is unclear what the
        // correct behavior is.
        Ok(next_addr)
    }
}
