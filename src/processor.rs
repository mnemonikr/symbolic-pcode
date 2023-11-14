use crate::emulator::{ControlFlow, Destination, PcodeEmulator};
use sla::{Address, OpCode, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};

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

    pub fn read_register<T>(&mut self, register_name: impl AsRef<str>) -> T
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error + 'static,
    {
        let input = self.sleigh.register_from_name(register_name);
        self.emulator
            .memory()
            .read_concrete_value::<T>(&input)
            .expect("failed to read value")
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

    pub fn write_instructions(&mut self, base_address: u64, instructions: impl AsRef<[u8]>) {
        let varnode = VarnodeData {
            address: Address {
                offset: base_address,
                address_space: self.sleigh.default_code_space(),
            },
            size: instructions.as_ref().len(),
        };
        self.write_concrete(varnode, instructions);
    }

    pub fn single_step(
        &mut self,
        instruction_register_name: impl AsRef<str>,
    ) -> crate::emulator::Result<()> {
        let rip: u64 = self.read_register(&instruction_register_name);
        let next_instr = self.emulate(rip)?;
        let next_instr: Vec<u8> = SymbolicBitVec::constant(new_rip.try_into().unwrap(), 64)
            .into_parts(8)
            .into_iter()
            .map(|byte| u8::try_from(byte))
            .collect()?;
        self.write_register_concrete(instruction_register_name, new_instruction)?;
        Ok(())
    }

    pub fn emulate(&mut self, offset: u64) -> crate::emulator::Result<u64> {
        let pcode = self.sleigh.pcode(&self.emulator, offset as u64)?;
        let next_addr = offset + pcode.num_bytes_consumed as u64;
        for instruction in pcode.pcode_instructions {
            println!("Emulating {instruction}");
            match self.emulator.emulate(&instruction)? {
                ControlFlow::Jump(destination) => match destination {
                    Destination::MachineAddress(addr) => {
                        assert_eq!(addr.address_space, self.sleigh.default_code_space());
                        return Ok(addr.offset);
                    }
                    Destination::PcodeAddress(offset) => {
                        todo!("handle p-code relative branch: {offset}")
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
                                    todo!("handle p-code relative branch: {offset}")
                                }
                            }
                        }
                    } else {
                        panic!("symbolic condition in branch: {condition:?}");
                    }
                }
                ControlFlow::NextInstruction => (),
            }
        }

        Ok(next_addr)
    }
}
