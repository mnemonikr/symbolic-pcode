use std::fs;

use pcode::emulator::{ControlFlow, Destination, PcodeEmulator};
use sla::{Address, Sleigh, VarnodeData};
use sym::SymbolicBitVec;

pub struct Processor<'a> {
    sleigh: Sleigh<'a>,
    emulator: PcodeEmulator,
}

impl<'a> Processor<'a> {
    pub fn new() -> Self {
        let sleigh = x86_64_sleigh();
        let emulator = PcodeEmulator::new(sleigh.address_spaces());
        Processor { sleigh, emulator }
    }

    pub fn write_register(&mut self, register_name: impl AsRef<str>, data: impl AsRef<[u8]>) {
        let output = self.sleigh.register_from_name(register_name);
        self.write_data(output, data);
    }

    pub fn read_register<T: From<usize>>(&mut self, register_name: impl AsRef<str>) -> T {
        let input = self.sleigh.register_from_name(register_name);
        self.emulator
            .memory()
            .read_concrete_value::<T>(&input)
            .expect("failed to read value")
    }

    pub fn write_memory(&mut self, offset: u64, data: impl AsRef<[u8]>) {
        let ram = self
            .sleigh
            .address_spaces()
            .into_iter()
            .find(|addr_space| addr_space.name == "ram")
            .expect("failed to find ram");

        let output = VarnodeData {
            address: Address {
                address_space: ram,
                offset,
            },
            size: data.as_ref().len(),
        };

        self.write_data(output, data);
    }

    pub fn write_data(&mut self, output: VarnodeData, data: impl AsRef<[u8]>) {
        let bytes: Vec<SymbolicBitVec> = data
            .as_ref()
            .into_iter()
            .copied()
            .map(Into::<SymbolicBitVec>::into)
            .collect();

        self.emulator
            .memory_mut()
            .write_bytes(bytes, &output)
            .expect("failed to write data");
    }

    pub fn write_instructions(&mut self, base_address: u64, instructions: impl AsRef<[u8]>) {
        let output = VarnodeData {
            address: Address {
                offset: base_address,
                address_space: self.sleigh.default_code_space(),
            },
            size: instructions.as_ref().len(),
        };

        self.write_data(output, instructions);
    }

    pub fn emulate(&mut self, offset: u64) -> Result<u64, String> {
        let pcode = self
            .sleigh
            .pcode(std::ptr::NonNull::from(&self.emulator), offset as u64)?;
        let next_addr = offset + pcode.num_bytes_consumed as u64;
        println!("Bytes consumed: {}", pcode.num_bytes_consumed);
        println!("Next address: {:016x}", next_addr);

        for instruction in pcode.pcode_instructions {
            println!("Emulating {instruction}");
            let result = self.emulator.emulate(&instruction);
            // self.emulator.memory().dump();

            if let Err(err) = result {
                println!("Emulation error for instruction: {instruction}");
                return Err(err.to_string());
            }

            match result.unwrap() {
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

pub fn x86_64_sleigh<'a>() -> Sleigh<'a> {
    let mut sleigh = Sleigh::new();
    let sleigh_spec =
        fs::read_to_string("tests/data/x86-64.sla").expect("failed to read processor spec file");
    let processor_spec =
        fs::read_to_string("sla/ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
            .expect("failed to read processor spec file");
    sleigh
        .initialize(&sleigh_spec, &processor_spec)
        .expect("failed to initialize sleigh");
    sleigh
}
