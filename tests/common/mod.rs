use std::{collections::BTreeMap, fs};

use sla::{Address, OpCode, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};
use symbolic_pcode::emulator::{ControlFlow, Destination, PcodeEmulator, StandardPcodeEmulator};
use symbolic_pcode::mem::{Memory, SymbolicMemory};

pub struct Processor {
    sleigh: Sleigh,
    emulator: StandardPcodeEmulator,
    memory: Memory,
    executed_instructions: BTreeMap<OpCode, usize>,
}

impl Processor {
    pub fn new() -> Self {
        let sleigh = x86_64_sleigh();
        let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
        let memory = Memory::new(sleigh.address_spaces());

        Processor {
            sleigh,
            emulator,
            memory,
            executed_instructions: Default::default(),
        }
    }

    pub fn init_registers(&mut self) {
        let mut bitvar = 0;
        let registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI"]
            .into_iter()
            .map(str::to_owned)
            .chain((8..16).map(|n| format!("R{n}")))
            .collect::<Vec<_>>();

        for register in registers {
            let output = self.sleigh.register_from_name(register);

            let mut bytes = Vec::with_capacity(8);
            for _ in 0..8 {
                let byte: SymbolicByte = [
                    SymbolicBit::Variable(bitvar),
                    SymbolicBit::Variable(bitvar + 1),
                    SymbolicBit::Variable(bitvar + 2),
                    SymbolicBit::Variable(bitvar + 3),
                    SymbolicBit::Variable(bitvar + 4),
                    SymbolicBit::Variable(bitvar + 5),
                    SymbolicBit::Variable(bitvar + 6),
                    SymbolicBit::Variable(bitvar + 7),
                ]
                .into();
                bytes.push(byte);
                bitvar += 8;
            }

            self.memory
                .write(&output, bytes)
                .expect("failed to write data");
        }
    }

    pub fn write_register(&mut self, register_name: impl AsRef<str>, data: impl AsRef<[u8]>) {
        let output = self.sleigh.register_from_name(register_name);
        self.write_data(output, data);
    }

    pub fn read_register<T>(&mut self, register_name: impl AsRef<str>) -> T
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error + 'static,
    {
        let input = self.sleigh.register_from_name(register_name);
        self.memory
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
        let bytes = data
            .as_ref()
            .into_iter()
            .copied()
            .map(Into::<SymbolicByte>::into)
            .collect::<Vec<_>>();

        self.memory
            .write(&output, bytes)
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

    pub fn executed_instructions(&self) -> impl Iterator<Item = (OpCode, usize)> + '_ {
        self.executed_instructions
            .iter()
            .map(|(&op, &count)| (op, count))
    }

    pub fn single_step(&mut self) {
        let rip: u64 = self.read_register("RIP");
        let address = Address {
            offset: rip,
            address_space: self.sleigh.default_code_space(),
        };
        let new_rip = self
            .emulate(address)
            .unwrap_or_else(|err| panic!("failed to emulate {rip:#02x}: {err}"));
        let new_rip: Vec<u8> = SymbolicBitVec::constant(new_rip.try_into().unwrap(), 64)
            .into_parts(8)
            .into_iter()
            .map(|byte| u8::try_from(byte).expect("failed byte conversion"))
            .collect();
        self.write_register("RIP", new_rip);
    }

    fn emulate(&mut self, instruction_address: Address) -> Result<u64, String> {
        let pcode = self.sleigh.pcode(&self.memory, &instruction_address)?;
        let next_addr = instruction_address.offset + pcode.num_bytes_consumed as u64;

        if pcode.pcode_instructions.len() == 0 {
            println!("NOOP [{instruction_address}]");
        }

        for instruction in pcode.pcode_instructions {
            println!("Emulating {instruction}");
            let result = self.emulator.emulate(&mut self.memory, &instruction);
            *self
                .executed_instructions
                .entry(instruction.op_code.into())
                .or_default() += 1;
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

pub fn x86_64_sleigh() -> Sleigh {
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
