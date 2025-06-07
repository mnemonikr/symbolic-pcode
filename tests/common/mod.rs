use std::{collections::BTreeMap, fs};

use libsla::{Address, AddressSpace, GhidraSleigh, OpCode, PcodeInstruction, Sleigh, VarnodeData};
use pcode_ops::PcodeOps;
use sym::SymbolicBitVec;
use symbolic_pcode::emulator::{self, ControlFlow, PcodeEmulator, StandardPcodeEmulator};
use symbolic_pcode::mem::{GenericMemory, MemoryBranch, VarnodeDataStore};
use symbolic_pcode::processor::{self, PcodeExecution, ProcessorResponseHandler};

pub type Memory = GenericMemory<SymbolicBitVec>;

#[derive(Debug, Clone)]
pub struct ProcessorHandlerX86 {
    rip: VarnodeData,
    instruction_address_space: AddressSpace,
}

impl ProcessorHandlerX86 {
    pub fn new(sleigh: &impl Sleigh) -> Self {
        let rip = sleigh
            .register_from_name("RIP")
            .expect("failed to get RIP register");
        let instruction_address_space = sleigh.default_code_space();
        Self {
            rip,
            instruction_address_space,
        }
    }
}

impl ProcessorResponseHandler<Memory> for ProcessorHandlerX86 {
    fn fetched(&mut self, memory: &mut MemoryBranch<Memory>) -> processor::Result<Address> {
        let offset = memory.read(&self.rip)?.try_into().map_err(|err| {
            processor::Error::InternalError(format!(
                "failed to convert instruction to concrete address: {err:?}"
            ))
        })?;
        Ok(Address::new(self.instruction_address_space.clone(), offset))
    }

    fn decoded(
        &mut self,
        memory: &mut MemoryBranch<Memory>,
        execution: &PcodeExecution,
    ) -> processor::Result<()> {
        let rip_value: u64 = memory.read(&self.rip)?.try_into().map_err(|err| {
            processor::Error::InternalError(format!(
                "failed to convert instruction to concrete address: {err:?}"
            ))
        })?;
        let bytes_read = u64::try_from(execution.origin().size).map_err(|err| {
            processor::Error::InternalError(format!(
                "RIP value {rip_value:016x} failed to convert to u64: {err}",
            ))
        })?;
        let rip_value = rip_value.checked_add(bytes_read).ok_or_else(|| {
            processor::Error::InternalError(format!(
                "RIP {rip_value:016x} + {len} overflowed",
                len = execution.origin().size
            ))
        })?;

        memory.write(&self.rip, rip_value.into())?;

        Ok(())
    }

    fn jumped(
        &mut self,
        memory: &mut MemoryBranch<Memory>,
        address: &Address,
    ) -> processor::Result<()> {
        memory.write(&self.rip, address.offset.into())?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TracingEmulator {
    inner: StandardPcodeEmulator,
    executed_instructions: std::cell::RefCell<BTreeMap<OpCode, usize>>,
}

impl PcodeEmulator for TracingEmulator {
    fn emulate<T: VarnodeDataStore>(
        &self,
        memory: &mut T,
        instruction: &PcodeInstruction,
    ) -> emulator::Result<ControlFlow> {
        //println!("Executing: {instruction}");
        match &instruction.op_code {
            OpCode::Store => (),
            OpCode::Branch
            | OpCode::BranchIndirect
            | OpCode::BranchConditional
            | OpCode::Call
            | OpCode::CallIndirect
            | OpCode::Return => (),
            _ => {
                for instr_input in instruction.inputs.iter() {
                    let input_result = memory.read(instr_input).unwrap();
                    let num_bytes = input_result.num_bytes();

                    const U128_BYTES: usize = const { (u128::BITS / u8::BITS) as usize };
                    let input_result = if input_result.num_bytes() < U128_BYTES {
                        input_result.zero_extend(U128_BYTES)
                    } else {
                        input_result
                    };

                    let input_bytes = input_result
                        .into_le_bytes()
                        .map(|byte| byte.try_into())
                        .collect::<Result<Vec<u8>, _>>();
                    if let Ok(input_bytes) = input_bytes {
                        let byte_array: Result<[u8; U128_BYTES], _> = input_bytes.try_into();
                        if let Ok(bytes) = byte_array {
                            let input = u128::from_le_bytes(bytes);
                            println!(
                                "Input {instr_input} = {input:0width$x}",
                                width = 2 * instr_input.size
                            );
                        } else {
                            println!("Input {instr_input} = Large value");
                        }
                    } else {
                        println!("Input {instr_input} = Symbolic value");
                    }
                    //<<T as VarnodeDataStore>::Value as TryInto<u128>>::try_into(input_result);
                    /*if let Ok(input) = input {
                        println!(
                            "Input {instr_input} = {input:0width$x}",
                            width = 2 * instr_input.size
                        );
                    } else if num_bytes > 16 {
                        println!("Input {instr_input} = Large value");
                    } else {
                        println!("Input {instr_input} = Symbolic value");
                    }*/
                }
            }
        };

        let result = self.inner.emulate(memory, instruction)?;

        /*
        match &instruction.op_code {
            OpCode::Store => println!("Store"),
            OpCode::Branch
            | OpCode::BranchIndirect
            | OpCode::BranchConditional
            | OpCode::Call
            | OpCode::CallIndirect
            | OpCode::Return => {
                println!("Branch: {result:?}")
            }
            _ => {
                let output_result = memory.read(instruction.output.as_ref().unwrap()).unwrap();
                let output =
                    <<T as VarnodeDataStore>::Value as TryInto<u64>>::try_into(output_result);
                if let Ok(output) = output {
                    println!(
                        "Output: {output:0width$x}",
                        width = 2 * instruction.output.as_ref().unwrap().size
                    );
                } else {
                    println!("Output: Symbolic");
                }
            }
        };
        */

        *self
            .executed_instructions
            .borrow_mut()
            .entry(instruction.op_code)
            .or_default() += 1;
        Ok(result)
    }
}

impl TracingEmulator {
    pub fn new(inner: StandardPcodeEmulator) -> Self {
        Self {
            inner,
            executed_instructions: Default::default(),
        }
    }

    pub fn executed_instructions(&self) -> BTreeMap<OpCode, usize> {
        self.executed_instructions
            .borrow()
            .iter()
            .map(|(&op, &count)| (op, count))
            .collect()
    }
}

pub fn x86_64_sleigh() -> libsla::Result<GhidraSleigh> {
    let sleigh_spec =
        fs::read_to_string("data/x86-64.sla").expect("failed to read processor spec file");
    let processor_spec = fs::read_to_string(
        "../crates/libsla/ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec",
    )
    .expect("failed to read processor spec file");

    GhidraSleigh::builder()
        .sleigh_spec(&sleigh_spec)?
        .processor_spec(&processor_spec)?
        .build()
}
