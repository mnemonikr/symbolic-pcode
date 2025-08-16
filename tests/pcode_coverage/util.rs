use std::collections::BTreeMap;

use libsla::{OpCode, PcodeInstruction};
use symbolic_pcode::{
    emulator::{self, ControlFlow, PcodeEmulator, StandardPcodeEmulator},
    mem::VarnodeDataStore,
};

#[derive(Debug, Clone)]
pub struct TracingEmulator {
    inner: StandardPcodeEmulator,
    executed_instructions: std::cell::RefCell<BTreeMap<OpCode, usize>>,
}

impl PcodeEmulator for TracingEmulator {
    fn emulate<T: VarnodeDataStore>(
        &mut self,
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
                /*
                for instr_input in instruction.inputs.iter() {
                    let input_result = memory.read(instr_input);
                    let input_result = match input_result {
                        Ok(x) => PcodeValue::from(x),
                        Err(err) => {
                            println!("Failed to read input {instr_input}: {err}");
                            break;
                        }
                    };

                    match u128::try_from(input_result) {
                        Ok(value) => {
                            println!(
                                "Input {instr_input} = {value:0width$x}",
                                width = 2 * instr_input.size
                            );
                        }
                        Err(TryFromPcodeValueError::InvalidSize) => {
                            println!("Input {instr_input} = Large value")
                        }
                        Err(TryFromPcodeValueError::InvalidByte { index }) => {
                            println!("Input {instr_input} = Symbolic value @ {index}")
                        }
                    }
                }
                    */
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
