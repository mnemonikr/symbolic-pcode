use std::collections::BTreeMap;
use std::ops::ControlFlow;

use symbolic_pcode::libsla::{OpCode, PcodeInstruction};
use symbolic_pcode::{
    emulator::BranchResult,
    mem::VarnodeDataStore,
    processor::{ControlFlowResult, EmulatorHandler, default_after_emulate},
};

#[derive(Debug, Clone)]
pub struct EmulatorTraceHandler {
    executed_instructions: std::cell::RefCell<BTreeMap<OpCode, usize>>,
}

impl EmulatorHandler for EmulatorTraceHandler {
    fn before_emulate<T: VarnodeDataStore>(
        &mut self,
        _memory: &mut T,
        instruction: &PcodeInstruction,
    ) -> ControlFlowResult {
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

        Ok(ControlFlow::Continue(()))
    }

    fn after_emulate<M: VarnodeDataStore>(
        &mut self,
        _memory: &mut M,
        instruction: &PcodeInstruction,
        result: BranchResult,
    ) -> ControlFlowResult {
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
        default_after_emulate(result)
    }
}

impl EmulatorTraceHandler {
    pub fn new() -> Self {
        Self {
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
