use std::rc::Rc;

use libsla::{OpCode, PcodeInstruction, PseudoOp, Sleigh};
use pcode_ops::PcodeOps;
use pcode_ops::convert::{PcodeValue, TryFromPcodeValueError};

use crate::emulator::{ControlFlow, Error, PcodeEmulator, Result, StandardPcodeEmulator};
use crate::kernel::{Kernel, NoKernel};
use crate::mem::VarnodeDataStore;

#[repr(u64)]
enum CallOtherOps {
    SupervisorCall = 0x19,
    DataMemoryBarrier = 0x1f,
    ExclusiveMonitorPass = 0x3f,
    ExclusiveMonitorsStatus = 0x40,
}

#[derive(Debug)]
pub struct Emulator<S, K = NoKernel>
where
    S: Sleigh,
    K: Kernel,
{
    sleigh: Rc<S>,
    kernel: K,
    emulator: StandardPcodeEmulator,
}

impl<S: Sleigh, K: Kernel + Clone> Clone for Emulator<S, K> {
    fn clone(&self) -> Self {
        Self {
            sleigh: self.sleigh.clone(),
            kernel: self.kernel.clone(),
            emulator: self.emulator.clone(),
        }
    }
}

impl<S: Sleigh> Emulator<S, NoKernel> {
    /// Create an emulator without a kernel. Syscalls will trigger [Error::UnsupportedInstruction].
    pub fn without_kernel(sleigh: Rc<S>) -> Self {
        Self {
            emulator: StandardPcodeEmulator::new(sleigh.address_spaces()),
            kernel: Default::default(),
            sleigh,
        }
    }
}

impl<S: Sleigh, K: Kernel> Emulator<S, K> {
    /// Create an emulator with syscalls implemented by the given kernel
    pub fn with_kernel(sleigh: Rc<S>, kernel: K) -> Self {
        Self {
            emulator: StandardPcodeEmulator::new(sleigh.address_spaces()),
            kernel,
            sleigh,
        }
    }

    /// Get a reference to the kernel
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

impl<S: Sleigh, K: Kernel> PcodeEmulator for Emulator<S, K> {
    fn emulate<M: VarnodeDataStore>(
        &mut self,
        memory: &mut M,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
        println!("Executing: {instruction}");
        for instr_input in instruction.inputs.iter() {
            let input = PcodeValue::from(memory.read(instr_input).unwrap());
            match u128::try_from(input) {
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
        let result = self.emulator.emulate(memory, instruction);

        if let Err(Error::UnsupportedInstruction { instruction }) = &result {
            if instruction.op_code == OpCode::Pseudo(PseudoOp::CallOther) {
                let arg = instruction.inputs.first().and_then(|input| {
                    if input.address.address_space.is_constant() {
                        Some(input.address.offset)
                    } else {
                        None
                    }
                });

                match arg {
                    Some(x) if x == CallOtherOps::SupervisorCall as u64 => {
                        return self.kernel.syscall(self.sleigh.as_ref(), memory);
                    }
                    Some(x) if x == CallOtherOps::ExclusiveMonitorPass as u64 => {
                        if let Some(output) = instruction.output.as_ref() {
                            // 1 Indicates that the monitor passes and the register can be written
                            // See usage in AARCH64base.sinc
                            memory.write(output, M::Value::from_le(1u8))?;
                        }
                        return Ok(ControlFlow::NextInstruction);
                    }
                    Some(x) if x == CallOtherOps::ExclusiveMonitorsStatus as u64 => {
                        if let Some(output) = instruction.output.as_ref() {
                            // 0 on success
                            // See usage in AARCH64base.sinc
                            memory.write(output, M::Value::from_le(0u8))?;
                        }
                        return Ok(ControlFlow::NextInstruction);
                    }
                    Some(x) if x == CallOtherOps::DataMemoryBarrier as u64 => {
                        return Ok(ControlFlow::NextInstruction);
                    }
                    _ => (),
                }
            }
        }

        if result.is_ok() {
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
                OpCode::Pseudo(x) => {
                    println!("PseudoCall: {x:?}")
                }
                _ => {
                    let output_result = memory.read(instruction.output.as_ref().unwrap()).unwrap();
                    let output = PcodeValue::from(output_result);
                    let output: std::result::Result<u128, _> = output.try_into();
                    if let Ok(output) = output {
                        println!(
                            "Output: {output:0width$x}",
                            width = 2 * instruction.output.as_ref().unwrap().size
                        );
                    } else {
                        println!("Output: Symbolic");
                    }
                }
            }
        }

        result
    }
}
