use std::rc::Rc;

use libsla::{OpCode, PcodeInstruction, PseudoOp, Sleigh};

use crate::emulator::{ControlFlow, Error, PcodeEmulator, Result, StandardPcodeEmulator};
use crate::kernel::model::{KernelModel, NoKernel};
use crate::mem::VarnodeDataStore;

#[repr(u64)]
enum CallOtherOps {
    _Segment = 0x00,
    _In = 0x01,
    _Out = 0x02,
    _SysEnter = 0x03,
    _SysExit = 0x04,
    SysCall = 0x05,
    Lock = 0x11,
}

#[derive(Debug)]
pub struct EmulatorX86<S, K = NoKernel>
where
    S: Sleigh,
    K: KernelModel,
{
    sleigh: Rc<S>,
    kernel: K,
    emulator: StandardPcodeEmulator,
}

impl<S: Sleigh, K: KernelModel + Clone> Clone for EmulatorX86<S, K> {
    fn clone(&self) -> Self {
        Self {
            sleigh: self.sleigh.clone(),
            kernel: self.kernel.clone(),
            emulator: self.emulator.clone(),
        }
    }
}

impl<S: Sleigh, K: KernelModel + Default> EmulatorX86<S, K> {
    pub fn new(sleigh: Rc<S>) -> Self {
        Self {
            emulator: StandardPcodeEmulator::new(sleigh.address_spaces()),
            kernel: Default::default(),
            sleigh,
        }
    }
}

impl<S: Sleigh, K: KernelModel> EmulatorX86<S, K> {
    pub fn with_kernel(sleigh: Rc<S>, kernel: K) -> Self {
        Self {
            emulator: StandardPcodeEmulator::new(sleigh.address_spaces()),
            kernel,
            sleigh,
        }
    }
}

impl<S: Sleigh, K: KernelModel> PcodeEmulator for EmulatorX86<S, K> {
    fn emulate<M: VarnodeDataStore>(
        &mut self,
        memory: &mut M,
        instruction: &PcodeInstruction,
    ) -> Result<ControlFlow> {
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
                    // TODO This is dependent on the x86 sleigh implementation
                    // This should be made more generic around x86 and then support supplying
                    // different kernels instead of treating this as a Linux emulator specifically
                    Some(x) if x == CallOtherOps::SysCall as u64 => {
                        return self.kernel.syscall(self.sleigh.as_ref(), memory)
                    }
                    Some(x) if x == CallOtherOps::Lock as u64 => {
                        // Lock instruction. Multithreading not supported so just ignore
                        return Ok(ControlFlow::NextInstruction);
                    }
                    _ => (),
                }
            }
        }

        result
    }
}
