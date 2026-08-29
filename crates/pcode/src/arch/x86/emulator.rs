use std::ops::ControlFlow;
use std::rc::Rc;

use libsla::{OpCode, PcodeInstruction, PseudoOp, Sleigh};

use crate::emulator::{BranchResult, Error};
use crate::kernel::{Kernel, NoKernel};
use crate::mem::VarnodeDataStore;
use crate::processor::{ControlFlowResult, EmulatorHandler, default_after_emulate};

#[repr(u64)]
enum CallOtherOps {
    _Segment = 0x00,
    _In = 0x01,
    _Out = 0x02,
    _SysEnter = 0x03,
    _SysExit = 0x04,
    SysCall = 0x05,
    Lock = 0x11,
    Unlock = 0x12,
}

#[derive(Debug)]
pub struct EmulatorX86<S, K = NoKernel>
where
    S: Sleigh,
    K: Kernel,
{
    sleigh: Rc<S>,
    kernel: K,
}

impl<S: Sleigh, K: Kernel + Clone> Clone for EmulatorX86<S, K> {
    fn clone(&self) -> Self {
        Self {
            sleigh: self.sleigh.clone(),
            kernel: self.kernel.clone(),
        }
    }
}

impl<S: Sleigh> EmulatorX86<S, NoKernel> {
    /// Create an emulator without a kernel. Syscalls will trigger [Error::UnsupportedInstruction].
    pub fn without_kernel(sleigh: Rc<S>) -> Self {
        Self {
            kernel: Default::default(),
            sleigh,
        }
    }
}

impl<S: Sleigh, K: Kernel> EmulatorX86<S, K> {
    /// Create an emulator with syscalls implemented by the given kernel
    pub fn with_kernel(sleigh: Rc<S>, kernel: K) -> Self {
        Self { kernel, sleigh }
    }

    /// Get a reference to the kernel
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
}

impl<S: Sleigh, K: Kernel> EmulatorHandler for EmulatorX86<S, K> {
    fn after_emulate<M: VarnodeDataStore>(
        &mut self,
        memory: &mut M,
        _instruction: &PcodeInstruction,
        result: BranchResult,
    ) -> ControlFlowResult {
        if let Err(Error::UnsupportedInstruction { instruction }) = &result
            && instruction.op_code == OpCode::Pseudo(PseudoOp::CallOther)
        {
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
                    return self.kernel.syscall(self.sleigh.as_ref(), memory);
                }
                Some(x) if x == CallOtherOps::Lock as u64 => {
                    // Lock instruction. Multithreading not supported so just ignore
                    return Ok(ControlFlow::Continue(()));
                }
                Some(x) if x == CallOtherOps::Unlock as u64 => {
                    // Unlock instruction. Multithreading not supported so just ignore
                    return Ok(ControlFlow::Continue(()));
                }
                _ => (),
            }
        }

        default_after_emulate(result)
    }
}
