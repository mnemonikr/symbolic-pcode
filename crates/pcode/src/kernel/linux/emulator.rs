use std::rc::Rc;

use libsla::{OpCode, PcodeInstruction, PseudoOp, Sleigh};

use super::model::LinuxModel;
use crate::emulator::{ControlFlow, Error, PcodeEmulator, Result, StandardPcodeEmulator};
use crate::mem::VarnodeDataStore;

#[repr(u64)]
enum CallOtherOps {
    Segment = 0x00,
    In = 0x01,
    Out = 0x02,
    SysEnter = 0x03,
    SysExit = 0x04,
    SysCall = 0x05,
    Lock = 0x11,
}

#[derive(Debug)]
pub struct LinuxEmulator<S> {
    sleigh: Rc<S>,
    linux: LinuxModel,
    emulator: StandardPcodeEmulator,
}

impl<S> Clone for LinuxEmulator<S> {
    fn clone(&self) -> Self {
        Self {
            sleigh: self.sleigh.clone(),
            linux: self.linux.clone(),
            emulator: self.emulator.clone(),
        }
    }
}

impl<S: Sleigh> LinuxEmulator<S> {
    pub fn new(sleigh: Rc<S>) -> Self {
        Self {
            emulator: StandardPcodeEmulator::new(sleigh.address_spaces()),
            linux: Default::default(),
            sleigh,
        }
    }
}

impl<S: Sleigh> PcodeEmulator for LinuxEmulator<S> {
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
                        return self
                            .linux
                            .syscall(self.sleigh.as_ref(), memory)
                            .map_err(|err| Error::DependencyError(Box::new(err)));
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
