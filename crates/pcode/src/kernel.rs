use libsla::Sleigh;

use crate::emulator::{ControlFlow, Error, Result};
use crate::mem::VarnodeDataStore;

pub mod linux;

pub trait Kernel: Clone {
    fn syscall(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut impl VarnodeDataStore,
    ) -> Result<ControlFlow>;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct NoKernel {}

impl Kernel for NoKernel {
    fn syscall(
        &mut self,
        _sleigh: &impl Sleigh,
        _memory: &mut impl VarnodeDataStore,
    ) -> Result<ControlFlow> {
        Err(Error::InternalError("no kernel configured".to_string()))
    }
}
