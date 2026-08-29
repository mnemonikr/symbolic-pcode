use libsla::Sleigh;

use crate::mem::VarnodeDataStore;
use crate::processor::{ControlFlowResult, Error};

pub mod linux;

pub trait Kernel: Clone {
    fn syscall(
        &mut self,
        sleigh: &impl Sleigh,
        memory: &mut impl VarnodeDataStore,
    ) -> ControlFlowResult;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct NoKernel {}

impl Kernel for NoKernel {
    fn syscall(
        &mut self,
        _sleigh: &impl Sleigh,
        _memory: &mut impl VarnodeDataStore,
    ) -> ControlFlowResult {
        Err(Error::InternalError("no kernel configured".to_string()))
    }
}
