use std::{collections::BTreeMap, fs};

use sla::{Address, OpCode, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};
use symbolic_pcode::emulator::{
    self, ControlFlow, Destination, PcodeEmulator, StandardPcodeEmulator,
};
use symbolic_pcode::mem::{ExecutableMemory, Memory, SymbolicMemory, SymbolicMemoryWriter};

pub struct TracingEmulator {
    inner: StandardPcodeEmulator,
    executed_instructions: std::cell::RefCell<BTreeMap<OpCode, usize>>,
}

impl PcodeEmulator for TracingEmulator {
    fn emulate(
        &self,
        memory: &mut impl SymbolicMemory,
        instruction: &sla::PcodeInstruction,
    ) -> emulator::Result<ControlFlow> {
        let result = self.inner.emulate(memory, instruction)?;
        *self
            .executed_instructions
            .borrow_mut()
            .entry(instruction.op_code.into())
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

    pub fn executed_instructions(&self) -> impl IntoIterator<Item = (OpCode, usize)> {
        self.executed_instructions
            .borrow()
            .iter()
            .map(|(&op, &count)| (op, count))
            .collect::<Vec<_>>()
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
