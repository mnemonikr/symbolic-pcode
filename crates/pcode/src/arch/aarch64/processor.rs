use libsla::{Address, AddressSpace, Sleigh, VarnodeData};

use crate::mem::VarnodeDataStore;
use crate::processor::{Error, PcodeExecution, ProcessorResponseHandler, Result};
use pcode_ops::{PcodeOps, convert::PcodeValue};

#[derive(Debug, Clone)]
pub struct ProcessorHandler {
    pc: VarnodeData,
    instruction_address_space: AddressSpace,
}

impl ProcessorHandler {
    pub fn new(sleigh: &impl Sleigh) -> Self {
        let pc = sleigh
            .register_from_name("pc")
            .expect("failed to get pc register");
        let instruction_address_space = sleigh.default_code_space();
        Self {
            pc,
            instruction_address_space,
        }
    }
}

impl ProcessorResponseHandler for ProcessorHandler {
    fn fetched<M: VarnodeDataStore>(&mut self, memory: &mut M) -> Result<Address> {
        let offset = PcodeValue::from(memory.read(&self.pc)?)
            .try_into()
            .map_err(|err| {
                Error::InternalError(format!(
                    "failed to convert instruction to concrete address: {err:?}"
                ))
            })?;
        Ok(Address::new(self.instruction_address_space.clone(), offset))
    }

    fn decoded<M: VarnodeDataStore>(
        &mut self,
        memory: &mut M,
        execution: &PcodeExecution,
    ) -> Result<()> {
        let pc_value: u64 = PcodeValue::from(memory.read(&self.pc)?)
            .try_into()
            .map_err(|err| {
                Error::InternalError(format!(
                    "failed to convert instruction to concrete address: {err:?}"
                ))
            })?;
        let bytes_read = u64::try_from(execution.origin().size).map_err(|err| {
            Error::InternalError(format!(
                "RIP value {pc_value:016x} failed to convert to u64: {err}",
            ))
        })?;
        let pc_value = pc_value.checked_add(bytes_read).ok_or_else(|| {
            Error::InternalError(format!(
                "RIP {pc_value:016x} + {len} overflowed",
                len = execution.origin().size
            ))
        })?;

        memory.write(&self.pc, M::Value::from_le(pc_value))?;

        Ok(())
    }

    fn jumped<M: VarnodeDataStore>(&mut self, memory: &mut M, address: &Address) -> Result<()> {
        memory.write(&self.pc, M::Value::from_le(address.offset))?;
        Ok(())
    }
}
