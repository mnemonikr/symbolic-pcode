use std::collections::BTreeMap;

use thiserror;

use crate::sym;
use crate::sym::{ConcretizationError, SymbolicBitVec};
use sla::{Address, AddressSpace, AddressSpaceType, VarnodeData};

type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("space id {0:0width$x} does not refer to a known address space", width = 2 * std::mem::size_of::<usize>())]
    UnknownAddressSpace(usize),

    #[error("address {address} address space is not {expected:?}")]
    InvalidAddressSpaceType {
        address: Address,
        expected: AddressSpaceType,
    },

    #[error("no data defined at address {0}")]
    UndefinedData(Address),

    #[error("address {0} has non-literal bit at index {1}")]
    SymbolicDataError(Address, usize),

    #[error("arguments provided are not valid: {0}")]
    InvalidArguments(String),

    #[error("an internal error occurred: {0}")]
    InternalError(String),
}

struct AddressSpaceData {
    address_space: AddressSpace,
    data: BTreeMap<usize, SymbolicBitVec>,
}

impl AddressSpaceData {
    pub fn new(address_space: AddressSpace) -> Self {
        Self {
            address_space,
            data: Default::default(),
        }
    }
}

pub struct Memory {
    data: BTreeMap<usize, AddressSpaceData>,
}

impl Memory {
    pub fn new(address_spaces: Vec<AddressSpace>) -> Self {
        Self {
            data: address_spaces
                .into_iter()
                .map(|space| (space.id, AddressSpaceData::new(space)))
                .collect(),
        }
    }

    pub fn read_bytes(&self, input: &VarnodeData) -> Result<Vec<&SymbolicBitVec>> {
        let space_id = input.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        let mut data = Vec::with_capacity(input.size as usize);
        for offset in input.range() {
            data.push(memory.data.get(&offset).ok_or_else(|| {
                Error::UndefinedData(Address {
                    offset,
                    address_space: input.address.address_space.clone(),
                })
            })?);
        }

        Ok(data)
    }

    pub fn read_concrete_value<T>(&self, input: &VarnodeData) -> Result<T>
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error,
    {
        // TODO Handle this result properly
        sym::concretize_bit_iter(
            self.read_bytes(&input)?
                .iter()
                .map(|byte| byte.iter())
                .flatten(),
        )
        .map_err(|err| match err {
            ConcretizationError::NonLiteralBit(index) => {
                Error::SymbolicDataError(input.address.clone(), index)
            }
            _ => Error::InternalError(format!("{}", err)),
        })
    }

    fn read_const(&self, input: &VarnodeData) -> Vec<SymbolicBitVec> {
        sym::SymbolicBitVec::constant(input.address.offset, 8 * input.size).into_parts(8)
    }

    pub fn read_bytes_owned(&self, input: &VarnodeData) -> Result<Vec<SymbolicBitVec>> {
        if input.address.address_space.space_type == AddressSpaceType::Constant {
            return Ok(self.read_const(&input));
        }

        let space_id = input.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        let mut data = Vec::with_capacity(input.size as usize);
        for offset in input.range() {
            data.push(
                memory
                    .data
                    .get(&offset)
                    .ok_or_else(|| {
                        Error::UndefinedData(Address {
                            offset,
                            address_space: input.address.address_space.clone(),
                        })
                    })
                    .map(SymbolicBitVec::clone)?,
            );
        }

        Ok(data)
    }

    pub fn write_bytes(&mut self, input: Vec<SymbolicBitVec>, output: &VarnodeData) -> Result<()> {
        if input.len() != output.size {
            return Err(Error::InvalidArguments(format!(
                "requested {} bytes to be written, provided {} bytes",
                output.size,
                input.len()
            )));
        }

        let space_id = output.address.address_space.id;
        let memory = self
            .data
            .get_mut(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        let mut offset = output.address.offset;
        for data in input.into_iter() {
            memory.data.insert(offset, data);
            offset += 1;
        }

        Ok(())
    }

    pub fn address_space(&self, input: &VarnodeData) -> Result<&AddressSpace> {
        if input.address.address_space.space_type != AddressSpaceType::Constant {
            return Err(Error::InvalidAddressSpaceType {
                expected: AddressSpaceType::Constant,
                address: input.address.clone(),
            });
        }

        let space_id = input.address.offset;
        self.data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))
            .map(|data| &data.address_space)
    }

    pub fn dump(&self) {
        for addr_space_data in self.data.values() {
            println!(
                "Begin memory dump for {}",
                addr_space_data.address_space.name
            );
            for key in addr_space_data.data.keys() {
                let byte: u8 = addr_space_data
                    .data
                    .get(key)
                    .unwrap()
                    .try_into()
                    .expect("failed to convert to byte");
                println!("{key:016x}: {byte:02x}");
            }
            println!("End memory dump for {}", addr_space_data.address_space.name);
        }
    }
}
