use std::collections::BTreeMap;

use thiserror;

use sla::{Address, AddressSpace, AddressSpaceType, VarnodeData};
use sym::{self, ConcretizationError, SymbolicBitVec};

type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("space id {0:0width$x} does not refer to a known address space", width = 2 * std::mem::size_of::<usize>())]
    UnknownAddressSpace(usize),

    #[error("space id {space_id:0width$x} is not a valid address space id", width = 2 * std::mem::size_of::<u64>())]
    InvalidAddressSpaceId {
        space_id: u64,
        source: <u64 as TryFrom<usize>>::Error,
    },

    #[error("address space for {address} does not match expected: {expected:?}")]
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
    data: BTreeMap<u64, SymbolicBitVec>,
}

impl AddressSpaceData {
    pub fn new(address_space: AddressSpace) -> Self {
        Self {
            address_space,
            data: Default::default(),
        }
    }

    pub fn read_bytes(
        &self,
        range: impl std::ops::RangeBounds<u64>,
    ) -> impl Iterator<Item = (&u64, &SymbolicBitVec)> {
        self.data.range(range)
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

        // Collect into a Vec or return the first undefined offset
        let result: std::result::Result<Vec<&SymbolicBitVec>, &u64> = memory
            .read_bytes(input.range())
            .enumerate()
            .map(|(i, (offset, v))| {
                if *offset == i as u64 + input.address.offset {
                    Ok(v)
                } else {
                    // Undefined offset
                    Err(offset)
                }
            })
            .collect();

        result.map_err(|&offset| {
            Error::UndefinedData(Address {
                offset,
                address_space: input.address.address_space.clone(),
            })
        })
    }

    pub fn read_concrete_value<T>(&self, input: &VarnodeData) -> Result<T>
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error + 'static,
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
        SymbolicBitVec::constant(
            input.address.offset.try_into().unwrap_or_else(|err| {
                panic!(
                    "unable to represent {offset} as symbolic constant: {err}",
                    offset = input.address.offset
                )
            }),
            8 * input.size,
        )
        .into_parts(8)
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

        // Collect into a Vec or return the first undefined offset
        let result: std::result::Result<Vec<SymbolicBitVec>, &u64> = memory
            .read_bytes(input.range())
            .enumerate()
            .map(|(i, (offset, v))| {
                if *offset == i as u64 + input.address.offset {
                    Ok(v.clone())
                } else {
                    // Undefined offset
                    Err(offset)
                }
            })
            .collect();

        result.map_err(|&offset| {
            Error::UndefinedData(Address {
                offset,
                address_space: input.address.address_space.clone(),
            })
        })
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

        let space_id =
            input
                .address
                .offset
                .try_into()
                .map_err(|err| Error::InvalidAddressSpaceId {
                    space_id: input.address.offset,
                    source: err,
                })?;

        self.data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))
            .map(|data| &data.address_space)
    }

    /// Dumps the contents of memory to stdout.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn address_space(id: usize) -> AddressSpace {
        AddressSpace {
            id,
            name: String::from("test_space"),
            word_size: 1,
            address_size: 8, // 64-bit
            space_type: AddressSpaceType::Processor,
            big_endian: false,
        }
    }

    fn const_space() -> AddressSpace {
        AddressSpace {
            id: 0,
            name: String::from("constant_space"),
            word_size: 1,
            address_size: 8, // 64-bit
            space_type: AddressSpaceType::Constant,
            big_endian: false,
        }
    }

    #[test]
    fn read_and_write() -> Result<()> {
        let addr_space = address_space(0);
        let varnode = VarnodeData {
            address: Address {
                offset: 0,
                address_space: addr_space.clone(),
            },
            size: 8,
        };
        let value = SymbolicBitVec::constant(0x0123456789abcdef, 64).into_parts(8);

        // Read and write value into memory
        let mut memory = Memory::new(vec![addr_space]);
        memory.write_bytes(value.clone(), &varnode)?;
        let read_value = memory.read_bytes(&varnode)?;

        // Confirm read value matches written value
        assert_eq!(read_value.len(), value.len());
        for (expected, actual) in value.into_iter().zip(read_value.into_iter()) {
            assert_eq!(
                actual.clone().equals(expected),
                sym::SymbolicBit::Literal(true)
            );
        }

        Ok(())
    }
}
