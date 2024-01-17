use std::{collections::BTreeMap, rc::Rc};

use thiserror;

use sla::{Address, AddressSpace, AddressSpaceType, VarnodeData};
use sym::{self, ConcretizationError, SymbolicBit, SymbolicBitVec, SymbolicByte};

pub type Result<T> = std::result::Result<T, Error>;

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
    UnexpectedSymbolicData(Address, usize),

    #[error("arguments provided are not valid: {0}")]
    InvalidArguments(String),

    #[error("an internal error occurred: {0}")]
    InternalError(String),
}

/// A branching memory model which branches on `SymbolicBit` predicates. Note that this does not
/// enforce the predicate. It simply tracks the assumptions made at various branches.
pub struct MemoryTree {
    predicate: SymbolicBit,
    parent: Option<Rc<Self>>,
    memory: Memory,
}

/// A memory model that stores bytes associated with an AddressSpace.
pub struct Memory {
    /// Structure for looking up data based on the id of an AddressSpace.
    data: BTreeMap<usize, AddressSpaceData>,
}

impl Memory {
    /// Create a new instance of memory for the provided AddressSpaces.
    pub fn new(address_spaces: Vec<AddressSpace>) -> Self {
        Self {
            data: address_spaces
                .into_iter()
                .map(|space| (space.id, AddressSpaceData::new(space)))
                .collect(),
        }
    }

    /// Read the bytes from the addresses specified by the varnode. This function returns `Ok` if
    /// and only if data is successfully read from the requested addresses.
    ///
    /// The values returned here are references. For owned values see [Self::read_bytes_owned].
    pub fn read(&self, varnode: &VarnodeData) -> Result<Vec<&SymbolicByte>> {
        let space_id = varnode.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        // Collect into a Vec or return the first undefined offset
        let result = memory
            .read_bytes(varnode.range())
            .enumerate()
            .map(|(i, (offset, v))| {
                if *offset == i as u64 + varnode.address.offset {
                    Ok(v)
                } else {
                    // Undefined offset
                    Err(offset)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>();

        let bytes = result.map_err(|&offset| {
            Error::UndefinedData(Address {
                offset,
                address_space: varnode.address.address_space.clone(),
            })
        })?;

        if bytes.len() == varnode.size {
            Ok(bytes)
        } else {
            Err(Error::UndefinedData(Address {
                offset: varnode.address.offset + bytes.len() as u64,
                address_space: varnode.address.address_space.clone(),
            }))
        }
    }

    /// Read the value specified by the varnode and convert it into the concrete type `T`. If any
    /// portion of the data read is symbolic then an `Error::UnexpectedSymbolicData` will be returned.
    pub fn read_concrete_value<T>(&self, varnode: &VarnodeData) -> Result<T>
    where
        T: TryFrom<usize>,
        <T as TryFrom<usize>>::Error: std::error::Error + 'static,
    {
        sym::concretize_bit_iter(
            self.read(&varnode)?
                .iter()
                .map(|byte| byte.iter())
                .flatten(),
        )
        .map_err(|err| match err {
            ConcretizationError::NonLiteralBit(index) => {
                Error::UnexpectedSymbolicData(varnode.address.clone(), index)
            }
            _ => Error::InternalError(format!("{}", err)),
        })
    }

    /// Read from a varnode with an address in the constant address space.
    fn read_const(&self, varnode: &VarnodeData) -> Vec<SymbolicByte> {
        SymbolicBitVec::constant(
            varnode.address.offset.try_into().unwrap_or_else(|err| {
                panic!(
                    "unable to represent {offset} as symbolic constant: {err}",
                    offset = varnode.address.offset
                )
            }),
            8 * varnode.size,
        )
        .into_bytes()
    }

    /// Read the least significant bit from the address referenced by the varnode.
    pub fn read_bit(&self, varnode: &VarnodeData) -> Result<SymbolicBit> {
        if varnode.address.address_space.space_type == AddressSpaceType::Constant {
            return match varnode.address.offset {
                0 => Ok(SymbolicBit::Literal(false)),
                1 => Ok(SymbolicBit::Literal(true)),
                value => Err(Error::InvalidArguments(format!(
                    "constant {value} is not a valid bit"
                ))),
            };
        }

        let space_id = varnode.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        memory
            .read_byte(varnode.address.offset)
            .map(|byte| byte[0].clone())
            .ok_or_else(|| Error::UndefinedData(varnode.address.clone()))
    }

    /// Functions identically to [Self::read_bytes] except the bytes returned are cloned.
    pub fn read_bytes_owned(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        if varnode.address.address_space.space_type == AddressSpaceType::Constant {
            return Ok(self.read_const(&varnode));
        }

        let space_id = varnode.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UnknownAddressSpace(space_id))?;

        // Collect into a Vec or return the first undefined offset
        let result = memory
            .read_bytes(varnode.range())
            .enumerate()
            .map(|(i, (offset, v))| {
                if *offset == i as u64 + varnode.address.offset {
                    Ok(v.clone())
                } else {
                    // Undefined offset
                    Err(offset)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>();

        let bytes = result.map_err(|&offset| {
            Error::UndefinedData(Address {
                offset,
                address_space: varnode.address.address_space.clone(),
            })
        })?;

        if bytes.len() == varnode.size {
            Ok(bytes)
        } else {
            Err(Error::UndefinedData(Address {
                offset: varnode.address.offset + bytes.len() as u64,
                address_space: varnode.address.address_space.clone(),
            }))
        }
    }

    /// Writes the given data to the location specified by the provided varnode. The number of
    /// bytes provided must match the size of the varnode or else an error will be returned.
    pub fn write_bytes(&mut self, input: Vec<SymbolicByte>, output: &VarnodeData) -> Result<()> {
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

    /// Get the address space associated with the given varnode.
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

    /// Dumps the contents of memory to stdout. This is a helper function for debugging.
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

impl MemoryTree {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            predicate: SymbolicBit::Literal(true),
            parent: None,
        }
    }

    /// Get the predicate associated with this branch.
    pub fn predicate(&self) -> &SymbolicBit {
        &self.predicate
    }

    /// Read the bytes for this varnode.
    pub fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        let result = self.memory.read_bytes_owned(&varnode);
        if let Some(ref parent) = self.parent {
            if let Err(Error::UndefinedData(address)) = result {
                let num_valid_bytes = (address.offset - varnode.address.offset) as usize;

                // Special case to defer to the parent if the data is entirely undefined
                if num_valid_bytes == 0 {
                    return parent.read(&varnode);
                }

                // Read the known valid data
                let valid_input = VarnodeData {
                    address: Address {
                        offset: varnode.address.offset,
                        address_space: varnode.address.address_space.clone(),
                    },
                    size: num_valid_bytes,
                };
                let mut data = self.memory.read_bytes_owned(&valid_input)?;

                // Read the missing data from parent
                let parent_varnode = VarnodeData {
                    address,
                    size: varnode.size - num_valid_bytes,
                };
                let mut parent_data = parent.read(&parent_varnode)?;

                // Combine the two and return the result
                data.append(&mut parent_data);
                return Ok(data);
            }
        }

        result
    }

    /// Write the data to the location specified by the varnode.
    pub fn write(&mut self, varnode: &VarnodeData, data: Vec<SymbolicByte>) -> Result<()> {
        self.memory.write_bytes(data, varnode)
    }

    /// Create a branch in the memory tree based on the given predicate. The left branch is the
    /// MemoryTree with the predicate and the right memory model is the MemoryTree with its
    /// negation.
    pub fn branch(self, predicate: SymbolicBit) -> (Self, Self) {
        let address_spaces = self
            .memory
            .data
            .values()
            .map(|data| data.address_space.clone())
            .collect::<Vec<_>>();

        let rc = Rc::new(self);
        let positive = Self {
            predicate: predicate.clone(),
            parent: Some(Rc::clone(&rc)),
            memory: Memory::new(address_spaces.clone()),
        };

        let negative = Self {
            predicate: !predicate,
            parent: Some(rc),
            memory: Memory::new(address_spaces),
        };

        (positive, negative)
    }
}

struct AddressSpaceData {
    address_space: AddressSpace,
    data: BTreeMap<u64, SymbolicByte>,
}

impl AddressSpaceData {
    pub fn new(address_space: AddressSpace) -> Self {
        Self {
            address_space,
            data: Default::default(),
        }
    }

    pub fn read_byte(&self, offset: u64) -> Option<&SymbolicByte> {
        self.data.get(&offset)
    }

    pub fn read_bytes(
        &self,
        range: impl std::ops::RangeBounds<u64>,
    ) -> impl Iterator<Item = (&u64, &SymbolicByte)> {
        self.data.range(range)
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
        let value = SymbolicBitVec::constant(0x0123456789abcdef, 64).into_bytes();

        // Read and write value into memory
        let mut memory = Memory::new(vec![addr_space]);
        memory.write_bytes(value.clone(), &varnode)?;
        let read_value = memory.read(&varnode)?;

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

    /// This test should test memory tree branching, branch inheritance and branch independence.
    #[test]
    fn memory_tree_branch() -> Result<()> {
        // Setup memory with an address space
        let addr_space = address_space(0);
        let mut memory = Memory::new(vec![addr_space.clone()]);

        // Write an initial value to this address space
        let varnode = VarnodeData {
            address: Address {
                offset: 0,
                address_space: addr_space.clone(),
            },
            size: 1,
        };
        memory.write_bytes(SymbolicBitVec::constant(0xff, 8).into_bytes(), &varnode)?;

        // Create a memory tree from this initialized memory and branch on a predicate
        let tree = MemoryTree::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let (mut left, right) = tree.branch(predicate.clone());

        // Confirm the correctness of the left and right branch predicates
        assert_eq!(*left.predicate(), SymbolicBit::Literal(true));
        assert_eq!(*right.predicate(), SymbolicBit::Literal(false));

        // Overwrite the initialized value in the left branch only
        left.write(&varnode, SymbolicBitVec::constant(0x00, 8).into_bytes())?;

        // Show that the left memory branch has the updated value
        assert_eq!(
            u8::try_from(left.read(&varnode)?.pop().unwrap()).expect("failed to convert to byte"),
            0x00
        );

        // Show that the right memory branch value was not impacted by changes to the left
        // and that it inherited the original value.
        assert_eq!(
            u8::try_from(right.read(&varnode)?.pop().unwrap()).expect("failed to convert to byte"),
            0xff
        );

        Ok(())
    }

    #[test]
    fn memory_tree_read() -> Result<()> {
        // Setup memory with an address space
        let addr_space = address_space(0);
        let mut memory = Memory::new(vec![addr_space.clone()]);

        // Write an initial value to this address space
        let mut varnode = VarnodeData {
            address: Address {
                offset: 0,
                address_space: addr_space.clone(),
            },
            size: 2,
        };
        memory.write_bytes(SymbolicBitVec::constant(0xbeef, 16).into_bytes(), &varnode)?;

        // Create a memory tree from this initialized memory
        let tree = MemoryTree::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let (mut left, _) = tree.branch(predicate.clone());

        // Overwite part of the initial value
        varnode.size = 1;
        left.write(&varnode, SymbolicBitVec::constant(0xed, 8).into_bytes())?;

        // Read the entire value and confirm the overwritten portion is read along with the portion
        // that is not overwritten
        varnode.size = 2;
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(left.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            0xbeed
        );

        Ok(())
    }
}
