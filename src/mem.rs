use std::{collections::BTreeMap, rc::Rc};

use thiserror;

use sla::{Address, AddressSpace, AddressSpaceType, VarnodeData};
use sym::{self, ConcretizationError, SymbolicBit, SymbolicByte};

pub type Result<T> = std::result::Result<T, Error>;

pub trait SymbolicMemoryReader {
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>>;

    fn read_bit(&self, varnode: &VarnodeData) -> Result<SymbolicBit> {
        if varnode.size != 1 {
            return Err(Error::InvalidArguments(format!(
                "expected varnode size to be 1, actual {size}",
                size = varnode.size
            )));
        }

        let byte = &self.read(varnode)?[0];
        Ok(byte[0].clone())
    }
}

pub trait SymbolicMemoryWriter {
    fn write(
        &mut self,
        output: &VarnodeData,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()>;

    fn write_address(
        &mut self,
        address: Address,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.write(&VarnodeData::new(address, data.len()), data)
    }
}

pub trait SymbolicMemory: SymbolicMemoryReader + SymbolicMemoryWriter {}

#[derive(thiserror::Error, Debug)]
pub enum Error {
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
#[derive(Default)]
pub struct Memory {
    /// Structure for looking up data based on the id of an AddressSpace.
    data: BTreeMap<usize, BTreeMap<u64, SymbolicByte>>,
}

pub enum UndefinedDataBehavior<'a> {
    ReadFromAuthority(&'a dyn SymbolicMemoryReader),
    WriteSymbolicData,
    Error,
}

pub struct FallbackMemory<'a, M: SymbolicMemory> {
    address_space_behaviors: BTreeMap<usize, UndefinedDataBehavior<'a>>,
    memory: std::cell::RefCell<M>,
}

impl<'a, M: SymbolicMemory> FallbackMemory<'a, M> {
    pub fn undefined_data_behavior(
        &mut self,
        address_space: &AddressSpace,
        behavior: UndefinedDataBehavior<'a>,
    ) {
        self.address_space_behaviors
            .insert(address_space.id, behavior);
    }
}

impl<'a, M: SymbolicMemory> SymbolicMemoryReader for FallbackMemory<'a, M> {
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        let result = self.memory.borrow().read(varnode);
        if matches!(result, Err(Error::UndefinedData(_))) {
            let behavior = self
                .address_space_behaviors
                .get(&varnode.address.address_space.id)
                .unwrap_or(&UndefinedDataBehavior::Error);

            match behavior {
                UndefinedDataBehavior::WriteSymbolicData => {
                    todo!("create symbolic data, write to memory")
                }
                UndefinedDataBehavior::Error => {
                    return result;
                }
                UndefinedDataBehavior::ReadFromAuthority(authority) => {
                    return authority.read(varnode);
                }
            }
        } else {
            // Data is not undefined, no special handling required
            return result;
        }
    }
}

impl<'a, M> SymbolicMemoryWriter for FallbackMemory<'a, M>
where
    M: SymbolicMemory,
{
    fn write(
        &mut self,
        output: &VarnodeData,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.memory.borrow_mut().write(output, data)
    }
}

impl<T: SymbolicMemoryReader + SymbolicMemoryWriter> SymbolicMemory for T {}

impl SymbolicMemoryReader for Memory {
    /// Read the bytes from the addresses specified by the varnode. This function returns `Ok` if
    /// and only if data is successfully read from the requested addresses.
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        if varnode.address.address_space.space_type == AddressSpaceType::Constant {
            // This is just a constant value defined by the offset
            let bytes: sym::SymbolicBitBuf<64> = varnode.address.offset.into();
            let mut result: Vec<SymbolicByte> = bytes.into();

            // The varnode size defines the number of bytes in the result. Since offset
            // is always 64 bits then the number of bytes is at most 8
            if varnode.size <= 8 {
                // Remove extra bytes
                result.drain(varnode.size..);
                return Ok(result);
            } else {
                return Err(Error::InvalidArguments(format!(
                    "varnode size {size} exceeds maximum allowed for constant address space",
                    size = varnode.size
                )));
            }
        }

        let space_id = varnode.address.address_space.id;
        let memory = self
            .data
            .get(&space_id)
            .ok_or(Error::UndefinedData(varnode.address.clone()))?;

        // Collect into a Vec or return the first undefined offset
        let result = memory
            .range(varnode.range())
            .enumerate()
            .map(|(i, (&offset, v))| {
                let i = u64::try_from(i).map_err(|_| offset)?;
                if offset == i as u64 + varnode.address.offset {
                    Ok(v.clone())
                } else {
                    // Undefined offset
                    Err(offset)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>();

        let bytes = result.map_err(|offset| {
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
}

impl SymbolicMemoryWriter for Memory {
    /// Writes the given data to the location specified by the provided varnode. The number of
    /// bytes provided must match the size of the varnode or else an error will be returned.
    fn write(
        &mut self,
        varnode: &VarnodeData,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        if data.len() != varnode.size {
            return Err(Error::InvalidArguments(format!(
                "requested {} bytes to be written, provided {} bytes",
                varnode.size,
                data.len()
            )));
        }

        let space_id = varnode.address.address_space.id;
        let memory = self.data.entry(space_id).or_default();

        let mut offset = varnode.address.offset;
        for byte in data {
            memory.insert(offset, byte.into());
            offset += 1;
        }

        Ok(())
    }
}

impl Memory {
    /// Create a new instance of memory for the provided AddressSpaces.
    pub fn new() -> Self {
        Self::default()
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

    /// Dumps the contents of memory to stdout. This is a helper function for debugging.
    pub fn dump(&self) {
        for address_space_id in self.data.keys() {
            println!("Begin memory dump for {}", address_space_id);

            let addr_space_data = self.data.get(&address_space_id).unwrap();
            for key in addr_space_data.keys() {
                let byte: u8 = addr_space_data
                    .get(key)
                    .unwrap()
                    .try_into()
                    .expect("failed to convert to byte");
                println!("{key:016x}: {byte:02x}");
            }

            println!("End memory dump for {}", address_space_id);
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

    /// Create a branch in the memory tree based on the given predicate. The left branch is the
    /// MemoryTree with the predicate and the right memory model is the MemoryTree with its
    /// negation.
    pub fn branch(self, predicate: SymbolicBit) -> (Self, Self) {
        let rc = Rc::new(self);
        let positive = Self {
            predicate: predicate.clone(),
            parent: Some(Rc::clone(&rc)),
            memory: Memory::new(),
        };

        let negative = Self {
            predicate: !predicate,
            parent: Some(rc),
            memory: Memory::new(),
        };

        (positive, negative)
    }
}

impl SymbolicMemoryReader for MemoryTree {
    /// Read the bytes for this varnode.
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        let result = self.memory.read(&varnode);
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
                let mut data = self.memory.read(&valid_input)?;

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
}

impl SymbolicMemoryWriter for MemoryTree {
    /// Write the data to the location specified by the varnode.
    fn write(
        &mut self,
        varnode: &VarnodeData,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.memory.write(varnode, data)
    }
}

pub struct ExecutableMemory<M: SymbolicMemory>(pub M);

impl<M: SymbolicMemory> SymbolicMemoryReader for ExecutableMemory<M> {
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        self.0.read(varnode)
    }
}

impl<M: SymbolicMemory> SymbolicMemoryWriter for ExecutableMemory<M> {
    fn write(
        &mut self,
        output: &VarnodeData,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.0.write(output, data)
    }
}

/// Implementation of the LoadImage trait to enable loading instructions from memory
impl<M: SymbolicMemory> sla::LoadImage for ExecutableMemory<M> {
    fn instruction_bytes(&self, input: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        let bytes = self.read(&input);

        // The number of bytes requested may exceed valid data in memory.
        // In that case only read and return the defined bytes.
        let bytes = match bytes {
            Err(Error::UndefinedData(addr)) => {
                let input = VarnodeData {
                    // SAFETY: This new size MUST be less than the existing input size
                    size: unsafe {
                        (addr.offset - input.address.offset)
                            .try_into()
                            .unwrap_unchecked()
                    },
                    address: input.address.clone(),
                };
                self.read(&input)
            }
            _ => bytes,
        };

        bytes
            .map_err(|err| err.to_string())?
            .into_iter()
            .map(|x| x.try_into())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_err| "symbolic byte".to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use sla::AddressSpace;
    use sym::SymbolicBitVec;

    use super::*;

    fn address_space(id: usize) -> AddressSpace {
        AddressSpace {
            id,
            name: Cow::Borrowed("test_space"),
            word_size: 1,
            address_size: 8, // 64-bit
            space_type: AddressSpaceType::Processor,
            big_endian: false,
        }
    }

    fn const_space() -> AddressSpace {
        AddressSpace {
            id: 0,
            name: Cow::Borrowed("constant_space"),
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
        let mut memory = Memory::new();
        memory.write(&varnode, value.clone().into_iter())?;
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
        let mut memory = Memory::new();

        // Write an initial value to this address space
        let varnode = VarnodeData {
            address: Address {
                offset: 0,
                address_space: addr_space.clone(),
            },
            size: 1,
        };
        memory.write(
            &varnode,
            SymbolicBitVec::constant(0xff, 8).into_bytes().into_iter(),
        )?;

        // Create a memory tree from this initialized memory and branch on a predicate
        let tree = MemoryTree::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let (mut left, right) = tree.branch(predicate.clone());

        // Confirm the correctness of the left and right branch predicates
        assert_eq!(*left.predicate(), SymbolicBit::Literal(true));
        assert_eq!(*right.predicate(), SymbolicBit::Literal(false));

        // Overwrite the initialized value in the left branch only
        left.write(
            &varnode,
            SymbolicBitVec::constant(0x00, 8).into_bytes().into_iter(),
        )?;

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
        let mut memory = Memory::new();

        // Write an initial value to this address space
        let mut varnode = VarnodeData {
            address: Address {
                offset: 0,
                address_space: addr_space.clone(),
            },
            size: 2,
        };
        memory.write(
            &varnode,
            SymbolicBitVec::constant(0xbeef, 16)
                .into_bytes()
                .into_iter(),
        )?;

        // Create a memory tree from this initialized memory
        let tree = MemoryTree::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let (mut left, _) = tree.branch(predicate.clone());

        // Overwite part of the initial value
        varnode.size = 1;
        left.write(
            &varnode,
            SymbolicBitVec::constant(0xed, 8).into_bytes().into_iter(),
        )?;

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
