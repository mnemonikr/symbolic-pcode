use std::{collections::BTreeMap, rc::Rc};

use thiserror;

use sla::{Address, AddressSpaceId, AddressSpaceType, VarnodeData};
use sym::{self, SymbolicBit, SymbolicBitVec, SymbolicByte};

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
        data: impl IntoIterator<Item = impl Into<SymbolicByte>>,
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

    #[error("arguments provided are not valid: {0}")]
    InvalidArguments(String),

    #[error("an internal error occurred: {0}")]
    InternalError(String),
}

/// Collection of all memory branches into a single tree. Tree is composed of both live and dead
/// branches. A dead branch is a branch of memory that has no bearing on an outcome. It is
/// important to include these branches so that they are appropriately excluded from the outcome.
pub struct MemoryTree<'b, 'd> {
    branches: Vec<&'b MemoryBranch>,
    dead_branches: Vec<&'d MemoryBranch>,
}

impl<'b, 'd> MemoryTree<'b, 'd> {
    pub fn new(
        branches: impl Iterator<Item = &'b MemoryBranch>,
        dead_branches: impl Iterator<Item = &'d MemoryBranch>,
    ) -> Self {
        Self {
            branches: branches.collect(),
            dead_branches: dead_branches.collect(),
        }
    }

    pub fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        let result = self
            .branches
            .iter()
            .map(|&branch| {
                branch
                    .predicated_read(varnode)
                    .map(|bytes| bytes.into_iter().collect::<SymbolicBitVec>())
            })
            .reduce(|x, y| {
                // Must check first argument first since it is the accumulator
                // If an error is propagating it will propagate here.
                if let Ok(x) = x {
                    if let Ok(y) = y {
                        Ok(x & y)
                    } else {
                        y
                    }
                } else {
                    x
                }
            })
            .ok_or_else(|| Error::InternalError("memory tree has no branches".to_string()))??;

        Ok(result.into_bytes())
    }

    /// Assert that none of the dead branches in this tree are taken in the provided condition.
    pub fn prune_dead_branches(&self, condition: SymbolicBit) -> SymbolicBit {
        let dead_branch_taken = self
            .dead_branches
            .iter()
            .map(|&branch| branch.branch_predicate())
            .fold(sym::FALSE, |x, y| x.clone() | y.clone());

        condition & !dead_branch_taken
    }
}

/// A branching memory model which branches on `SymbolicBit` predicates. Note that this does not
/// enforce the predicate. It simply tracks the assumptions made at various branches.
#[derive(Debug)]
pub struct MemoryBranch {
    leaf_predicate: SymbolicBit,
    branch_predicate: SymbolicBit,
    parent: Option<Rc<Self>>,
    memory: Memory,
}

impl Default for MemoryBranch {
    fn default() -> Self {
        Self {
            leaf_predicate: sym::TRUE,
            branch_predicate: sym::TRUE,
            parent: Default::default(),
            memory: Default::default(),
        }
    }
}

/// A memory model that stores bytes associated with an AddressSpace.
#[derive(Debug, Default)]
pub struct Memory {
    /// Structure for looking up data based on the id of an AddressSpace.
    data: BTreeMap<AddressSpaceId, BTreeMap<u64, SymbolicByte>>,
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
        data: impl IntoIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        let data = data.into_iter();
        let (lower_size, upper_size) = data.size_hint();

        if upper_size
            .filter(|&size| size == lower_size && size == varnode.size)
            .is_none()
        {
            // Bounds do not exactly match varnode size. Check if size falls out of bounds
            if varnode.size < lower_size {
                return Err(Error::InvalidArguments(format!(
                    "requested {} bytes to be written, provided lower bound {lower_size} bytes",
                    varnode.size
                )));
            }

            if let Some(upper_size) = upper_size {
                if upper_size < varnode.size {
                    return Err(Error::InvalidArguments(format!(
                        "requested {} bytes to be written, provided upper bound {upper_size} bytes",
                        varnode.size
                    )));
                }
            }
        }

        // Make sure the varnode does not overflow the offset
        let overflows = u64::try_from(varnode.size)
            .ok()
            .map(|size| varnode.address.offset.checked_add(size))
            .flatten()
            .is_none();

        if overflows {
            return Err(Error::InvalidArguments(format!(
                "varnode size {size} overflows address offset {offset}",
                size = varnode.size,
                offset = varnode.address.offset
            )));
        }

        let space_id = varnode.address.address_space.id;
        let memory = self.data.entry(space_id).or_default();
        let rollback_fn = |memory: &mut BTreeMap<u64, SymbolicByte>, rollback| {
            let mut offset = varnode.address.offset;
            for rollback_value in rollback {
                if let Some(value) = rollback_value {
                    memory.insert(offset, value);
                }
                offset += 1;
            }
        };

        let mut offset = varnode.address.offset;
        let mut rollback = Vec::with_capacity(varnode.size);
        for byte in data {
            if rollback.len() == varnode.size {
                rollback_fn(memory, rollback);
                return Err(Error::InvalidArguments(format!(
                    "too many bytes provided, expected {}",
                    varnode.size
                )));
            }
            rollback.push(memory.insert(offset, byte.into()));
            offset += 1;
        }

        let bytes_written = rollback.len();
        if bytes_written < varnode.size {
            rollback_fn(memory, rollback);
            return Err(Error::InvalidArguments(format!(
                "not enough bytes provided, {bytes_written} < expected {size}",
                size = varnode.size
            )));
        }

        Ok(())
    }
}

impl Memory {
    /// Create a new instance of memory for the provided AddressSpaces.
    pub fn new() -> Self {
        Self::default()
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

impl MemoryBranch {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            leaf_predicate: sym::TRUE,
            branch_predicate: sym::TRUE,
            parent: None,
        }
    }

    pub fn branch_predicate(&self) -> &SymbolicBit {
        &self.branch_predicate
    }

    /// Get the predicate associated with this branch.
    pub fn leaf_predicate(&self) -> &SymbolicBit {
        &self.leaf_predicate
    }

    /// Branch this tree on the given predicate. The current tree branch predicate is updated to
    /// the given predicate. A new branch of the tree is returned with a negation of the predicate.
    pub fn new_branch(&mut self, predicate: SymbolicBit) -> Self {
        // Build the new shared parent
        let mut parent = Self {
            leaf_predicate: predicate.clone(),
            branch_predicate: self.branch_predicate.clone() & predicate.clone(),
            parent: None,
            memory: Memory::new(),
        };

        // Update the shared parent to hold contents of this branch
        std::mem::swap(self, &mut parent);

        // Point this branch to new shared parent
        let rc = Rc::new(parent);
        self.parent = Some(Rc::clone(&rc));

        // Create new branch with negated predicate
        Self {
            leaf_predicate: !predicate.clone(),
            branch_predicate: rc.branch_predicate.clone() & !predicate,
            parent: Some(rc),
            memory: Memory::new(),
        }
    }

    /// Read a value from memory with the necessary predicates applied to the result. For example,
    /// if a value `V` is predicated on `X`, then the response is the implication `X => V`, or
    /// equivalently `!X | V`.
    ///
    /// If a portion of a value `V` has not been updated in this branch, then that portion is only
    /// predicated on the parent predicate in which it is stored.
    pub fn predicated_read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        let result = self.memory.read(varnode);
        if let Some(parent) = &self.parent {
            if let Err(Error::UndefinedData(address)) = result {
                let num_valid_bytes = (address.offset - varnode.address.offset) as usize;

                // Special case to defer to the parent if the data is entirely undefined
                if num_valid_bytes == 0 {
                    return parent.predicated_read(varnode);
                }

                // Read the known valid data
                let valid_input = VarnodeData {
                    address: Address {
                        offset: varnode.address.offset,
                        address_space: varnode.address.address_space.clone(),
                    },
                    size: num_valid_bytes,
                };
                let mut data = self.predicated_value(self.memory.read(&valid_input)?);

                // Read the missing data from parent
                let parent_varnode = VarnodeData {
                    address,
                    size: varnode.size - num_valid_bytes,
                };
                let mut parent_data = parent.predicated_read(&parent_varnode)?;

                // Combine the two and return the result
                data.append(&mut parent_data);
                return Ok(data);
            }
        }

        result.map(|value| self.predicated_value(value))
    }

    fn predicated_value(&self, value: Vec<SymbolicByte>) -> Vec<SymbolicByte> {
        let bitvec: SymbolicBitVec = value.into_iter().collect();
        let negated_predicate: SymbolicBitVec = std::iter::repeat(!self.branch_predicate.clone())
            .take(bitvec.len())
            .collect();
        let conditional_value = bitvec | negated_predicate;
        conditional_value.into_bytes()
    }
}

impl SymbolicMemoryReader for MemoryBranch {
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

impl SymbolicMemoryWriter for MemoryBranch {
    /// Write the data to the location specified by the varnode.
    fn write(
        &mut self,
        varnode: &VarnodeData,
        data: impl IntoIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.memory.write(varnode, data)
    }
}

pub struct ExecutableMemory<'a, M: SymbolicMemory>(pub &'a M);

impl<'a, M: SymbolicMemory> SymbolicMemoryReader for ExecutableMemory<'a, M> {
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        self.0.read(varnode)
    }
}

/// Implementation of the LoadImage trait to enable loading instructions from memory
impl<'a, M: SymbolicMemory> sla::LoadImage for ExecutableMemory<'a, M> {
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
            id: AddressSpaceId::new(id),
            name: Cow::Borrowed("test_space"),
            word_size: 1,
            address_size: 8, // 64-bit
            space_type: AddressSpaceType::Processor,
            big_endian: false,
        }
    }

    fn const_space() -> AddressSpace {
        AddressSpace {
            id: AddressSpaceId::new(0),
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
        let mut tree = MemoryBranch::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let _ = tree.new_branch(predicate.clone());

        // Overwite part of the initial value
        varnode.size = 1;
        tree.write(
            &varnode,
            SymbolicBitVec::constant(0xed, 8).into_bytes().into_iter(),
        )?;

        // Read the entire value and confirm the overwritten portion is read along with the portion
        // that is not overwritten
        varnode.size = 2;
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(tree.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            0xbeed
        );

        Ok(())
    }

    #[test]
    fn new_branch() -> Result<()> {
        // Setup memory with an address space
        let addr_space = address_space(0);
        let mut memory = Memory::new();
        const TEST_VALUE: u16 = 0xbeef;

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
            SymbolicBitVec::constant(TEST_VALUE.into(), 16)
                .into_bytes()
                .into_iter(),
        )?;

        // Create a new branch
        let mut tree = MemoryBranch::new(memory);
        let predicate = sym::SymbolicBit::Variable(0);
        let false_branch = tree.new_branch(predicate.clone());

        // Confirm predicates are correct
        assert_eq!(*tree.leaf_predicate(), predicate);
        assert_eq!(*tree.branch_predicate(), predicate);
        assert_eq!(*false_branch.leaf_predicate(), !predicate.clone());
        assert_eq!(*false_branch.branch_predicate(), !predicate);

        // Overwite part of the initial value
        varnode.size = 1;
        tree.write(
            &varnode,
            SymbolicBitVec::constant(0xed, 8).into_bytes().into_iter(),
        )?;

        // Read the entire value and confirm the overwritten portion is read along with the portion
        // that is not overwritten
        varnode.size = 2;
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(tree.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            0xbeed
        );

        // Confirm false branch value is unchanged
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(false_branch.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            TEST_VALUE
        );

        Ok(())
    }
}
