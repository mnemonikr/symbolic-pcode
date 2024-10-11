use std::{collections::BTreeMap, rc::Rc};

use thiserror;

use sla::{Address, AddressSpaceId, AddressSpaceType, VarnodeData};
use sym::{self, SymbolicBit, SymbolicBitVec, SymbolicByte};

/// Memory result type
pub type Result<T> = std::result::Result<T, Error>;

/// Possible memory errors
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// There is no data defined at a particular address
    #[error("no data defined at address {0}")]
    UndefinedData(Address),

    /// The arguments provided for a given request are invalid
    #[error("arguments provided are not valid: {0}")]
    InvalidArguments(String),

    /// An internal error that cannot be handled
    #[error("an internal error occurred: {0}")]
    InternalError(String),
}

/// Trait for memory with symbolic bytes that can be read from.
pub trait SymbolicMemoryReader {
    /// Read [SymbolicByte] values from the requested source.
    fn read(&self, source: &VarnodeData) -> Result<Vec<SymbolicByte>>;

    /// Read [SymbolicBit] from the requested source. Since [VarnodeData] can only represent
    /// byte-sized requests, the source's `size` must be `1`.
    fn read_bit(&self, source: &VarnodeData) -> Result<SymbolicBit> {
        if source.size != 1 {
            return Err(Error::InvalidArguments(format!(
                "expected varnode size to be 1, actual {size}",
                size = source.size
            )));
        }

        let byte = &self.read(source)?[0];
        Ok(byte[0].clone())
    }
}

/// Trait for memory with symbolic bytes that can be written to.
pub trait SymbolicMemoryWriter {
    /// Write [SymbolicByte] values to the destination in memory.
    fn write(
        &mut self,
        destination: &VarnodeData,
        data: impl IntoIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()>;

    /// Write [SymbolicByte] values to the address in memory.
    fn write_address(
        &mut self,
        address: Address,
        data: impl ExactSizeIterator<Item = impl Into<SymbolicByte>>,
    ) -> Result<()> {
        self.write(&VarnodeData::new(address, data.len()), data)
    }
}

/// Trait for memory with symbolic bytes that can be read from and written to.
pub trait SymbolicMemory: SymbolicMemoryReader + SymbolicMemoryWriter {}

impl<T: SymbolicMemoryReader + SymbolicMemoryWriter> SymbolicMemory for T {}

/// A memory model that stores bytes associated with an AddressSpace.
#[derive(Debug, Default)]
pub struct Memory {
    /// Structure for looking up data based on the id of an AddressSpace.
    data: BTreeMap<AddressSpaceId, BTreeMap<u64, SymbolicByte>>,
}

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
                if offset == i + varnode.address.offset {
                    Ok(v.clone())
                } else {
                    // Undefined offset
                    Err(i + varnode.address.offset)
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
            .and_then(|size| varnode.address.offset.checked_add(size))
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
}

/// A branching memory model which branches on `SymbolicBit` predicates. Note that this does not
/// enforce the predicate. It simply tracks the assumptions made. Note that `[Self::read()]` is
/// *not* conditioned on the branch predicate. See `[Self::predicated_read()]`.
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
        self.read(varnode).map(|value| self.predicated_value(value))
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
        // Not using a question mark operator here in order to check for undefined data
        let result = self.memory.read(varnode);

        if let Some(parent) = &self.parent {
            if let Err(Error::UndefinedData(address)) = result {
                let num_valid_bytes = (address.offset - varnode.address.offset) as usize;

                // Special case to defer to the parent if the data is entirely undefined
                if num_valid_bytes == 0 {
                    return parent.read(varnode);
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

/// Collection of all memory branches into a single tree. Tree is composed of both live and dead
/// branches. A dead branch is a branch of memory that has no bearing on an outcome. These branches
/// are necessary to include so that their predicates are appropriately excluded from the outcome.
pub struct MemoryTree<'b, 'd> {
    branches: Vec<&'b MemoryBranch>,
    dead_branches: Vec<&'d MemoryBranch>,
}

impl<'b, 'd> MemoryTree<'b, 'd> {
    pub fn new(
        branches: impl IntoIterator<Item = &'b MemoryBranch>,
        dead_branches: impl IntoIterator<Item = &'d MemoryBranch>,
    ) -> Self {
        Self {
            branches: branches.into_iter().collect(),
            dead_branches: dead_branches.into_iter().collect(),
        }
    }

    fn dead_branches_not_taken_predicate(&self) -> SymbolicBit {
        let dead_branch_taken = self
            .dead_branches
            .iter()
            .map(|&branch| branch.branch_predicate())
            .fold(sym::FALSE, |x, y| x.clone() | y.clone());

        !dead_branch_taken
    }
}

impl<'b, 'd> SymbolicMemoryReader for MemoryTree<'b, 'd> {
    /// Read data from the requested source. This will read data from each live branch using a
    /// [predicated read](MemoryBranch::predicated_read()) and return the conjunction these
    /// conditional values. These values are subsequently conditioned on none of the dead branches
    /// being taken.
    ///
    /// For example, consider the following:
    ///
    /// ```
    /// fn f(x: bool, y: bool) -> u8 {
    ///     if x {
    ///         if y {
    ///             1
    ///         } else {
    ///             2
    ///         }
    ///     } else {
    ///         panic!("Invalid path");
    ///     }
    /// }
    /// ```
    ///
    /// If investigating the return value of `f`, then the branch where `x` is `false` cannot be
    /// taken since that induces a panic. In that model, the return register will have data
    /// unrelated to the return value. This branch is considered a dead branch.
    ///
    /// The read of this return value is `!(!x) & ((x & y) => 1) & ((x & !y) => 2)` where `!(!x)`
    /// is asserting that the `else` branch is _not_ taken.
    ///
    /// # Branch errors
    ///
    /// If any of the live branches return an error, then this read will also be an error. That
    /// means all branches must contain a fully defined value for the source being read.
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
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
            .ok_or_else(|| {
                Error::InternalError("memory tree has no live branches".to_string())
            })??;

        let result: SymbolicBitVec = result
            .into_iter()
            .zip(std::iter::repeat(self.dead_branches_not_taken_predicate()))
            .map(|(bit, dead_branches_not_taken)| bit & dead_branches_not_taken)
            .collect();

        Ok(result.into_bytes())
    }
}

/// Memory that holds binary-encoded executable instructions.
pub struct ExecutableMemory<'a, M: SymbolicMemoryReader>(pub &'a M);

impl<'a, M: SymbolicMemory> SymbolicMemoryReader for ExecutableMemory<'a, M> {
    fn read(&self, varnode: &VarnodeData) -> Result<Vec<SymbolicByte>> {
        self.0.read(varnode)
    }
}

/// Implementation of the LoadImage trait to enable loading instructions from memory
impl<'a, M: SymbolicMemory> sla::LoadImage for ExecutableMemory<'a, M> {
    fn instruction_bytes(&self, input: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        let bytes = self.read(input);

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

    use sla::{AddressSpace, LoadImage};
    use sym::SymbolicBitVec;

    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestIterator {
        size_hint: (usize, Option<usize>),
        value: SymbolicByte,
        size: usize,
    }

    impl TestIterator {
        pub fn with_single_value(value: u8) -> Self {
            Self::with_value(value, 1)
        }

        pub fn with_symbolic_value(size: usize) -> Self {
            Self {
                size_hint: (size, Some(size)),
                size,
                value: SymbolicByte::default() | (SymbolicBit::Variable(0).into()),
            }
        }

        pub fn with_value(value: u8, size: usize) -> Self {
            Self {
                size_hint: (size, Some(size)),
                value: SymbolicBitVec::constant(value.into(), 8)
                    .into_bytes()
                    .pop()
                    .unwrap(),
                size,
            }
        }

        pub fn with_size(size: usize) -> Self {
            Self {
                size_hint: (size, Some(size)),
                size,
                ..Default::default()
            }
        }

        pub fn without_hint(size: usize) -> Self {
            Self {
                size,
                ..Default::default()
            }
        }
    }

    impl ExactSizeIterator for TestIterator {}

    impl Iterator for TestIterator {
        type Item = SymbolicByte;

        fn next(&mut self) -> Option<Self::Item> {
            if self.size == 0 {
                return None;
            }

            self.size -= 1;
            Some(self.value.clone())
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.size_hint
        }
    }

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

        // Read and write value into memory
        let mut memory = Memory::new();
        memory.write(&varnode, TestIterator::with_value(0x5a, 8))?;
        let read_value = memory.read(&varnode)?;

        // Confirm read value matches written value
        assert_eq!(read_value.len(), varnode.size);
        for actual in read_value {
            let actual: u8 = sym::concretize_into(std::iter::once(actual)).unwrap();
            assert_eq!(actual, 0x5a);
        }

        Ok(())
    }

    #[test]
    fn memory_branch_read() -> Result<()> {
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
        memory.write(&varnode, TestIterator::with_value(0xbe, 2))?;

        // Create a memory tree from this initialized memory
        let mut tree = MemoryBranch::new(memory);
        let predicate = sym::SymbolicBit::Literal(true);
        let _ = tree.new_branch(predicate.clone());

        // Overwite part of the initial value
        varnode.size = 1;
        tree.write(&varnode, TestIterator::with_single_value(0xef))?;

        // Read the entire value and confirm the overwritten portion is read along with the portion
        // that is not overwritten
        varnode.size = 2;
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(tree.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            0xbeef
        );

        Ok(())
    }

    #[test]
    fn new_branch() -> Result<()> {
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
        memory.write(&varnode, TestIterator::with_value(0xbe, 2))?;

        // Create a new branch
        let mut true_branch = MemoryBranch::new(memory);
        let predicate = sym::SymbolicBit::Variable(0);
        let false_branch = true_branch.new_branch(predicate.clone());

        // Confirm predicates are correct
        assert_eq!(*true_branch.leaf_predicate(), predicate);
        assert_eq!(*true_branch.branch_predicate(), predicate);
        assert_eq!(*false_branch.leaf_predicate(), !predicate.clone());
        assert_eq!(*false_branch.branch_predicate(), !predicate);

        // Overwite part of the initial value
        varnode.size = 1;
        true_branch.write(&varnode, TestIterator::with_single_value(0xef))?;
        varnode.size = 2;

        // Read the entire value and confirm the overwritten portion is read along with the portion
        // that is not overwritten
        let old_value: u16 = 0xbebe;
        let new_value: u16 = 0xbeef;
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(true_branch.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            new_value
        );
        // Confirm false branch value is unchanged
        assert_eq!(
            u16::try_from(
                sym::SymbolicBitBuf::<16>::try_from(false_branch.read(&varnode)?)
                    .expect("buffer conversion failed")
            )
            .expect("failed to make value concrete"),
            old_value
        );

        Ok(())
    }

    #[test]
    fn memory_tree() -> Result<()> {
        let mut memory = MemoryBranch::default();
        let x = sym::TRUE;
        let y = sym::TRUE;
        let panic_memory = memory.new_branch(x);
        let mut y_branch = memory.new_branch(y);
        let return_address = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&return_address, TestIterator::with_single_value(1))?;
        y_branch.write(&return_address, TestIterator::with_single_value(2))?;
        let live_branches = [memory, y_branch];
        let dead_branches = [panic_memory];

        let tree = MemoryTree::new(live_branches.iter(), dead_branches.iter());
        let return_value = tree
            .read(&return_address)?
            .pop()
            .expect("did not read 1 byte");
        assert_eq!(return_value.equals(sym::TRUE.into()), sym::TRUE);

        Ok(())
    }

    #[test]
    fn memory_tree_false_branch_unwritten() -> Result<()> {
        let mut memory = Memory::new();
        let value = 0xbe;
        let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 2);
        memory.write(&varnode, TestIterator::with_value(value, 2))?;

        let mut true_branch = MemoryBranch::new(memory);
        let false_branch = true_branch.new_branch(sym::TRUE);

        varnode.size = 1;
        true_branch.write(&varnode, TestIterator::with_single_value(0xef))?;
        varnode.size = 2;

        let live_branches = [true_branch, false_branch];
        let tree = MemoryTree::new(live_branches.iter(), std::iter::empty());
        let tree_value = tree.read(&varnode)?;
        let tree_value: u16 = sym::concretize_into(tree_value).expect("symbolic tree value");

        // The tree should result in evaluating the following
        // (!true | 0xbeef) & (!false | 0xbebe) & true
        let expected_value = 0xbeef;
        assert_eq!(
            tree_value, expected_value,
            "got {tree_value:0x}, expected {expected_value:0x}"
        );

        Ok(())
    }

    #[test]
    fn memory_tree_dead_branch() -> Result<()> {
        let mut memory = MemoryBranch::default();
        let condition = SymbolicBit::Variable(0);
        let memory_else = memory.new_branch(condition.clone());
        let return_address = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let written_return_value = SymbolicBitVec::constant(1, 8);
        memory.write(&return_address, TestIterator::with_single_value(1))?;
        let live_branches = [memory];
        let dead_branches = [memory_else];

        let tree = MemoryTree::new(live_branches.iter(), dead_branches.iter());
        let return_value = tree
            .read(&return_address)?
            .pop()
            .expect("did not read 1 byte");

        // The expected return value should be conjunction of:
        //
        // * live branch taken (!x | R)
        // * dead branch NOT taken: !(!x), or x
        //
        // so x & (!x | R) reduces to x & R.
        let expected_return_value = std::iter::repeat(condition)
            .zip(written_return_value)
            .map(|(x, r)| (x & r))
            .collect::<SymbolicBitVec>()
            .into_bytes()
            .pop()
            .unwrap();

        assert_eq!(return_value, expected_return_value);
        Ok(())
    }

    #[test]
    fn write_too_few_bytes() -> Result<()> {
        let mut memory = Memory::new();
        let destination = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let result = memory.write(&destination, TestIterator::with_size(0));
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == "requested 1 bytes to be written, provided upper bound 0 bytes")
        );
        Ok(())
    }

    #[test]
    fn write_too_few_bytes_rollback() -> Result<()> {
        let mut memory = Memory::new();
        let destination = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&destination, TestIterator::with_single_value(42))?;

        let result = memory.write(&destination, TestIterator::without_hint(0));
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == "not enough bytes provided, 0 < expected 1")
        );

        // Verify rollback
        let value: u8 = sym::concretize_into(memory.read(&destination)?).unwrap();
        assert_eq!(value, 42);

        Ok(())
    }

    #[test]
    fn write_too_many_bytes() -> Result<()> {
        let mut memory = Memory::new();
        let destination = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let result = memory.write(&destination, TestIterator::with_size(2));
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == "requested 1 bytes to be written, provided lower bound 2 bytes")
        );

        Ok(())
    }

    #[test]
    fn write_too_many_bytes_rollback() -> Result<()> {
        let mut memory = Memory::new();
        let destination = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&destination, TestIterator::with_single_value(42))?;

        let result = memory.write(&destination, TestIterator::without_hint(2));
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == "too many bytes provided, expected 1")
        );

        // Verify rollback
        let value: u8 = sym::concretize_into(memory.read(&destination)?).unwrap();
        assert_eq!(value, 42);

        Ok(())
    }

    #[test]
    fn write_overflows_address() -> Result<()> {
        let mut memory = Memory::new();
        let destination = VarnodeData::new(Address::new(address_space(0), u64::MAX), 1);
        let result = memory.write(&destination, TestIterator::with_single_value(0));
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("varnode size 1 overflows address offset {offset}", offset = u64::MAX))
        );
        Ok(())
    }

    #[test]
    fn memory_branch_read_from_parent() -> Result<()> {
        let mut branch = MemoryBranch::default();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let value = 1;
        branch.write(&varnode, TestIterator::with_single_value(value))?;

        let _child = branch.new_branch(sym::TRUE);
        let read_value: u8 = sym::concretize_into(branch.read(&varnode)?)
            .expect("unexpected symbolic value from memory");
        assert_eq!(read_value, value);
        Ok(())
    }

    #[test]
    fn memory_branch_partial_read_failure() -> Result<()> {
        let mut branch = MemoryBranch::default();
        let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let value = 1;
        let _child = branch.new_branch(sym::TRUE);
        branch.write(&varnode, TestIterator::with_single_value(value))?;

        // Increase varnode size such that will now read undefined data
        varnode.size = 2;
        let result = branch.read(&varnode);
        let mut expected_failure_address = varnode.address.clone();
        expected_failure_address.offset += 1;
        assert!(
            matches!(result, Err(Error::UndefinedData(addr)) if addr == expected_failure_address)
        );
        Ok(())
    }

    #[test]
    fn memory_branch_predicated_read() -> Result<()> {
        let mut branch = MemoryBranch::default();
        let _child = branch.new_branch(sym::TRUE);
        let varnode = VarnodeData::new(Address::new(address_space(0), u64::MAX), 1);
        let result = branch.read(&varnode);
        assert!(matches!(result, Err(Error::UndefinedData(addr)) if addr == varnode.address));
        Ok(())
    }

    #[test]
    fn memory_tree_branches_with_undefined_data() -> Result<()> {
        let value = 0xbe;
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let mut true_branch = MemoryBranch::default();
        let false_branch = true_branch.new_branch(sym::TRUE);
        true_branch.write(&varnode, TestIterator::with_single_value(value))?;

        let live_branches = [true_branch, false_branch];
        let tree = MemoryTree::new(live_branches.iter(), std::iter::empty());
        let result = tree.read(&varnode);
        assert!(matches!(result, Err(Error::UndefinedData(addr)) if addr == varnode.address));

        let tree = MemoryTree::new(live_branches.iter().rev(), std::iter::empty());
        let result = tree.read(&varnode);
        assert!(matches!(result, Err(Error::UndefinedData(addr)) if addr == varnode.address));
        Ok(())
    }

    #[test]
    fn memory_tree_without_branches() -> Result<()> {
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let tree = MemoryTree::new(std::iter::empty(), std::iter::empty());
        let result = tree.read(&varnode);
        assert!(
            matches!(result, Err(Error::InternalError(msg)) if msg == "memory tree has no live branches")
        );
        Ok(())
    }

    #[test]
    fn executable_memory_instruction_bytes() -> Result<()> {
        // Setup memory with an address space
        let value = 0xbe;
        let mut memory = Memory::new();

        // Write an initial value to this address space
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&varnode, TestIterator::with_value(value, 1))?;
        let exec_mem = ExecutableMemory(&memory);
        let data = exec_mem
            .instruction_bytes(&varnode)
            .expect("failed to get instruction bytes");

        assert_eq!(data.len(), 1);
        assert_eq!(data[0], value);

        Ok(())
    }

    #[test]
    fn executable_memory_undefined_data() -> Result<()> {
        let memory = Memory::new();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let exec_mem = ExecutableMemory(&memory);

        let result = exec_mem.instruction_bytes(&varnode);
        assert!(
            matches!(result, Err(msg) if msg == format!("no data defined at address {address}", address = varnode.address))
        );

        Ok(())
    }

    #[test]
    fn executable_memory_symbolic_data() -> Result<()> {
        let mut memory = Memory::new();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&varnode, TestIterator::with_symbolic_value(1))?;
        let exec_mem = ExecutableMemory(&memory);

        let result = exec_mem.instruction_bytes(&varnode);
        assert!(matches!(result, Err(msg) if msg == "symbolic byte"));

        Ok(())
    }

    #[test]
    fn executable_memory_partial_undefined_data() -> Result<()> {
        // Setup memory with an address space
        let value = 0xbe;
        let mut memory = Memory::new();

        // Write an initial value to this address space
        let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&varnode, TestIterator::with_value(value, 1))?;
        let exec_mem = ExecutableMemory(&memory);

        // Increase varnode size by 1 to trigger undefined data during read
        varnode.size += 1;

        let data = exec_mem
            .instruction_bytes(&varnode)
            .expect("failed to get instruction bytes");

        assert_eq!(data.len(), 1);
        assert_eq!(data[0], value);

        Ok(())
    }

    #[test]
    fn read_const_addr_space() -> Result<()> {
        let expected_value = 0xdeadbeef;
        let varnode = VarnodeData::new(Address::new(const_space(), expected_value.into()), 4);
        let memory = Memory::new();
        let value: u32 = sym::concretize_into(memory.read(&varnode)?).expect("symbolic byte");
        assert_eq!(value, expected_value);
        Ok(())
    }

    #[test]
    fn read_const_addr_space_invalid() -> Result<()> {
        let varnode = VarnodeData::new(Address::new(const_space(), 0), 9);
        let memory = Memory::new();
        let result = memory.read(&varnode);
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("varnode size {size} exceeds maximum allowed for constant address space", size = varnode.size))
        );
        Ok(())
    }

    #[test]
    fn read_data_gap() -> Result<()> {
        let mut memory = Memory::new();

        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        memory.write(&varnode, TestIterator::with_single_value(0x5a))?;

        let varnode = VarnodeData::new(Address::new(address_space(0), 2), 1);
        memory.write(&varnode, TestIterator::with_single_value(0xa5))?;

        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 4);
        let result = memory.read(&varnode);

        let undefined_addr = Address::new(address_space(0), 1);
        assert!(matches!(result, Err(Error::UndefinedData(addr)) if addr == undefined_addr));
        Ok(())
    }

    #[test]
    fn write_address() -> Result<()> {
        let mut memory = Memory::new();
        let address = Address::new(address_space(0), 0);
        let value = 0xbe;
        memory.write_address(address.clone(), TestIterator::with_single_value(value))?;
        let read_value: u8 = sym::concretize_into(memory.read(&VarnodeData::new(address, 1))?)
            .expect("symbolic value");
        assert_eq!(read_value, value);

        Ok(())
    }

    #[test]
    fn read_bit() -> Result<()> {
        let mut memory = Memory::new();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let value = 0x01;
        memory.write(&varnode, TestIterator::with_single_value(value))?;
        let read_value: bool = memory.read_bit(&varnode)?.try_into().expect("symbolic bit");
        assert!(read_value, "expected to read true bit from memory");
        Ok(())
    }

    #[test]
    fn read_bit_invalid_varnode_size() -> Result<()> {
        let memory = Memory::new();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 2);
        let result = memory.read_bit(&varnode);
        assert!(
            matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("expected varnode size to be 1, actual {size}", size = varnode.size))
        );
        Ok(())
    }

    #[test]
    fn read_bit_undefined() -> Result<()> {
        let memory = Memory::new();
        let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
        let result = memory.read_bit(&varnode);
        assert!(matches!(result, Err(Error::UndefinedData(addr)) if addr == varnode.address));
        Ok(())
    }
}
