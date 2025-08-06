use std::{collections::BTreeMap, rc::Rc};

use thiserror;

use libsla::{Address, AddressSpaceId, AddressSpaceType, LoadImage, VarnodeData};
use pcode_ops::{BitwisePcodeOps, PcodeOps};

/// Memory result type
pub type Result<T> = std::result::Result<T, Error>;

/// Possible memory errors
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// There is no data defined at a particular address
    #[error("data not defined at {target} + {relative_offset}")]
    UndefinedData {
        target: VarnodeData,
        relative_offset: usize,
    },

    /// The arguments provided for a given request are invalid
    #[error("arguments provided are not valid: {0}")]
    InvalidArguments(String),

    /// An internal error that cannot be handled
    #[error("an internal error occurred: {0}")]
    InternalError(String),
}

pub trait VarnodeDataStore {
    type Value: PcodeOps;

    fn read(&self, source: &VarnodeData) -> Result<Self::Value>;
    fn read_bit(&self, source: &VarnodeData) -> Result<<Self::Value as PcodeOps>::Bit> {
        if source.size > 1 {
            Err(Error::InvalidArguments(format!(
                "expected varnode size to be 1, actual {size}",
                size = source.size
            )))
        } else {
            self.read(source).map(PcodeOps::lsb)
        }
    }

    fn write(&mut self, destination: &VarnodeData, data: Self::Value) -> Result<()>;
    fn write_bit(
        &mut self,
        destination: &VarnodeData,
        data: <Self::Value as PcodeOps>::Bit,
    ) -> Result<()>;
}

/// Generic memory structure that stores [PcodeOps::Byte] keyed on [AddressSpaceId]. This structure
/// is the foundation for other memory structures.
pub struct GenericMemory<T: PcodeOps> {
    data: BTreeMap<AddressSpaceId, BTreeMap<u64, T::Byte>>,
}

impl<T: PcodeOps> Default for GenericMemory<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<T: PcodeOps> GenericMemory<T> {
    fn read_bytes(
        &self,
        source: &VarnodeData,
    ) -> Result<Vec<<<Self as VarnodeDataStore>::Value as PcodeOps>::Byte>> {
        // Sources with a constant address space are 64-bit constants encoded in the address offset
        if source.address.address_space.space_type == AddressSpaceType::Constant {
            const MAX_BYTES: usize = (u64::BITS / 8) as usize;
            if source.size <= MAX_BYTES {
                return Ok(source
                    .address
                    .offset
                    .to_le_bytes()
                    .into_iter()
                    .take(source.size)
                    .map(<<Self as VarnodeDataStore>::Value as PcodeOps>::Byte::from)
                    .collect());
            } else {
                return Err(Error::InvalidArguments(format!(
                    "varnode size {size} exceeds maximum allowed for constant address space",
                    size = source.size
                )));
            }
        }

        let space_id = source.address.address_space.id;
        let memory = self.data.get(&space_id).ok_or(Error::UndefinedData {
            target: source.clone(),
            relative_offset: 0,
        })?;

        // Collect into a Vec or return the first undefined offset
        let result = memory
            .range(source.range())
            .enumerate()
            .map(|(i, (&offset, v))| {
                let i = u64::try_from(i).map_err(|_| offset)?;
                if offset == i + source.address.offset {
                    Ok(v.clone())
                } else {
                    // Undefined offset
                    Err(i)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>();

        let bytes = result.map_err(|relative_offset| Error::UndefinedData {
            target: source.clone(),
            relative_offset: relative_offset as usize,
        })?;

        if bytes.len() == source.size {
            Ok(bytes)
        } else {
            Err(Error::UndefinedData {
                target: source.clone(),
                relative_offset: bytes.len(),
            })
        }
    }

    fn write_bytes(
        &mut self,
        destination: &VarnodeData,
        data: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = <T as PcodeOps>::Byte>>,
    ) -> Result<()> {
        // Make sure data provided matches expected amount for destination
        let data = data.into_iter();
        if data.len() != destination.size {
            return Err(Error::InvalidArguments(format!(
                "expected to write {expected} byte(s), got {actual} byte(s)",
                expected = destination.size,
                actual = data.len()
            )));
        }

        // Make sure the varnode does not overflow the offset
        let overflows = u64::try_from(destination.size)
            .ok()
            .and_then(|size| destination.address.offset.checked_add(size))
            .is_none();

        if overflows {
            return Err(Error::InvalidArguments(format!(
                "varnode size {size} overflows address offset {offset}",
                size = destination.size,
                offset = destination.address.offset
            )));
        }

        let space_id = destination.address.address_space.id;
        let memory = self.data.entry(space_id).or_default();

        let mut offset = destination.address.offset;
        for byte in data {
            memory.insert(offset, byte);
            offset += 1;
        }

        Ok(())
    }
}

impl<T: PcodeOps> VarnodeDataStore for GenericMemory<T> {
    type Value = T;

    fn read(&self, source: &VarnodeData) -> Result<Self::Value> {
        Ok(self.read_bytes(source)?.into_iter().collect())
    }

    fn write(&mut self, destination: &VarnodeData, data: Self::Value) -> Result<()> {
        self.write_bytes(destination, data.into_le_bytes())
    }

    fn write_bit(
        &mut self,
        destination: &VarnodeData,
        bit: <Self::Value as PcodeOps>::Bit,
    ) -> Result<()> {
        self.write_bytes(destination, [<Self::Value as PcodeOps>::Byte::from(bit)])
    }
}

/// A branching memory model which branches on [PcodeOps::Bit] predicates. Note that this only
/// tracks the predicate value, it does not enforce it.
///
/// # Reading values
///
/// [Self::read] is *not* conditioned on the branch predicate. See [Self::predicated_read].
#[derive(Debug)]
pub struct MemoryBranch<M: VarnodeDataStore + Default> {
    leaf_predicate: <M::Value as PcodeOps>::Bit,
    branch_predicate: <M::Value as PcodeOps>::Bit,
    parent: Option<Rc<Self>>,
    memory: M,
}

impl<M: VarnodeDataStore + Default> Default for MemoryBranch<M> {
    fn default() -> Self {
        Self {
            leaf_predicate: true.into(),
            branch_predicate: true.into(),
            parent: Default::default(),
            memory: Default::default(),
        }
    }
}

impl<M: VarnodeDataStore + Default> MemoryBranch<M> {
    pub fn new(memory: M) -> Self {
        Self {
            memory,
            ..Default::default()
        }
    }

    /// Get the parent memory branch if it exists
    pub fn parent(&self) -> Option<&MemoryBranch<M>> {
        self.parent.as_deref()
    }

    /// Get the predicate associated with the entire branch. The branch predicate is the
    /// conjunction of all predicates leading to and including the leaf predicate of this branch.
    ///
    /// The root memory branch has a branch predicate of `true` as [PcodeOps::Bit].
    pub fn branch_predicate(&self) -> &<M::Value as PcodeOps>::Bit {
        &self.branch_predicate
    }

    /// Get the predicate associated with the leaf of this branch. Other predicates in the branch
    /// are **not** included.
    ///
    /// The root memory branch has a leaf predicate of `true` as [PcodeOps::Bit].
    pub fn leaf_predicate(&self) -> &<M::Value as PcodeOps>::Bit {
        &self.leaf_predicate
    }

    /// Branch this tree on the given predicate. The current tree branch predicate is updated to
    /// the given predicate. A new branch of the tree is returned with a negation of the predicate.
    pub fn new_branch(&mut self, predicate: <M::Value as PcodeOps>::Bit) -> Self {
        // Build the new shared parent
        let mut parent = Self {
            leaf_predicate: predicate.clone(),
            branch_predicate: self.branch_predicate.clone().and(predicate.clone()),
            parent: None,
            memory: M::default(),
        };

        // Update the shared parent to hold contents of this branch
        std::mem::swap(self, &mut parent);

        // Point this branch to new shared parent
        let rc = Rc::new(parent);
        self.parent = Some(Rc::clone(&rc));

        // Create new branch with negated predicate
        Self {
            leaf_predicate: predicate.clone().not(),
            branch_predicate: rc.branch_predicate.clone().and(predicate.not()),
            parent: Some(rc),
            memory: M::default(),
        }
    }

    /// Read a value from memory with the necessary predicates applied to the result. For example,
    /// if a value `V` is predicated on `X`, then the response is the implication `X => V`, or
    /// equivalently `!X | V`.
    ///
    /// If a portion of a value `V` has not been updated in this branch, then that portion is only
    /// predicated on the parent predicate in which it is stored.
    pub fn predicated_read(&self, varnode: &VarnodeData) -> Result<M::Value> {
        let value: M::Value = self.read(varnode)?;
        let predicate = M::Value::fill_bytes_with(self.branch_predicate.clone(), varnode.size);
        Ok(predicate.not().or(value))
    }
}

impl<M: VarnodeDataStore + Default> VarnodeDataStore for MemoryBranch<M> {
    type Value = M::Value;

    /// Read the bytes for this varnode.
    fn read(&self, varnode: &VarnodeData) -> Result<M::Value> {
        // Not using a question mark operator here in order to check for undefined data
        let result = self.memory.read(varnode);

        if let Some(parent) = &self.parent {
            if let Err(Error::UndefinedData {
                target,
                relative_offset: num_valid_bytes,
            }) = result
            {
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
                let data = self.memory.read(&valid_input)?;

                // Read the missing data from parent
                let mut parent_address = target.address.clone();
                parent_address.offset += num_valid_bytes as u64;

                let parent_varnode = VarnodeData {
                    address: parent_address,
                    size: varnode.size - num_valid_bytes,
                };
                let parent_data = parent.read(&parent_varnode)?;

                // Combine the two and return the result
                return Ok(parent_data.piece(data));
            }
        }

        result
    }

    /// Write the data to the location specified by the varnode.
    fn write(&mut self, destination: &VarnodeData, data: Self::Value) -> Result<()> {
        self.memory.write(destination, data)
    }

    fn write_bit(
        &mut self,
        destination: &VarnodeData,
        data: <Self::Value as PcodeOps>::Bit,
    ) -> Result<()> {
        self.memory.write_bit(destination, data)
    }
}

/// Collection of all memory branches into a single tree. Tree is composed of both live and dead
/// branches. A *dead branch* is a branch of memory that has no bearing on an outcome. These branches
/// are necessary to include so that their predicates are appropriately excluded from the outcome.
pub struct MemoryTree<'b, 'd, M: VarnodeDataStore + Default> {
    branches: Vec<&'b MemoryBranch<M>>,
    dead_branches: Vec<&'d MemoryBranch<M>>,
}

impl<'b, 'd, M: VarnodeDataStore + Default> MemoryTree<'b, 'd, M> {
    /// Create a new memory tree composed of the given (live) branches and dead branches.
    pub fn new(
        branches: impl IntoIterator<Item = &'b MemoryBranch<M>>,
        dead_branches: impl IntoIterator<Item = &'d MemoryBranch<M>>,
    ) -> Self {
        Self {
            branches: branches.into_iter().collect(),
            dead_branches: dead_branches.into_iter().collect(),
        }
    }

    /// Read data from the requested source. This will read data from each live branch using a
    /// [predicated read](MemoryBranch::predicated_read) and return the conjunction these
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
    pub fn read(&self, varnode: &VarnodeData) -> Result<M::Value> {
        let result = self
            .branches
            .iter()
            .map(|&branch| branch.predicated_read(varnode))
            .reduce(|x, y| {
                // Must check first argument first since it is the accumulator
                // If an error is propagating it will propagate here.
                if let Ok(x) = x {
                    if let Ok(y) = y { Ok(x.and(y)) } else { y }
                } else {
                    x
                }
            })
            .ok_or_else(|| {
                Error::InternalError("memory tree has no live branches".to_string())
            })??;

        // Assert dead branches not taken
        Ok(result.and(M::Value::fill_bytes_with(
            self.dead_branches_not_taken_predicate(),
            varnode.size,
        )))
    }

    fn dead_branches_not_taken_predicate(&self) -> <M::Value as PcodeOps>::Bit {
        let dead_branch_taken = self
            .dead_branches
            .iter()
            .map(|&branch| branch.branch_predicate())
            .fold(<M::Value as PcodeOps>::Bit::from(false), |x, y| {
                x.clone().or(y.clone())
            });

        dead_branch_taken.not()
    }
}

/// Memory that holds binary-encoded executable instructions.
pub struct ExecutableMemory<'a, M: VarnodeDataStore>(pub &'a M);

/// Implementation of the LoadImage trait to enable loading instructions from memory
impl<M: VarnodeDataStore> LoadImage for ExecutableMemory<'_, M> {
    fn instruction_bytes(&self, input: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        let value = self.0.read(input);

        // The number of bytes requested may exceed valid data in memory.
        // In that case only read and return the defined bytes.
        let value = match value {
            Err(Error::UndefinedData {
                target,
                relative_offset,
            }) if relative_offset > 0 => {
                let input = VarnodeData {
                    // The relative offset is effectively the number of valid bytes
                    size: relative_offset,
                    address: target.address.clone(),
                };
                self.0.read(&input)
            }
            _ => value,
        };

        let value = value.map_err(|err| err.to_string())?;

        value
            .into_le_bytes()
            .map(|x| x.try_into())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_err| "symbolic byte".to_string())
    }
}
