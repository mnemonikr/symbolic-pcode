use std::borrow::Cow;

use super::test_fixture::SymbolicValue;
use crate::mem::{Error, Result, *};
use libsla::*;
use pcode_ops::*;

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
    let mut memory = GenericMemory::<Pcode128>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 8);
    let value: Pcode128 = 0x8877665544332211u64.into();

    // Read and write value into memory
    memory.write(&varnode, value)?;
    let read_value = memory.read(&varnode)?;

    // Confirm bytes read matches varnode size
    let iter = read_value.into_le_bytes();
    assert_eq!(iter.len(), varnode.size);

    // Confirm bytes are expected values
    for (i, actual) in iter.enumerate() {
        let expected = u8::from_str_radix(&format!("{x}{x}", x = i + 1), 16).unwrap();
        assert_eq!(actual, expected);
    }

    Ok(())
}

#[test]
fn memory_branch_read() -> Result<()> {
    // Setup memory with an address space
    let mut memory = GenericMemory::<Pcode128>::default();

    // Write an initial value to this address space
    let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 2);

    memory.write(&varnode, 0xbebeu16.into())?;

    // Create a memory tree from this initialized memory
    let mut tree = MemoryBranch::new(memory);
    let predicate = true;
    let _ = tree.new_branch(predicate);

    // Overwite part of the initial value
    varnode.size = 1;
    tree.write(&varnode, 0xefu8.into())?;

    // Read the entire value and confirm the overwritten portion is read along with the portion
    // that is not overwritten
    varnode.size = 2;
    assert_eq!(u16::try_from(tree.read(&varnode)?).unwrap(), 0xbeefu16);

    Ok(())
}

#[test]
fn new_branch() -> Result<()> {
    // Setup memory with an address space
    let addr_space = address_space(0);
    let mut memory = GenericMemory::<Pcode128>::default();

    // Write an initial value to this address space
    let mut varnode = VarnodeData {
        address: Address {
            offset: 0,
            address_space: addr_space.clone(),
        },
        size: 2,
    };
    memory.write(&varnode, 0xbebeu16.into())?;

    // Create a new branch
    let mut true_branch = MemoryBranch::new(memory);
    let predicate = true;
    let false_branch = true_branch.new_branch(predicate);

    // Confirm predicates are correct
    assert_eq!(*true_branch.leaf_predicate(), predicate);
    assert_eq!(*true_branch.branch_predicate(), predicate);
    assert_eq!(*false_branch.leaf_predicate(), !predicate);
    assert_eq!(*false_branch.branch_predicate(), !predicate);

    // Overwite part of the initial value
    varnode.size = 1;
    true_branch.write(&varnode, 0xefu8.into())?;
    varnode.size = 2;

    // Read the entire value and confirm the overwritten portion is read along with the portion
    // that is not overwritten
    let old_value: u16 = 0xbebe;
    let new_value: u16 = 0xbeef;
    assert_eq!(
        u16::try_from(true_branch.read(&varnode)?).expect("buffer conversion failed"),
        new_value
    );
    // Confirm false branch value is unchanged
    assert_eq!(
        u16::try_from(false_branch.read(&varnode)?).expect("buffer conversion failed"),
        old_value
    );

    Ok(())
}

#[test]
fn memory_tree() -> Result<()> {
    let mut memory = MemoryBranch::new(GenericMemory::<Pcode128>::default());
    let x = true;
    let y = true;
    let panic_memory = memory.new_branch(x);
    let mut y_branch = memory.new_branch(y);
    let return_address = VarnodeData::new(Address::new(address_space(0), 0), 1);
    memory.write(&return_address, 1u8.into())?;
    y_branch.write(&return_address, 2u8.into())?;
    let live_branches = [memory, y_branch];
    let dead_branches = [panic_memory];

    let tree = MemoryTree::new(live_branches.iter(), dead_branches.iter());
    let return_value = tree.read(&return_address)?;
    assert_eq!(u8::try_from(return_value).unwrap(), 1);

    Ok(())
}

#[test]
fn memory_tree_false_branch_unwritten() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 2);
    memory.write(&varnode, 0xbebeu16.into())?;

    let mut true_branch = MemoryBranch::new(memory);
    let false_branch = true_branch.new_branch(true);

    varnode.size = 1;
    true_branch.write(&varnode, 0xefu8.into())?;
    varnode.size = 2;

    let live_branches = [true_branch, false_branch];
    let tree = MemoryTree::new(live_branches.iter(), std::iter::empty());
    let tree_value = u16::try_from(tree.read(&varnode)?).unwrap();

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
fn memory_tree_live_branch_taken() -> Result<()> {
    let mut memory = MemoryBranch::<GenericMemory<Pcode128>>::default();
    let condition = true;
    let memory_else = memory.new_branch(condition);
    let return_address = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let written_return_value = 0x5au8;
    memory.write(&return_address, written_return_value.into())?;
    let live_branches = [memory];
    let dead_branches = [memory_else];

    let tree = MemoryTree::new(live_branches.iter(), dead_branches.iter());
    let return_value = tree.read(&return_address)?;

    // The expected return value should be conjunction of:
    //
    // * live branch taken (!c | R)
    // * dead branch NOT taken: !(!c), or c
    //
    // so c & (!c | R) reduces to c & R.
    let expected_return_value = written_return_value;

    assert_eq!(u8::try_from(return_value).unwrap(), expected_return_value);
    Ok(())
}

#[test]
fn memory_tree_live_branch_not_taken() -> Result<()> {
    let mut memory = MemoryBranch::<GenericMemory<Pcode128>>::default();
    let condition = false;
    let memory_else = memory.new_branch(condition);
    let return_address = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let written_return_value = 0x5au8;
    memory.write(&return_address, written_return_value.into())?;
    let live_branches = [memory];
    let dead_branches = [memory_else];

    let tree = MemoryTree::new(live_branches.iter(), dead_branches.iter());
    let return_value = tree.read(&return_address)?;

    // The expected return value should be conjunction of:
    //
    // * live branch taken (!c | R)
    // * dead branch NOT taken: !(!c), or c
    //
    // so c & (!c | R) reduces to c & R.
    let expected_return_value = 0;

    assert_eq!(u8::try_from(return_value).unwrap(), expected_return_value);
    Ok(())
}

#[test]
fn write_too_few_bytes() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let destination = VarnodeData::new(Address::new(address_space(0), 0), 2);
    let original_value = 0xbeefu16;
    memory.write(&destination, original_value.into())?;

    let result = memory.write(&destination, 0xffu8.into());
    assert!(
        matches!(result, Err(Error::InvalidArguments(msg)) if msg == "expected to write 2 byte(s), got 1 byte(s)")
    );

    // Validate that the original value in memory is not modified by the failed write
    let value = memory.read(&destination)?;
    assert_eq!(u16::try_from(value).unwrap(), original_value);

    Ok(())
}

#[test]
fn write_too_many_bytes() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let destination = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let original_value = 0xbeu8;
    memory.write(&destination, original_value.into())?;

    let result = memory.write(&destination, 0xfadeu16.into());
    assert!(
        matches!(result, Err(Error::InvalidArguments(msg)) if msg == "expected to write 1 byte(s), got 2 byte(s)")
    );

    // Validate that the original value in memory is not modified by the failed write
    let value = memory.read(&destination)?;
    assert_eq!(u8::try_from(value).unwrap(), original_value);

    Ok(())
}

#[test]
fn write_overflows_address() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let destination = VarnodeData::new(Address::new(address_space(0), u64::MAX), 1);
    let result = memory.write(&destination, 0xffu8.into());
    assert!(
        matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("varnode size {size} overflows address offset {offset}", offset = u64::MAX, size = destination.size))
    );
    Ok(())
}

#[test]
fn memory_branch_read_from_parent() -> Result<()> {
    let mut branch = MemoryBranch::<GenericMemory<_>>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let value = 1;
    branch.write(&varnode, Pcode128::from(value))?;

    let _child = branch.new_branch(true);
    let read_value: u8 = branch.read(&varnode)?.try_into().unwrap();
    assert_eq!(read_value, value);
    Ok(())
}

#[test]
fn memory_branch_partial_read_failure() -> Result<()> {
    let mut branch = MemoryBranch::<GenericMemory<_>>::default();
    let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let value = 1u8;
    let _child = branch.new_branch(true);
    branch.write(&varnode, Pcode128::from(value))?;

    // Increase varnode size such that will now read undefined data
    varnode.size = 2;
    let result = branch.read(&varnode);

    // The failure varnode will occur due to reading from the parent branch.
    let mut expected_failure_varnode = varnode.clone();
    expected_failure_varnode.address.offset += 1;
    expected_failure_varnode.size -= 1;

    if let Err(Error::UndefinedData {
        target,
        relative_offset,
    }) = result
    {
        assert_eq!(
            target, expected_failure_varnode,
            "undefined data target should match varnode used to read"
        );
        assert_eq!(
            relative_offset, 0,
            "relative offset of failed parent read should be 0"
        );
    }
    Ok(())
}

#[test]
fn memory_branch_predicated_read() -> Result<()> {
    let mut branch = MemoryBranch::<GenericMemory<Pcode128>>::default();
    let _child = branch.new_branch(true);
    let varnode = VarnodeData::new(Address::new(address_space(0), u64::MAX), 1);
    let result = branch.read(&varnode);
    assert!(matches!(result, Err(Error::UndefinedData { target, .. }) if target == varnode));
    Ok(())
}

#[test]
fn memory_tree_branches_with_undefined_data() -> Result<()> {
    let value = 0xbeu8;
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let mut true_branch = MemoryBranch::<GenericMemory<_>>::default();
    let false_branch = true_branch.new_branch(true);
    true_branch.write(&varnode, Pcode128::from(value))?;

    let live_branches = [true_branch, false_branch];
    let tree = MemoryTree::new(live_branches.iter(), std::iter::empty());
    let result = tree.read(&varnode);
    assert!(matches!(result, Err(Error::UndefinedData { target, .. }) if target == varnode));

    let tree = MemoryTree::new(live_branches.iter().rev(), std::iter::empty());
    let result = tree.read(&varnode);
    assert!(matches!(result, Err(Error::UndefinedData { target, .. }) if target == varnode));
    Ok(())
}

#[test]
fn memory_tree_without_branches() -> Result<()> {
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let tree = MemoryTree::<GenericMemory<Pcode128>>::new(std::iter::empty(), std::iter::empty());
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
    let mut memory = GenericMemory::<Pcode128>::default();

    // Write an initial value to this address space
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    memory.write(&varnode, value.into())?;
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
    let memory = GenericMemory::<Pcode128>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let exec_mem = ExecutableMemory(&memory);

    let result = exec_mem.instruction_bytes(&varnode);
    let msg = result.expect_err("result should be undefined data error");
    assert_eq!(
        msg,
        format!("data not defined at {varnode} + {offset}", offset = 0)
    );
    Ok(())
}

#[test]
fn executable_memory_symbolic_data() -> Result<()> {
    let mut memory = GenericMemory::<SymbolicValue>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    memory.write(&varnode, SymbolicValue::default())?;
    let exec_mem = ExecutableMemory(&memory);

    let result = exec_mem.instruction_bytes(&varnode);
    assert!(matches!(result, Err(msg) if msg == "symbolic byte"));

    Ok(())
}

#[test]
fn executable_memory_partial_undefined_data() -> Result<()> {
    // Setup memory with an address space
    let value = 0xbe;
    let mut memory = GenericMemory::<Pcode128>::default();

    // Write an initial value to this address space
    let mut varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    memory.write(&varnode, value.into())?;
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
    let memory = GenericMemory::<Pcode128>::default();
    let value: u32 = memory.read(&varnode)?.try_into().unwrap();
    assert_eq!(value, expected_value);
    Ok(())
}

#[test]
fn read_const_addr_space_invalid() -> Result<()> {
    let varnode = VarnodeData::new(Address::new(const_space(), 0), 9);
    let memory = GenericMemory::<Pcode128>::default();
    let result = memory.read(&varnode);
    assert!(
        matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("varnode size {size} exceeds maximum allowed for constant address space", size = varnode.size))
    );
    Ok(())
}

#[test]
fn read_data_gap() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();

    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    memory.write(&varnode, 0x5au8.into())?;

    let varnode = VarnodeData::new(Address::new(address_space(0), 2), 1);
    memory.write(&varnode, 0xa5u8.into())?;

    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 4);
    let result = memory.read(&varnode);

    assert!(
        matches!(result, Err(Error::UndefinedData { target, relative_offset }) if target == varnode && relative_offset == 1)
    );
    Ok(())
}

#[test]
fn read_bit() -> Result<()> {
    let mut memory = GenericMemory::<Pcode128>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let value = 0x01u8;
    memory.write(&varnode, value.into())?;
    let read_value = memory.read_bit(&varnode)?;
    assert!(read_value, "expected to read true bit from memory");
    Ok(())
}

#[test]
fn read_bit_invalid_varnode_size() -> Result<()> {
    let memory = GenericMemory::<Pcode128>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 2);
    let result = memory.read_bit(&varnode);
    assert!(
        matches!(result, Err(Error::InvalidArguments(msg)) if msg == format!("expected varnode size to be 1, actual {size}", size = varnode.size))
    );
    Ok(())
}

#[test]
fn read_bit_undefined() -> Result<()> {
    let memory = GenericMemory::<Pcode128>::default();
    let varnode = VarnodeData::new(Address::new(address_space(0), 0), 1);
    let result = memory.read_bit(&varnode);
    assert!(matches!(result, Err(Error::UndefinedData { target, .. }) if target == varnode));
    Ok(())
}
