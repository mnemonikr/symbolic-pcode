use libsla::{Address, Sleigh, VarnodeData};
use sym::SymbolicBitVec;
use symbolic_pcode::mem::VarnodeDataStore;

use crate::common::{INITIAL_STACK, Memory};

pub fn initialize_libc_stack(memory: &mut Memory, sleigh: &impl Sleigh) {
    // The stack for libc programs:
    // * argc
    // * argv - list must be terminated by NULL pointer
    // * envp - list must be terminated by NULL pointer
    // * auxv - list must be terminated by NULL pointer
    let ram = sleigh
        .address_space_by_name("ram")
        .expect("failed to find ram");
    let argc = VarnodeData {
        address: Address {
            offset: INITIAL_STACK,
            address_space: ram.clone(),
        },
        size: 8,
    };
    memory
        .write(&argc, SymbolicBitVec::constant(1, u64::BITS as usize))
        .expect("failed to initialize argc on stack");

    // The argv list must be terminated by null pointer. Setting program name to null AND
    // terminating the list with NULL, whence 16 bytes
    //
    // musl has support for null program name:
    // https://git.musl-libc.org/cgit/musl/tree/src/env/__libc_start_main.c
    let argv = VarnodeData {
        address: Address {
            offset: argc.address.offset + argc.size as u64,
            address_space: ram.clone(),
        },
        size: 16,
    };
    memory
        .write(&argv, SymbolicBitVec::constant(0, (2 * u64::BITS) as usize))
        .expect("failed to initialize argv");

    let envp = VarnodeData {
        address: Address {
            offset: argv.address.offset + argv.size as u64,
            address_space: ram.clone(),
        },
        size: 8,
    };
    memory
        .write(&envp, SymbolicBitVec::constant(0, u64::BITS as usize))
        .expect("failed to initialize envp");

    // musl targets initialize the libc pagesize using aux[AT_PAGESZ]. For architectures without a
    // hardcoded value the libc pagesize is used. For example x86-64 has a hardcoded value of 4096
    // and so this is ignored. However for aarch64 there is no hardcoded value. This must be
    // supplied otherwise a pagesize of 0 will be used in some cases.
    let mut auxv = VarnodeData {
        address: Address {
            offset: envp.address.offset + envp.size as u64,
            address_space: ram.clone(),
        },
        size: 8,
    };

    // The index for AT_PAGESZ
    let at_pagesz = 6;
    memory
        .write(
            &auxv,
            SymbolicBitVec::constant(at_pagesz, u64::BITS as usize),
        )
        .expect("failed to write AT_PAGESZ into auxv");
    auxv.address.offset += auxv.size as u64;

    let page_size = 4096;
    memory
        .write(
            &auxv,
            SymbolicBitVec::constant(page_size, u64::BITS as usize),
        )
        .expect("failed to write page size into auxv");
    auxv.address.offset += auxv.size as u64;

    memory
        .write(&auxv, SymbolicBitVec::constant(0, u64::BITS as usize))
        .expect("failed to initialize auxv");
}
