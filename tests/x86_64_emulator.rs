mod common;

use std::path::Path;

use common::{x86_64_sleigh, TracingEmulator};
use sla::{Address, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};
use symbolic_pcode::{
    emulator::{PcodeEmulator, StandardPcodeEmulator},
    mem::{Memory, SymbolicMemory, SymbolicMemoryWriter},
    processor::{self, Processor},
};

const INITIAL_STACK: u64 = 0x8000000000;
const EXIT_RIP: u64 = 0xFEEDBEEF0BADF00D;

fn processor_with_image(image: impl AsRef<Path>, entry: u64) -> Processor<TracingEmulator, Memory> {
    let sleigh = x86_64_sleigh();
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let memory = Memory::new();
    let mut processor = Processor::new(sleigh, TracingEmulator::new(emulator), memory);

    //let mut processor = common::Processor::new();

    // Write image into memory
    let data = std::fs::read(image).expect("failed to read image file");
    let data_location = VarnodeData {
        address: Address {
            offset: 0,
            address_space: processor.default_code_space(),
        },
        size: data.len(),
    };
    processor
        .memory_mut()
        .write(&data_location, data.into_iter().map(SymbolicByte::from))
        .expect("failed to write image into memory");

    // Init RIP to entry
    let rip = processor.register("RIP");
    let data = entry.to_le_bytes().into_iter().map(SymbolicByte::from);
    processor
        .memory_mut()
        .write(&rip, data)
        .expect("failed to initialize RIP");

    // Init RBP to magic EXIT_RIP value
    let rbp = processor.register("RBP");
    let data = EXIT_RIP.to_le_bytes().into_iter().map(SymbolicByte::from);
    processor
        .memory_mut()
        .write(&rbp, data.clone())
        .expect("failed to initialize RBP");

    // Init stack address in memory to magic EXIT_RIP value
    let stack_addr = VarnodeData {
        address: Address {
            offset: INITIAL_STACK,
            address_space: processor.address_space("ram").expect("failed to find ram"),
        },
        size: data.len(),
    };
    processor
        .memory_mut()
        .write(&stack_addr, data)
        .expect("failed to initialize stack");

    // Init RSP to stack address
    let rsp = processor.register("RSP");
    let data = INITIAL_STACK
        .to_le_bytes()
        .into_iter()
        .map(SymbolicByte::from);
    processor
        .memory_mut()
        .write(&rsp, data)
        .expect("failed to initialize RSP");

    processor
}

fn init_registers<E: PcodeEmulator, M: SymbolicMemory>(processor: &mut Processor<E, M>) {
    let mut bitvar = 0;
    let registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI"]
        .into_iter()
        .map(str::to_owned)
        .chain((8..16).map(|n| format!("R{n}")))
        .collect::<Vec<_>>();

    for register in registers {
        let mut bytes = Vec::with_capacity(8);
        for _ in 0..8 {
            let byte: SymbolicByte = [
                SymbolicBit::Variable(bitvar),
                SymbolicBit::Variable(bitvar + 1),
                SymbolicBit::Variable(bitvar + 2),
                SymbolicBit::Variable(bitvar + 3),
                SymbolicBit::Variable(bitvar + 4),
                SymbolicBit::Variable(bitvar + 5),
                SymbolicBit::Variable(bitvar + 6),
                SymbolicBit::Variable(bitvar + 7),
            ]
            .into();
            bytes.push(byte);
            bitvar += 8;
        }

        processor
            .write_register(register, bytes.into_iter())
            .expect("failed to write data");
    }
}

/// Confirms the functionality of general-purpose x86-64 registers and overlapping behavior.
#[test]
fn x86_64_registers() {
    let sleigh = x86_64_sleigh();
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let memory = Memory::new();
    let mut processor = Processor::new(sleigh, emulator, memory);

    let registers = vec!['A', 'B', 'C', 'D'];
    for register in registers {
        processor
            .write_register(
                format!("R{register}X"),
                vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88].into_iter(),
            )
            .expect("failed to write register");
        let rax: u64 = processor
            .read_register(format!("R{register}X"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(rax, 0x8877665544332211);
        let eax: u32 = processor
            .read_register(format!("E{register}X"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(eax, 0x44332211);
        let ax: u16 = processor
            .read_register(format!("{register}X"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(ax, 0x2211);
        let ah: u8 = processor
            .read_register(format!("{register}H"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(ah, 0x22);
        let al: u8 = processor
            .read_register(format!("{register}L"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(al, 0x11);
    }

    let registers = vec!["SI", "DI", "BP", "SP"];
    for register in registers {
        processor
            .write_register(
                format!("R{register}"),
                vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88].into_iter(),
            )
            .expect("failed to write register");
        let r: u64 = processor
            .read_register(format!("R{register}"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);
        let e: u32 = processor
            .read_register(format!("E{register}"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(e, 0x44332211);
        let b: u16 = processor
            .read_register(format!("{register}"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(b, 0x2211);
        let l: u8 = processor
            .read_register(format!("{register}L"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(l, 0x11);
    }

    for register in 8..=15 {
        processor
            .write_register(
                format!("R{register}"),
                vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88].into_iter(),
            )
            .expect("failed to write register");
        let r: u64 = processor
            .read_register(format!("R{register}"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);
        let rd: u32 = processor
            .read_register(format!("R{register}D"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(rd, 0x44332211);
        let rw: u16 = processor
            .read_register(format!("R{register}W"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(rw, 0x2211);
        let rb: u8 = processor
            .read_register(format!("R{register}B"))
            .expect("failed to read register")
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(rb, 0x11);
    }
}

/// Emulates the following x86-64 instructions:
///
/// ram:0000000000000000 | PUSH RBP
/// ram:0000000000000001 | MOV RBP,RSP
/// ram:0000000000000004 | MOV dword ptr [RBP + -0x4],EDI
/// ram:0000000000000007 | MOV EAX,dword ptr [RBP + -0x4]
/// ram:000000000000000a | ADD EAX,EAX
/// ram:000000000000000c | POP RBP
/// ram:000000000000000d | RET
#[test]
fn doubler_32b() -> Result<(), processor::Error> {
    let sleigh = x86_64_sleigh();
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let memory = Memory::new();
    let mut processor = Processor::new(sleigh, emulator, memory);
    let base_addr = 0x84210000;
    let num_instructions = 7;

    let code_space = processor.default_code_space();
    processor.memory_mut().write_address(
        Address::new(code_space, base_addr),
        b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3\x00\x00"
            .into_iter()
            .copied(),
    )?;

    processor.write_register(
        "RSP",
        b"\x00\x01\x01\x01\x01\x01\x01\x00".into_iter().copied(),
    )?;

    processor.write_register(
        "RSP",
        vec![0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00].into_iter(),
    )?;

    processor.write_register(
        "RBP",
        vec![0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x00].into_iter(),
    )?;

    processor.write_address(
        Address::new(processor.address_space("ram")?, 0x0001010101010100),
        vec![0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66].into_iter(),
    )?;

    let initial_value: u32 = 0x99;
    processor.write_register("EDI", initial_value.to_le_bytes().into_iter())?;
    processor.write_register("RIP", base_addr.to_le_bytes().into_iter())?;
    for _ in 0..num_instructions {
        processor.single_step("RIP")?;
    }

    let rip: u64 = processor
        .read_register("RIP")?
        .into_iter()
        .collect::<SymbolicBitVec>()
        .try_into()
        .expect("failed to convert rip value to u64");
    assert_eq!(rip, 0x66778899aabbccdd, "return address on stack");

    let rax: u32 = processor
        .read_register("RAX")?
        .into_iter()
        .collect::<SymbolicBitVec>()
        .try_into()
        .expect("failed to convert rax value to u32");
    assert_eq!(
        rax,
        2 * initial_value,
        "result should be double initial value: {initial_value}",
    );

    Ok(())
}

// This test requires the coverage file to be compiled ahead of time.
// The coverage file can be compiled with gcc like the following:
//
// gcc -fno-stack-protector -fPIC -mpopcnt -o tests/data/coverage/coverage tests/data/coverage/coverage.c
//
// Then check the resulting binary for the main function
//
// objdump -t tests/data/coverage/coverage | grep main
#[test]
fn pcode_coverage() -> processor::Result<()> {
    // Use address of the main function for the entry point
    let mut processor = processor_with_image("tests/data/coverage/coverage", 0x1675);
    init_registers(&mut processor);

    loop {
        processor.single_step("RIP")?;

        // Check if RIP is the magic value
        let rip: u64 = processor
            .read_register("RIP")?
            .into_iter()
            .collect::<SymbolicBitVec>()
            .try_into()
            .expect("failed to convert RIP to u64");
        if rip == EXIT_RIP {
            break;
        }
    }

    let rax: u64 = processor
        .read_register("RAX")?
        .into_iter()
        .collect::<SymbolicBitVec>()
        .try_into()
        .expect("failed to convert RAX to u64");
    assert_eq!(rax, 0);

    processor
        .emulator()
        .executed_instructions()
        .into_iter()
        .for_each(|(opcode, count)| println!("Executed {opcode:?}: {count}"));

    // Currently the following p-code instructions are not covered by this test:
    //
    // Piece
    // Bool(Xor)
    // Int(LessThanOrEqual(Signed))
    // Int(LessThanOrEqual(Unsigned))
    assert_eq!(
        processor
            .emulator()
            .executed_instructions()
            .into_iter()
            .count(),
        38
    );
    Ok(())
}
