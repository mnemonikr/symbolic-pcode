mod common;

use std::path::Path;

use common::{x86_64_sleigh, ProcessorHandlerX86, TracingEmulator};
use sla::{Address, Sleigh, VarnodeData};
use sym::{self, SymbolicBit, SymbolicBitVec, SymbolicByte};
use symbolic_pcode::{
    emulator::StandardPcodeEmulator,
    mem::{Memory, MemoryTree, SymbolicMemoryReader, SymbolicMemoryWriter},
    processor::{self, Processor, ProcessorManager, ProcessorState},
};

const INITIAL_STACK: u64 = 0x8000000000;
const EXIT_RIP: u64 = 0xFEEDBEEF0BADF00D;

fn processor_with_image(
    image: impl AsRef<Path>,
    entry: u64,
) -> ProcessorManager<TracingEmulator, ProcessorHandlerX86> {
    let sleigh = x86_64_sleigh();
    let mut memory = Memory::new();

    // Write image into memory
    let data = std::fs::read(image).expect("failed to read image file");
    let data_location = VarnodeData {
        address: Address {
            offset: 0,
            address_space: sleigh.default_code_space(),
        },
        size: data.len(),
    };

    memory
        .write(&data_location, data.into_iter().map(SymbolicByte::from))
        .expect("failed to write image into memory");

    // Init RIP to entry

    let rip = sleigh.register_from_name("RIP").expect("invalid register");
    let data = entry.to_le_bytes().into_iter().map(SymbolicByte::from);
    memory.write(&rip, data).expect("failed to initialize RIP");

    // Init RBP to magic EXIT_RIP value
    let rbp = sleigh.register_from_name("RBP").expect("invalid register");
    let data = EXIT_RIP.to_le_bytes().into_iter().map(SymbolicByte::from);
    memory
        .write(&rbp, data.clone())
        .expect("failed to initialize RBP");

    // Init stack address in memory to magic EXIT_RIP value
    let stack_addr = VarnodeData {
        address: Address {
            offset: INITIAL_STACK,
            address_space: sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
        },
        size: data.len(),
    };
    memory
        .write(&stack_addr, data)
        .expect("failed to initialize stack");

    // Init RSP to stack address
    let rsp = sleigh.register_from_name("RSP").expect("invalid register");
    let data = INITIAL_STACK
        .to_le_bytes()
        .into_iter()
        .map(SymbolicByte::from);
    memory.write(&rsp, data).expect("failed to initialize RSP");

    init_registers(&sleigh, &mut memory);

    let handler = ProcessorHandlerX86::new(&sleigh);
    let processor = Processor::new(
        memory,
        TracingEmulator::new(StandardPcodeEmulator::new(sleigh.address_spaces())),
        handler,
    );
    ProcessorManager::new(sleigh, processor)
}

fn init_registers(sleigh: &impl Sleigh, memory: &mut impl SymbolicMemoryWriter) {
    let mut bitvar = 0;
    let registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI"]
        .into_iter()
        .map(str::to_owned)
        .chain((8..16).map(|n| format!("R{n}")))
        .collect::<Vec<_>>();

    for register_name in registers {
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

        let register = sleigh
            .register_from_name(&register_name)
            .unwrap_or_else(|err| panic!("invalid register {register_name}: {err}"));
        memory
            .write(&register, bytes.into_iter())
            .unwrap_or_else(|err| panic!("failed to write register {register_name}: {err}"));
    }
}

/// Confirms the functionality of general-purpose x86-64 registers and overlapping behavior.
#[test]
fn x86_64_registers() {
    let sleigh = x86_64_sleigh();

    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .expect(&format!("invalid register {name}"));
        memory
            .write(&register, data.into_iter().cloned())
            .expect(&format!("failed to write register {name}"));
    };

    let read_register = |memory: &Memory, name: &str| {
        let register = sleigh
            .register_from_name(name)
            .expect(&format!("invalid register {name}"));
        memory
            .read(&register)
            .expect(&format!("failed to read register {name}"))
    };

    let mut memory = Memory::new();
    let registers = vec!['A', 'B', 'C', 'D'];
    for register in registers {
        let name = format!("R{register}X");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let rax: u64 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u64");
        assert_eq!(rax, 0x8877665544332211);

        let name = format!("E{register}X");
        let eax: u32 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u32");
        assert_eq!(eax, 0x44332211);

        let name = format!("{register}X");
        let ax: u16 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u16");
        assert_eq!(ax, 0x2211);

        let name = format!("{register}H");
        let ah: u8 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u8");
        assert_eq!(ah, 0x22);

        let name = format!("{register}L");
        let al: u8 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u8");
        assert_eq!(al, 0x11);
    }

    let registers = vec!["SI", "DI", "BP", "SP"];
    for register in registers {
        let name = format!("R{register}");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let r: u64 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);

        let name = format!("E{register}");
        let e: u32 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u32");
        assert_eq!(e, 0x44332211);

        let name = format!("{register}");
        let b: u16 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u16");
        assert_eq!(b, 0x2211);

        let name = format!("{register}L");
        let l: u8 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u8");
        assert_eq!(l, 0x11);
    }

    for register in 8..=15 {
        let name = format!("R{register}");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let r: u64 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);

        let name = format!("R{register}D");
        let rd: u32 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u32");
        assert_eq!(rd, 0x44332211);

        let name = format!("R{register}W");
        let rw: u16 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u16");
        assert_eq!(rw, 0x2211);

        let name = format!("R{register}B");
        let rb: u8 =
            sym::concretize_into(read_register(&memory, &name)).expect("failed to convert to u8");
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
fn doubler_32b() -> processor::Result<()> {
    let sleigh = x86_64_sleigh();
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::new();
    let base_addr = 0x84210000;
    let num_instructions = 7;

    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
        memory
            .write(&register, data.iter().copied())
            .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));
    };

    let code_space = sleigh.default_code_space();
    memory.write_address(
        Address::new(code_space, base_addr),
        b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3\x00\x00"
            .iter()
            .copied(),
    )?;

    write_register(&mut memory, "RSP", b"\x00\x01\x01\x01\x01\x01\x01\x00");
    write_register(&mut memory, "RBP", b"\x00\x02\x02\x02\x02\x02\x02\x00");

    memory.write_address(
        Address::new(
            sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
            0x0001010101010100,
        ),
        vec![0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66].into_iter(),
    )?;

    let initial_value: u32 = 0x99;
    write_register(&mut memory, "EDI", &initial_value.to_le_bytes());
    write_register(&mut memory, "RIP", &base_addr.to_le_bytes());

    let handler = ProcessorHandlerX86::new(&sleigh);
    let mut processor = Processor::new(memory, emulator, handler);

    for _ in 0..num_instructions {
        loop {
            let step_response = processor.step(&sleigh)?;
            if step_response.is_some() {
                panic!("Unexpected branch: {state:?}", state = processor.state());
            }

            if matches!(processor.state(), processor::ProcessorState::Fetch) {
                break;
            }
        }
    }

    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to get RIP register");
    let rip_value: u64 = sym::concretize_into(processor.memory().read(&rip)?)
        .expect("failed to convert rip value to u64");
    assert_eq!(rip_value, 0x66778899aabbccdd, "return address on stack");

    let rax = sleigh
        .register_from_name("RAX")
        .expect("failed to get RAX register");
    let rax_value: u64 = sym::concretize_into(processor.memory().read(&rax)?)
        .expect("failed to convert rax value to u64");
    assert_eq!(
        u32::try_from(rax_value)
            .unwrap_or_else(|err| panic!("failed to convert rax to u32: {rax_value:016x}: {err}")),
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
    let mut manager = processor_with_image("tests/data/coverage/coverage", 0x1675);

    let rip = manager
        .sleigh()
        .register_from_name("RIP")
        .expect("failed to get RIP register");
    loop {
        manager.step_all()?;

        // Check if RIP is the magic value
        let processors: Vec<_> = manager.processors().collect();
        if processors.len() != 1 {
            panic!(
                "Unexpected number of processors: {len}",
                len = processors.len()
            );
        }

        let processor = processors[0];
        let instruction_pointer: u64 = sym::concretize_into(processor.memory().read(&rip)?)?;

        if instruction_pointer == EXIT_RIP && matches!(processor.state(), ProcessorState::Fetch) {
            break;
        }
    }

    let processor = manager.processors().next().expect("no processors");
    let rax = manager
        .sleigh()
        .register_from_name("RAX")
        .expect("failed to get RAX register");
    let return_value: u64 = sym::concretize_into(processor.memory().read(&rax)?)?;
    assert_eq!(
        return_value, 0,
        "unexpected return value: {return_value:016x}"
    );

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

#[test]
fn memory_tree() -> processor::Result<()> {
    let sleigh = x86_64_sleigh();
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::new();
    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .expect(&format!("invalid register {name}"));
        memory
            .write(&register, data.into_iter().cloned())
            .expect(&format!("failed to write register {name}"));
    };

    // Small program: if x > 0 { x + x } else { x * x }
    // The input x is provided in the register EDI and is signed 32-bit
    let program_bytes: Vec<u8> = vec![
        0x55, // PUSH RBP
        0x48, 0x89, 0xe5, // MOV RBP, RSP
        0x89, 0x7d, 0xfc, // MOV DWORD PTR [RBP - 4], EDI
        0x83, 0x7d, 0xfc, 0x00, // CMP DWORD PTR [RBP - 4], 0
        0x7e, 0x07, // JLE [START + 0x14]
        0x8b, 0x45, 0xfc, // MOV EAX, DWORD PTR [RBP - 0x4]
        0x01, 0xc0, // ADD EAX, EAX
        0xeb, 0x06, // JMP [START + 0x1a]
        0x8b, 0x45, 0xfc, // MOV EAX, DWORD PTR [RBP - 0x4]
        0x0f, 0xaf, 0xc0, // IMUL EAX, EAX
        0x5d, // POP RBP
        0xc3, // RET
    ];

    let code_offset = 0;
    memory.write_address(
        Address::new(sleigh.default_code_space(), code_offset),
        program_bytes.into_iter(),
    )?;
    write_register(&mut memory, "RIP", &code_offset.to_le_bytes());

    // Initialize stack and base pointer registers
    write_register(&mut memory, "RSP", &INITIAL_STACK.to_le_bytes());
    write_register(&mut memory, "RBP", &INITIAL_STACK.to_le_bytes());

    // Put EXIT_RIP onto the stack. The final RET will trigger this.
    memory.write_address(
        Address::new(
            sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
            INITIAL_STACK,
        ),
        EXIT_RIP.to_le_bytes().into_iter(),
    )?;

    // Set input to concrete value
    // TODO: Have a way to set this symbolically and then evaluate it with different values
    let input_value = -sym::SymbolicBitVec::constant(3, 32);

    let name = "EDI";
    let register = sleigh
        .register_from_name(name)
        .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
    memory
        .write(&register, input_value.into_bytes())
        .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));
    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to find RIP");

    let handler = ProcessorHandlerX86::new(&sleigh);
    let processor = Processor::new(memory, emulator, handler);
    let mut manager = ProcessorManager::new(sleigh, processor);

    let mut finished = Vec::new();
    loop {
        finished.append(&mut manager.remove_all(|p| {
            let rip_value: u64 =
                sym::concretize_into(p.memory().read(&rip).expect("failed to read RIP"))
                    .expect("failed to concretize RIP");
            rip_value == EXIT_RIP
        }));

        if manager.is_empty() {
            break;
        }

        manager.step_all()?;
    }

    // Output register is EAX
    let eax = manager
        .sleigh()
        .register_from_name("EAX")
        .unwrap_or_else(|err| panic!("failed to find EAX: {err}"));

    let memory_tree = MemoryTree::new(finished.iter().map(|p| p.memory()), std::iter::empty());
    let result = memory_tree.read(&eax)?;
    let result: SymbolicBitVec = result.into_iter().collect();

    let assertion = result.equals(SymbolicBitVec::constant(9, 32));
    assert_eq!(assertion, sym::TRUE);

    Ok(())
}
