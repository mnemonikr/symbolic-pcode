mod common;

use std::{path::Path, rc::Rc};

use common::{x86_64_sleigh, Memory, TracingEmulator};
use libsla::{Address, GhidraSleigh, Sleigh, VarnodeData};
use pcode_ops::{convert::PcodeValue, PcodeOps};
use sym::{self, Evaluator, SymbolicBit, SymbolicBitVec, SymbolicByte, VariableAssignments};
use symbolic_pcode::{
    arch::x86::{emulator::EmulatorX86, processor::ProcessorHandlerX86},
    emulator::StandardPcodeEmulator,
    kernel::linux::LinuxKernel,
    mem::{MemoryTree, VarnodeDataStore},
    processor::{self, Processor, ProcessorState},
};

const INITIAL_STACK: u64 = 0x8000000000;
const EXIT_RIP: u64 = 0xFEEDBEEF0BADF00D;

fn initialize_libc_stack(memory: &mut Memory, sleigh: &impl Sleigh) {
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
    // MUSL has support for null program name:
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

    let auxv = VarnodeData {
        address: Address {
            offset: envp.address.offset + envp.size as u64,
            address_space: ram.clone(),
        },
        size: 8,
    };
    memory
        .write(&auxv, SymbolicBitVec::constant(0, u64::BITS as usize))
        .expect("failed to initialize envp");
}

fn memory_with_image(sleigh: &GhidraSleigh, image: impl AsRef<Path>) -> Memory {
    use elf::abi::PT_LOAD;
    use elf::endian::AnyEndian;
    use elf::ElfBytes;

    let mut memory = Memory::default();

    // Write image into memory
    let data = std::fs::read(image).expect("failed to read image file");

    let elf = ElfBytes::<AnyEndian>::minimal_parse(&data).expect("failed to parse elf");
    for segment in elf.segments().expect("failed to get segments") {
        if segment.p_type != PT_LOAD {
            // Not a loadable segment
            continue;
        }

        let data_location = VarnodeData {
            address: Address {
                offset: segment.p_vaddr,
                address_space: sleigh.default_code_space(),
            },

            // The initial set of data is from the file
            size: segment.p_filesz as usize,
        };

        // Write the segment from the file into memory
        let offset = segment.p_offset as usize;
        let file_size = segment.p_filesz as usize;
        memory
            .write(
                &data_location,
                data[offset..offset + file_size].iter().copied().collect(),
            )
            .expect("failed to write image section into memory");

        // If the virtual size is larger than the file size then zero out the remainder
        let zero_size = (segment.p_memsz - segment.p_filesz) as usize;
        if zero_size > 0 {
            let zeros = vec![0u8; zero_size];
            let zeros_location = VarnodeData {
                address: Address {
                    offset: segment.p_vaddr + segment.p_filesz,
                    address_space: sleigh.default_code_space(),
                },
                size: zero_size,
            };
            memory
                .write(&zeros_location, zeros.into_iter().collect())
                .expect("failed to write zeros into memory");
        }
    }

    // Init RIP to entry
    let rip = sleigh.register_from_name("RIP").expect("invalid register");
    memory
        .write(&rip, elf.ehdr.e_entry.into())
        .expect("failed to initialize RIP");

    memory
}

fn init_registers(sleigh: &impl Sleigh, memory: &mut Memory) {
    let mut bitvar = 0;
    let registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP"]
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
            .write(&register, bytes.into_iter().collect())
            .unwrap_or_else(|err| panic!("failed to write register {register_name}: {err}"));
    }

    // Init RSP to stack address
    let rsp = sleigh.register_from_name("RSP").expect("invalid register");
    memory
        .write(&rsp, INITIAL_STACK.into())
        .expect("failed to initialize RSP");

    // Initialize the DF register to 0. It appears to be a convention that whenever STD is called
    // CLD is called thereafter to reset it. There is a bug (?) in MUSL where REP STOS is used
    // without ensuring the flag is cleared

    // 0000000000449f4d <__init_libc>:
    //   449f4d:       53                      push   rbx
    //   449f4e:       48 89 fa                mov    rdx,rdi
    //   449f51:       31 c0                   xor    eax,eax
    //   449f53:       b9 4c 00 00 00          mov    ecx,0x4c
    //   449f58:       48 81 ec 50 01 00 00    sub    rsp,0x150
    //   449f5f:       48 8d 7c 24 20          lea    rdi,[rsp+0x20]
    //   449f64:       f3 ab                   rep stos DWORD PTR es:[rdi],eax

    let df_register = sleigh
        .register_from_name("DF")
        .expect("failed to get DF register");
    memory
        .write(&df_register, 0u8.into())
        .expect("failed to init DF register");
}

fn find_test_fixture(
    messages: escargot::CommandMessages,
) -> Result<std::path::PathBuf, Vec<escargot::error::CargoResult<escargot::Message>>> {
    let messages: Vec<_> = messages.into_iter().collect();
    let executable_path = messages.iter().find_map(|result| {
        result
            .as_ref()
            .ok()
            .into_iter()
            .filter_map(|msg| {
                msg.decode()
                    .ok()
                    .into_iter()
                    .filter_map(|msg| match msg {
                        escargot::format::Message::CompilerArtifact(artifact) => {
                            artifact.executable.map(|p| p.into_owned())
                        }
                        _ => None,
                    })
                    .next()
            })
            .next()
    });

    executable_path.ok_or(messages)
}

/// Confirms the functionality of general-purpose x86-64 registers and overlapping behavior.
#[test]
fn x86_64_registers() {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");

    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .expect(&format!("invalid register {name}"));
        memory
            .write(&register, data.into_iter().copied().collect())
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

    let mut memory = Memory::default();
    let registers = vec!['A', 'B', 'C', 'D'];
    for register in registers {
        let name = format!("R{register}X");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let rax: u64 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(rax, 0x8877665544332211);

        let name = format!("E{register}X");
        let eax: u32 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(eax, 0x44332211);

        let name = format!("{register}X");
        let ax: u16 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(ax, 0x2211);

        let name = format!("{register}H");
        let ah: u8 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(ah, 0x22);

        let name = format!("{register}L");
        let al: u8 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(al, 0x11);
    }

    let registers = vec!["SI", "DI", "BP", "SP"];
    for register in registers {
        let name = format!("R{register}");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let r: u64 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);

        let name = format!("E{register}");
        let e: u32 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(e, 0x44332211);

        let name = format!("{register}");
        let b: u16 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(b, 0x2211);

        let name = format!("{register}L");
        let l: u8 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u8");
        assert_eq!(l, 0x11);
    }

    for register in 8..=15 {
        let name = format!("R{register}");
        let data = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        write_register(&mut memory, &name, &data);

        let r: u64 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u64");
        assert_eq!(r, 0x8877665544332211);

        let name = format!("R{register}D");
        let rd: u32 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u32");
        assert_eq!(rd, 0x44332211);

        let name = format!("R{register}W");
        let rw: u16 = read_register(&memory, &name)
            .try_into()
            .expect("failed to convert to u16");
        assert_eq!(rw, 0x2211);

        let name = format!("R{register}B");
        let rb: u8 = read_register(&memory, &name)
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
fn doubler_32b() -> processor::Result<()> {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::default();
    let base_addr = 0x84210000;
    let num_instructions = 7;

    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
        memory
            .write(&register, data.iter().copied().collect())
            .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));
    };

    let code_space = sleigh.default_code_space();
    let code = b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3\x00\x00";
    let destination = VarnodeData::new(Address::new(code_space, base_addr), code.len());
    memory.write(&destination, code.into_iter().copied().collect())?;

    write_register(&mut memory, "RSP", b"\x00\x01\x01\x01\x01\x01\x01\x00");
    write_register(&mut memory, "RBP", b"\x00\x02\x02\x02\x02\x02\x02\x00");

    memory.write(
        &VarnodeData::new(
            Address::new(
                sleigh
                    .address_space_by_name("ram")
                    .expect("failed to find ram"),
                0x0001010101010100,
            ),
            8,
        ),
        0x66778899aabbccddu64.into(),
    )?;

    let initial_value: u32 = 0x99;
    write_register(&mut memory, "EDI", &initial_value.to_le_bytes());
    write_register(&mut memory, "RIP", &base_addr.to_le_bytes());

    let handler = ProcessorHandlerX86::new(&sleigh);
    let mut processor = Processor::new(memory, emulator, handler);

    for _ in 0..num_instructions {
        loop {
            processor.step(&sleigh)?;
            if matches!(processor.state(), ProcessorState::Fetch) {
                break;
            }
        }
    }

    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to get RIP register");
    let rip_value: u64 = processor
        .memory()
        .read(&rip)?
        .try_into()
        .expect("failed to convert rip value to u64");
    assert_eq!(rip_value, 0x66778899aabbccdd, "return address on stack");

    let rax = sleigh
        .register_from_name("RAX")
        .expect("failed to get RAX register");
    let rax_value: u64 = processor
        .memory()
        .read(&rax)?
        .try_into()
        .expect("failed to convert rax value to u64");
    assert_eq!(
        u32::try_from(rax_value)
            .unwrap_or_else(|err| panic!("failed to convert rax to u32: {rax_value:016x}: {err}")),
        2 * initial_value,
        "result should be double initial value: {initial_value}",
    );

    Ok(())
}

#[test]
fn hello_world_linux() -> processor::Result<()> {
    // Build test fixture first
    let messages = escargot::CargoBuild::new()
        .bin("linux-syscalls")
        .manifest_path("./test-fixtures/linux-syscalls/Cargo.toml")
        .target("x86_64-unknown-linux-musl")
        .env(
            "RUSTFLAGS",
            "-Ctarget-feature=+crt-static -Crelocation-model=static",
        )
        .exec()
        .unwrap();

    let image = match find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = Rc::new(x86_64_sleigh().expect("failed to build sleigh"));
    let mut memory = memory_with_image(&sleigh, image);
    init_registers(sleigh.as_ref(), &mut memory);
    initialize_libc_stack(&mut memory, sleigh.as_ref());

    let handler = ProcessorHandlerX86::new(sleigh.as_ref());
    let emulator = EmulatorX86::with_kernel(sleigh.clone(), LinuxKernel::default());
    let mut processor = Processor::new(memory, emulator, handler);

    loop {
        processor.step(sleigh.as_ref())?;

        // Debug
        if matches!(processor.state(), ProcessorState::Decode(_)) {
            let disassembly = processor.disassemble(sleigh.as_ref())?;
            let encoded_instr = processor
                .memory()
                .read(&disassembly.origin)?
                .into_le_bytes()
                .map(|byte| u8::try_from(byte).map_or("xx".to_string(), |b| format!("{b:02x}")))
                .collect::<Vec<_>>()
                .join(" ");

            println!("Encoded instruction from memory: {encoded_instr}");
            println!("Decoded: {disassembly}");
        }

        if matches!(processor.state(), ProcessorState::Halt) {
            assert_eq!(
                processor.emulator().kernel().exit_status(),
                Some(0),
                "exit code should be 0"
            );
            return Ok(());
        }
    }
}

#[test]
fn pcode_coverage() -> processor::Result<()> {
    // Build test fixture first
    let messages = escargot::CargoBuild::new()
        .bin("pcode-coverage")
        .manifest_path("./test-fixtures/pcode-coverage/Cargo.toml")
        .target("x86_64-unknown-none")
        .exec()
        .unwrap();

    let image = match find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let mut memory = memory_with_image(&sleigh, image);
    init_registers(&sleigh, &mut memory);

    // Init stack address in memory to magic EXIT_RIP value
    let stack_addr = VarnodeData {
        address: Address {
            offset: INITIAL_STACK,
            address_space: sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
        },
        size: EXIT_RIP.to_le_bytes().len(),
    };
    memory
        .write(&stack_addr, EXIT_RIP.into())
        .expect("failed to initialize stack");

    let handler = ProcessorHandlerX86::new(&sleigh);
    let emulator = TracingEmulator::new(StandardPcodeEmulator::new(sleigh.address_spaces()));
    let mut processor = Processor::new(memory, emulator, handler);

    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to get RIP register");
    loop {
        processor.step(&sleigh)?;

        // Check if RIP is the magic value
        if matches!(processor.state(), ProcessorState::Decode(_)) {
            let disassembly = processor.disassemble(&sleigh)?;
            let encoded_instr = processor
                .memory()
                .read(&disassembly.origin)?
                .into_le_bytes()
                .map(|byte| u8::try_from(byte).map_or("xx".to_string(), |b| format!("{b:02x}")))
                .collect::<Vec<_>>()
                .join(" ");

            println!("Encoded instruction from memory: {encoded_instr}");
            println!("Decoded: {disassembly}");
        }

        let instruction_pointer: u64 = processor
            .memory()
            .read(&rip)?
            .try_into()
            .expect("failed to concretize RIP");

        if instruction_pointer == EXIT_RIP && matches!(processor.state(), ProcessorState::Fetch) {
            break;
        }
    }

    let rax = sleigh
        .register_from_name("RAX")
        .expect("failed to get RAX register");
    let return_value: u64 = processor
        .memory()
        .read(&rax)?
        .try_into()
        .expect("failed to concretize RAX");
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
    // Int(GreaterThan(Signed))
    // Int(GreaterThan(Unsigned))
    // Int(GreaterThanOrEqual(Signed))
    // Int(GreaterThanOrEqual(Unsigned))
    // BranchIndirect -- though technically this is covered via Return
    assert_eq!(processor.emulator().executed_instructions().len(), 36);
    Ok(())
}

#[test]
fn z3_integration() -> processor::Result<()> {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::default();
    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .expect(&format!("invalid register {name}"));
        memory
            .write(&register, data.into_iter().copied().collect())
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
    memory.write(
        &VarnodeData::new(
            Address::new(sleigh.default_code_space(), code_offset),
            program_bytes.len(),
        ),
        program_bytes.into_iter().collect(),
    )?;
    write_register(&mut memory, "RIP", &code_offset.to_le_bytes());

    // Initialize stack and base pointer registers
    write_register(&mut memory, "RSP", &INITIAL_STACK.to_le_bytes());
    write_register(&mut memory, "RBP", &INITIAL_STACK.to_le_bytes());

    // Put EXIT_RIP onto the stack. The final RET will trigger this.
    memory.write(
        &VarnodeData::new(
            Address::new(
                sleigh
                    .address_space_by_name("ram")
                    .expect("failed to find ram"),
                INITIAL_STACK,
            ),
            EXIT_RIP.to_le_bytes().len(),
        ),
        EXIT_RIP.into(),
    )?;

    let input_value = sym::SymbolicBitVec::with_size(32);

    // Input register is EDI
    let name = "EDI";
    let register = sleigh
        .register_from_name(name)
        .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
    memory
        .write(&register, input_value)
        .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));

    let handler = ProcessorHandlerX86::new(&sleigh);
    let processor = processor::BranchingProcessor::new(memory, emulator, handler);
    let mut processors = vec![processor];

    let mut finished = Vec::new();
    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to find RIP");

    loop {
        if processors.is_empty() {
            break;
        }

        let mut i = 0;
        while i < processors.len() {
            let rip_value: u64 = processors[i]
                .processor()
                .memory()
                .read(&rip)
                .expect("failed ot read RIP")
                .try_into()
                .expect("failed to concretize RIP");
            if rip_value == EXIT_RIP {
                finished.push(processors.swap_remove(i));

                // Do not update index since processor from end was moved here
                continue;
            }

            if let Some(new_processor) = processors[i].step(&sleigh)? {
                processors.push(new_processor);
            }

            i += 1;
        }
    }

    // Output register is EAX
    let eax = sleigh
        .register_from_name("EAX")
        .unwrap_or_else(|err| panic!("failed to find EAX: {err}"));

    let memory_tree = MemoryTree::new(
        finished.iter().map(|p| p.processor().memory()),
        std::iter::empty(),
    );
    let result = memory_tree.read(&eax)?;
    let result: SymbolicBitVec = result.into_iter().collect();

    let assertion = result.equals(SymbolicBitVec::constant(8, 32));
    let aiger = sym::aiger::Aiger::from_bits(std::iter::once(assertion));
    let cfg = z3::Config::new();
    let ctx = z3::Context::new(&cfg);

    let mut z3_ast = std::collections::BTreeMap::new();
    for input in aiger.inputs() {
        if input.is_negated() {
            panic!("input literal is negated: {input}");
        }

        let z3_name = format!("x{index}", index = input.index());
        z3_ast.insert(input, z3::ast::Bool::new_const(&ctx, z3_name));
    }

    for gate in aiger.gates() {
        let aiger_literal = gate.input_lhs();
        if !z3_ast.contains_key(&aiger_literal) {
            assert!(aiger_literal.is_negated());
            let z3_value = z3_ast
                .get(&aiger_literal.negated())
                .unwrap_or_else(|| panic!("missing literal (negated): {aiger_literal}"));
            z3_ast.insert(aiger_literal, z3::ast::Bool::not(z3_value));
        };

        let aiger_literal = gate.input_rhs();
        if !z3_ast.contains_key(&aiger_literal) {
            assert!(aiger_literal.is_negated());
            let z3_value = z3_ast
                .get(&aiger_literal.negated())
                .unwrap_or_else(|| panic!("missing literal (negated): {aiger_literal}"));
            z3_ast.insert(aiger_literal, z3::ast::Bool::not(z3_value));
        };

        let z3_lhs = z3_ast.get(&gate.input_lhs()).expect("missing lhs");
        let z3_rhs = z3_ast.get(&gate.input_rhs()).expect("missing rhs");
        let z3_gate = z3::ast::Bool::and(&ctx, &[z3_lhs, z3_rhs]);

        if gate.gate_literal().is_negated() {
            panic!("gate literal is negated: {gate:?}");
        }

        z3_ast.insert(gate.gate_literal(), z3_gate);
    }

    let solver = z3::Solver::new(&ctx);
    for output in aiger.outputs() {
        if output.is_const() {
            println!("Output is constant: {output}");
            break;
        }

        if output.is_negated() {
            let z3_output = z3_ast
                .get(&output.negated())
                .expect("missing output (negated)");
            solver.assert(&z3::ast::Bool::not(z3_output));
        } else {
            let z3_output = z3_ast.get(&output).expect("missing output");
            solver.assert(z3_output);
        }
    }

    let sat_result = solver.check();
    assert_eq!(sat_result, z3::SatResult::Sat);

    let model = solver.get_model().expect("model not returned by solver");
    for input in aiger.inputs() {
        let z3_input = z3_ast.get(&input).expect("missing input");
        if let Some(z3_value) = model.get_const_interp(z3_input) {
            let variable_id = aiger
                .input_variable_id(input)
                .unwrap_or_else(|| panic!("input {input} does not map to a variable"));
            let solved_value = z3_value.as_bool().expect("non-bool assignment for {input}");

            // The expected input answer is 4.
            if variable_id == 2 {
                assert!(
                    solved_value,
                    "expected input variable {variable_id} to be set"
                );
            } else {
                assert!(
                    !solved_value,
                    "expected input variable {variable_id} to be unset"
                );
            }
        }
    }

    Ok(())
}

#[test]
fn take_the_path_not_taken() -> processor::Result<()> {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::default();
    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
        memory
            .write(&register, data.iter().copied().collect())
            .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));
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
    memory.write(
        &VarnodeData::new(
            Address::new(sleigh.default_code_space(), code_offset),
            program_bytes.len(),
        ),
        program_bytes.into_iter().collect(),
    )?;
    write_register(&mut memory, "RIP", &code_offset.to_le_bytes());

    // Initialize stack and base pointer registers
    write_register(&mut memory, "RSP", &INITIAL_STACK.to_le_bytes());
    write_register(&mut memory, "RBP", &INITIAL_STACK.to_le_bytes());

    // Put EXIT_RIP onto the stack. The final RET will trigger this.
    memory.write(
        &VarnodeData::new(
            Address::new(
                sleigh
                    .address_space_by_name("ram")
                    .expect("failed to find ram"),
                INITIAL_STACK,
            ),
            EXIT_RIP.to_le_bytes().len(),
        ),
        EXIT_RIP.into(),
    )?;

    // Create symbolic input
    let input_value: [_; 32] = std::array::from_fn(sym::SymbolicBit::Variable);
    let input_value = input_value.into_iter().collect();

    // The test will emulate with a symbolic value. However if we encounter any branch that uses a
    // symbolic value, we will evaluate the branch using the concrete value instead.
    let concrete_input = -3i32;
    let concrete_input = SymbolicBitVec::from(concrete_input as u32);
    let mut evaluator = Evaluator::new(VariableAssignments::from_bitvecs(
        &input_value,
        &concrete_input,
    ));

    // Input register is EDI
    let name = "EDI";
    let register = sleigh
        .register_from_name(name)
        .unwrap_or_else(|err| panic!("invalid register {name}: {err}"));
    memory
        .write(&register, input_value)
        .unwrap_or_else(|err| panic!("failed to write register {name}: {err}"));

    let handler = ProcessorHandlerX86::new(&sleigh);
    let mut processor = Processor::new(memory, emulator, handler);
    let rip = sleigh
        .register_from_name("RIP")
        .expect("failed to get RIP register");

    let mut branches = Vec::new();
    loop {
        if let Err(e) = processor.step(&sleigh) {
            if let processor::Error::SymbolicBranch { condition_origin } = &e {
                let condition = processor.memory().read(condition_origin)?;
                let evaluation =
                    evaluator.evaluate(&processor.memory().read_bit(condition_origin)?);
                if evaluation {
                    branches.push(condition.not_equals(0u8.into()));
                } else {
                    branches.push(condition.equals(0u8.into()));
                }

                processor.step_branch(&sleigh, evaluation)?;
            } else {
                return Err(e);
            }
        }

        // Exit loop when we reach the magic RIP
        let instruction_pointer: u64 = processor
            .memory()
            .read(&rip)?
            .try_into()
            .expect("failed to concretize RIP");
        if instruction_pointer == EXIT_RIP && matches!(processor.state(), ProcessorState::Fetch) {
            break;
        }
    }

    // The processor was evaluated with the concrete value for branching. Now lets ask Z3 what
    // concrete input value could have been used to take the other path of the most recent branch.
    let last_branch = branches.pop().expect("should have at least one branch");
    let parent_path = branches
        .into_iter()
        .reduce(|x, y| x & y)
        .unwrap_or(sym::TRUE);
    let other_branch = parent_path & !last_branch;

    // Solve with Z3
    let aiger = sym::aiger::Aiger::from_bits(std::iter::once(other_branch));
    let cfg = z3::Config::new();
    let ctx = z3::Context::new(&cfg);

    let mut z3_ast = std::collections::BTreeMap::new();
    for input in aiger.inputs() {
        if input.is_negated() {
            panic!("input literal is negated: {input}");
        }

        let z3_name = format!("x{index}", index = input.index());
        z3_ast.insert(input, z3::ast::Bool::new_const(&ctx, z3_name));
    }

    for gate in aiger.gates() {
        let aiger_literal = gate.input_lhs();
        if !z3_ast.contains_key(&aiger_literal) {
            assert!(aiger_literal.is_negated());
            let z3_value = z3_ast
                .get(&aiger_literal.negated())
                .unwrap_or_else(|| panic!("missing literal (negated): {aiger_literal}"));
            z3_ast.insert(aiger_literal, z3::ast::Bool::not(z3_value));
        };

        let aiger_literal = gate.input_rhs();
        if !z3_ast.contains_key(&aiger_literal) {
            assert!(aiger_literal.is_negated());
            let z3_value = z3_ast
                .get(&aiger_literal.negated())
                .unwrap_or_else(|| panic!("missing literal (negated): {aiger_literal}"));
            z3_ast.insert(aiger_literal, z3::ast::Bool::not(z3_value));
        };

        let z3_lhs = z3_ast.get(&gate.input_lhs()).expect("missing lhs");
        let z3_rhs = z3_ast.get(&gate.input_rhs()).expect("missing rhs");
        let z3_gate = z3::ast::Bool::and(&ctx, &[z3_lhs, z3_rhs]);

        if gate.gate_literal().is_negated() {
            panic!("gate literal is negated: {gate:?}");
        }

        z3_ast.insert(gate.gate_literal(), z3_gate);
    }

    let solver = z3::Solver::new(&ctx);
    for output in aiger.outputs() {
        if output.is_const() {
            println!("Output is constant: {output}");
            break;
        }

        if output.is_negated() {
            let z3_output = z3_ast
                .get(&output.negated())
                .expect("missing output (negated)");
            solver.assert(&z3::ast::Bool::not(z3_output));
        } else {
            let z3_output = z3_ast.get(&output).expect("missing output");
            solver.assert(z3_output);
        }
    }

    let sat_result = solver.check();
    assert_eq!(sat_result, z3::SatResult::Sat);

    let model = solver.get_model().expect("model not returned by solver");
    let mut nonzero_lower_bit = false;
    for input in aiger.inputs() {
        let z3_input = z3_ast.get(&input).expect("missing input");
        if let Some(z3_value) = model.get_const_interp(z3_input) {
            let variable_id = aiger
                .input_variable_id(input)
                .unwrap_or_else(|| panic!("input {input} does not map to a variable"));
            let solved_value = z3_value.as_bool().expect("non-bool assignment for {input}");
            println!("Input variable id {variable_id} has value {solved_value}");

            // The initial concrete value was a negative number, so the branch taken was x <= 0.
            // Taking the other branch would entail x > 0 meaning the most-significant bit should
            // be unset. The other bits could be anything as long as x != 0
            if variable_id == 31 {
                assert!(
                    !solved_value,
                    "expected input variable {variable_id} to be unset"
                );
            } else if solved_value {
                nonzero_lower_bit = true;
            }
        }
    }

    assert!(
        nonzero_lower_bit,
        "expected at least one lower bit to be set"
    );

    Ok(())
}
