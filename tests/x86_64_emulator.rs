mod common;

use std::path::Path;

use common::{x86_64_sleigh, Memory, ProcessorHandlerX86, TracingEmulator};
use libsla::{Address, Sleigh, VarnodeData};
use sym::{self, Evaluator, SymbolicBit, SymbolicBitVec, SymbolicByte, VariableAssignments};
use symbolic_pcode::{
    emulator::StandardPcodeEmulator,
    mem::{MemoryBranch, MemoryTree, VarnodeDataStore},
    processor::{self, Processor, ProcessorManager, ProcessorState},
};

const INITIAL_STACK: u64 = 0x8000000000;
const EXIT_RIP: u64 = 0xFEEDBEEF0BADF00D;

fn processor_with_image(
    image: impl AsRef<Path>,
    entry: u64,
) -> ProcessorManager<TracingEmulator, Memory, ProcessorHandlerX86> {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let mut memory = Memory::default();

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
        .write(&data_location, data.into_iter().collect())
        .expect("failed to write image into memory");

    // Init RIP to entry

    let rip = sleigh.register_from_name("RIP").expect("invalid register");
    memory
        .write(&rip, entry.into())
        .expect("failed to initialize RIP");

    // Init RBP to magic EXIT_RIP value
    let rbp = sleigh.register_from_name("RBP").expect("invalid register");
    memory
        .write(&rbp, EXIT_RIP.into())
        .expect("failed to initialize RBP");

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

    // Init RSP to stack address
    let rsp = sleigh.register_from_name("RSP").expect("invalid register");
    memory
        .write(&rsp, INITIAL_STACK.into())
        .expect("failed to initialize RSP");

    init_registers(&sleigh, &mut memory);

    let handler = ProcessorHandlerX86::new(&sleigh);
    let processor = Processor::new(
        memory,
        TracingEmulator::new(StandardPcodeEmulator::new(sleigh.address_spaces())),
        handler,
    );
    ProcessorManager::new(sleigh, processor)
}

fn init_registers(sleigh: &impl Sleigh, memory: &mut Memory) {
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
            .write(&register, bytes.into_iter().collect())
            .unwrap_or_else(|err| panic!("failed to write register {register_name}: {err}"));
    }
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
    let mut manager = processor_with_image("data/coverage/coverage", 0x1675);

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
        if matches!(processor.state(), ProcessorState::Decode(_)) {
            println!("Executing: {}", processor.disassemble(manager.sleigh())?);
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

    let processor = manager.processors().next().expect("no processors");
    let rax = manager
        .sleigh()
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
    let processor = Processor::new(memory, emulator, handler);
    let mut manager = ProcessorManager::new(sleigh, processor);

    let mut finished = Vec::new();
    let rip = manager
        .sleigh()
        .register_from_name("RIP")
        .expect("failed to find RIP");

    loop {
        finished.append(&mut manager.remove_all(|p| {
            let rip_value: u64 = p
                .memory()
                .read(&rip)
                .expect("failed to read RIP")
                .try_into()
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

    loop {
        if let Some(false_processor) = processor.step(&sleigh)? {
            // Evaluate the branch using the concrete value instead of the symbolic value
            let concrete_evaluation = evaluator.evaluate(processor.memory().leaf_predicate());

            // The processor will step forward using the true branch and will return the false
            // branch as the other option. If the concrete value indicates we should take the false
            // branch then change processors
            if !concrete_evaluation {
                processor = false_processor;
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
    let parent_predicate = processor
        .memory()
        .parent()
        .map(MemoryBranch::branch_predicate)
        .unwrap_or(&sym::TRUE);
    let other_branch = parent_predicate.clone() & !processor.memory().leaf_predicate().clone();

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
            // be unset. The other bits could be anything.
            if variable_id == 31 {
                assert!(
                    !solved_value,
                    "expected input variable {variable_id} to be unset"
                );
            }
        }
    }

    Ok(())
}
