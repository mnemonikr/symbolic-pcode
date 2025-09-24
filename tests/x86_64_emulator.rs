use pcode_ops::PcodeOps;
use sym::{self, Evaluator, SymbolicBitVec, VariableAssignments};
use symbolic_pcode::{
    arch::x86::processor::ProcessorHandlerX86,
    emulator::StandardPcodeEmulator,
    libsla::{Address, Sleigh, VarnodeData},
    mem::{MemoryTree, VarnodeDataStore},
    processor::{self, BranchingProcessor, Processor, ProcessorState},
};

use crate::common::{self, Memory, x86_64_sleigh};

/// Confirms the functionality of general-purpose x86-64 registers and overlapping behavior.
#[test]
fn x86_64_registers() {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");

    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|_| panic!("invalid register {name}"));
        memory
            .write(&register, data.iter().copied().collect())
            .unwrap_or_else(|_| panic!("failed to write register {name}"));
    };

    let read_register = |memory: &Memory, name: &str| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|_| panic!("invalid register {name}"));
        memory
            .read(&register)
            .unwrap_or_else(|_| panic!("failed to read register {name}"))
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

        let b: u16 = read_register(&memory, register)
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
fn z3_integration() -> processor::Result<()> {
    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let emulator = StandardPcodeEmulator::new(sleigh.address_spaces());
    let mut memory = Memory::default();
    let write_register = |memory: &mut Memory, name: &str, data: &[u8]| {
        let register = sleigh
            .register_from_name(name)
            .unwrap_or_else(|_| panic!("invalid register {name}"));
        memory
            .write(&register, data.iter().copied().collect())
            .unwrap_or_else(|_| panic!("failed to write register {name}"));
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
    write_register(&mut memory, "RSP", &common::INITIAL_STACK.to_le_bytes());
    write_register(&mut memory, "RBP", &common::INITIAL_STACK.to_le_bytes());

    // Put EXIT_IP_ADDR onto the stack. The final RET will trigger this.
    memory.write(
        &VarnodeData::new(
            Address::new(
                sleigh
                    .address_space_by_name("ram")
                    .expect("failed to find ram"),
                common::INITIAL_STACK,
            ),
            common::EXIT_IP_ADDR.to_le_bytes().len(),
        ),
        common::EXIT_IP_ADDR.into(),
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
    let processor = BranchingProcessor::new(memory, emulator, handler);
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
                .expect("failed to read RIP")
                .try_into()
                .expect("failed to concretize RIP");
            if rip_value == common::EXIT_IP_ADDR {
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
    write_register(&mut memory, "RSP", &common::INITIAL_STACK.to_le_bytes());
    write_register(&mut memory, "RBP", &common::INITIAL_STACK.to_le_bytes());

    // Put EXIT_IP_ADDR onto the stack. The final RET will trigger this.
    memory.write(
        &VarnodeData::new(
            Address::new(
                sleigh
                    .address_space_by_name("ram")
                    .expect("failed to find ram"),
                common::INITIAL_STACK,
            ),
            common::EXIT_IP_ADDR.to_le_bytes().len(),
        ),
        common::EXIT_IP_ADDR.into(),
    )?;

    // Create symbolic input
    let input_value: [_; 32] = std::array::from_fn(sym::SymbolicBit::Variable);
    let input_value = input_value.into_iter().collect();

    // The test will emulate with a symbolic value. However if we encounter any branch that uses a
    // symbolic value, we will evaluate the branch using the concrete value instead.
    let concrete_input = -3i32;
    let concrete_input = SymbolicBitVec::from(concrete_input as u32);
    let evaluator = Evaluator::new(VariableAssignments::from_bitvecs(
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
                let evaluation = evaluator
                    .evaluate(&processor.memory().read_bit(condition_origin)?)
                    .response
                    .expect("evaluation should be concrete");
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
        if instruction_pointer == common::EXIT_IP_ADDR
            && matches!(processor.state(), ProcessorState::Fetch)
        {
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
