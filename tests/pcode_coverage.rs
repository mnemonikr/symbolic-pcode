mod util;

use pcode_ops::PcodeOps;
use symbolic_pcode::libsla::{Address, Sleigh, VarnodeData};
use symbolic_pcode::{
    arch::{self, x86::processor::ProcessorHandlerX86},
    emulator::StandardPcodeEmulator,
    mem::VarnodeDataStore,
    processor::{self, Processor, ProcessorState},
};

use crate::common::{self, x86_64_sleigh};
use util::TracingEmulator;

#[test]
fn pcode_coverage() -> processor::Result<()> {
    // Build test fixture first
    let messages = escargot::CargoBuild::new()
        .bin("pcode-coverage")
        .manifest_path("./test-fixtures/pcode-coverage/Cargo.toml")
        .env("RUSTFLAGS", "")
        .target("x86_64-unknown-none")
        .exec()
        .unwrap();

    let image = match common::find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = x86_64_sleigh().expect("failed to build sleigh");
    let rip = sleigh.register_from_name("RIP").expect("invalid register");
    let mut memory = common::memory_with_image(&sleigh, image, &rip);
    common::init_registers_x86_64(&sleigh, &mut memory);

    // Init stack address in memory to magic EXIT_IP_ADDR value
    let stack_addr = VarnodeData {
        address: Address {
            offset: common::INITIAL_STACK,
            address_space: sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
        },
        size: common::EXIT_IP_ADDR.to_le_bytes().len(),
    };
    memory
        .write(&stack_addr, common::EXIT_IP_ADDR.into())
        .expect("failed to initialize stack");

    let handler = ProcessorHandlerX86::new(&sleigh);
    let emulator = TracingEmulator::new(StandardPcodeEmulator::new(sleigh.address_spaces()));
    let mut processor = Processor::new(memory, emulator, handler);

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

        if instruction_pointer == common::EXIT_IP_ADDR
            && matches!(processor.state(), ProcessorState::Fetch)
        {
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
fn pcode_coverage_aarch64() -> processor::Result<()> {
    // Build test fixture first
    let messages = escargot::CargoBuild::new()
        .bin("pcode-coverage")
        .manifest_path("./test-fixtures/pcode-coverage/Cargo.toml")
        .env("RUSTFLAGS", "")
        .target("aarch64-unknown-none")
        .exec()
        .unwrap();

    let image = match common::find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = common::aarch64_sleigh().expect("failed to build sleigh");
    let pc = sleigh.register_from_name("pc").expect("invalid register");
    let mut memory = common::memory_with_image(&sleigh, image, &pc);
    common::init_registers_aarch64(&sleigh, &mut memory);

    // Init stack address in memory to magic EXIT_IP_ADDR value
    let stack_addr = VarnodeData {
        address: Address {
            offset: common::INITIAL_STACK,
            address_space: sleigh
                .address_space_by_name("ram")
                .expect("failed to find ram"),
        },
        size: common::EXIT_IP_ADDR.to_le_bytes().len(),
    };
    memory
        .write(&stack_addr, common::EXIT_IP_ADDR.into())
        .expect("failed to initialize stack");

    let handler = arch::aarch64::processor::ProcessorHandler::new(&sleigh);
    let emulator = TracingEmulator::new(StandardPcodeEmulator::new(sleigh.address_spaces()));
    let mut processor = Processor::new(memory, emulator, handler);

    loop {
        processor.step(&sleigh)?;

        // Check if PC is the magic value
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
            .read(&pc)?
            .try_into()
            .expect("failed to concretize pc register");

        if instruction_pointer == common::EXIT_IP_ADDR
            && matches!(processor.state(), ProcessorState::Fetch)
        {
            break;
        }
    }

    let return_register = sleigh
        .register_from_name("x0")
        .expect("failed to get x0 register");
    let return_value: u64 = processor
        .memory()
        .read(&return_register)?
        .try_into()
        .expect("failed to concretize return register");
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
    assert_eq!(processor.emulator().executed_instructions().len(), 33);
    Ok(())
}
