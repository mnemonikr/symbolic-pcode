mod util;

use std::rc::Rc;

use pcode_ops::PcodeOps;
use symbolic_pcode::arch;
use symbolic_pcode::kernel::linux::LinuxKernel;
use symbolic_pcode::libsla::Sleigh;
use symbolic_pcode::mem::VarnodeDataStore;
use symbolic_pcode::processor::{self, Processor, ProcessorState};

use crate::common;
use util::initialize_libc_stack;

#[test]
fn hello_world_x86_linux() -> processor::Result<()> {
    common::initialize_logger();

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

    let image = match common::find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = Rc::new(common::x86_64_sleigh().expect("failed to build sleigh"));
    let rip = sleigh.register_from_name("RIP").expect("invalid register");
    let mut memory = common::memory_with_image(&sleigh, image, &rip);
    common::init_registers_x86_64(sleigh.as_ref(), &mut memory);
    initialize_libc_stack(&mut memory, sleigh.as_ref());

    let handler = arch::x86::processor::ProcessorHandlerX86::new(sleigh.as_ref());
    let emulator =
        arch::x86::emulator::EmulatorX86::with_kernel(sleigh.clone(), LinuxKernel::default());
    let mut processor = Processor::new(memory, emulator, handler);

    loop {
        let print_pcode = matches!(processor.state(), ProcessorState::Decode(_));
        processor.step(sleigh.as_ref())?;

            if print_pcode && let ProcessorState::Execute(e) = processor.state() {
                for instr in &e.pcode().instructions {
                    println!("{instr}");
                }
            }

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
fn hello_world_aarch64_linux() -> processor::Result<()> {
    common::initialize_logger();

    // Build test fixture first
    let messages = escargot::CargoBuild::new()
        .bin("linux-syscalls")
        .manifest_path("./test-fixtures/linux-syscalls/Cargo.toml")
        .target("aarch64-unknown-linux-musl")
        .env(
            "RUSTFLAGS",
            "-Ctarget-feature=+crt-static -Crelocation-model=static",
        )
        // On Ubuntu can install with sudo apt install gcc-aarch64-linux-gnu
        .env(
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER",
            "aarch64-linux-gnu-gcc",
        )
        .exec()
        .unwrap();

    let image = match common::find_test_fixture(messages) {
        Ok(image) => image,
        Err(messages) => {
            panic!("Failed to find test fixture: {messages:?}");
        }
    };

    let sleigh = Rc::new(common::aarch64_sleigh().expect("failed to build sleigh"));
    let pc = sleigh.register_from_name("pc").expect("invalid register");
    let mut memory = common::memory_with_image(&sleigh, image, &pc);
    common::init_registers_aarch64(sleigh.as_ref(), &mut memory);
    initialize_libc_stack(&mut memory, sleigh.as_ref());

    let handler = arch::aarch64::processor::ProcessorHandler::new(sleigh.as_ref());
    let emulator = arch::aarch64::emulator::Emulator::with_kernel(
        sleigh.clone(),
        LinuxKernel::with_config(arch::aarch64::linux::config()),
    );
    let mut processor = Processor::new(memory, emulator, handler);

    loop {
        let print_pcode = matches!(processor.state(), ProcessorState::Decode(_));
        processor.step(sleigh.as_ref())?;

        if print_pcode {
            if let ProcessorState::Execute(e) = processor.state() {
                for instr in &e.pcode().instructions {
                    println!("{instr}");
                }
            }
        }

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
            println!("Decoded: {instr}", instr = disassembly.instruction);
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
