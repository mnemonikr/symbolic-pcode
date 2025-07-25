use std::sync::OnceLock;
use std::{borrow::Cow, fs};

use crate::ffi::sys;
use crate::*;

static X86_64_SLA: OnceLock<String> = OnceLock::new();

struct LoadImageImpl(Vec<u8>);

impl LoadImage for LoadImageImpl {
    fn instruction_bytes(&self, data: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        let start: usize = data.address.offset.try_into().expect("invalid offset");
        if start >= self.0.len() {
            return Err("Requested fill outside image".to_string());
        }

        // Never exceed image
        let end = usize::min(start + data.size, self.0.len());
        Ok(self.0[start..end].to_vec())
    }
}

fn dump_pcode_response(response: &Disassembly<PcodeInstruction>) {
    for instruction in &response.instructions {
        print!(
            "{}:{:016x} | {:?}",
            instruction.address.address_space.name, instruction.address.offset, instruction.op_code
        );

        if let Some(output) = instruction.output.as_ref() {
            print!(
                " [{}:{:016x}]#{} <-",
                output.address.address_space.name, output.address.offset, output.size
            );
        }

        for input in instruction.inputs.iter() {
            print!(
                " [{}:{:016x}]#{} <-",
                input.address.address_space.name, input.address.offset, input.size
            );
        }

        println!();
    }
}

fn compile_x86_64_slaspec() -> sleigh_compiler::Result<String> {
    use assert_fs::fixture::PathChild;
    let temp = assert_fs::TempDir::new().unwrap();
    let sla_path = temp.child("x86-64.sla");

    let mut compiler = sleigh_compiler::SleighCompiler::default();
    let slaspec_path =
        std::path::Path::new("../../ghidra/Ghidra/Processors/x86/data/languages/x86-64.slaspec");
    compiler.compile(slaspec_path, &sla_path)?;
    let output = fs::read_to_string(sla_path).expect("failed to read sla file");
    temp.close().unwrap();
    Ok(output)
}

fn x86_64_sleigh() -> Result<GhidraSleigh> {
    let sleigh_spec = X86_64_SLA
        .get_or_init(|| compile_x86_64_slaspec().expect("failed to compile x86-64.slaspec"));
    let processor_spec =
        fs::read_to_string("../../ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
            .expect("Failed to read processor spec file");
    let sleigh = GhidraSleigh::builder()
        .sleigh_spec(&sleigh_spec)?
        .processor_spec(&processor_spec)?
        .build()?;
    Ok(sleigh)
}

#[test]
fn test_pcode() -> Result<()> {
    const NUM_INSTRUCTIONS: usize = 7;
    let load_image =
        LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x0f\xaf\xc0\x5d\xc3".to_vec());
    let sleigh = x86_64_sleigh()?;
    let mut offset = 0;
    for _ in 0..NUM_INSTRUCTIONS {
        let address = Address {
            offset,
            address_space: sleigh.default_code_space(),
        };

        let response = sleigh
            .disassemble_pcode(&load_image, address)
            .expect("Failed to decode instruction");
        dump_pcode_response(&response);
        offset += response.origin.size as u64;
    }
    assert_eq!(offset, 15, "Expected 15 bytes to be decoded");
    Ok(())
}

#[test]
fn test_assembly() -> Result<()> {
    let load_image =
        LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3".to_vec());
    let sleigh = x86_64_sleigh()?;
    let mut offset = 0;
    let expected = vec![
        ("ram".to_string(), 0, "PUSH".to_string(), "RBP".to_string()),
        (
            "ram".to_string(),
            1,
            "MOV".to_string(),
            "RBP,RSP".to_string(),
        ),
        (
            "ram".to_string(),
            4,
            "MOV".to_string(),
            "dword ptr [RBP + -0x4],EDI".to_string(),
        ),
        (
            "ram".to_string(),
            7,
            "MOV".to_string(),
            "EAX,dword ptr [RBP + -0x4]".to_string(),
        ),
        (
            "ram".to_string(),
            10,
            "ADD".to_string(),
            "EAX,EAX".to_string(),
        ),
        ("ram".to_string(), 12, "POP".to_string(), "RBP".to_string()),
        ("ram".to_string(), 13, "RET".to_string(), "".to_string()),
    ];

    for expected_entry in expected {
        let address = Address {
            offset,
            address_space: sleigh.default_code_space(),
        };

        let response = sleigh
            .disassemble_native(&load_image, address)
            .expect("Failed to decode instruction");
        let instruction = &response.instructions[0];
        assert_eq!(instruction.address.address_space.name, expected_entry.0);
        assert_eq!(instruction.address.offset, expected_entry.1);
        assert_eq!(instruction.mnemonic, expected_entry.2);
        assert_eq!(instruction.body, expected_entry.3);
        println!(
            "{}:{:016x} | {} {}",
            expected_entry.0, expected_entry.1, expected_entry.2, expected_entry.3
        );
        offset += response.origin.size as u64;
    }

    Ok(())
}

#[test]
pub fn register_from_name() -> Result<()> {
    let sleigh = x86_64_sleigh()?;
    let rax = sleigh.register_from_name("RAX").expect("invalid register");
    assert_eq!(rax.address.address_space.name, "register");
    assert_eq!(rax.address.offset, 0);
    assert_eq!(rax.size, 8);
    assert_eq!(sleigh.register_name(&rax), Some("RAX".to_string()));
    Ok(())
}

#[test]
pub fn register_name_of_non_register() -> Result<()> {
    let sleigh = x86_64_sleigh()?;
    let mut register = sleigh
        .register_from_name("RAX")
        .expect("RAX should be a valid register");

    // Change offset to something absurd. Make sure not to trigger the overflow check
    // so that this request will actually go to Ghidra
    register.address.offset = u64::MAX - register.size as u64;

    let result = sleigh.register_name(&register);
    assert!(result.is_none(), "{result:?} should be None");
    Ok(())
}

#[test]
pub fn register_name_of_overflowing_non_register() -> Result<()> {
    let sleigh = x86_64_sleigh()?;
    let mut register = sleigh
        .register_from_name("RAX")
        .expect("RAX should be a valid register");

    // Note that the lookup will perform offset + size without overflow checks
    // There is a guard in our impl against this before calling Ghidra.
    //
    // See ghidra/Ghidra/Features/Decompiler/src/decompile/cpp/sleighbase.cc
    register.address.offset = u64::MAX;

    let result = sleigh.register_name(&register);
    assert!(result.is_none(), "{result:?} should be None");
    Ok(())
}

#[test]
pub fn addr_space_type() -> Result<()> {
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_IOP),
        AddressSpaceType::PcodeOp
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_CONSTANT),
        AddressSpaceType::Constant
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_PROCESSOR),
        AddressSpaceType::Processor
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_JOIN),
        AddressSpaceType::Join
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_FSPEC),
        AddressSpaceType::FuncCallSpecs
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_INTERNAL),
        AddressSpaceType::Internal
    );
    assert_eq!(
        AddressSpaceType::from(sys::spacetype::IPTR_SPACEBASE),
        AddressSpaceType::BaseRegister
    );

    Ok(())
}

#[test]
pub fn invalid_register_name() -> Result<()> {
    let sleigh = x86_64_sleigh()?;
    let invalid_register_name = "invalid_register";
    let err = sleigh
        .register_from_name(invalid_register_name)
        .expect_err(&format!(
            "register '{invalid_register_name}' should be invalid"
        ));

    let expected_message: Cow<'static, str> =
        Cow::Owned(format!("failed to get register {invalid_register_name}"));
    match err {
        Error::DependencyError { message, .. } => {
            assert_eq!(message, expected_message);
        }
        _ => panic!("Expected dependency error, got {err:?}"),
    }

    Ok(())
}

#[test]
pub fn insufficient_data() -> Result<()> {
    let load_image = LoadImageImpl(b"\x00".to_vec());
    let sleigh = x86_64_sleigh()?;
    let offset = 0;
    let address = Address {
        offset,
        address_space: sleigh.default_code_space(),
    };

    let err = sleigh
        .disassemble_native(&load_image, address)
        .expect_err("Expected decoding error");
    println!("{err:?}");

    assert!(matches!(err, Error::InsufficientData { .. }));

    Ok(())
}

#[test]
pub fn invalid_instruction() -> Result<()> {
    let load_image = LoadImageImpl(std::iter::repeat_n(0xFF, 16).collect());
    let sleigh = x86_64_sleigh()?;
    let offset = 0;
    let address = Address {
        offset,
        address_space: sleigh.default_code_space(),
    };

    let err = sleigh
        .disassemble_native(&load_image, address)
        .expect_err("Expected decoding error");
    println!("{err:?}");

    assert!(matches!(
        err,
        Error::DependencyError {
            message: Cow::Borrowed("failed to decode instruction"),
            ..
        }
    ));

    Ok(())
}

#[test]
fn all_register_names() -> Result<()> {
    let sleigh = x86_64_sleigh()?;
    let expected_name = ["RAX", "EAX", "AX", "AL"];
    for (i, (reg, name)) in sleigh.register_name_map().iter().take(4).enumerate() {
        assert_eq!(
            reg.address.offset, 0,
            "address offset should be 0 for {name}: {reg:?}"
        );
        assert_eq!(name, expected_name[i]);
    }
    Ok(())
}
