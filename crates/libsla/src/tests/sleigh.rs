use std::io::Read;

use flate2::{
    Compression,
    bufread::{ZlibDecoder, ZlibEncoder},
};
use libsla_sys::sys;
use sleigh_config::processor_x86::PSPEC_X86_64 as PROCESSOR_SPEC;
use sleigh_config::processor_x86::SLA_X86_64 as SLEIGH_SPEC;

use crate::*;

struct InstructionLoader;

impl LoadImage for InstructionLoader {
    fn instruction_bytes(&self, _data: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
        // PUSH RBP
        Ok(vec![0x55])
    }
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
fn build_sla() -> Result<()> {
    // Confirm the original spec builds successfully
    let sleigh = GhidraSleigh::builder()
        .processor_spec(PROCESSOR_SPEC)?
        .build(SLEIGH_SPEC)?;
    verify_sleigh(sleigh);
    Ok(())
}

#[test]
fn build_sla_recompressed() -> Result<()> {
    const SLA_VERSION: u8 = 4;
    const HEADER_SIZE: usize = 4;

    assert!(SLEIGH_SPEC.len() > HEADER_SIZE);
    assert_eq!(SLEIGH_SPEC[0], b's');
    assert_eq!(SLEIGH_SPEC[1], b'l');
    assert_eq!(SLEIGH_SPEC[2], b'a');
    assert_eq!(SLEIGH_SPEC[3], SLA_VERSION);

    // Decompress input
    let mut decoder = ZlibDecoder::new(&SLEIGH_SPEC[4..]);
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .expect("failed to decode zlib compressed sla spec data");
    assert!(!decoded.is_empty(), "decoded data should not be empty");

    // Recompress input
    let mut encoder = ZlibEncoder::new(std::io::Cursor::new(decoded), Compression::fast());
    let mut compressed_data = Vec::with_capacity(4096);
    encoder
        .read_to_end(&mut compressed_data)
        .expect("failed to compress data");

    let mut test_spec = Vec::with_capacity(compressed_data.len() + HEADER_SIZE);
    test_spec.push(b's');
    test_spec.push(b'l');
    test_spec.push(b'a');
    test_spec.push(SLA_VERSION);
    test_spec.append(&mut compressed_data);

    // Confirm the recompressed spec with header builds successfully
    let sleigh = GhidraSleigh::builder()
        .processor_spec(PROCESSOR_SPEC)?
        .build(test_spec)?;
    verify_sleigh(sleigh);
    Ok(())
}

#[test]
fn build_raw_sla() -> Result<()> {
    const SLA_VERSION: u8 = 4;
    const HEADER_SIZE: usize = 4;

    assert!(SLEIGH_SPEC.len() > HEADER_SIZE);
    assert_eq!(SLEIGH_SPEC[0], b's');
    assert_eq!(SLEIGH_SPEC[1], b'l');
    assert_eq!(SLEIGH_SPEC[2], b'a');
    assert_eq!(SLEIGH_SPEC[3], SLA_VERSION);

    // Decompress input
    let mut decoder = ZlibDecoder::new(&SLEIGH_SPEC[4..]);
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .expect("failed to decode zlib compressed sla spec data");
    assert!(!decoded.is_empty(), "decoded data should not be empty");

    let sleigh = GhidraSleigh::builder()
        .processor_spec(PROCESSOR_SPEC)?
        .sla_decoder(SlaDecoder::Raw)
        .build(&decoded)?;
    verify_sleigh(sleigh);
    Ok(())
}

fn verify_sleigh(sleigh: GhidraSleigh) {
    let loader = InstructionLoader;
    let address = Address::new(sleigh.default_code_space(), 0);
    let disassembly = sleigh
        .disassemble_native(&loader, address)
        .expect("disassembly should succeed");

    let instruction = &disassembly.instructions[0];
    assert_eq!(instruction.mnemonic, "PUSH");
    assert_eq!(instruction.body, "RBP");
}
