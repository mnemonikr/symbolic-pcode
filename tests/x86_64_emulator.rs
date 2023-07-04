mod common;

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
fn doubler_32b() -> Result<(), String> {
    let mut processor = common::Processor::new();
    let base_addr = 0x84210000;
    let num_instructions = 7;

    processor.write_instructions(
        base_addr,
        b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3\x00\x00".to_vec(),
    );

    processor.write_register("RSP", vec![0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00]);
    processor.write_register("RBP", vec![0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x00]);
    processor.write_memory(
        0x0001010101010100,
        vec![0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66],
    );

    let initial_value = 0x99;
    processor.write_register("EDI", vec![initial_value, 0x00, 0x00, 0x00]);

    let mut addr = base_addr;
    for _ in 0..num_instructions {
        addr = processor.emulate(addr)?;
    }

    assert_eq!(addr, 0x66778899aabbccdd, "return address on stack");
    let result: usize = processor.read_register("RAX");
    assert_eq!(
        result,
        2 * initial_value as usize,
        "result should be double initial value: {initial_value}",
    );

    Ok(())
}
