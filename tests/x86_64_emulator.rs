mod common;

/// Confirms the functionality of general-purpose x86-64 registers and overlapping behavior.
#[test]
fn x86_64_registers() {
    let mut processor = common::Processor::new();

    let registers = vec!['A', 'B', 'C', 'D'];
    for register in registers {
        processor.write_register(
            format!("R{register}X"),
            vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        );
        let rax: u64 = processor.read_register(format!("R{register}X"));
        assert_eq!(rax, 0x8877665544332211);
        let eax: u32 = processor.read_register(format!("E{register}X"));
        assert_eq!(eax, 0x44332211);
        let ax: u16 = processor.read_register(format!("{register}X"));
        assert_eq!(ax, 0x2211);
        let ah: u8 = processor.read_register(format!("{register}H"));
        assert_eq!(ah, 0x22);
        let al: u8 = processor.read_register(format!("{register}L"));
        assert_eq!(al, 0x11);
    }

    let registers = vec!["SI", "DI", "BP", "SP"];
    for register in registers {
        processor.write_register(
            format!("R{register}"),
            vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        );
        let r: u64 = processor.read_register(format!("R{register}"));
        assert_eq!(r, 0x8877665544332211);
        let e: u32 = processor.read_register(format!("E{register}"));
        assert_eq!(e, 0x44332211);
        let b: u16 = processor.read_register(format!("{register}"));
        assert_eq!(b, 0x2211);
        let l: u8 = processor.read_register(format!("{register}L"));
        assert_eq!(l, 0x11);
    }

    for register in 8..=15 {
        processor.write_register(
            format!("R{register}"),
            vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        );
        let r: u64 = processor.read_register(format!("R{register}"));
        assert_eq!(r, 0x8877665544332211);
        let rd: u32 = processor.read_register(format!("R{register}D"));
        assert_eq!(rd, 0x44332211);
        let rw: u16 = processor.read_register(format!("R{register}W"));
        assert_eq!(rw, 0x2211);
        let rb: u8 = processor.read_register(format!("R{register}B"));
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
