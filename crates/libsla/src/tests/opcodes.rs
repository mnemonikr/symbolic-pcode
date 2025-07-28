use crate::OpCode;
use libsla_sys::sys;

#[test]
fn type_conversion() {
    let min_opcode: i32 = unsafe { ::std::mem::transmute(sys::OpCode::CPUI_COPY) };
    let max_opcode: i32 = unsafe { ::std::mem::transmute(sys::OpCode::CPUI_MAX) };
    let mut btreeset: std::collections::BTreeSet<_> = Default::default();
    let mut hashset: std::collections::HashSet<_> = Default::default();
    for opcode in min_opcode..max_opcode {
        if opcode == 45 {
            // opcode 45 is unused
            continue;
        }

        let sys_opcode: sys::OpCode = unsafe { ::std::mem::transmute(opcode) };
        let sla_opcode: OpCode = sys_opcode.into();
        assert_eq!(
            sys_opcode,
            sla_opcode.into(),
            "failed to convert {sla_opcode:?}"
        );

        // Derivation tests
        println!("Opcode: {opcode:?}");

        // Copy derivation test here too. This will not compile without Copy.
        btreeset.insert(sla_opcode);
        hashset.insert(sla_opcode);
    }

    assert_eq!(sys::OpCode::from(OpCode::Unknown(0)), sys::OpCode::CPUI_MAX);
    assert_eq!(
        OpCode::from(sys::OpCode::CPUI_MAX),
        OpCode::Unknown(max_opcode)
    );
}
