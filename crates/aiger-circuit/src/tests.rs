use crate::circuit::*;
use crate::model::*;
use crate::*;

#[test]
fn true_and_false_are_const() {
    assert!(TRUE.is_const());
    assert!(FALSE.is_const());
}

#[test]
fn input_and_gate_literals_not_negated() {
    let aiger = Aiger::with_output(&(SimpleCircuit::var(0) & SimpleCircuit::var(1)));

    aiger
        .inputs()
        .for_each(|input| assert!(!input.is_negated()));
    aiger
        .gates()
        .for_each(|gate| assert!(!gate.gate_literal().is_negated()));
}

#[test]
fn output_false() {
    let aiger = Aiger::with_output(&SimpleCircuit::lit(false));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert!(inputs.is_empty());

    let gates = aiger.gates().collect::<Vec<_>>();
    assert!(gates.is_empty());

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![FALSE]);

    assert_eq!(
        aiger.serialize_binary(),
        Vec::from("aag 0 0 0 1 0\n0\n".as_bytes())
    );
}

#[test]
fn output_true() {
    let aiger = Aiger::with_output(&SimpleCircuit::lit(true));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert!(inputs.is_empty());

    let gates = aiger.gates().collect::<Vec<_>>();
    assert!(gates.is_empty());

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![TRUE]);

    assert_eq!(
        aiger.serialize_binary(),
        Vec::from("aag 0 0 0 1 0\n1\n".as_bytes())
    );
}

#[test]
fn output_variable() {
    let aiger = Aiger::with_output(&SimpleCircuit::var(0));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert_eq!(inputs, vec![AigerLiteral::new(1)]);

    let gates = aiger.gates().collect::<Vec<_>>();
    assert!(gates.is_empty());

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![AigerLiteral::new(1)]);

    assert_eq!(
        aiger.serialize_binary(),
        Vec::from("aag 1 1 0 1 0\n2\n".as_bytes())
    );
}

#[test]
fn output_negated_variable() {
    let aiger = Aiger::with_output(&!SimpleCircuit::var(0));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert_eq!(inputs, vec![AigerLiteral::new(1)]);

    let gates = aiger.gates().collect::<Vec<_>>();
    assert!(gates.is_empty());

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![AigerLiteral::new(1).negated()]);

    assert_eq!(
        aiger.serialize_binary(),
        Vec::from("aag 1 1 0 1 0\n3\n".as_bytes())
    );
}

#[test]
fn output_and_variables() {
    let aiger = Aiger::with_output(&(SimpleCircuit::var(0) & SimpleCircuit::var(1)));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

    let gates = aiger.gates().collect::<Vec<_>>();
    assert_eq!(
        gates,
        vec![AigerGate::new(
            AigerLiteral::new(3),
            (AigerLiteral::new(1), AigerLiteral::new(2))
        )]
    );

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![AigerLiteral::new(3)]);

    let mut expected = Vec::from("aag 3 2 0 1 1\n6\n".as_bytes());
    expected.push(0x02);
    expected.push(0x02);
    assert_eq!(aiger.serialize_binary(), expected);
}

#[test]
fn output_or_variables() {
    let aiger = Aiger::with_output(&(!(!SimpleCircuit::var(0) & !SimpleCircuit::var(1))));
    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

    let gates = aiger.gates().collect::<Vec<_>>();
    assert_eq!(
        gates,
        vec![AigerGate::new(
            AigerLiteral::new(3),
            (
                AigerLiteral::new(1).negated(),
                AigerLiteral::new(2).negated()
            )
        )]
    );

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![AigerLiteral::new(3).negated()]);

    let mut expected = Vec::from("aag 3 2 0 1 1\n7\n".as_bytes());
    expected.push(0x01);
    expected.push(0x02);
    assert_eq!(aiger.serialize_binary(), expected);
}

#[test]
fn half_adder() {
    let x = SimpleCircuit::var(0);
    let y = SimpleCircuit::var(1);
    let both = x.clone() & y.clone();
    let neither = !x & !y;
    let bits = [!both.clone() & !neither, both];
    let aiger = Aiger::with_outputs(bits.iter());

    let inputs = aiger.inputs().collect::<Vec<_>>();
    assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

    let gates = aiger.gates().collect::<Vec<_>>();
    assert_eq!(
        gates,
        vec![
            AigerGate::new(
                AigerLiteral::new(3),
                (AigerLiteral::new(1), AigerLiteral::new(2))
            ),
            AigerGate::new(
                AigerLiteral::new(4),
                (
                    AigerLiteral::new(1).negated(),
                    AigerLiteral::new(2).negated()
                )
            ),
            AigerGate::new(
                AigerLiteral::new(5),
                (
                    AigerLiteral::new(3).negated(),
                    AigerLiteral::new(4).negated()
                )
            ),
        ]
    );

    let outputs = aiger.outputs().collect::<Vec<_>>();
    assert_eq!(outputs, vec![AigerLiteral::new(5), AigerLiteral::new(3)]);

    let mut expected = Vec::from("aag 5 2 0 2 3\n10\n6\n".as_bytes());
    expected.extend_from_slice(&[0x02, 0x02, 0x03, 0x02, 0x01, 0x02]);
    assert_eq!(aiger.serialize_binary(), expected);
}

#[test]
fn gate_serialization() {
    let serialized = AigerGate::new(
        AigerLiteral::new(2),
        (
            AigerLiteral::new(1).negated(),
            AigerLiteral::new(1).negated(),
        ),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x1, 0x0]);

    let serialized = AigerGate::new(
        AigerLiteral::new(0x80 >> 1),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x80, 0x01, 0x0]);

    let serialized = AigerGate::new(
        AigerLiteral::new(0x7F >> 1).negated(),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x7F, 0x0]);

    let serialized = AigerGate::new(
        AigerLiteral::new(0x102 >> 1),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x82, 0x02, 0x0]);

    // 2^14 - 1
    let serialized = AigerGate::new(
        AigerLiteral::new((0b1_0000000_0000000 - 1) >> 1).negated(),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0xFF, 0x7F, 0x0]);

    // 2^14 + 3
    let serialized = AigerGate::new(
        AigerLiteral::new((0b1_0000000_0000000 + 3) >> 1).negated(),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x83, 0x80, 0x01, 0x0]);

    // 2^28 - 1
    let serialized = AigerGate::new(
        AigerLiteral::new((0b1_0000000_0000000_0000000_0000000 - 1) >> 1).negated(),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0xFF, 0xFF, 0xFF, 0x7F, 0x00]);

    // 2^28 + 7
    let serialized = AigerGate::new(
        AigerLiteral::new((0b1_0000000_0000000_0000000_0000000 + 7) >> 1).negated(),
        (AigerLiteral::new(0), AigerLiteral::new(0)),
    )
    .serialize_binary();
    assert_eq!(serialized, vec![0x87, 0x80, 0x80, 0x80, 0x01, 0x00]);
}
