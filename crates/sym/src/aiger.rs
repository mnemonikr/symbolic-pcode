use std::rc::Rc;

use aiger_circuit::circuit::{AigerCircuit, AndOperand, AsAigerCircuit};

use crate::bit::SymbolicBit;

impl<'a> AsAigerCircuit<'a> for SymbolicBit {
    type Inner = Self;

    fn as_aiger_circuit(&'a self) -> AigerCircuit<'a, Self::Inner> {
        match self {
            SymbolicBit::Literal(value) => AigerCircuit::Literal(*value),
            SymbolicBit::Variable(id) => AigerCircuit::Variable(*id),
            SymbolicBit::Not(x) => AigerCircuit::Not(x.as_ref()),
            SymbolicBit::And(x, y) => AigerCircuit::And(
                AndOperand {
                    id: Rc::as_ptr(x) as usize,
                    value: x.as_ref(),
                },
                AndOperand {
                    id: Rc::as_ptr(y) as usize,
                    value: y.as_ref(),
                },
            ),
        }
    }
}
