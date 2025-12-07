# Overview

Library for converting and serializing any symbolic circuit into [AIGER format](https://github.com/arminbiere/aiger/blob/master/FORMAT). The data structure can also be used to query and iterate over the AIGER inputs, gates, and outputs which could be used to generate an AST for a solver.

# Using Aiger

```rust
# use aiger_circuit::Aiger;
# use aiger_circuit::circuit::SimpleCircuit;
# let output = SimpleCircuit::lit(false); // Pretend this is a complex circuit
let aiger = Aiger::with_output(&output);
for input in aiger.inputs() {
    // Iterates over all variables used in the circuit
}

for gate in aiger.gates() {
    // Iterates over all unique gates in the circuit
}

for output in aiger.outputs() {
    // Iterate over all outputs
}

// Serialize data into AIGER binary format
let serialized = aiger.serialize_binary();
```

# Simple Circuit

The `SimpleCircuit` object can be used to build circuits for `Aiger`.

```rust
# use aiger_circuit::Aiger;
# use aiger_circuit::circuit::{AsAigerCircuit, SimpleCircuit};
let output = !SimpleCircuit::var(0) & SimpleCircuit::lit(true);
let aiger = Aiger::with_output(&output);
let serialized = aiger.serialize_binary();
```

# Circuit Conversion

This library does not prescribe a circuit representation. Simply implement the `AsAigerCircuit` trait to enable support for your own custom circuit type.

```rust
# use aiger_circuit::circuit::{AsAigerCircuit, AigerCircuit};
# struct YourCircuitType { }
impl<'a> AsAigerCircuit<'a> for YourCircuitType {
    type Inner = YourCircuitType;

    fn as_circuit(&'a self) -> AigerCircuit<'a, Self::Inner> {
        // Generate approprite AigerCircuit type (And, Not, Variable, Literal)
        todo!();
    }
}
```

