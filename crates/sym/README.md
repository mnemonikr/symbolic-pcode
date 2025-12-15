# Overview

A symbolic bitvectors[^bitvec] library for use with theorem provers and other binary analysis tools. Supported operations include:

* Bitwise arithmetic (`&`, `|`, `^`, `!`)
* Bit manipulation (`<<`, `>>`, extension, truncation)
* Bit segmentation (split, join)
* Integer arithmetic (`+`, `-`, `×`, `÷`)
* Integer comparison (`<`, `>`, `=`, `≠`, `≤`, `≥`)

## Example

```rust
# use symbit::{Evaluator, SymbolicBitVec};
# fn f() -> Option<()> {
// Create 32-bit symbolic bitvecs. Each bit is an independent variable.
let x = SymbolicBitVec::with_size(32);
let y = SymbolicBitVec::with_size(32);

// Compute the (symbolic) product of x and y.
let z = x.clone() * y.clone();

// Create an evaluator assuming both x and y are negative.
// This means their most-significant bits (msb) are 1 (true).
let x_msb = x.msb()?.maybe_variable()?;
let y_msb = y.msb()?.maybe_variable()?;
let eval = Evaluator::new([(x_msb, true), (y_msb, true)]);

// If x and y are negative then z should be positive (msb = 0)
let e = eval.evaluate(z.msb()?);
assert_eq!(e.response, Some(false));

// Only the msb from x and y were used in this determination
assert_eq!(e.used_variables.len(), 2);

// The least significant bit (lsb) of the product is a variable.
let e = eval.evaluate(z.lsb()?);
assert!(e.response.is_none());

// The product lsb is dependent only on the lsb of x and y.
let x_lsb = x.lsb()?.maybe_variable()?;
let y_lsb = x.lsb()?.maybe_variable()?;
assert_eq!(e.unassigned_variables.len(), 2);
assert!(e.unassigned_variables.contains(&x_lsb));
assert!(e.unassigned_variables.contains(&y_lsb));
# Some(())
# }
```


# Features

## `aiger`

AIGER is a standardized format for an and-inverter gate circuit. The `aiger` feature can be used to enable integration with the [aiger-circuit](https://crates.io/crates/aiger-circuit) crate for converting constraints on outputs into an AIGER formatted circuit.

[^bitvec]: A bitvector is a fixed sized number of bits that models the semantics of signed and unsigned two's complement arithmetic.
