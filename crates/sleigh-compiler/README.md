## Overview

This crate is a thin wrapper around the Ghidra SLEIGH compiler for compiling `.slaspec` files into `.sla` files.

## Usage

```rust
let mut compiler = SleighCompiler::default();
let input_file = std::path::Path::new("Ghidra/Processors/x86/data/languages/x86-64.slaspec");
let output_file = std::path::Path::new("x86-64.sla");
compiler.compile(&input_file, &output_file)?;

// Success! Can now use compiled x86-64.sla file
```

The SLEIGH compiler may report warnings on stderr. For details on any reported switches, refer to [SLEIGH compiler usage](../../ghidra/Ghidra/Features/Decompiler/src/decompile/cpp/slgh_compile.cc#L3687-L3701).

## Related Work

The [libsla](https://crates.io/crates/libsla) crate wraps the Ghidra SLEIGH library to disassemble native instructions into pcode.
