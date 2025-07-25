## Overview

This crate is a small wrapper around the Ghidra SLEIGH compiler for compiling `.slaspec` files into `.sla` files.

## Usage

```rust
let mut compiler = SleighCompiler::default();
let input_file = std::path::Path::new("Ghidra/Processors/x86/data/languages/x86-64.slaspec");
let output_file = std::path::Path::new("x86-64.sla");
compiler.compile(&input_file, &output_file)?;

// Success! Can now use compiled x86-64.sla file
```

## Related Work

See [libsla](https://crates.io/crates/libsla) for using Ghidra libsla to using `.sla` files to perform native and pcode disassembly.
