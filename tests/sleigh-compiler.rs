use sleigh_compiler::SleighCompiler;

#[test]
fn compile_x86_x64() -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = SleighCompiler::default();
    let mut input_file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    input_file.push("../crates/libsla/ghidra/Ghidra/Processors/x86/data/languages/x86-64.slaspec");
    assert!(input_file.exists(), "input slaspec file should exist");

    let mut output_file = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    output_file.push("x86-64.sla");

    if output_file.exists() {
        std::fs::remove_file(&output_file)?;
    }

    compiler.compile(input_file, &output_file)?;

    assert!(output_file.exists(), "compiled sla file should exist");
    Ok(())
}
