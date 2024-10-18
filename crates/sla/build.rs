use std::path::Path;

fn main() {
    let source_path = Path::new("ghidra/Ghidra/Features/Decompiler/src/decompile/cpp");

    // The following sources were pulled from the Makefile
    const LIBSLA_SOURCE_FILES: &[&str] = &[
        // CORE=	xml marshal space float address pcoderaw translate opcodes globalcontext
        "xml.cc",
        "marshal.cc",
        "space.cc",
        "float.cc",
        "address.cc",
        "pcoderaw.cc",
        "translate.cc",
        "opcodes.cc",
        "globalcontext.cc",
        // SLEIGH=	sleigh pcodeparse pcodecompile sleighbase slghsymbol \
        // slghpatexpress slghpattern semantics context filemanage
        "sleigh.cc",
        "pcodeparse.cc",
        "pcodecompile.cc",
        "sleighbase.cc",
        "slghsymbol.cc",
        "slghpatexpress.cc",
        "slghpattern.cc",
        "semantics.cc",
        "context.cc",
        "filemanage.cc",
        // LIBSLA_NAMES
        "loadimage.cc",
        "memstate.cc",
        "emulate.cc",
        "opbehavior.cc",
    ];

    cxx_build::bridge("src/ffi/sys.rs")
        .flag_if_supported("-std=c++14")
        .files(LIBSLA_SOURCE_FILES.iter().map(|s| source_path.join(s)))
        .file("src/ffi/cpp/bridge.cc")
        .include(&source_path) // Header files coexist with cpp files
        .warnings(false) // Not interested in the warnings for Ghidra code
        .compile("libsla.a");

    // Rerun if any of the C++ to Rust bindings have changed
    println!("cargo:rerun-if-changed=src/ffi/sys.rs");

    // Rerun if any of the C++ bridge code has changed
    println!("cargo:rerun-if-changed=src/ffi/cpp");

    // Rerun if any of the Rust to C++ bindings have changed
    println!("cargo:rerun-if-changed=src/ffi/rust.rs");
}
