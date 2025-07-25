use std::path::Path;

fn main() {
    // TODO
    let source_path = Path::new("../libsla/ghidra/Ghidra/Features/Decompiler/src/decompile/cpp");

    // The following sources were pulled from the Makefile
    const SLACOMP_SOURCE_FILES: &[&str] = &[
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
        // SLACOMP=slgh_compile slghparse slghscan
        //"slgh_compile.cc",
        "slghparse.cc",
        "slghscan.cc",
    ];

    cxx_build::bridge("src/ffi/sys.rs")
        .flag_if_supported("-std=c++14")
        .files(SLACOMP_SOURCE_FILES.iter().map(|s| source_path.join(s)))
        .file("src/ffi/cpp/bridge.cc")
        .file("src/ffi/cpp/slgh_compile_nomain.cc")
        .include(source_path) // Header files coexist with cpp files
        .warnings(false) // Not interested in the warnings for Ghidra code
        .compile("libsla.a");

    // Rerun if any of the bindings have changed
    println!("cargo:rerun-if-changed=src/ffi/sys.rs");

    // Rerun if any of the C++ bridge code has changed
    println!("cargo:rerun-if-changed=src/ffi/cpp");
}
