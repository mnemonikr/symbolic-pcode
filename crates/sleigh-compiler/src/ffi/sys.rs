pub use bridge::*;

#[cxx::bridge]
mod bridge {
    struct PreprocessorDefine {
        pub name: String,
        pub value: String,
    }

    unsafe extern "C++" {
        include!("sleigh-compiler/src/ffi/cpp/bridge.hh");

        type SleighCompileProxy;

        #[rust_name = "new_sleigh_compiler"]
        fn construct_new_sleigh_compile() -> UniquePtr<SleighCompileProxy>;

        // Allowing too many arguments to mirror the C++ API
        #[rust_name = "set_all_options"]
        #[allow(clippy::too_many_arguments)]
        fn setAllOptionsProxy(
            self: Pin<&mut SleighCompileProxy>,
            defines: &CxxVector<PreprocessorDefine>,
            unnecessaryPcodeWarning: bool,
            lenientConflict: bool,
            allCollisionWarning: bool,
            allNopWarning: bool,
            deadTempWarning: bool,
            enforceLocalKeyWord: bool,
            largeTemporaryWarning: bool,
            caseSensitiveRegisterNames: bool,
        );

        #[rust_name = "run_compilation"]
        fn run_compilation_proxy(
            self: Pin<&mut SleighCompileProxy>,
            filein: Pin<&CxxString>,
            fileout: Pin<&CxxString>,
        ) -> Result<i32>;
    }
}
