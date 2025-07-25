#[cxx::bridge]
pub mod bridge {
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
            defines: UniquePtr<DefineOptions>,
            unnecessaryPcodeWarning: bool,
            lenientConflict: bool,
            allCollisionWarning: bool,
            allNopWarning: bool,
            deadTempWarning: bool,
            enforceLocalKeyWord: bool,
            largeTemporaryWarning: bool,
            caseSensitiveRegisterNames: bool,
        );
        fn run_compilation(
            self: Pin<&mut SleighCompileProxy>,
            filein: Pin<&CxxString>,
            fileout: Pin<&CxxString>,
        ) -> Result<i32>;

        type DefineOptions;

        #[rust_name = "new_define_options"]
        fn construct_new_define_options() -> UniquePtr<DefineOptions>;

        #[rust_name = "define_option"]
        fn defineOption(
            self: Pin<&mut DefineOptions>,
            name: Pin<&mut CxxString>,
            value: Pin<&mut CxxString>,
        );
    }
}
