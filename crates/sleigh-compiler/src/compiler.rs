use std::{collections::BTreeMap, path::Path};

use crate::ffi::sys::{self, PreprocessorDefine};
use cxx::UniquePtr;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("non-zero exit code: {0}")]
    NonZeroExitCode(i32),

    #[error("compiler error: {0}")]
    CompilerError(Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The primary interface for interacting with the Sleigh compiler
pub struct SleighCompiler {
    compiler: UniquePtr<sys::SleighCompileProxy>,
}

/// Available compiler options exposed by the Sleigh compiler
pub struct SleighCompilerOptions {
    /// Map of variable to value that is passed to the preprocessor
    pub defines: BTreeMap<String, String>,

    /// Set to `true` to enable individual warnings about unnecessary p-code ops
    pub unnecessary_pcode_warnings: bool,

    /// Set to `false` to report indistinguishable patterns as errors
    pub lenient_conflict: bool,

    /// Set to `true` for individual warnings about constructors with colliding operands
    pub all_collision_warning: bool,

    /// Set to `true` for individual warnings about NOP constructors
    pub all_nop_warning: bool,

    /// Set to `true` for individual warnings about dead temporary varnodes
    pub dead_temp_warning: bool,

    /// Set to `true` to force all local variable definitions to use the `local` keyword
    pub enforce_local_keyword: bool,

    /// Set to `true` for individual warnings about temporary varnodes that are too large
    pub large_temporary_warning: bool,

    /// Set to `true` if register names are allowed to be case-sensitive
    pub case_sensitive_register_names: bool,
}

impl Default for SleighCompilerOptions {
    fn default() -> Self {
        // These are the default options defined in slgh_compile.cc
        Self {
            defines: Default::default(),
            unnecessary_pcode_warnings: false,
            lenient_conflict: true,
            all_collision_warning: false,
            all_nop_warning: false,
            dead_temp_warning: false,
            enforce_local_keyword: false,
            large_temporary_warning: false,
            case_sensitive_register_names: false,
        }
    }
}

impl SleighCompiler {
    /// Construct a new compiler instance configured with the given options. Use
    /// [SleighCompiler::default] to create an instance with default options.
    pub fn new(options: SleighCompilerOptions) -> Self {
        let mut compiler = sys::new_sleigh_compiler();

        let mut defines = cxx::CxxVector::new();
        for (name, value) in options.defines {
            defines.pin_mut().push(PreprocessorDefine { name, value });
        }

        compiler.pin_mut().set_all_options(
            defines.as_ref().unwrap(),
            options.unnecessary_pcode_warnings,
            options.lenient_conflict,
            options.all_collision_warning,
            options.all_nop_warning,
            options.dead_temp_warning,
            options.enforce_local_keyword,
            options.large_temporary_warning,
            options.case_sensitive_register_names,
        );

        Self { compiler }
    }

    /// Invoke the compiler on the provided `.slaspec` input file. The output `.sla` file will be
    /// written to the given output path.
    ///
    /// ### Return value
    ///
    /// The return value for this function
    pub fn compile(
        &mut self,
        input_slaspec_path: impl AsRef<Path>,
        output_sla_path: impl AsRef<Path>,
    ) -> Result<()> {
        cxx::let_cxx_string!(filein = input_slaspec_path.as_ref().as_os_str().as_encoded_bytes());
        cxx::let_cxx_string!(fileout = output_sla_path.as_ref().as_os_str().as_encoded_bytes());

        let exit_code = self
            .compiler
            .pin_mut()
            .run_compilation(filein.as_ref(), fileout.as_ref())
            .map_err(|err| Error::CompilerError(Box::new(err)))?;

        if exit_code == 0 {
            Ok(())
        } else {
            Err(Error::NonZeroExitCode(exit_code))
        }
    }
}

impl Default for SleighCompiler {
    fn default() -> Self {
        Self::new(SleighCompilerOptions::default())
    }
}
