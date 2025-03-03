use std::borrow::Cow;
use std::sync::Once;

use cxx::{let_cxx_string, UniquePtr};

use crate::ffi::api;
use crate::ffi::rust;
use crate::ffi::sys;
use crate::ffi::sys::AddrSpace;
use crate::opcodes::OpCode;

/// Tracks whether the one-time initialization required for libsla has been performed
static LIBSLA_INIT: Once = Once::new();

/// Errors returned by this crate. Note that some APIs that may pass through FFI boundaries return
/// [String] since those errors are ultimately serialized anyway.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("insufficient data at varnode {0}")]
    InsufficientData(VarnodeData),

    #[error("dependency error: {message} caused by {source}")]
    DependencyError {
        message: Cow<'static, str>,
        source: Box<dyn std::error::Error>,
    },

    #[error("internal error: {0}")]
    InternalError(String),
}

/// Result returned by Sleigh APIs
pub type Result<T> = std::result::Result<T, Error>;

/// Interface for the Sleigh API. See [GhidraSleigh] for the Ghidra implementation.
pub trait Sleigh {
    /// Get the default address space for code execution
    #[must_use]
    fn default_code_space(&self) -> AddressSpace;

    /// List all available address spaces
    #[must_use]
    fn address_spaces(&self) -> Vec<AddressSpace>;

    /// Get an address space by name (if it exists)
    #[must_use]
    fn address_space_by_name(&self, name: impl AsRef<str>) -> Option<AddressSpace> {
        let name = name.as_ref();
        self.address_spaces()
            .into_iter()
            .find(|addr_space| addr_space.name == name)
    }

    /// Get the [VarnodeData] that represents the named register.
    fn register_from_name(&self, name: impl AsRef<str>) -> Result<VarnodeData>;

    /// Disassemble the instructions at the given address into pcode.
    fn disassemble_pcode(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> Result<Disassembly<PcodeInstruction>>;

    /// Disassemble the instructions at the given address into native assembly instructions.
    fn disassemble_native(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> Result<Disassembly<AssemblyInstruction>>;
}

/// An address is represented by an offset into an address space
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Address {
    /// The standard interpretation of the offset is an index into the associated address space.
    /// However, when used in conjunction with the constant address space, the offset is the actual
    /// value. In some contexts this value may be signed, in which case the offset should be
    /// considered an [i64] value.
    pub offset: u64,
    pub address_space: AddressSpace,
}

impl Address {
    pub fn new(address_space: AddressSpace, offset: u64) -> Self {
        Self {
            address_space,
            offset,
        }
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{:0width$x}",
            self.address_space,
            self.offset,
            // Each byte is represented by 2 hex characters
            width = 2 * self.address_space.address_size
        )
    }
}

impl From<&sys::Address> for Address {
    fn from(address: &sys::Address) -> Self {
        Self {
            offset: address.offset(),
            address_space: unsafe { &*address.address_space() }.into(),
        }
    }
}

/// A VarnodeData represents the address and size of data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarnodeData {
    pub address: Address,
    pub size: usize,
}

impl std::fmt::Display for VarnodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]#{}", self.address, self.size)
    }
}

impl VarnodeData {
    pub fn new(address: Address, size: usize) -> Self {
        Self { address, size }
    }

    pub fn range(&self) -> std::ops::Range<u64> {
        let offset = self.address.offset * self.address.address_space.word_size as u64;
        let size: u64 = self
            .size
            .try_into()
            .unwrap_or_else(|err| panic!("invalid varnode size {size}: {err}", size = self.size));

        offset..offset + size
    }
}

impl From<&sys::VarnodeData> for VarnodeData {
    fn from(varnode: &sys::VarnodeData) -> Self {
        let size = sys::varnode_size(varnode);
        Self {
            address: sys::varnode_address(varnode).as_ref().unwrap().into(),
            size: size.try_into().unwrap_or_else(|err| {
                panic!("unable to convert Ghidra varnode size: {size}. {err}")
            }),
        }
    }
}

/// Address space identifier for an address space
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AddressSpaceId(usize);

impl std::fmt::Display for AddressSpaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:#0width$x}",
            self.0,
            // Each byte is represented by 2 hex characters
            width = 2 * std::mem::size_of::<usize>()
        )
    }
}

impl AddressSpaceId {
    /// Construct a new address space id
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the raw identifier representing this address space id. This identifier should be
    /// treated as an opaque value.
    pub const fn raw_id(self) -> usize {
        self.0
    }
}

/// Information about an address space
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddressSpace {
    pub id: AddressSpaceId,
    pub name: Cow<'static, str>,
    pub word_size: usize,
    pub address_size: usize,
    pub space_type: AddressSpaceType,
    pub big_endian: bool,
}

impl AddressSpace {
    pub fn is_constant(&self) -> bool {
        self.space_type == AddressSpaceType::Constant
    }

    /// Creates an address space from a Ghidra address space id.
    ///
    /// # Safety
    ///
    /// The address space id must have originated from the Ghidra library in the current process.
    pub unsafe fn from_ghidra_id(id: AddressSpaceId) -> AddressSpace {
        AddressSpace::from(&*(id.0 as *const sys::AddrSpace))
    }
}

impl std::fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl From<&sys::AddrSpace> for AddressSpace {
    fn from(address_space: &sys::AddrSpace) -> Self {
        Self {
            id: address_space.into(),
            name: Cow::Owned(address_space.name().to_string()),
            word_size: address_space.word_size().try_into().unwrap(),
            address_size: address_space.address_size().try_into().unwrap(),
            space_type: address_space.space_type().into(),
            big_endian: address_space.big_endian(),
        }
    }
}

impl From<&sys::AddrSpace> for AddressSpaceId {
    fn from(address_space: &sys::AddrSpace) -> Self {
        Self::new((address_space as *const _) as usize)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AddressSpaceType {
    /// Special space to represent constants
    Constant,
    /// Normal spaces modelled by processor
    Processor,
    /// addresses = offsets off of base register
    BaseRegister,
    /// Internally managed temporary space
    Internal,
    /// Special internal FuncCallSpecs reference
    FuncCallSpecs,
    /// Special internal PcodeOp reference
    PcodeOp,
    /// Special virtual space to represent split variables
    Join,
}

impl From<sys::spacetype> for AddressSpaceType {
    fn from(space_type: sys::spacetype) -> Self {
        match space_type {
            sys::spacetype::IPTR_CONSTANT => Self::Constant,
            sys::spacetype::IPTR_PROCESSOR => Self::Processor,
            sys::spacetype::IPTR_SPACEBASE => Self::BaseRegister,
            sys::spacetype::IPTR_INTERNAL => Self::Internal,
            sys::spacetype::IPTR_FSPEC => Self::FuncCallSpecs,
            sys::spacetype::IPTR_IOP => Self::PcodeOp,
            sys::spacetype::IPTR_JOIN => Self::Join,
            _ => panic!("Unknown address space type: {space_type:?}"),
        }
    }
}

/// A pcode instruction
#[derive(Clone, Debug)]
pub struct PcodeInstruction {
    /// The originating address for this instruction. This information is necessary to include for
    /// the [OpCode::BranchIndirect] operation, which determines the destination address space from
    /// the instruction address space.
    pub address: Address,

    /// The operation this pcode performs. The operation defines the semantics for the inputs and
    /// optional output of this instruction.
    pub op_code: OpCode,

    /// The inputs this pcode operation requires. The semantics for the inputs is determined by
    /// the [OpCode]. For example, the [OpCode::Load] operation requires the first input has an
    /// address in the [AddressSpaceType::Constant] address space, and is interpreted as an address
    /// space identifier for the ultimate address to load. The second input is interpreted as a
    /// pointer to the offset of the address to load, meaning its size must match the target
    /// address space.
    pub inputs: Vec<VarnodeData>,

    /// The output for the pcode operation. The semantics of the output and whether it is expected
    /// is determined by the [OpCode].
    pub output: Option<VarnodeData>,
}

impl std::fmt::Display for PcodeInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {:?} ", self.address, self.op_code)?;
        if let Some(output) = &self.output {
            write!(f, "{output} <- ")?;
        }

        for input in self.inputs.iter() {
            write!(f, "{input} ")?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AssemblyInstruction {
    pub address: Address,
    pub mnemonic: String,
    pub body: String,
}

impl std::fmt::Display for AssemblyInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{address}] {mnemonic} {body}",
            address = self.address,
            mnemonic = self.mnemonic,
            body = self.body
        )?;
        Ok(())
    }
}

/// A disassembly of instructions originating from a [VarnodeData].
#[derive(Debug, Clone)]
pub struct Disassembly<T> {
    /// The disassembled instructions
    pub instructions: Vec<T>,

    /// The origin of the instructions
    pub origin: VarnodeData,
}

impl<T> Disassembly<T> {
    /// Create a new disassembly
    pub fn new(instructions: Vec<T>, origin: VarnodeData) -> Self {
        Self {
            instructions,
            origin,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Disassembly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{origin}]: {count} instructions",
            origin = self.origin,
            count = self.instructions.len()
        )?;

        for instr in &self.instructions {
            writeln!(f, "{instr}")?;
        }

        Ok(())
    }
}

impl api::AssemblyEmit for Vec<AssemblyInstruction> {
    fn dump(&mut self, address: &sys::Address, mnemonic: &cxx::CxxString, body: &cxx::CxxString) {
        self.push(AssemblyInstruction {
            address: address.into(),
            mnemonic: mnemonic.to_string(),
            body: body.to_string(),
        });
    }
}

impl api::PcodeEmit for Vec<PcodeInstruction> {
    fn dump(
        &mut self,
        address: &sys::Address,
        op_code: sys::OpCode,
        output_variable: Option<&sys::VarnodeData>,
        input_variables: &cxx::CxxVector<sys::VarnodeData>,
    ) {
        self.push(PcodeInstruction {
            address: address.into(),
            op_code: op_code.into(),
            inputs: input_variables
                .into_iter()
                .map(Into::<VarnodeData>::into)
                .collect(),
            output: output_variable.map(Into::<VarnodeData>::into),
        });
    }
}

/// Wrapper around the public load image API so that it can be converted to the native API.
/// This is required in order to pass a trait object reference down into the native API.
struct InstructionLoader<'a>(&'a dyn LoadImage);

impl InstructionLoader<'_> {
    /// Returns true only if the requested number of instruction bytes are read.
    fn readable(&self, varnode: &VarnodeData) -> bool {
        self.0
            .instruction_bytes(varnode)
            .is_ok_and(|data| data.len() == varnode.size)
    }
}

impl api::LoadImage for InstructionLoader<'_> {
    fn load_fill(
        &self,
        data: &mut [u8],
        address: &sys::Address,
    ) -> std::result::Result<(), String> {
        let varnode = VarnodeData {
            size: data.len(),
            address: address.into(),
        };

        let loaded_data = self.0.instruction_bytes(&varnode)?;
        data[..loaded_data.len()].copy_from_slice(&loaded_data);

        Ok(())
    }
}

// TODO This should be renamed. The GhidraSleigh API is called LoadImage but the external trait we
// ask users to implement can be named in a more Rustic fashion.
pub trait LoadImage {
    fn instruction_bytes(&self, data: &VarnodeData) -> std::result::Result<Vec<u8>, String>;
}

enum DisassemblyKind<'a> {
    Native(&'a mut Vec<AssemblyInstruction>),
    Pcode(&'a mut Vec<PcodeInstruction>),
}

/// The sleigh or processor specification has not yet been provided
pub enum MissingSpec {}

/// The sleigh or processor specification has been provided
pub enum HasSpec {}

/// Builder for [GhidraSleigh]. The parameters `S` and `P` track whether the sleigh and processor
/// specifications have been provided.
pub struct GhidraSleighBuilder<S, P> {
    /// Document store for sleigh and processor specifications
    store: UniquePtr<sys::DocumentStorage>,

    /// Phantom data for type tracking whether sleigh specification has been provided
    sleigh_spec: std::marker::PhantomData<S>,

    /// Phantom data for type tracking whether processor specification has been provided
    processor_spec: std::marker::PhantomData<P>,
}

impl Default for GhidraSleighBuilder<MissingSpec, MissingSpec> {
    fn default() -> Self {
        // This global libsla initialization is required for parsing sleigh document
        LIBSLA_INIT.call_once(|| {
            sys::initialize_element_id();
            sys::initialize_attribute_id();
        });

        Self {
            store: sys::new_document_storage(),
            sleigh_spec: Default::default(),
            processor_spec: Default::default(),
        }
    }
}

impl<S> GhidraSleighBuilder<S, MissingSpec> {
    /// Use this processor specification for this sleigh instance.
    pub fn processor_spec(
        mut self,
        processor_spec: &str,
    ) -> Result<GhidraSleighBuilder<S, HasSpec>> {
        let_cxx_string!(processor_spec = processor_spec);
        sys::parse_document_and_register_root(self.store.pin_mut(), &processor_spec).map_err(
            |err| Error::DependencyError {
                message: Cow::Borrowed("failed to parse processor specification"),
                source: Box::new(err),
            },
        )?;

        Ok(GhidraSleighBuilder::<S, HasSpec> {
            store: self.store,
            sleigh_spec: std::marker::PhantomData,
            processor_spec: std::marker::PhantomData,
        })
    }
}

impl<P> GhidraSleighBuilder<MissingSpec, P> {
    /// Use this sleigh specification for this sleigh instance.
    pub fn sleigh_spec(mut self, sleigh_spec: &str) -> Result<GhidraSleighBuilder<HasSpec, P>> {
        let_cxx_string!(sleigh_spec = sleigh_spec);
        sys::parse_document_and_register_root(self.store.pin_mut(), &sleigh_spec).map_err(
            |err| Error::DependencyError {
                message: Cow::Borrowed("failed to parse sleigh specification"),
                source: Box::new(err),
            },
        )?;

        Ok(GhidraSleighBuilder::<HasSpec, P> {
            store: self.store,
            sleigh_spec: std::marker::PhantomData,
            processor_spec: std::marker::PhantomData,
        })
    }
}

impl GhidraSleighBuilder<HasSpec, HasSpec> {
    pub fn build(mut self) -> Result<GhidraSleigh> {
        let mut sleigh = sys::new_sleigh(sys::new_context_internal());

        sleigh
            .pin_mut()
            .initialize(self.store.pin_mut())
            .map_err(|err| Error::DependencyError {
                message: Cow::Borrowed("failed to initialize Ghidra sleigh"),
                source: Box::new(err),
            })?;

        sleigh
            .pin_mut()
            .parse_processor_config(&self.store)
            .map_err(|err| Error::DependencyError {
                message: Cow::Borrowed("failed to import processor config"),
                source: Box::new(err),
            })?;

        Ok(GhidraSleigh { sleigh })
    }
}

/// Sleigh instance that uses Ghidra libsla for its disassembly.
pub struct GhidraSleigh {
    /// The sleigh object. This object holds a reference to the image loader.
    sleigh: UniquePtr<sys::SleighProxy>,
}

impl GhidraSleigh {
    /// Create a new sleigh builder. Use this to construct a sleigh instance.
    pub fn builder() -> GhidraSleighBuilder<MissingSpec, MissingSpec> {
        Default::default()
    }

    fn sys_address_space(&self, space_id: AddressSpaceId) -> Option<*mut AddrSpace> {
        let num_spaces = self.sleigh.num_spaces();
        for i in 0..num_spaces {
            let addr_space = self.sleigh.address_space(i);

            // SAFETY: The address space returned here is safe to dereference
            if AddressSpaceId::from(unsafe { &*addr_space }) == space_id {
                return Some(addr_space);
            }
        }

        None
    }

    fn disassemble(
        &self,
        loader: &dyn LoadImage,
        address: Address,
        kind: DisassemblyKind,
    ) -> Result<VarnodeData> {
        let address = unsafe {
            sys::new_address(
                self.sys_address_space(address.address_space.id)
                    .expect("invalid space id"),
                address.offset,
            )
        };

        let loader = InstructionLoader(loader);
        let rust_loader = rust::RustLoadImage(&loader);

        let response = match kind {
            DisassemblyKind::Pcode(instructions) => {
                let mut emitter = rust::RustPcodeEmit(instructions);
                self.sleigh
                    .disassemble_pcode(&rust_loader, &mut emitter, address.as_ref().unwrap())
            }
            DisassemblyKind::Native(instructions) => {
                let mut emitter = rust::RustAssemblyEmit(instructions);
                self.sleigh.disassemble_native(
                    &rust_loader,
                    &mut emitter,
                    address.as_ref().unwrap(),
                )
            }
        };

        let bytes_consumed = response
            .map_err(|err| Error::DependencyError {
                message: Cow::Borrowed("failed to decode instruction"),
                source: Box::new(err),
            })?
            .try_into()
            .map_err(|err| {
                Error::InternalError(format!("instruction origin size is too large: {err}"))
            })?;

        let source = VarnodeData {
            address: (&*address).into(),
            size: bytes_consumed,
        };

        // Sleigh may attempt to read more bytes than are available to read.
        // Unfortuantely the callback API does not provide any mechanism to
        // inform the caller that only a subset of the requested bytes are valid.
        // Since many ISAs are variable-length instructions, it is possible the
        // valid subset will decode to a valid instruction, and the requested length
        // was an over-estimation.
        //
        // This is a sanity check to determine if the bytes Sleigh used for decoding
        // are all valid.
        if !loader.readable(&source) {
            return Err(Error::InsufficientData(source));
        }

        Ok(source)
    }
}

impl Sleigh for GhidraSleigh {
    fn default_code_space(&self) -> AddressSpace {
        unsafe { &*self.sleigh.default_code_space() }.into()
    }

    fn address_spaces(&self) -> Vec<AddressSpace> {
        let num_spaces = self.sleigh.num_spaces();
        let mut addr_spaces = Vec::with_capacity(num_spaces as usize);
        for i in 0..num_spaces {
            // SAFETY: Address spaces returned from sleigh are safe to dereference
            let raw_addr_space = unsafe { &*self.sleigh.address_space(i) };
            addr_spaces.push(raw_addr_space.into());
        }
        addr_spaces
    }

    fn register_from_name(&self, name: impl AsRef<str>) -> Result<VarnodeData> {
        let_cxx_string!(name = name.as_ref());
        self.sleigh
            .register_from_name(&name)
            .map(VarnodeData::from)
            .map_err(|err| Error::DependencyError {
                message: Cow::Owned(format!("failed to get register {name}")),
                source: Box::new(err),
            })
    }

    fn disassemble_pcode(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> Result<Disassembly<PcodeInstruction>> {
        let mut instructions = Vec::new();
        let origin =
            self.disassemble(loader, address, DisassemblyKind::Pcode(&mut instructions))?;
        Ok(Disassembly::new(instructions, origin))
    }

    fn disassemble_native(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> Result<Disassembly<AssemblyInstruction>> {
        let mut instructions = Vec::new();
        let origin =
            self.disassemble(loader, address, DisassemblyKind::Native(&mut instructions))?;
        Ok(Disassembly::new(instructions, origin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct LoadImageImpl(Vec<u8>);

    impl LoadImage for LoadImageImpl {
        fn instruction_bytes(&self, data: &VarnodeData) -> std::result::Result<Vec<u8>, String> {
            let start: usize = data.address.offset.try_into().expect("invalid offset");
            if start >= self.0.len() {
                return Err("Requested fill outside image".to_string());
            }

            // Never exceed image
            let end = usize::min(start + data.size, self.0.len());
            Ok(self.0[start..end].to_vec())
        }
    }

    fn dump_pcode_response(response: &Disassembly<PcodeInstruction>) {
        for instruction in &response.instructions {
            print!(
                "{}:{:016x} | {:?}",
                instruction.address.address_space.name,
                instruction.address.offset,
                instruction.op_code
            );

            if let Some(output) = instruction.output.as_ref() {
                print!(
                    " [{}:{:016x}]#{} <-",
                    output.address.address_space.name, output.address.offset, output.size
                );
            }

            for input in instruction.inputs.iter() {
                print!(
                    " [{}:{:016x}]#{} <-",
                    input.address.address_space.name, input.address.offset, input.size
                );
            }

            println!();
        }
    }

    #[test]
    fn test_pcode() -> Result<()> {
        const NUM_INSTRUCTIONS: usize = 7;
        let load_image =
            LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x0f\xaf\xc0\x5d\xc3".to_vec());
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read sleigh spec file");

        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;

        let mut offset = 0;
        for _ in 0..NUM_INSTRUCTIONS {
            let address = Address {
                offset,
                address_space: sleigh.default_code_space(),
            };

            let response = sleigh
                .disassemble_pcode(&load_image, address)
                .expect("Failed to decode instruction");
            dump_pcode_response(&response);
            offset += response.origin.size as u64;
        }
        assert_eq!(offset, 15, "Expected 15 bytes to be decoded");
        Ok(())
    }

    #[test]
    fn test_assembly() -> Result<()> {
        const NUM_INSTRUCTIONS: usize = 7;
        let load_image =
            LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3".to_vec());
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;

        let mut offset = 0;
        let expected = vec![
            ("ram".to_string(), 0, "PUSH".to_string(), "RBP".to_string()),
            (
                "ram".to_string(),
                1,
                "MOV".to_string(),
                "RBP,RSP".to_string(),
            ),
            (
                "ram".to_string(),
                4,
                "MOV".to_string(),
                "dword ptr [RBP + -0x4],EDI".to_string(),
            ),
            (
                "ram".to_string(),
                7,
                "MOV".to_string(),
                "EAX,dword ptr [RBP + -0x4]".to_string(),
            ),
            (
                "ram".to_string(),
                10,
                "ADD".to_string(),
                "EAX,EAX".to_string(),
            ),
            ("ram".to_string(), 12, "POP".to_string(), "RBP".to_string()),
            ("ram".to_string(), 13, "RET".to_string(), "".to_string()),
        ];

        for expected_entry in expected {
            let address = Address {
                offset,
                address_space: sleigh.default_code_space(),
            };

            let response = sleigh
                .disassemble_native(&load_image, address)
                .expect("Failed to decode instruction");
            let instruction = &response.instructions[0];
            assert_eq!(instruction.address.address_space.name, expected_entry.0);
            assert_eq!(instruction.address.offset, expected_entry.1);
            assert_eq!(instruction.mnemonic, expected_entry.2);
            assert_eq!(instruction.body, expected_entry.3);
            println!(
                "{}:{:016x} | {} {}",
                expected_entry.0, expected_entry.1, expected_entry.2, expected_entry.3
            );
            offset += response.origin.size as u64;
        }

        Ok(())
    }

    #[test]
    pub fn register_from_name() -> Result<()> {
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;

        let rax = sleigh.register_from_name("RAX").expect("invalid register");
        assert_eq!(rax.address.address_space.name, "register");
        assert_eq!(rax.address.offset, 0);
        assert_eq!(rax.size, 8);
        Ok(())
    }

    #[test]
    pub fn addr_space_type() -> Result<()> {
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_IOP),
            AddressSpaceType::PcodeOp
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_CONSTANT),
            AddressSpaceType::Constant
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_PROCESSOR),
            AddressSpaceType::Processor
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_JOIN),
            AddressSpaceType::Join
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_FSPEC),
            AddressSpaceType::FuncCallSpecs
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_INTERNAL),
            AddressSpaceType::Internal
        );
        assert_eq!(
            AddressSpaceType::from(sys::spacetype::IPTR_SPACEBASE),
            AddressSpaceType::BaseRegister
        );

        Ok(())
    }

    #[test]
    pub fn invalid_register_name() -> Result<()> {
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");
        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;

        let invalid_register_name = "invalid_register";
        let err = sleigh
            .register_from_name(invalid_register_name)
            .expect_err(&format!(
                "register '{invalid_register_name}' should be invalid"
            ));

        let expected_message: Cow<'static, str> =
            Cow::Owned(format!("failed to get register {invalid_register_name}"));
        match err {
            Error::DependencyError { message, .. } => {
                assert_eq!(message, expected_message);
            }
            _ => panic!("Expected dependency error, got {err:?}"),
        }

        Ok(())
    }

    #[test]
    pub fn insufficient_data() -> Result<()> {
        let load_image = LoadImageImpl(b"\x00".to_vec());
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");
        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;
        let offset = 0;
        let address = Address {
            offset,
            address_space: sleigh.default_code_space(),
        };

        let err = sleigh
            .disassemble_native(&load_image, address)
            .expect_err("Expected decoding error");
        println!("{err:?}");

        assert!(matches!(err, Error::InsufficientData { .. }));

        Ok(())
    }

    #[test]
    pub fn invalid_instruction() -> Result<()> {
        let load_image = LoadImageImpl(std::iter::repeat_n(0xFF, 16).collect());
        let sleigh_spec = fs::read_to_string("../../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");
        let sleigh = GhidraSleigh::builder()
            .sleigh_spec(&sleigh_spec)?
            .processor_spec(&processor_spec)?
            .build()?;
        let offset = 0;
        let address = Address {
            offset,
            address_space: sleigh.default_code_space(),
        };

        let err = sleigh
            .disassemble_native(&load_image, address)
            .expect_err("Expected decoding error");
        println!("{err:?}");

        assert!(matches!(
            err,
            Error::DependencyError {
                message: Cow::Borrowed("failed to decode instruction"),
                ..
            }
        ));

        Ok(())
    }
}
