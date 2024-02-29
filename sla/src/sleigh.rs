use std::borrow::Cow;
use std::sync::Once;

use crate::ffi::api;
use crate::ffi::rust;
use crate::ffi::sys;
use crate::ffi::sys::AddrSpace;
pub use crate::opcodes::OpCode;
use cxx::{let_cxx_string, UniquePtr};

static INIT: Once = Once::new();

pub enum Error {}

pub type Result<T> = std::result::Result<T, Error>;

// TODO Rename this to Sleigh and current Sleigh to GhidraSleigh
pub trait Slegh {
    /// Get the default address space for code execution
    fn default_code_space(&self) -> AddressSpace;

    /// List all available address spaces
    fn address_spaces(&self) -> Vec<AddressSpace>;

    fn register_from_name(&self, name: impl AsRef<str>)
        -> std::result::Result<VarnodeData, String>;

    fn disassemble_pcode(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> std::result::Result<DecodeResponse<PcodeInstruction>, String>;

    fn disassemble_native(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> std::result::Result<DecodeResponse<AssemblyInstruction>, String>;
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Address {
    /// The standard interpretation of the offset is an index into the associated address space.
    /// However, when used in conjunction with the constant address space, the offset is the actual
    /// value. In some contexts this value may be signed, in which case the offset should be
    /// considered an `i64` value.
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

#[derive(Clone, Debug)]
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
        let size = sys::varnode_size(&varnode);
        Self {
            address: sys::varnode_address(&varnode).as_ref().unwrap().into(),
            size: size.try_into().unwrap_or_else(|err| {
                panic!("unable to convert Ghidra varnode size: {size}. {err}")
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddressSpace {
    pub id: usize,
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
}

impl std::fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl From<&sys::AddrSpace> for AddressSpace {
    fn from(address_space: &sys::AddrSpace) -> Self {
        Self {
            id: (address_space as *const _) as usize,
            name: Cow::Owned(address_space.name().to_string()),
            word_size: address_space.word_size().try_into().unwrap(),
            address_size: address_space.address_size().try_into().unwrap(),
            space_type: address_space.space_type().into(),
            big_endian: address_space.big_endian(),
        }
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

#[derive(Clone, Debug)]
pub struct PcodeInstruction {
    // TODO This should be separate from the instruction. Note that in that case though the address
    // becomes a required argument for branch indirect instructions to execute them.
    pub address: Address,
    pub op_code: OpCode,
    pub inputs: Vec<VarnodeData>,
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

pub trait Disassembly {
    type Instruction;

    fn instructions(&self) -> impl AsRef<[Self::Instruction]>;
    fn origin(&self) -> &VarnodeData;
}

impl<T> Disassembly for DecodeResponse<T> {
    type Instruction = T;

    fn instructions(&self) -> impl AsRef<[Self::Instruction]> {
        &self.instructions
    }

    fn origin(&self) -> &VarnodeData {
        &self.origin
    }
}

pub struct AssemblyInstruction {
    pub address: Address,
    pub mnemonic: String,
    pub body: String,
}

pub struct DecodeResponse<T> {
    pub instructions: Vec<T>,
    pub origin: VarnodeData,
}

impl<T> DecodeResponse<T> {
    pub fn new(instructions: Vec<T>, origin: VarnodeData) -> Self {
        Self {
            instructions,
            origin,
        }
    }
}

#[derive(Default)]
pub struct PcodeResponse {
    pub pcode_instructions: Vec<PcodeInstruction>,
    pub num_bytes_consumed: usize,
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

impl<'a> InstructionLoader<'a> {
    /// Returns true only if the requested number of instruction bytes are read.
    fn readable(&self, varnode: &VarnodeData) -> bool {
        self.0
            .instruction_bytes(&varnode)
            .map_or(false, |data| data.len() == varnode.size)
    }
}

impl<'a> api::LoadImage for InstructionLoader<'a> {
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

pub trait LoadImage {
    fn instruction_bytes(&self, data: &VarnodeData) -> std::result::Result<Vec<u8>, String>;
}

pub struct Sleigh {
    /// The sleigh object. This object holds a reference to the image loader.
    sleigh: UniquePtr<sys::SleighProxy>,
}

enum DisassemblyKind<'a> {
    Native(&'a mut Vec<AssemblyInstruction>),
    Pcode(&'a mut Vec<PcodeInstruction>),
}

impl Sleigh {
    pub fn new() -> Self {
        Self {
            sleigh: sys::new_sleigh(sys::new_context_internal()),
        }
    }

    pub fn initialize(
        &mut self,
        sleigh_spec: &str,
        processor_spec: &str,
    ) -> std::result::Result<(), String> {
        // This global libsla initialization is required for parsing sleigh document
        INIT.call_once(|| {
            sys::initialize_element_id();
            sys::initialize_attribute_id();
        });

        let_cxx_string!(sleigh_spec = sleigh_spec);
        let_cxx_string!(processor_spec = processor_spec);

        let mut store = sys::new_document_storage();
        sys::parse_document_and_register_root(store.pin_mut(), &sleigh_spec)
            .map_err(|_err| "Failed to parse sleigh spec")?;

        sys::parse_document_and_register_root(store.pin_mut(), &processor_spec)
            .map_err(|_err| "Failed to parse processor spec")?;

        self.sleigh
            .pin_mut()
            .initialize(store.pin_mut())
            .map_err(|_err| "Failed to initialize sleigh")?;

        self.sleigh
            .pin_mut()
            .parse_processor_config(&store)
            .map_err(|err| format!("Failed to import processor config: {:?}", err))?;

        Ok(())
    }

    fn sys_address_space(&self, space_id: usize) -> Option<*mut AddrSpace> {
        let num_spaces = self.sleigh.num_spaces();
        for i in 0..num_spaces {
            let addr_space = self.sleigh.address_space(i);
            if (addr_space as usize) == space_id {
                return Some(addr_space);
            }
        }

        None
    }

    #[must_use]
    fn disassemble(
        &self,
        loader: &dyn LoadImage,
        address: Address,
        kind: DisassemblyKind,
    ) -> std::result::Result<VarnodeData, String> {
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
            .map_err(|err| format!("Failed to decode instruction: {err}"))?
            .try_into()
            .map_err(|err| format!("Invalid number of bytes consumed: {err}"))?;

        let source = VarnodeData {
            address: (&*address).into(),
            size: bytes_consumed,
        };

        if !loader.readable(&source) {
            return Err("Out-of-bounds read while decoding instruction".to_string());
        }

        Ok(source)
    }

    ///
    ///
    /// # Safety
    ///
    /// The `space_id` must be a valid identifier obtained from a live pcode dump. This identifier
    /// is NOT portable between sessions.
    pub unsafe fn address_space(&self, space_id: usize) -> AddressSpace {
        AddressSpace::from(&*(space_id as *const sys::AddrSpace))
    }
}

impl Slegh for Sleigh {
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

    fn register_from_name(
        &self,
        name: impl AsRef<str>,
    ) -> std::result::Result<VarnodeData, String> {
        let_cxx_string!(name = name.as_ref());
        self.sleigh
            .register_from_name(&name)
            .map(VarnodeData::from)
            .map_err(|err| format!("failed to get register {name}: {err}"))
    }

    fn disassemble_pcode(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> std::result::Result<DecodeResponse<PcodeInstruction>, String> {
        let mut instructions = Vec::new();
        let origin =
            self.disassemble(loader, address, DisassemblyKind::Pcode(&mut instructions))?;
        Ok(DecodeResponse::new(instructions, origin))
    }

    fn disassemble_native(
        &self,
        loader: &dyn LoadImage,
        address: Address,
    ) -> std::result::Result<DecodeResponse<AssemblyInstruction>, String> {
        let mut instructions = Vec::new();
        let origin =
            self.disassemble(loader, address, DisassemblyKind::Native(&mut instructions))?;
        Ok(DecodeResponse::new(instructions, origin))
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

    fn dump_pcode_response(response: &DecodeResponse<PcodeInstruction>) {
        for instruction in response.instructions().as_ref() {
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
    fn test_pcode() {
        const NUM_INSTRUCTIONS: usize = 7;
        let load_image =
            LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x0f\xaf\xc0\x5d\xc3".to_vec());
        let mut sleigh = Sleigh::new();
        let sleigh_spec = fs::read_to_string("../tests/data/x86-64.sla")
            .expect("Failed to read sleigh spec file");

        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        sleigh
            .initialize(&sleigh_spec, &processor_spec)
            .expect("Failed to initialize sleigh");

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
            offset += response.origin().size as u64;
        }
        assert_eq!(offset, 15, "Expected 15 bytes to be decoded");
    }

    #[test]
    fn test_assembly() {
        const NUM_INSTRUCTIONS: usize = 7;
        let load_image =
            LoadImageImpl(b"\x55\x48\x89\xe5\x89\x7d\xfc\x8b\x45\xfc\x01\xc0\x5d\xc3".to_vec());
        let mut sleigh = Sleigh::new();
        let sleigh_spec = fs::read_to_string("../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        sleigh
            .initialize(&sleigh_spec, &processor_spec)
            .expect("Failed to initialize sleigh");

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

        for i in 0..NUM_INSTRUCTIONS {
            let address = Address {
                offset,
                address_space: sleigh.default_code_space(),
            };

            let response = sleigh
                .disassemble_native(&load_image, address)
                .expect("Failed to decode instruction");
            let instruction = &response.instructions[0];
            assert_eq!(instruction.address.address_space.name, expected[i].0);
            assert_eq!(instruction.address.offset, expected[i].1);
            assert_eq!(instruction.mnemonic, expected[i].2);
            assert_eq!(instruction.body, expected[i].3);
            println!(
                "{}:{:016x} | {} {}",
                expected[i].0, expected[i].1, expected[i].2, expected[i].3
            );
            offset += response.origin().size as u64;
        }
    }

    #[test]
    pub fn register_from_name() {
        let mut sleigh = Sleigh::new();
        let sleigh_spec = fs::read_to_string("../tests/data/x86-64.sla")
            .expect("Failed to read processor spec file");
        let processor_spec =
            fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
                .expect("Failed to read processor spec file");

        sleigh
            .initialize(&sleigh_spec, &processor_spec)
            .expect("Failed to initialize sleigh");

        let rax = sleigh.register_from_name("RAX").expect("invalid register");
        assert_eq!(rax.address.address_space.name, "register");
        assert_eq!(rax.address.offset, 0);
        assert_eq!(rax.size, 8);
    }
}
