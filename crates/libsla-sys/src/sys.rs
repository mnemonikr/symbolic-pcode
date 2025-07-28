pub use default::*;

use super::rust::*;

#[cxx::bridge]
mod default {
    #[repr(i32)]
    #[derive(Debug)]
    #[namespace = "ghidra"]
    enum spacetype {
        /// Special space to represent constants
        IPTR_CONSTANT = 0,
        /// Normal spaces modelled by processor
        IPTR_PROCESSOR = 1,
        /// addresses = offsets off of base register
        IPTR_SPACEBASE = 2,
        /// Internally managed temporary space
        IPTR_INTERNAL = 3,
        /// Special internal FuncCallSpecs reference
        IPTR_FSPEC = 4,
        /// Special internal PcodeOp reference
        IPTR_IOP = 5,
        /// Special virtual space to represent split variables
        IPTR_JOIN = 6,
    }

    #[repr(i32)]
    #[derive(Debug)]
    #[namespace = "ghidra"]
    enum OpCode {
        /// Copy one operand to another
        CPUI_COPY = 1,
        /// Load from a pointer into a specified address space
        CPUI_LOAD = 2,
        /// Store at a pointer into a specified address space
        CPUI_STORE = 3,

        /// Always branch
        CPUI_BRANCH = 4,
        /// Conditional branch
        CPUI_CBRANCH = 5,
        /// Indirect branch (jumptable)
        CPUI_BRANCHIND = 6,

        /// Call to an absolute address
        CPUI_CALL = 7,
        /// Call through an indirect address
        CPUI_CALLIND = 8,
        /// User-defined operation
        CPUI_CALLOTHER = 9,
        /// Return from subroutine
        CPUI_RETURN = 10,

        // Integer/bit operations
        /// Integer comparison, equality (==)
        CPUI_INT_EQUAL = 11,
        /// Integer comparison, in-equality (!=)
        CPUI_INT_NOTEQUAL = 12,
        /// Integer comparison, signed less-than (<)
        CPUI_INT_SLESS = 13,
        /// Integer comparison, signed less-than-or-equal (<=)
        CPUI_INT_SLESSEQUAL = 14,
        /// Integer comparison, unsigned less-than (<)
        /// This also indicates a borrow on unsigned substraction
        CPUI_INT_LESS = 15,
        /// Integer comparison, unsigned less-than-or-equal (<=)
        CPUI_INT_LESSEQUAL = 16,
        /// Zero extension
        CPUI_INT_ZEXT = 17,
        /// Sign extension
        CPUI_INT_SEXT = 18,
        /// Addition, signed or unsigned (+)
        CPUI_INT_ADD = 19,
        /// Subtraction, signed or unsigned (-)
        CPUI_INT_SUB = 20,
        /// Test for unsigned carry
        CPUI_INT_CARRY = 21,
        /// Test for signed carry
        CPUI_INT_SCARRY = 22,
        /// Test for signed borrow
        CPUI_INT_SBORROW = 23,
        /// Twos complement
        CPUI_INT_2COMP = 24,
        /// Logical/bitwise negation (~)
        CPUI_INT_NEGATE = 25,
        /// Logical/bitwise exclusive-or (^)
        CPUI_INT_XOR = 26,
        /// Logical/bitwise and (&)
        CPUI_INT_AND = 27,
        /// Logical/bitwise or (|)
        CPUI_INT_OR = 28,
        /// Left shift (<<)
        CPUI_INT_LEFT = 29,
        /// Right shift, logical (>>)
        CPUI_INT_RIGHT = 30,
        /// Right shift, arithmetic (>>)
        CPUI_INT_SRIGHT = 31,
        /// Integer multiplication, signed and unsigned (*)
        CPUI_INT_MULT = 32,
        /// Integer division, unsigned (/)
        CPUI_INT_DIV = 33,
        /// Integer division, signed (/)
        CPUI_INT_SDIV = 34,
        /// Remainder/modulo, unsigned (%)
        CPUI_INT_REM = 35,
        /// Remainder/modulo, signed (%)
        CPUI_INT_SREM = 36,

        /// Boolean negate (!)
        CPUI_BOOL_NEGATE = 37,
        /// Boolean exclusive-or (^^)
        CPUI_BOOL_XOR = 38,
        /// Boolean and (&&)
        CPUI_BOOL_AND = 39,
        /// Boolean or (||)
        CPUI_BOOL_OR = 40,

        // Floating point operations
        /// Floating-point comparison, equality (==)
        CPUI_FLOAT_EQUAL = 41,
        /// Floating-point comparison, in-equality (!=)
        CPUI_FLOAT_NOTEQUAL = 42,
        /// Floating-point comparison, less-than (<)
        CPUI_FLOAT_LESS = 43,
        /// Floating-point comparison, less-than-or-equal (<=)
        CPUI_FLOAT_LESSEQUAL = 44,
        // Slot 45 is currently unused
        /// Not-a-number test (NaN)
        CPUI_FLOAT_NAN = 46,

        /// Floating-point addition (+)
        CPUI_FLOAT_ADD = 47,
        /// Floating-point division (/)
        CPUI_FLOAT_DIV = 48,
        /// Floating-point multiplication (*)
        CPUI_FLOAT_MULT = 49,
        /// Floating-point subtraction (-)
        CPUI_FLOAT_SUB = 50,
        /// Floating-point negation (-)
        CPUI_FLOAT_NEG = 51,
        /// Floating-point absolute value (abs)
        CPUI_FLOAT_ABS = 52,
        /// Floating-point square root (sqrt)
        CPUI_FLOAT_SQRT = 53,

        /// Convert an integer to a floating-point
        CPUI_FLOAT_INT2FLOAT = 54,
        /// Convert between different floating-point sizes
        CPUI_FLOAT_FLOAT2FLOAT = 55,
        /// Round towards zero
        CPUI_FLOAT_TRUNC = 56,
        /// Round towards +infinity
        CPUI_FLOAT_CEIL = 57,
        /// Round towards -infinity
        CPUI_FLOAT_FLOOR = 58,
        /// Round towards nearest
        CPUI_FLOAT_ROUND = 59,

        // Internal opcodes for simplification. Not
        // typically generated in a direct translation.

        // Data-flow operations
        /// Phi-node operator
        CPUI_MULTIEQUAL = 60,
        /// Copy with an indirect effect
        CPUI_INDIRECT = 61,
        /// Concatenate
        CPUI_PIECE = 62,
        /// Truncate
        CPUI_SUBPIECE = 63,

        /// Cast from one data-type to another
        CPUI_CAST = 64,
        /// Index into an array ([])
        CPUI_PTRADD = 65,
        /// Drill down to a sub-field  (->)
        CPUI_PTRSUB = 66,
        /// Look-up a \e segmented address
        CPUI_SEGMENTOP = 67,
        /// Recover a value from the \e constant \e pool
        CPUI_CPOOLREF = 68,
        /// Allocate a new object (new)
        CPUI_NEW = 69,
        /// Insert a bit-range
        CPUI_INSERT = 70,
        /// Extract a bit-range
        CPUI_EXTRACT = 71,
        /// Count the 1-bits
        CPUI_POPCOUNT = 72,
        /// Count the leading 0-bits
        CPUI_LZCOUNT = 73,
        /// Value indicating the end of the op-code values
        CPUI_MAX = 74,
    }

    extern "Rust" {
        type RustPcodeEmit<'a>;
        unsafe fn dump(
            self: &mut RustPcodeEmit,
            addr: &Address,
            op_code: OpCode,
            output: *mut VarnodeData,
            inputs: &CxxVector<VarnodeData>,
        );

        type RustLoadImage<'a>;
        unsafe fn load_fill(
            self: &RustLoadImage,
            ptr: *mut u8,
            size: i32,
            addr: &Address,
        ) -> Result<()>;

        type RustAssemblyEmit<'a>;
        fn dump(
            self: &mut RustAssemblyEmit,
            addr: &Address,
            mnemonic: &CxxString,
            body: &CxxString,
        );
    }

    unsafe extern "C++" {
        include!("libsla-sys/src/cpp/bridge.hh");

        #[namespace = "ghidra"]
        type OpCode;

        #[namespace = "ghidra"]
        type spacetype;

        #[namespace = "ghidra"]
        type Address;

        #[rust_name = "new_address"]
        unsafe fn construct_new(address_space: *mut AddrSpace, offset: u64) -> UniquePtr<Address>;

        #[rust_name = "offset"]
        fn getOffset(self: &Address) -> u64;
        #[rust_name = "address_space"]
        fn getSpace(self: &Address) -> *mut AddrSpace;

        #[namespace = "ghidra"]
        type AddrSpace;
        #[rust_name = "name"]
        fn getName(self: &AddrSpace) -> &CxxString;
        #[rust_name = "word_size"]
        fn getWordSize(self: &AddrSpace) -> u32;
        #[rust_name = "address_size"]
        fn getAddrSize(self: &AddrSpace) -> u32;
        #[rust_name = "space_type"]
        fn getType(self: &AddrSpace) -> spacetype;
        #[rust_name = "big_endian"]
        fn isBigEndian(self: &AddrSpace) -> bool;

        #[namespace = "ghidra"]
        type LoadImage;

        #[namespace = "ghidra"]
        type ContextInternal;

        #[namespace = "ghidra"]
        type PcodeEmit;

        #[namespace = "ghidra"]
        type VarnodeData;

        #[rust_name = "varnode_address"]
        fn getAddress(data: &VarnodeData) -> UniquePtr<Address>;
        #[rust_name = "varnode_size"]
        fn getSize(data: &VarnodeData) -> u32;

        #[namespace = "ghidra"]
        type ContextDatabase;
        #[rust_name = "new_context_internal"]
        fn construct_new_context() -> UniquePtr<ContextDatabase>;

        #[namespace = "ghidra"]
        type Element;
        fn initialize_element_id();
        fn initialize_attribute_id();

        #[namespace = "ghidra"]
        type Document;
        #[rust_name = "root"]
        fn getRoot(self: &Document) -> *mut Element;
        #[rust_name = "document_root"]
        fn getDocumentRoot(doc: &Document) -> &Element;

        #[namespace = "ghidra"]
        type DocumentStorage;
        #[rust_name = "new_document_storage"]
        fn construct_new() -> UniquePtr<DocumentStorage>;
        #[rust_name = "docstore_parse_document"]
        fn parseDocumentIntoStore<'a>(
            store: Pin<&'a mut DocumentStorage>,
            data: &CxxString,
        ) -> Result<&'a Document>;

        /// # Safety
        ///
        /// `element` must be a valid pointer.
        #[rust_name = "register_tag"]
        unsafe fn registerTag(self: Pin<&mut DocumentStorage>, element: *const Element);

        // Register varnode
        type RegisterVarnodeName;
        #[rust_name = "register"]
        fn getVarnode(self: &RegisterVarnodeName) -> &VarnodeData;
        #[rust_name = "name"]
        fn getName(self: &RegisterVarnodeName) -> &CxxString;

        // The Sleigh
        type SleighProxy;
        #[rust_name = "new_sleigh"]
        fn construct_new_sleigh(context: UniquePtr<ContextDatabase>) -> UniquePtr<SleighProxy>;
        fn initialize(self: Pin<&mut SleighProxy>, store: Pin<&mut DocumentStorage>) -> Result<()>;
        #[rust_name = "num_spaces"]
        fn numSpaces(self: &SleighProxy) -> i32;
        #[rust_name = "address_space"]
        fn getSpace(self: &SleighProxy, index: i32) -> *mut AddrSpace;
        #[rust_name = "default_code_space"]
        fn getDefaultCodeSpace(self: &SleighProxy) -> *mut AddrSpace;
        /// Throws an exception if the name does not correspond to an existing context variable
        #[rust_name = "set_context_default"]
        fn setContextDefault(
            self: Pin<&mut SleighProxy>,
            name: &CxxString,
            value: u32,
        ) -> Result<()>;
        #[rust_name = "parse_document_and_register_root"]
        fn parseDocumentAndRegisterRootElement(
            store: Pin<&mut DocumentStorage>,
            data: &CxxString,
        ) -> Result<()>;

        #[rust_name = "register_from_name"]
        fn getRegister<'a>(self: &'a SleighProxy, name: &CxxString) -> Result<&'a VarnodeData>;

        // This will return an empty string on failure
        #[rust_name = "register_name"]
        unsafe fn getRegisterNameProxy(
            self: &SleighProxy,
            base: *mut AddrSpace,
            offset: u64,
            size: i32,
        ) -> UniquePtr<CxxString>;

        // Need to proxy this because the C++ API interfaces with std::map. However as of now the
        // cxx crate does not expose std::map. So the FFI layer builds a vector of the map
        // key-value pairs instead in a custom class. The custom layer is necessary since the pair
        // API cannot be easily exposed through FFI either.
        #[rust_name = "all_register_names"]
        fn getAllRegistersProxy(self: &SleighProxy) -> UniquePtr<CxxVector<RegisterVarnodeName>>;

        #[rust_name = "parse_processor_config"]
        fn parseProcessorConfig(self: Pin<&mut SleighProxy>, store: &DocumentStorage)
            -> Result<()>;

        // TODO Can throw UnimplError from C++ which has lots of useful info that is lost here
        #[rust_name = "disassemble_pcode"]
        fn disassemblePcode(
            self: &SleighProxy,
            loadImage: &RustLoadImage,
            emit: &mut RustPcodeEmit,
            address: &Address,
        ) -> Result<i32>;

        #[rust_name = "disassemble_native"]
        fn disassembleNative(
            self: &SleighProxy,
            loadImage: &RustLoadImage,
            emit: &mut RustAssemblyEmit,
            address: &Address,
        ) -> Result<i32>;
    }
}
