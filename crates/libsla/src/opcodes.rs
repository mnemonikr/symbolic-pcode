//! The opcode for a p-code instruction determines the semantics of the instruction. The [OpCode]
//! enum contains the full list of possible opcodes. However, the [AnalysisOp] opcodes are only
//! ever emitted by analysis programs; they are not permitted in Sleigh processor specifications.
//! The [PseudoOp] opcodes may be emitted but do not have fully defined semantics.
use libsla_sys::sys;

/// A representation of opcodes for p-code instructions.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum OpCode {
    /// Copy a sequence of bytes from one fixed location to another.
    Copy,

    /// Load a sequence of bytes from a dynamic location to a fixed location.
    Load,

    /// Store a sequence of bytes from a fixed location to a dynamic location.
    Store,

    /// Jump to a fixed destination. The destination may be an absolute address or a relative
    /// p-code address for the current instruction.
    Branch,

    /// Jump to a fixed destination based on a condition. See [OpCode::Branch] for details on
    /// an explanation on possible destinations.
    BranchConditional,

    /// Jump to a dynamic destination.
    BranchIndirect,

    /// Semantically identical to [OpCode::Branch] but is used as a hint to analysis programs.
    Call,

    /// Semantically identical to [OpCode::BranchIndirect] but is used as a hint to analysis programs.
    CallIndirect,

    /// Semantically identical to [OpCode::BranchIndirect] but is used as a hint to analysis programs.
    Return,

    /// Concatenates two inputs together: `x:y`.
    Piece,

    /// Truncates an input: `x:y => x`.
    Subpiece,

    /// Counts the number of bits set in an input.
    Popcount,

    /// Count the number of leading 0-bits
    LzCount,

    /// Operations which operate on boolean (single bit) inputs.
    Bool(BoolOp),

    /// Operations which operate on integers.
    Int(IntOp),

    /// Operations which operate on floating-point numbers.
    Float(FloatOp),

    /// A pseudo operation.
    Pseudo(PseudoOp),

    /// An operation produced by analysis.
    Analysis(AnalysisOp),

    /// An unknown operation that holds the raw opcode.
    Unknown(i32),
}

/// Operations for boolean, single-bit inputs.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum BoolOp {
    /// Negate a single bit: `!x`.
    Negate,

    /// The and operation of two bits: `x & y`.
    And,

    /// The inclusive-or of two bits: `x | y`.
    Or,

    /// The exclusive-or of two bits: `x ^ y`.
    Xor,
}

/// Indicates whether an integer operation is operating on signed or unsigned inputs. If the
/// operation does not include `IntSign` as an argument, then distinguishing between signed and
/// unsigned is not applicable for the operation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum IntSign {
    /// An integer where the most significant bit (msb) indicates the sign of the integer. The integer is
    /// positive if the msb is `0` and negative if the msb is `1`. Signed integers are represented
    /// using the two's complement encoding.
    Signed,

    /// An integer that does not have a sign bit and therefore cannot be negative.
    Unsigned,
}

/// Operations on integers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum IntOp {
    /// Add two integers: `x + y`.
    Add,

    /// Negate an integer by converting it to its two's complement: `-x`.
    Negate,

    /// Subtract two integers: `x - y`.
    Subtract,

    /// Multiply two integers: `x * y`.
    Multiply,

    /// Divide two integers: `x / y`.
    Divide(IntSign),

    /// The remainder from integer division: `x % y`.
    Remainder(IntSign),

    /// Check if two integers are equal: `x == y`.
    Equal,

    /// Check if two integers are not equal: `x != y`.
    NotEqual,

    /// Check if an integer is less than another: `x < y`.
    LessThan(IntSign),

    /// Check if an integer is less than or equal to another: `x <= y`.
    LessThanOrEqual(IntSign),

    /// Extend an integer with additional bits. Extends with zero bits if unsigned and with the
    /// sign bit if the integer is signed.
    Extension(IntSign),

    /// The carry flag for an addition indicating an overflow would occur.
    Carry(IntSign),

    /// The borrow flag for a subtraction indicating an overflow would occur. The inputs for this
    /// operation are always signed. The equivalent unsigned check is
    /// `LessThan(IntSign::Unsigned)`.
    Borrow,

    /// Shift the integer left by some number of bits: x << y.
    ShiftLeft,

    /// Shift the integer right by some number of bits: x >> y. A signed shift right will shift in
    /// the sign bit of `x` instead of zero.
    ShiftRight(IntSign),

    /// Bitwise boolean operations applied to each bit of the integer.
    Bitwise(BoolOp),
}

/// Operations on floating-point numbers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum FloatOp {
    /// Check if two numbers are equal: `x == y`.
    Equal,

    /// Check if two numbers are not equal: `x != `y.
    NotEqual,

    /// Check if a number is less than another: `x < y`
    LessThan,

    /// Check if a number is less than or equal to another: `x <= y`
    LessThanOrEqual,

    /// Check if a number is interpreted as NaN.
    IsNaN,

    /// Add two numbers: `x + y`.
    Add,

    /// Subtract two numbers: `x - y`.
    Subtract,

    /// Multiply two numbers: `x + y`.
    Multiply,

    /// Divide two numbers: `x / y`.
    Divide,

    /// Negate a number: `-x`.
    Negate,

    /// Take the absolute value of a number: `|x|`.
    AbsoluteValue,

    /// Take the square root of a number: `x`<sup>0.5</sup>.
    SquareRoot,

    /// Convert an integer to a floating point number.
    IntToFloat,

    /// Convert a floating point number to another with different precision.
    FloatToFloat,

    /// Convert a floating point number to an integer.
    Truncate,

    /// Round a number towards positive infinity.
    Ceiling,

    /// Round a number towards negative infinity.
    Floor,

    /// Round a number to the closest integral value.
    Round,
}

/// Operations which represent black-box placeholders for some sequence of changes to the machine state.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum PseudoOp {
    /// A call that cannot be semantically represented in p-code. For example, a syscall.
    CallOther,

    /// Returns specific run-time dependent values from the constant pool. Used by object-oriented
    /// instruction sets and other managed code environments.
    ConstantPoolRef,

    /// Allocates memory for an object or set of objects.
    New,
}

/// Operations which are only generated by analysis programs. These operations are not permitted
/// for use in processor specifications and therefore will never be emitted when directly
/// translating machine instructions.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum AnalysisOp {
    /// Copies a sequence of bytes to a fixed location. There are multiple origins possible for the
    /// bytes. The selected origin depends on the execution path leading to this operation.
    MultiEqual,

    /// Copies a sequence of bytes to a fixed location. However, the value may be altered
    /// indirectly by another operation referenced by this one.
    CopyIndirect,

    /// Add an offset to a pointer: `(p + i)`
    PointerAdd,

    /// Access a subcomponent of a pointer: `p->x`.
    PointerSubcomponent,

    /// Identical to a [OpCode::Copy]. This operation is a signal that the output data-type
    /// interpretation has changed.
    Cast,

    /// Insert bits from one input into the section of another: `x[8..16] = y`
    Insert,

    /// Extract bits from the section of an input and copy them to another: `y = x[8..16]`.
    Extract,

    /// A placeholder for address mappings involving segments.
    SegmentOp,
}

impl From<sys::OpCode> for OpCode {
    fn from(value: sys::OpCode) -> Self {
        match value {
            sys::OpCode::CPUI_COPY => OpCode::Copy,
            sys::OpCode::CPUI_LOAD => OpCode::Load,
            sys::OpCode::CPUI_STORE => OpCode::Store,
            sys::OpCode::CPUI_BRANCH => OpCode::Branch,
            sys::OpCode::CPUI_CBRANCH => OpCode::BranchConditional,
            sys::OpCode::CPUI_BRANCHIND => OpCode::BranchIndirect,
            sys::OpCode::CPUI_CALL => OpCode::Call,
            sys::OpCode::CPUI_CALLIND => OpCode::CallIndirect,
            sys::OpCode::CPUI_CALLOTHER => OpCode::Pseudo(PseudoOp::CallOther),
            sys::OpCode::CPUI_RETURN => OpCode::Return,
            sys::OpCode::CPUI_INT_EQUAL => OpCode::Int(IntOp::Equal),
            sys::OpCode::CPUI_INT_NOTEQUAL => OpCode::Int(IntOp::NotEqual),
            sys::OpCode::CPUI_INT_SLESS => OpCode::Int(IntOp::LessThan(IntSign::Signed)),
            sys::OpCode::CPUI_INT_SLESSEQUAL => {
                OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed))
            }
            sys::OpCode::CPUI_INT_LESS => OpCode::Int(IntOp::LessThan(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_LESSEQUAL => {
                OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned))
            }
            sys::OpCode::CPUI_INT_ZEXT => OpCode::Int(IntOp::Extension(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_SEXT => OpCode::Int(IntOp::Extension(IntSign::Signed)),
            sys::OpCode::CPUI_INT_ADD => OpCode::Int(IntOp::Add),
            sys::OpCode::CPUI_INT_SUB => OpCode::Int(IntOp::Subtract),
            sys::OpCode::CPUI_INT_CARRY => OpCode::Int(IntOp::Carry(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_SCARRY => OpCode::Int(IntOp::Carry(IntSign::Signed)),
            sys::OpCode::CPUI_INT_SBORROW => OpCode::Int(IntOp::Borrow),
            sys::OpCode::CPUI_INT_2COMP => OpCode::Int(IntOp::Negate),
            sys::OpCode::CPUI_INT_NEGATE => OpCode::Int(IntOp::Bitwise(BoolOp::Negate)),
            sys::OpCode::CPUI_INT_XOR => OpCode::Int(IntOp::Bitwise(BoolOp::Xor)),
            sys::OpCode::CPUI_INT_AND => OpCode::Int(IntOp::Bitwise(BoolOp::And)),
            sys::OpCode::CPUI_INT_OR => OpCode::Int(IntOp::Bitwise(BoolOp::Or)),
            sys::OpCode::CPUI_INT_LEFT => OpCode::Int(IntOp::ShiftLeft),
            sys::OpCode::CPUI_INT_RIGHT => OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_SRIGHT => OpCode::Int(IntOp::ShiftRight(IntSign::Signed)),
            sys::OpCode::CPUI_INT_MULT => OpCode::Int(IntOp::Multiply),
            sys::OpCode::CPUI_INT_DIV => OpCode::Int(IntOp::Divide(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_SDIV => OpCode::Int(IntOp::Divide(IntSign::Signed)),
            sys::OpCode::CPUI_INT_REM => OpCode::Int(IntOp::Remainder(IntSign::Unsigned)),
            sys::OpCode::CPUI_INT_SREM => OpCode::Int(IntOp::Remainder(IntSign::Signed)),
            sys::OpCode::CPUI_BOOL_NEGATE => OpCode::Bool(BoolOp::Negate),
            sys::OpCode::CPUI_BOOL_XOR => OpCode::Bool(BoolOp::Xor),
            sys::OpCode::CPUI_BOOL_AND => OpCode::Bool(BoolOp::And),
            sys::OpCode::CPUI_BOOL_OR => OpCode::Bool(BoolOp::Or),
            sys::OpCode::CPUI_FLOAT_EQUAL => OpCode::Float(FloatOp::Equal),
            sys::OpCode::CPUI_FLOAT_NOTEQUAL => OpCode::Float(FloatOp::NotEqual),
            sys::OpCode::CPUI_FLOAT_LESS => OpCode::Float(FloatOp::LessThan),
            sys::OpCode::CPUI_FLOAT_LESSEQUAL => OpCode::Float(FloatOp::LessThanOrEqual),
            sys::OpCode::CPUI_FLOAT_NAN => OpCode::Float(FloatOp::IsNaN),
            sys::OpCode::CPUI_FLOAT_ADD => OpCode::Float(FloatOp::Add),
            sys::OpCode::CPUI_FLOAT_DIV => OpCode::Float(FloatOp::Divide),
            sys::OpCode::CPUI_FLOAT_MULT => OpCode::Float(FloatOp::Multiply),
            sys::OpCode::CPUI_FLOAT_SUB => OpCode::Float(FloatOp::Subtract),
            sys::OpCode::CPUI_FLOAT_NEG => OpCode::Float(FloatOp::Negate),
            sys::OpCode::CPUI_FLOAT_ABS => OpCode::Float(FloatOp::AbsoluteValue),
            sys::OpCode::CPUI_FLOAT_SQRT => OpCode::Float(FloatOp::SquareRoot),
            sys::OpCode::CPUI_FLOAT_INT2FLOAT => OpCode::Float(FloatOp::IntToFloat),
            sys::OpCode::CPUI_FLOAT_FLOAT2FLOAT => OpCode::Float(FloatOp::FloatToFloat),
            sys::OpCode::CPUI_FLOAT_TRUNC => OpCode::Float(FloatOp::Truncate),
            sys::OpCode::CPUI_FLOAT_CEIL => OpCode::Float(FloatOp::Ceiling),
            sys::OpCode::CPUI_FLOAT_FLOOR => OpCode::Float(FloatOp::Floor),
            sys::OpCode::CPUI_FLOAT_ROUND => OpCode::Float(FloatOp::Round),

            sys::OpCode::CPUI_MULTIEQUAL => OpCode::Analysis(AnalysisOp::MultiEqual),
            sys::OpCode::CPUI_INDIRECT => OpCode::Analysis(AnalysisOp::CopyIndirect),
            sys::OpCode::CPUI_PIECE => OpCode::Piece,
            sys::OpCode::CPUI_SUBPIECE => OpCode::Subpiece,

            sys::OpCode::CPUI_CAST => OpCode::Analysis(AnalysisOp::Cast),
            sys::OpCode::CPUI_PTRADD => OpCode::Analysis(AnalysisOp::PointerAdd),
            sys::OpCode::CPUI_PTRSUB => OpCode::Analysis(AnalysisOp::PointerSubcomponent),
            sys::OpCode::CPUI_SEGMENTOP => OpCode::Analysis(AnalysisOp::SegmentOp),
            sys::OpCode::CPUI_CPOOLREF => OpCode::Pseudo(PseudoOp::ConstantPoolRef),
            sys::OpCode::CPUI_NEW => OpCode::Pseudo(PseudoOp::New),
            sys::OpCode::CPUI_INSERT => OpCode::Analysis(AnalysisOp::Insert),
            sys::OpCode::CPUI_EXTRACT => OpCode::Analysis(AnalysisOp::Extract),
            sys::OpCode::CPUI_POPCOUNT => OpCode::Popcount,
            sys::OpCode::CPUI_LZCOUNT => OpCode::LzCount,
            sys::OpCode { repr } => OpCode::Unknown(repr),
        }
    }
}

impl From<OpCode> for sys::OpCode {
    fn from(value: OpCode) -> Self {
        match value {
            OpCode::Copy => sys::OpCode::CPUI_COPY,
            OpCode::Load => sys::OpCode::CPUI_LOAD,
            OpCode::Store => sys::OpCode::CPUI_STORE,
            OpCode::Branch => sys::OpCode::CPUI_BRANCH,
            OpCode::BranchConditional => sys::OpCode::CPUI_CBRANCH,
            OpCode::BranchIndirect => sys::OpCode::CPUI_BRANCHIND,
            OpCode::Call => sys::OpCode::CPUI_CALL,
            OpCode::CallIndirect => sys::OpCode::CPUI_CALLIND,
            OpCode::Return => sys::OpCode::CPUI_RETURN,
            OpCode::Subpiece => sys::OpCode::CPUI_SUBPIECE,
            OpCode::Piece => sys::OpCode::CPUI_PIECE,
            OpCode::Popcount => sys::OpCode::CPUI_POPCOUNT,
            OpCode::LzCount => sys::OpCode::CPUI_LZCOUNT,
            OpCode::Bool(BoolOp::Negate) => sys::OpCode::CPUI_BOOL_NEGATE,
            OpCode::Bool(BoolOp::Xor) => sys::OpCode::CPUI_BOOL_XOR,
            OpCode::Bool(BoolOp::And) => sys::OpCode::CPUI_BOOL_AND,
            OpCode::Bool(BoolOp::Or) => sys::OpCode::CPUI_BOOL_OR,
            OpCode::Int(IntOp::Equal) => sys::OpCode::CPUI_INT_EQUAL,
            OpCode::Int(IntOp::NotEqual) => sys::OpCode::CPUI_INT_NOTEQUAL,
            OpCode::Int(IntOp::LessThan(IntSign::Signed)) => sys::OpCode::CPUI_INT_SLESS,
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Signed)) => {
                sys::OpCode::CPUI_INT_SLESSEQUAL
            }
            OpCode::Int(IntOp::LessThan(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_LESS,
            OpCode::Int(IntOp::LessThanOrEqual(IntSign::Unsigned)) => {
                sys::OpCode::CPUI_INT_LESSEQUAL
            }
            OpCode::Int(IntOp::Extension(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_ZEXT,
            OpCode::Int(IntOp::Extension(IntSign::Signed)) => sys::OpCode::CPUI_INT_SEXT,
            OpCode::Int(IntOp::Add) => sys::OpCode::CPUI_INT_ADD,
            OpCode::Int(IntOp::Subtract) => sys::OpCode::CPUI_INT_SUB,
            OpCode::Int(IntOp::Carry(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_CARRY,
            OpCode::Int(IntOp::Carry(IntSign::Signed)) => sys::OpCode::CPUI_INT_SCARRY,
            OpCode::Int(IntOp::Borrow) => sys::OpCode::CPUI_INT_SBORROW,
            OpCode::Int(IntOp::Negate) => sys::OpCode::CPUI_INT_2COMP,
            OpCode::Int(IntOp::Bitwise(BoolOp::Negate)) => sys::OpCode::CPUI_INT_NEGATE,
            OpCode::Int(IntOp::Bitwise(BoolOp::Xor)) => sys::OpCode::CPUI_INT_XOR,
            OpCode::Int(IntOp::Bitwise(BoolOp::And)) => sys::OpCode::CPUI_INT_AND,
            OpCode::Int(IntOp::Bitwise(BoolOp::Or)) => sys::OpCode::CPUI_INT_OR,
            OpCode::Int(IntOp::ShiftLeft) => sys::OpCode::CPUI_INT_LEFT,
            OpCode::Int(IntOp::ShiftRight(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_RIGHT,
            OpCode::Int(IntOp::ShiftRight(IntSign::Signed)) => sys::OpCode::CPUI_INT_SRIGHT,
            OpCode::Int(IntOp::Multiply) => sys::OpCode::CPUI_INT_MULT,
            OpCode::Int(IntOp::Divide(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_DIV,
            OpCode::Int(IntOp::Divide(IntSign::Signed)) => sys::OpCode::CPUI_INT_SDIV,
            OpCode::Int(IntOp::Remainder(IntSign::Unsigned)) => sys::OpCode::CPUI_INT_REM,
            OpCode::Int(IntOp::Remainder(IntSign::Signed)) => sys::OpCode::CPUI_INT_SREM,
            OpCode::Float(FloatOp::Equal) => sys::OpCode::CPUI_FLOAT_EQUAL,
            OpCode::Float(FloatOp::NotEqual) => sys::OpCode::CPUI_FLOAT_NOTEQUAL,
            OpCode::Float(FloatOp::LessThan) => sys::OpCode::CPUI_FLOAT_LESS,
            OpCode::Float(FloatOp::LessThanOrEqual) => sys::OpCode::CPUI_FLOAT_LESSEQUAL,
            OpCode::Float(FloatOp::IsNaN) => sys::OpCode::CPUI_FLOAT_NAN,
            OpCode::Float(FloatOp::Add) => sys::OpCode::CPUI_FLOAT_ADD,
            OpCode::Float(FloatOp::Divide) => sys::OpCode::CPUI_FLOAT_DIV,
            OpCode::Float(FloatOp::Multiply) => sys::OpCode::CPUI_FLOAT_MULT,
            OpCode::Float(FloatOp::Subtract) => sys::OpCode::CPUI_FLOAT_SUB,
            OpCode::Float(FloatOp::Negate) => sys::OpCode::CPUI_FLOAT_NEG,
            OpCode::Float(FloatOp::AbsoluteValue) => sys::OpCode::CPUI_FLOAT_ABS,
            OpCode::Float(FloatOp::SquareRoot) => sys::OpCode::CPUI_FLOAT_SQRT,
            OpCode::Float(FloatOp::IntToFloat) => sys::OpCode::CPUI_FLOAT_INT2FLOAT,
            OpCode::Float(FloatOp::FloatToFloat) => sys::OpCode::CPUI_FLOAT_FLOAT2FLOAT,
            OpCode::Float(FloatOp::Truncate) => sys::OpCode::CPUI_FLOAT_TRUNC,
            OpCode::Float(FloatOp::Ceiling) => sys::OpCode::CPUI_FLOAT_CEIL,
            OpCode::Float(FloatOp::Floor) => sys::OpCode::CPUI_FLOAT_FLOOR,
            OpCode::Float(FloatOp::Round) => sys::OpCode::CPUI_FLOAT_ROUND,
            OpCode::Pseudo(PseudoOp::CallOther) => sys::OpCode::CPUI_CALLOTHER,
            OpCode::Pseudo(PseudoOp::ConstantPoolRef) => sys::OpCode::CPUI_CPOOLREF,
            OpCode::Pseudo(PseudoOp::New) => sys::OpCode::CPUI_NEW,
            OpCode::Analysis(AnalysisOp::MultiEqual) => sys::OpCode::CPUI_MULTIEQUAL,
            OpCode::Analysis(AnalysisOp::CopyIndirect) => sys::OpCode::CPUI_INDIRECT,
            OpCode::Analysis(AnalysisOp::Cast) => sys::OpCode::CPUI_CAST,
            OpCode::Analysis(AnalysisOp::PointerAdd) => sys::OpCode::CPUI_PTRADD,
            OpCode::Analysis(AnalysisOp::PointerSubcomponent) => sys::OpCode::CPUI_PTRSUB,
            OpCode::Analysis(AnalysisOp::SegmentOp) => sys::OpCode::CPUI_SEGMENTOP,
            OpCode::Analysis(AnalysisOp::Insert) => sys::OpCode::CPUI_INSERT,
            OpCode::Analysis(AnalysisOp::Extract) => sys::OpCode::CPUI_EXTRACT,
            OpCode::Unknown(_) => sys::OpCode::CPUI_MAX,
        }
    }
}
