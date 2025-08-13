use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::{collections::BTreeMap, fs};

use libsla::{Address, GhidraSleigh, OpCode, PcodeInstruction, Sleigh, VarnodeData};
use sym::{SymbolicBit, SymbolicBitVec, SymbolicByte};
use symbolic_pcode::emulator::{self, ControlFlow, PcodeEmulator, StandardPcodeEmulator};
use symbolic_pcode::mem::{GenericMemory, VarnodeDataStore};

static X86_64_SLA: OnceLock<PathBuf> = OnceLock::new();
static AARCH_64_SLA: OnceLock<PathBuf> = OnceLock::new();
static LOGGER_INIT: OnceLock<flexi_logger::LoggerHandle> = OnceLock::new();

pub fn initialize_logger() -> &'static flexi_logger::LoggerHandle {
    LOGGER_INIT.get_or_init(|| {
        flexi_logger::Logger::try_with_env()
            .unwrap()
            .start()
            .unwrap()
    })
}

pub const INITIAL_STACK: u64 = 0x8000000000;
pub const EXIT_IP_ADDR: u64 = 0xFEEDBEEF0BADF00D;

pub type Memory = GenericMemory<SymbolicBitVec>;

#[derive(Debug, Clone)]
pub struct TracingEmulator {
    inner: StandardPcodeEmulator,
    executed_instructions: std::cell::RefCell<BTreeMap<OpCode, usize>>,
}

impl PcodeEmulator for TracingEmulator {
    fn emulate<T: VarnodeDataStore>(
        &mut self,
        memory: &mut T,
        instruction: &PcodeInstruction,
    ) -> emulator::Result<ControlFlow> {
        //println!("Executing: {instruction}");
        match &instruction.op_code {
            OpCode::Store => (),
            OpCode::Branch
            | OpCode::BranchIndirect
            | OpCode::BranchConditional
            | OpCode::Call
            | OpCode::CallIndirect
            | OpCode::Return => (),
            _ => {
                /*
                for instr_input in instruction.inputs.iter() {
                    let input_result = memory.read(instr_input);
                    let input_result = match input_result {
                        Ok(x) => PcodeValue::from(x),
                        Err(err) => {
                            println!("Failed to read input {instr_input}: {err}");
                            break;
                        }
                    };

                    match u128::try_from(input_result) {
                        Ok(value) => {
                            println!(
                                "Input {instr_input} = {value:0width$x}",
                                width = 2 * instr_input.size
                            );
                        }
                        Err(TryFromPcodeValueError::InvalidSize) => {
                            println!("Input {instr_input} = Large value")
                        }
                        Err(TryFromPcodeValueError::InvalidByte { index }) => {
                            println!("Input {instr_input} = Symbolic value @ {index}")
                        }
                    }
                }
                    */
            }
        };

        let result = self.inner.emulate(memory, instruction)?;

        /*
        match &instruction.op_code {
            OpCode::Store => println!("Store"),
            OpCode::Branch
            | OpCode::BranchIndirect
            | OpCode::BranchConditional
            | OpCode::Call
            | OpCode::CallIndirect
            | OpCode::Return => {
                println!("Branch: {result:?}")
            }
            _ => {
                let output_result = memory.read(instruction.output.as_ref().unwrap()).unwrap();
                let output =
                    <<T as VarnodeDataStore>::Value as TryInto<u64>>::try_into(output_result);
                if let Ok(output) = output {
                    println!(
                        "Output: {output:0width$x}",
                        width = 2 * instruction.output.as_ref().unwrap().size
                    );
                } else {
                    println!("Output: Symbolic");
                }
            }
        };
        */

        *self
            .executed_instructions
            .borrow_mut()
            .entry(instruction.op_code)
            .or_default() += 1;
        Ok(result)
    }
}

impl TracingEmulator {
    pub fn new(inner: StandardPcodeEmulator) -> Self {
        Self {
            inner,
            executed_instructions: Default::default(),
        }
    }

    pub fn executed_instructions(&self) -> BTreeMap<OpCode, usize> {
        self.executed_instructions
            .borrow()
            .iter()
            .map(|(&op, &count)| (op, count))
            .collect()
    }
}

fn compile_x86_64_slaspec() -> sleigh_compiler::Result<PathBuf> {
    let sla_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("x86-64.sla");

    let mut compiler = sleigh_compiler::SleighCompiler::default();
    let slaspec_path =
        std::path::Path::new("ghidra/Ghidra/Processors/x86/data/languages/x86-64.slaspec");
    compiler.compile(slaspec_path, &sla_path)?;
    Ok(sla_path)
}

fn compile_aarch64_slaspec() -> sleigh_compiler::Result<PathBuf> {
    let sla_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("aarch64.sla");

    let mut compiler = sleigh_compiler::SleighCompiler::default();
    let slaspec_path =
        std::path::Path::new("ghidra/Ghidra/Processors/AARCH64/data/languages/AARCH64.slaspec");
    compiler.compile(slaspec_path, &sla_path)?;
    Ok(sla_path)
}

pub fn x86_64_sleigh() -> libsla::Result<GhidraSleigh> {
    let sleigh_spec = X86_64_SLA
        .get_or_init(|| compile_x86_64_slaspec().expect("failed to compile x86-64.slaspec"));
    let processor_spec =
        fs::read_to_string("ghidra/Ghidra/Processors/x86/data/languages/x86-64.pspec")
            .expect("Failed to read processor spec file");
    let sleigh = GhidraSleigh::builder()
        .sleigh_spec(sleigh_spec)?
        .processor_spec(&processor_spec)?
        .build()?;
    Ok(sleigh)
}

pub fn aarch64_sleigh() -> libsla::Result<GhidraSleigh> {
    let sleigh_spec = AARCH_64_SLA
        .get_or_init(|| compile_aarch64_slaspec().expect("failed to compile aarch64.slaspec"));
    let processor_spec =
        fs::read_to_string("ghidra/Ghidra/Processors/AARCH64/data/languages/AARCH64.pspec")
            .expect("Failed to read processor spec file");
    let sleigh = GhidraSleigh::builder()
        .sleigh_spec(sleigh_spec)?
        .processor_spec(&processor_spec)?
        .build()?;
    Ok(sleigh)
}

pub fn memory_with_image(
    sleigh: &GhidraSleigh,
    image: impl AsRef<Path>,
    pc_register: &VarnodeData,
) -> Memory {
    use elf::ElfBytes;
    use elf::abi::PT_LOAD;
    use elf::endian::AnyEndian;

    let mut memory = Memory::default();

    // Write image into memory
    let data = std::fs::read(image).expect("failed to read image file");

    let elf = ElfBytes::<AnyEndian>::minimal_parse(&data).expect("failed to parse elf");
    for segment in elf.segments().expect("failed to get segments") {
        if segment.p_type != PT_LOAD {
            // Not a loadable segment
            continue;
        }

        let data_location = VarnodeData {
            address: Address {
                offset: segment.p_vaddr,
                address_space: sleigh.default_code_space(),
            },

            // The initial set of data is from the file
            size: segment.p_filesz as usize,
        };

        // Write the segment from the file into memory
        let offset = segment.p_offset as usize;
        let file_size = segment.p_filesz as usize;
        memory
            .write(
                &data_location,
                data[offset..offset + file_size].iter().copied().collect(),
            )
            .expect("failed to write image section into memory");

        println!(
            "Loaded segment {vaddr:#x} from {offset:#x} ({file_size})",
            vaddr = segment.p_vaddr
        );

        // If the virtual size is larger than the file size then zero out the remainder
        let zero_size = (segment.p_memsz - segment.p_filesz) as usize;
        if zero_size > 0 {
            let zeros = vec![0u8; zero_size];
            let zeros_location = VarnodeData {
                address: Address {
                    offset: segment.p_vaddr + segment.p_filesz,
                    address_space: sleigh.default_code_space(),
                },
                size: zero_size,
            };
            println!(
                "Zeroing segment {vaddr:#x} at {zeros_location}",
                vaddr = segment.p_vaddr
            );
            memory
                .write(&zeros_location, zeros.into_iter().collect())
                .expect("failed to write zeros into memory");
        }
    }

    // Init RIP to entry
    memory
        .write(pc_register, elf.ehdr.e_entry.into())
        .expect("failed to initialize PC register");

    memory
}

pub fn init_registers_x86_64(sleigh: &impl Sleigh, memory: &mut Memory) {
    let rsp = sleigh.register_from_name("RSP").expect("invalid register");
    let registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP"]
        .into_iter()
        .map(str::to_owned)
        .chain((8..16).map(|n| format!("R{n}")));

    init_registers(sleigh, memory, &rsp, registers);

    // Initialize the DF register to 0. It appears to be a convention that whenever STD is called
    // CLD is called thereafter to reset it. There is a bug (?) in musl where REP STOS is used
    // without ensuring the flag is cleared

    // 0000000000449f4d <__init_libc>:
    //   449f4d:       53                      push   rbx
    //   449f4e:       48 89 fa                mov    rdx,rdi
    //   449f51:       31 c0                   xor    eax,eax
    //   449f53:       b9 4c 00 00 00          mov    ecx,0x4c
    //   449f58:       48 81 ec 50 01 00 00    sub    rsp,0x150
    //   449f5f:       48 8d 7c 24 20          lea    rdi,[rsp+0x20]
    //   449f64:       f3 ab                   rep stos DWORD PTR es:[rdi],eax

    let df_register = sleigh
        .register_from_name("DF")
        .expect("failed to get DF register");
    memory
        .write(&df_register, 0u8.into())
        .expect("failed to init DF register");
}

pub fn init_registers_aarch64(sleigh: &impl Sleigh, memory: &mut Memory) {
    let sp = sleigh
        .register_from_name("sp")
        .expect("invalid stack register");
    let registers = (0..30).map(|n| format!("x{n}"));

    init_registers(sleigh, memory, &sp, registers);

    // Init link register value to final return address
    let link_register = sleigh
        .register_from_name("x30")
        .expect("invalid link register");
    memory
        .write(&link_register, EXIT_IP_ADDR.into())
        .expect("failed to initialize link register");

    // Initialize system register `dczid_el0`
    //
    // Per DDI0487L_b_a-profile_architecture_reference_manual.pdf
    //
    // Indicates the block size that is written with byte values of 0 by the DC ZVA (Data Cache
    // Zero by Address) System instruction.
    //
    // - Bits [63:5] Reserved, RES0.
    // - Bit [4] Data Zero Prohibited (DZP). This field indicates whether the use of `DC ZVA`
    // instruction is permitted (0b0) or prohibited (0b1).
    // - Bits [3:0] Block Size (BS). Log2 of the block size in words.
    //
    // Writing 0x10 to prohibit DC ZVA. This instruction is not supported directly in pcode and
    // would need to be modeled.
    let dczid_el0 = sleigh
        .register_from_name("dczid_el0")
        .expect("unknown register");
    memory
        .write(&dczid_el0, 0x10u64.into())
        .expect("failed to initialize dczid_el0");
}

fn init_registers(
    sleigh: &impl Sleigh,
    memory: &mut Memory,
    stack_register: &VarnodeData,
    registers: impl IntoIterator<Item = String>,
) {
    let mut bitvar = 0;
    for register_name in registers.into_iter() {
        let mut bytes = Vec::with_capacity(8);
        for _ in 0..8 {
            let byte: SymbolicByte = [
                SymbolicBit::Variable(bitvar),
                SymbolicBit::Variable(bitvar + 1),
                SymbolicBit::Variable(bitvar + 2),
                SymbolicBit::Variable(bitvar + 3),
                SymbolicBit::Variable(bitvar + 4),
                SymbolicBit::Variable(bitvar + 5),
                SymbolicBit::Variable(bitvar + 6),
                SymbolicBit::Variable(bitvar + 7),
            ]
            .into();
            bytes.push(byte);
            bitvar += 8;
        }

        let register = sleigh
            .register_from_name(&register_name)
            .unwrap_or_else(|err| panic!("invalid register {register_name}: {err}"));
        memory
            .write(&register, bytes.into_iter().collect())
            .unwrap_or_else(|err| panic!("failed to write register {register_name}: {err}"));
    }

    // Init stack register to stack address
    memory
        .write(stack_register, INITIAL_STACK.into())
        .expect("failed to initialize stack register");
}

pub fn find_test_fixture(
    messages: escargot::CommandMessages,
) -> Result<std::path::PathBuf, Vec<escargot::error::CargoResult<escargot::Message>>> {
    let messages: Vec<_> = messages.into_iter().collect();
    let executable_path = messages.iter().find_map(|result| {
        result
            .as_ref()
            .ok()
            .into_iter()
            .filter_map(|msg| {
                msg.decode()
                    .ok()
                    .into_iter()
                    .filter_map(|msg| match msg {
                        escargot::format::Message::CompilerArtifact(artifact) => {
                            artifact.executable.map(|p| p.into_owned())
                        }
                        _ => None,
                    })
                    .next()
            })
            .next()
    });

    executable_path.ok_or(messages)
}
